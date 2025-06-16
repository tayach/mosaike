"""This module implements the basic communication protocol for mosaik.
"""

from __future__ import annotations

import asyncio
from asyncio import StreamReader, StreamWriter
import itertools
from json import JSONDecoder, JSONEncoder
import sys
import traceback
from typing import Any, Dict, Iterator, Optional, Tuple, Union
import weakref


REQUEST = 0
SUCCESS = 1
FAILURE = 2


_encoder = JSONEncoder()


def encode(obj: Any) -> bytes:
    """Encode an object as expected by mosaik communicaction protocol.

    The object is first converted to a JSON string. This string is
    prepended with its length as a 4-byte integer in big endian order.
    """
    # JSONEncoder encodes to string, then encode that string into bytes
    obj_bytes = _encoder.encode(obj).encode()
    return len(obj_bytes).to_bytes(4, "big") + obj_bytes


_decoder = JSONDecoder()


async def decode(reader: asyncio.StreamReader) -> Any:
    """Decode the next object from the StreamReader, in the format
    produced by `encode`.

    If the connection is closed before the message is complete, the
    asyncio.IncompleteReadError is passed on."""
    msg_length_bytes = await reader.readexactly(4)
    msg_length = int.from_bytes(msg_length_bytes, "big")
    msg_bytes = await reader.readexactly(msg_length)
    return _decoder.decode(msg_bytes.decode())


class RemoteException(Exception):
    """An exception due to a failure from a remote connected through a
    Channel."""

    remote_type: str
    """The exception type reported by the remote."""
    remote_msg: str
    """The message from the remote."""
    further_args: Tuple[str, ...]
    """Additional information from the remote, like a stack trace."""
    source: Optional[str]
    """The source of the exception."""

    def __init__(
        self,
        remote_type: str,
        remote_msg: str,
        *further_args: str,
        source: Optional[str] = None,
    ):
        super().__init__(remote_type, remote_msg, *further_args, source)
        self.remote_type = remote_type
        self.remote_msg = remote_msg
        self.further_args = further_args
        self.source = source


class Request:
    """An incoming request from a channel.

    Respond to the request by calling set_result (if successful) or
    set_exception (if unsuccessful)."""

    _channel: weakref.ref[Channel]
    _msg_id: int
    content: Any

    def __init__(self, channel: Channel, msg_id: int, content: Any):
        self._channel = weakref.ref(channel)
        self._msg_id = msg_id
        self.content = content

    async def set_result(self, result: Any):
        """Respond to the request with a successful result."""
        channel = self._channel()
        if channel:
            await channel._write_success(self._msg_id, result)

    async def set_exception(self, exception: Exception):
        """Respond to the request with a failure."""
        channel = self._channel()
        if channel:
            await channel._write_failure(
                self._msg_id, 
                [
                    type(exception).__name__,
                    str(exception),
                    *traceback.format_exception(*sys.exc_info())
                ],
            )


class EndOfRequests(Exception):
    pass


class Channel:
    """A Channel for mosaik-style request-response communication.

    To send out a request, call send and await the result.

    To react to incoming requests, get the next incoming request from
    next_request as a Request object and respond to it using the
    object's set_result and set_exception methods."""

    _msg_counter: Iterator[int]
    _outgoing_request_futures: Dict[int, asyncio.Future[Any]]
    _incoming_requests: asyncio.Queue[Union[Request, EndOfRequests]]
    _reader: asyncio.StreamReader
    _writer: asyncio.StreamWriter
    _receiver_task: asyncio.Task[None]
    _name: Optional[str]

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        name: Optional[str] = None,
    ):
        """Initialize a new channel.

        This method sets up a listener task for incoming requests and
        can therefore only be called while an asyncio loop is running.

        (reader, writer) should be a pair returned by
        asyncio.open_connection or received by an asycnio.start_server
        callback.
        """
        self._reader = reader
        self._writer = writer
        self._msg_counter = itertools.count()
        self._outgoing_request_futures = {}
        self._incoming_requests = asyncio.Queue()
        self._name = name
        try:
            self._receiver_task = asyncio.get_running_loop().create_task(
                self._receive_forever(),
                name=f"Reader task for {name}",
            )
        except RuntimeError as e:
            raise RuntimeError(
                "can only create Channel while event loop is running"
            ) from e

    async def _receive_forever(self):
        try:
            while True:
                msg_type, msg_id, content = await decode(self._reader)
                if msg_type == REQUEST:
                    await self._incoming_requests.put(Request(self, msg_id, content))
                elif msg_type == SUCCESS:
                    self._outgoing_request_futures.pop(msg_id).set_result(content)
                elif msg_type == FAILURE:
                    self._outgoing_request_futures.pop(msg_id).set_exception(
                        RemoteException(*content, source=self._name)
                    )
        except asyncio.IncompleteReadError as e:
            for future in self._outgoing_request_futures.values():
                if not future.cancelled():
                    future.set_exception(e)
            self._incoming_requests.put_nowait(EndOfRequests())

    async def send(self, value: Any) -> Any:
        """Sends a request over the channel and waits for the response.

        On a successful response, its value is returned.

        On an unsuccessful respose, the error is raised as a
        RemoteException.

        If the connection is closed while waiting for the response,
        an asyncio.IncompleteReadError is raised.
        """
        msg_id = next(self._msg_counter)
        result_future: asyncio.Future[Any] = asyncio.Future()
        self._outgoing_request_futures[msg_id] = result_future
        self._writer.write(encode([REQUEST, msg_id, value]))
        await self._writer.drain()
        return await result_future

    async def next_request(self) -> Request:
        """Get the next incoming request as a Request object.

        If the connection is closed while waiting for a request,
        an asyncio.IncompleteReadError is raised.
        """
        request_or_error = await self._incoming_requests.get()
        if isinstance(request_or_error, Request):
            return request_or_error
        else:
            raise request_or_error

    async def _write_success(self, msg_id: int, value: Any):
        """Write the given value to the channel as a successful response
        to the request msg_id.
        """
        self._writer.write(encode([SUCCESS, msg_id, value]))
        await self._writer.drain()

    async def _write_failure(self, msg_id: int, value: Any):
        """Write the given value to the channel as a failure response
        to the request msg_id."""
        self._writer.write(encode([FAILURE, msg_id, value]))
        await self._writer.drain()

    async def close(self):
        """Close the connection."""
        self._writer.close()
        await self._writer.wait_closed()
        self._receiver_task.cancel()


async def single_channel(host: str, port: int) -> Channel:
    """Runs a server that will only accept a single connection and
    returns this connection as a Channel."""
    channel_future: asyncio.Future[Channel] = asyncio.Future()

    async def connected_cb(reader: StreamReader, writer: StreamWriter):
        channel_future.set_result(Channel(reader, writer))

    server = await asyncio.start_server(connected_cb, host, port)
    try:
        return await channel_future
    finally:
        server.close()
