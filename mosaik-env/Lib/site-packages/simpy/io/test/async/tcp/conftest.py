import pytest

from simpy.io import asyncio
from simpy.io.network import Protocol
from simpy.io.packet import PacketUTF8


@pytest.fixture()
def env(request):
    env = asyncio.Environment()
    request.addfinalizer(env.close)
    return env


@pytest.fixture()
def socket_type(env, request):
    return asyncio.TCPSocket

@pytest.fixture()
def protocol(env, request):
    return Protocol(env, PacketUTF8, asyncio.TCPSocket)

