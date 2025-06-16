"""
Mosaik API for simulations written in Python.

"""
from __future__ import annotations

import abc
import asyncio
import inspect
import re
import socket
import sys
import traceback
from typing import Any, Dict, List, Optional, Union

import docopt
from loguru import logger

from .connection import Channel, RemoteException
from .types import (
    Attr,
    CreateResult,
    CreateResultChild,
    EntityGraph,
    EntitySpec,
    FullId,
    InputData,
    Meta,
    ModelName,
    OutputData,
    OutputRequest,
    SimId,
    Time,
)

__version__ = '3.0.13'
__api_version__ = __version__
__all__ = [
    "__version__",
    "__api_version__",
    "Simulator",
    "start_simulation",
    "MosaikProxy",
    "Attr",
    "CreateResult",
    "CreateResultChild",
    "EntityGraph",
    "EntitySpec",
    "FullId",
    "InputData",
    "OutputRequest",
    "OutputData",
    "Meta",
    "ModelName",
    "SimId",
    "Time",
]


_HELP = """%(desc)s

Usage:
    %(prog)s [options] HOST:PORT

Options:
    HOST:PORT   Connect to this address
    -l LEVEL, --log-level LEVEL
                Log level for simulator (%(levels)s)
    -r, --remote
                Simulator is to be started on a machine remote from mosaik
    -t TIME, --timeout TIME
                Timeout in seconds for mosaik handshake [default: 60]
%(extra_opts)s
"""
_LOG_LEVELS = {
    'trace': 'TRACE',
    'debug': 'DEBUG',
    'info': 'INFO',
    'warning': 'WARNING',
    'error': 'ERROR',
    'critical': 'CRITICAL',
}


logger.disable(__name__)


# NOTE: We don't use an ABC here, because the effort of making it py2 AND py3
# compatible (-> meta classes) outweighs the benefits.
class Simulator(object):
    """This is the base class that you need to inherit from and implement the
    API calls."""

    meta: Meta
    """Meta data describing the simulator (the same that is returned by
    :meth:`init()`).

    ::

        {
            'api_version': 'x.y',
            'type': 'time-based'|'event-based'|'hybrid',
            'models': {
                'ModelName': {
                    'public': True|False,
                    'params': ['param_1', ...],
                    'attrs': ['attr_1', ...],
                    'any_inputs': True|False,
                    'trigger': ['attr_1', ...],
                    'non-persistent': ['attr_2', ...],
                },
                ...
            },
            'extra_methods': [
                'do_cool_stuff',
                'set_static_data'
            ]
        }

    The *api_version* is a string that defines which version of the mosaik API
    the simulator implements.  Since mosaik API version 2.3, the simulator's
    `major version <http://semver.org/>`_ ("x", in the snippet above) has to be
    equal to mosaik's.  Mosaik will cancel the simulation if a version mismatch
    occurs.

    The *type* defines how the simulator is advanced through time and whether
    its attributes are persistent in time or transient.

    *models* is a dictionary describing the models provided by this simulator.
    The entry *public* determines whether a model can be instantiated by a user
    (``True``) or if it is a sub-model that cannot be created directly
    (``False``). *params* is a list of parameter names that can be passed to
    the model when creating it. *attrs* is a list of attribute names that can
    be accessed (reading or writing).  If the optional *any_inputs* flag is set
    to ``true``, any attributes can be connected to the model, even if they are
    not *attrs*. This may, for example, be useful for databases that don't know
    in advance which attributes of an entity they'll receive. *trigger* is a
    list of attribute names that cause the simulator to be stepped when
    another simulator provides output which is connected to one of those.


    *extra_methods* is an optional list of methods that a simulator provides in
    addition to the standard API calls (``init()``, ``create()`` and so on).
    These methods can be called while the scenario is being created and can be
    used for operations that don't really belong into ``init()`` or
    ``create()``.

    """

    mosaik: MosaikProxy = None  # type: ignore # Will be set by "start_simulation()"
    # TODO: Maybe introduce abstract MosaikProxy?
    """An RPC proxy to mosaik."""

    def __init__(self, meta):
        self.meta = {
            'api_version': __api_version__,
            'models': {},
            'extra_methods': [],
            'type': 'time-based',
        }
        self.meta.update(meta)

    def init(self, sid: SimId, time_resolution: float = 1., **sim_params) -> Meta:
        """Initialize the simulator with the ID *sid* and pass the
         *time_resolution* and additional parameters *(sim_params)*
         sent by mosaik. Return the meta data :attr:`meta`.

        If your simulator has no *sim_params*, you don't need to override this
        method.

        """
        return self.meta

    def create(self, num: int, model: ModelName, **model_params) -> List[CreateResult]:
        """Create *num* instances of *model* using the provided *model_params*.

        *num* is an integer for the number of model instances to create.

        *model* needs to be a public entry in the simulator's
        ``meta['models']``.

        *model_params* is a dictionary mapping parameters (from
        ``meta['models'][model]['params']``) to their values.

        Return a (nested) list of dictionaries describing the created model
        instances (entities). The root list must contain exactly *num*
        elements. The number of objects in sub-lists is not constrained::

            [
                {
                    'eid': 'eid_1',
                    'type': 'model_name',
                    'rel': ['eid_2', ...],
                    'children': [
                        {'eid': 'child_1', 'type': 'child'},
                        ...
                    ],
                },
                ...
            ]

        The entity ID (*eid*) of an object must be unique within a simulator
        instance. For entities in the root list, *type* must be the same as the
        *model* parameter. The type for objects in sub-lists may be anything
        that can be found in ``meta['models']``. *rel* is an optional list of
        related entities; "related" means that two entities are somehow connect
        within the simulator, either logically or via a real data-flow (e.g.,
        grid nodes are related to their adjacent branches). The *children*
        entry is optional and may contain a sub-list of entities.

        """
        raise NotImplementedError

    def setup_done(self) -> None:
        """Callback that indicates that the scenario setup is done and the
        actual simulation is about to start.

        At this point, all entities and all connections between them are know
        but no simulator has been stepped yet.

        Implementing this method is optional.

        *Added in mosaik API version 2.3*

        """

    def step(self, time: Time, inputs: InputData, max_advance: Time) -> Optional[Time]:
        """Perform the next simulation step from time *time* using input values
        from *inputs* and return the new simulation time (the time at which
        ``step()`` should be called again).

        *time* and the time returned are integers. Their unit is arbitrary,
        e.g. *seconds* (from simulation start), but has to be consistent among
        all simulators used in a simulation.

        *inputs* is a dict of dicts mapping entity IDs to attributes and
        dicts of values (each simulator has to decide on its own how to reduce
        the values (e.g., as its sum, average or maximum)::

            {
                'dest_eid': {
                    'attr': {'src_fullid': val, ...},
                    ...
                },
                ...
            }

        *max_advance* tells the simulator how far it can advance its time
        without risking any causality error, i.e. it is guaranteed that no
        external step will be triggered before max_advance + 1, unless the
        simulator activates an output loop earlier than that. For time-based
        simulators (or hybrid ones without any triggering input) *max_advance*
        is always equal to the end of the simulation (*until*).
        """
        raise NotImplementedError

    def get_data(self, outputs: OutputRequest) -> OutputData:
        """Return the data for the requested attributes in *outputs*

        *outputs* is a dict mapping entity IDs to lists of attribute names
        whose values are requested::

            {
                'eid_1': ['attr_1', 'attr_2', ...],
                ...
            }

        The return value needs to be a dict of dicts mapping entity IDs and
        attribute names to their values::

            {
                'eid_1: {
                    'attr_1': 'val_1',
                    'attr_2': 'val_2',
                    ...
                },
                ...
                'time': output_time (for event-based sims, optional)
            }

        Time-based simulators have set an entry for all requested attributes,
        whereas for event-based and hybrid simulators this is optional (e.g.
        if there's no new event).
        Event-based and hybrid simulators can optionally set a timing of their
        non-persistent output attributes via a *time* entry, which is valid
        for all given (non-persistent) attributes. If not given, it defaults
        to the current time of the step. Thus only one output time is possible
        per step. For further output times the simulator has to schedule
        another self-step (via the step's return value).

        """
        raise NotImplementedError

    def configure(self, args):
        """This method can be overridden to configure the simulation with the
        command line *args* as created by `docopt <http://docopt.org/>`_.

        *backend* and *env* are the *simpy.io* backend and environment used
        for networking. You can use them to start extra processes (e.g., a
        web server).

        The default implementation simply ignores them.

        """
        pass

    def event_setter(self):
        """This method can be overridden to allow the simulator to
        asynchronously set events for itself at time step event_time via the
        set_event api call::
            yield self.mosaik.set_event(event_time)

        *env* is the *simpy.io* environment.

        The default implementation does nothing by just yielding a triggered
        event.

        """
        pass

    def finalize(self):
        """This method can be overridden to do some clean-up operations after
        the simulation finished (e.g., shutting down external processes).

        """
        pass


class MosaikProxy(abc.ABC):
    @abc.abstractmethod
    async def get_progress(self) -> float:
        pass

    @abc.abstractmethod
    async def get_related_entities(self, entities: Union[None, FullId, List[FullId]]) -> Union[EntityGraph, Dict[FullId, EntitySpec], Dict[FullId, Dict[FullId, EntitySpec]]]:
        pass

    @abc.abstractmethod
    async def get_data(self, attrs: Dict[FullId, Attr]) -> Dict[FullId, Dict[Attr, Any]]:
        pass

    @abc.abstractmethod
    async def set_data(self, data: Dict[FullId, Dict[FullId, Dict[Attr, Any]]]) -> None:
        pass

    @abc.abstractmethod
    async def set_event(self, event_time: Time) -> None:
        pass


class RemoteMosaikProxy(MosaikProxy):
    def __init__(self, channel: Channel):
        self._channel = channel

    async def get_progress(self) -> float:
        return await self._channel.send(["get_progress", [], {}])

    async def get_related_entities(self, entities: Union[None, FullId, List[FullId]]) -> Union[EntityGraph, Dict[FullId, EntitySpec], Dict[FullId, Dict[FullId, EntitySpec]]]:
        return await self._channel.send(["get_related_entities", [entities], {}])

    async def get_data(self, attrs: Dict[FullId, Attr]) -> Dict[FullId, Dict[Attr, Any]]:
        return await self._channel.send(["get_data", [attrs], {}])

    async def set_data(self, data: Dict[FullId, Dict[FullId, Dict[Attr, Any]]]) -> None:
        return await self._channel.send(["set_data", [data], {}])

    async def set_event(self, event_time: Time) -> None:
        return await self._channel.send(["set_event", [event_time], {}])


def start_simulation(
    simulator: Simulator,
    description: str = '',
    extra_options=None,
    configure_logging: bool = True,
) -> int:
    """Start the simulation process for ``simulator``.

    *simulator* is the instance of your API implementation (see
        :class:`Simulator`).

    *description* may override the default description printed with the help on
    the command line.

    *extra_option* may be a list of options for `docopt <http://docopt.org/>`_
    (example: ``['-e, --example     Enable example mode']``). Commandline
    arguments are passed to :meth:`Simulator.configure()` so that your API
    implementation can handle them.

    *configure_logging* determines whether the API sets up the loguru
    logger. If the user specifies the log level using the --log-level
    command line flag, logging will always be configured.
    """
    return asyncio.run(start_simulation_async(simulator, description, extra_options, configure_logging=configure_logging))

async def start_simulation_async(
    simulator: Simulator,
    description: str = '',
    extra_options=None,
    configure_logging: bool = False,
) -> int:
    OK, ERR = 0, 1

    args = _parse_args(
        description or 'Start the simulation service.',
        extra_options or [],
    )

    log_level = args.get("--log-level")
    if configure_logging or log_level:
        logger.enable(__name__)
        logger.remove()
        logger.add(sys.stderr, level=_LOG_LEVELS.get(log_level, "INFO"))
    remote_flag = args['--remote'] if '--remote' in args.keys() else False
    sim_name = simulator.__class__.__name__

    # Check if simulator has implemented *time_resolution* and *max_advance*:
    global api_compliant
    api_compliant = check_api_compliance(simulator)

    writer = None

    try:
        logger.info('Starting %s ...' % sim_name)
        simulator.configure(args)

        # Setup simpy.io and start the event loop.
        host, port = _parse_addr(args['HOST:PORT'])
        # Interception for remote simulators
        if remote_flag:
            incoming_connection: asyncio.Future[Channel] = asyncio.Future()
            async def connected_cb(reader, writer):
                incoming_connection.set_result(Channel(reader, writer))
            server = await asyncio.start_server(connected_cb, host, port)
            start_timeout = int(args['--timeout'])
            logger.info("Waiting for connection from mosaik")
            try:
                channel = await asyncio.wait_for(
                    incoming_connection,
                    start_timeout,
                )
            except asyncio.TimeoutError:
                raise RuntimeError('Connection from mosaik not received in time')
            finally:
                server.close()
        else:
            reader, writer = await asyncio.open_connection(host, port)
            channel = Channel(reader, writer)
        simulator.mosaik = RemoteMosaikProxy(channel)
        await init(channel, simulator)
        asyncio.create_task(get_wrapper(simulator.event_setter)())
        await run(channel, simulator)
    except ConnectionRefusedError:
        logger.error('Could not connect to mosaik.')
        errstr = 'INFO:mosaik_api_v3:Starting ExampleSim ...\n' + 'ERROR:mosaik_api_v3:Could not connect to mosaik.\n'
        return errstr
    except (ConnectionError, KeyboardInterrupt):
        pass  # Exit silently.
    except Exception as exc:
        if type(exc) is OSError and exc.errno == 10057:
            # ConnectionRefusedError in Windows O.o
            logger.error('Could not connect to mosaik.')
            return ERR

        print('Error in %s:' % sim_name)
        traceback.print_exc()  # Exit loudly
        print('---------%s-' % ('-' * len(sim_name)))
        return ERR
    finally:
        await get_wrapper(simulator.finalize)()
        if writer is not None:
            writer.close()
            await writer.wait_closed()

    return OK


def check_api_compliance(simulator):
    """Checks for compliance with API 3:
    i.e. if meta contains 'type' and if the new parameters,
    namely *time_resolution* for the init method and *max_advance* for step
    are implemented.

    """
    compliant = True
    sim_name = simulator.__class__.__name__

    for func_name, param in [('init', 'time_resolution'),
                             ('step', 'max_advance')]:
        func = getattr(simulator, func_name)
        signature = inspect.signature(func)
        for isig in signature.parameters.values():
            if isig.name == param or str(isig.kind) == 'VAR_KEYWORD':
                break
        else:
            compliant = False
            print("DEPRECATION WARNING: '%s' is not implemented as argument "
                  "of %s's %s function. This might cause an error in "
                  "future API versions." % (param, sim_name, func_name))
    return compliant


async def init(channel: Channel, sim) -> None:
    init_func = get_wrapper(sim.init)
    request = await channel.next_request()
    func, args, kwargs = request.content
    logger.debug('Calling %s(*%s, **%s)' % (func, args, kwargs))
    assert func == 'init'
    sim.time_resolution = kwargs['time_resolution']
    if not api_compliant:
        kwargs.pop('time_resolution')
    ret = await init_func(*args, **kwargs)
    await request.set_result(ret)


async def run(channel: Channel, sim):
    """Main simulator process. Send a greeting message to mosaik and wait
    for requests to step the simulation, get data or whatever.

    *channel* is a :class:`simpy.io.message.Message` instance.

    *sim* is the instance of an :class:`Simulator` implementation.

    """
    funcs = {
        'init': sim.init,
        'create': sim.create,
        'setup_done': sim.setup_done,
        'step': sim.step,
        'get_data': sim.get_data,
    }
    extra_funcs = {
        name: getattr(sim, name) for name in sim.meta.get('extra_methods', [])
    }
    funcs.update(extra_funcs)
    for name, func in funcs.items():
        funcs[name] = get_wrapper(func)

    logger.debug('Entering event loop ...')
    while True:
        try:
            request = await channel.next_request()
        except asyncio.IncompleteReadError:
            logger.info("mosaik closed its connection unexpectedly")
            break
        func, args, kwargs = request.content
        logger.debug('Calling %s(*%s, **%s)' % (func, args, kwargs))
        if func == 'stop':
            break

        try:
            func = funcs[func]
            ret = await func(*args, **kwargs)
            await request.set_result(ret)
        except Exception as e:
            await request.set_exception(e)


def get_wrapper(func):
    if inspect.isgeneratorfunction(func):
        async def wrapper(*args, **kwargs):
            gen = func(*args, **kwargs)
            try:
                # Requests from the simulator to mosaik are encoded as
                # coroutines that the simulator yields (not awaits!).
                # So we will have to do the awaiting.
                # (This is due to the way things worked in simpy to
                # avoid breaking existing simulators.)
                request = next(gen)
                while True:
                    try:
                        # Awaiting the request gives the result or
                        # raises an exception.
                        remote_result = await request
                    except RemoteException as remote_exception:
                        request = gen.throw(remote_exception)
                    else:
                        request = gen.send(remote_result)
            except StopIteration as stop:
                return stop.value
    else:
        async def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    return wrapper


def _parse_args(desc, extra_options):
    """Fill-in the values into :data:`_HELP` and parse and return the arguments
    using :func:`~docopt.docopt()`.

    """
    msg = _HELP % {
        'desc': desc,
        'prog': sys.argv[0],
        'levels': ', '.join(_LOG_LEVELS.keys()),
        'extra_opts': '\n'.join('    %s' % opt
                                for opt in extra_options),
    }
    args = docopt.docopt(msg)
    args['--timeout'] = args.get('--timeout', 60)
    return args


def _parse_addr(addr):
    """Parse ``addr`` and returns a ``('host', port)`` tuple.

    If the host does not look like an IP(v4) address, resolve its name
    to an IP address.

    Raise a :exc:`ValueError` if resolving the hostname fails or if the
    address contains no host or port.

    """
    try:
        host, port = addr.strip().split(':')
        # Resolve hostname if it doesn't look like an IP address
        if not re.match(r'^(\d{1,3}\.){3}\d{1,3}$', host):
            host = socket.gethostbyname(host)
        addr = (host, int(port))
        return addr

    except (ValueError):
        raise ValueError('Error parsing "%s"' % addr)

    except (IOError, OSError):
        raise ValueError('Could not resolve "%s"' % addr[0])
