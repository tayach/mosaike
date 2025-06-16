import os

import pytest

from simpy.io import asyncio
from simpy.io.network import Protocol
from simpy.io.packet import PacketUTF8
from simpy.io import test


testdir = os.path.dirname(test.__file__)


class DefaultSSLSocket(asyncio.SSLSocket):
    def __init__(self, env, sock=None, **kwargs):
        kwargs['keyfile'] = os.path.join(testdir, 'server.key')
        kwargs['certfile'] = os.path.join(testdir, 'cacert.pem')
        asyncio.SSLSocket.__init__(self, env, sock, **kwargs)


@pytest.fixture()
def env(request):
    env = asyncio.Environment()
    request.addfinalizer(env.close)
    return env


@pytest.fixture()
def socket_type(env, request):
    return DefaultSSLSocket


@pytest.fixture()
def protocol(env, request):
    return Protocol(env, PacketUTF8, DefaultSSLSocket)
