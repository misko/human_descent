import json
import os
import subprocess
import sys

import pytest

from hudes.lobby.protocol import LobbyError, decode_frame, validate_request


def command(**overrides):
    value = {"v": 1, "type": "snapshot", "requestId": "request-1", "payload": {}}
    value.update(overrides)
    return value


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {},
        command(v=True),
        command(v="1"),
        command(v=2),
        command(type="team.join"),
        command(type=[]),
        command(requestId=""),
        command(requestId="x" * 81),
        command(requestId=123),
        command(payload=[]),
        command(payload={"playerId": "spoofed"}),
        command(type="match.ready", payload={"matchId": "x", "ready": 1}),
        command(type="match.ready", payload={"matchId": "x", "ready": "false"}),
        command(type="hello", payload={"name": "x", "resumeToken": None}),
        command(type="team.create", payload={"name": ""}),
        command(type="team.invite", payload={"playerId": "x", "captainId": "spoofed"}),
    ],
)
def test_rejects_unversioned_unbounded_and_forged_requests(value):
    with pytest.raises(LobbyError) as error:
        validate_request(value)
    assert error.value.code == "invalid_request"


@pytest.mark.parametrize(
    "frame",
    [
        b"protobuf",
        "{",
        "[] NaN",
        '{"x":NaN}',
        '{"x":Infinity}',
        '{"x":1,"x":2}',
        '{"x":{"y":1,"y":2}}',
        "[" * 2000 + "]" * 2000,
    ],
    ids=lambda value: str(value)[:50],
)
def test_json_parser_rejects_binary_duplicate_keys_constants_and_deep_nesting(frame):
    with pytest.raises(LobbyError):
        decode_frame(frame)


def test_message_size_counts_utf8_bytes_and_canonical_envelope_works():
    frame = json.dumps(command(), separators=(",", ":"))
    assert validate_request(decode_frame(frame)) == ("snapshot", "request-1", {})
    with pytest.raises(LobbyError):
        decode_frame('"' + "é" * 40 + '"', max_bytes=50)


def test_lobby_import_and_standalone_server_do_not_import_torch_protobuf_or_models(
    tmp_path,
):
    # A fresh interpreter proves independence even when another test already
    # imported Torch. Block forbidden modules rather than checking only timing.
    source = """
import sys
class Forbidden:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torchvision', 'numpy', 'google'} or fullname in {'hudes.websocket_server', 'hudes.hudes_pb2', 'hudes.model_data_and_subspace'}:
            raise AssertionError('Forbidden dependency: ' + fullname)
sys.meta_path.insert(0, Forbidden())
import hudes.lobby
import hudes.lobby.server
import hudes.lobby.fake
import asyncio
from hudes.lobby.fake import FakeGameControl
from hudes.lobby.server import serve_lobby
async def startup():
    async with serve_lobby(FakeGameControl()) as server:
        assert server.sockets
asyncio.run(startup())
assert 'torch' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", source],
        cwd=tmp_path,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
