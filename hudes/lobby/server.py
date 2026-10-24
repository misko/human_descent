"""Standalone lobby composition. No Torch, protobuf, datasets or game server.

Run ``python -m hudes.lobby.server --port 18110 --fake-game`` for social testing;
inject a real GameControl into ``serve_lobby`` for an actual game deployment.
"""

import argparse
import asyncio
from contextlib import asynccontextmanager

from .service import LobbyService
from .websocket import LobbyWebSocketHandler


@asynccontextmanager
async def serve_lobby(
    game_control, *, host="127.0.0.1", port=0, service=None, **service_options
):
    from websockets.asyncio.server import serve

    lobby = service or LobbyService(game_control, **service_options)
    await lobby.start()
    try:
        async with serve(
            LobbyWebSocketHandler(lobby),
            host,
            port,
            max_size=lobby.policy.max_frame_bytes,
            max_queue=16,
            write_limit=32768,
            ping_interval=20,
            ping_timeout=20,
        ) as server:
            # Attach the service for embedders without hiding the real socket
            # server API (including OS-assigned ephemeral socket addresses).
            server.lobby_service = lobby
            yield server
    finally:
        await lobby.close()


async def _main(args):
    from .fake import FakeGameControl

    async with serve_lobby(FakeGameControl(), host=args.host, port=args.port) as server:
        print(
            f"Standalone lobby (fake game control) listening on {server.sockets[0].getsockname()}",
            flush=True,
        )
        await asyncio.Future()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18110)
    parser.add_argument(
        "--fake-game",
        action="store_true",
        required=True,
        help="A standalone demo requires explicit fake game control",
    )
    args = parser.parse_args()
    try:
        asyncio.run(_main(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
