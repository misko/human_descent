"""Real lobby/game transports with deterministic inference for browser tests.

This executable is never used by the production entry point. All admission,
ownership, votes, timing, and lifecycle behavior use the application services.
Only numerical inference is substituted.
"""

import argparse
import asyncio
import json
import math
import signal

from hudes.game import TeamGameService
from hudes.game.websocket import TeamWebSocketHandler
from hudes.game_control import MatchPreset
from hudes.lobby import LobbyPolicy, LobbyService, LobbyWebSocketHandler
from websockets.asyncio.server import serve

from tests.multiplayer_support import DeterministicEngine


async def serve_test_backend(port, health_port, duration=3.0):
    engine = DeterministicEngine()
    game = TeamGameService(engine)
    lobby = LobbyService(
        game,
        preset=MatchPreset(
            duration_seconds=duration,
            countdown_seconds=0.25,
            mesh_grid_size=7,
        ),
        policy=LobbyPolicy(maintenance_seconds=0.05),
    )
    lobby_handler = LobbyWebSocketHandler(lobby)
    game_handler = TeamWebSocketHandler(game)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    async def dispatch(websocket):
        endpoint = websocket.request.path.split("?", 1)[0]
        if endpoint == "/lobby/ws":
            await lobby_handler(websocket)
        elif endpoint == "/ws":
            await game_handler(websocket)
        else:
            await websocket.close(code=1008, reason="unknown endpoint")

    async def http(reader, writer):
        try:
            request = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 5)
            endpoint = request.split(b" ", 2)[1]
            if endpoint == b"/health":
                payload = {"ok": True, "engine": "deterministic-test"}
                status = "200 OK"
            elif endpoint == b"/__test__/state":
                payload = {
                    "sessions": engine.sessions,
                    "jobs": engine.jobs,
                    "matches": {
                        key: value.phase for key, value in game.matches.items()
                    },
                }
                status = "200 OK"
            else:
                payload, status = {"error": "not found"}, "404 Not Found"
            body = json.dumps(payload).encode()
            writer.write(
                f"HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
                + body
            )
            await writer.drain()
        except (asyncio.IncompleteReadError, TimeoutError):
            pass
        finally:
            writer.close()
            await writer.wait_closed()

    await lobby.start()
    try:
        async with serve(dispatch, "0.0.0.0", port, max_size=1024 * 1024):
            async with await asyncio.start_server(http, "0.0.0.0", health_port):
                print(
                    f"Multiplayer test backend ready on {port}/{health_port}",
                    flush=True,
                )
                await stop.wait()
    finally:
        await lobby.close()
        await game.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--health-port", type=int, required=True)
    parser.add_argument("--duration", type=float, default=3.0)
    args = parser.parse_args()
    if any(
        port in {10000, 10001} or not 1024 <= port <= 65535
        for port in (args.port, args.health_port)
    ):
        parser.error("Use isolated test ports outside the production listeners")
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("Test duration must be positive and finite")
    asyncio.run(serve_test_backend(args.port, args.health_port, args.duration))


if __name__ == "__main__":
    main()
