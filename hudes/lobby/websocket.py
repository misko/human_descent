"""Bounded JSON WebSocket transport, with no gameplay imports."""

import asyncio
import json
from contextlib import suppress

from .protocol import LobbyError, decode_frame, response


class LobbyWebSocketHandler:
    def __init__(self, service):
        self.service = service

    async def __call__(self, websocket, path=None):
        request = getattr(websocket, "request", None)
        path = (
            path or getattr(request, "path", None) or getattr(websocket, "path", None)
        )
        if path is not None and path.split("?", 1)[0] != "/lobby/ws":
            await websocket.close(code=1008, reason="Unknown lobby endpoint")
            return
        try:
            conn = await self.service.connect()
        except LobbyError as exc:
            await websocket.send(
                json.dumps(
                    response(None, error=exc), allow_nan=False, ensure_ascii=False
                )
            )
            await websocket.close(code=1013, reason="Lobby is full")
            return

        async def writer():
            try:
                while not conn.closed:
                    event = await conn.outgoing.get()
                    if event is None or conn.closed:
                        break
                    await asyncio.wait_for(
                        websocket.send(
                            json.dumps(
                                event,
                                separators=(",", ":"),
                                allow_nan=False,
                                ensure_ascii=False,
                            )
                        ),
                        self.service.policy.operation_seconds,
                    )
            finally:
                await websocket.close(
                    code=1008
                    if conn.close_reason == "replaced"
                    else 1013
                    if conn.close_reason == "slow_consumer"
                    else 1000,
                    reason=conn.close_reason,
                )

        writer_task = asyncio.create_task(writer())
        try:
            async for frame in websocket:
                try:
                    request = decode_frame(frame, self.service.policy.max_frame_bytes)
                except LobbyError as exc:
                    # Malformed/binary input cannot create an unbounded error
                    # stream or cross over into the gameplay protocol.
                    await asyncio.wait_for(
                        websocket.send(json.dumps(response(None, error=exc))),
                        self.service.policy.operation_seconds,
                    )
                    self.service._close_connection(conn, "invalid_request")
                    await websocket.close(code=1008, reason="invalid_request")
                    break
                await self.service.handle(conn, request)
                if conn.closed:
                    break
        except Exception as exc:
            # ConnectionClosed isn't imported at module load, keeping the core
            # usable without the optional network dependency.
            if exc.__class__.__name__ not in {
                "ConnectionClosed",
                "ConnectionClosedOK",
                "ConnectionClosedError",
            }:
                raise
        finally:
            await self.service.disconnect(conn)
            writer_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await writer_task

    handler = __call__
