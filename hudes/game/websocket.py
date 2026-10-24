"""Binary-only team gameplay transport; no social commands or solo fallback."""

import time

from google.protobuf.message import DecodeError
from websockets.exceptions import ConnectionClosed

from hudes import hudes_pb2 as pb
from hudes.control_security import validate_control


class TeamWebSocketHandler:
    def __init__(self, service):
        self.service = service

    async def __call__(self, websocket, first_message=None):
        session = player_id = connection = None
        window_start, count = time.monotonic(), 0
        try:
            first = (
                first_message if first_message is not None else await websocket.recv()
            )
            admission = self._decode(first)
            if (
                admission.type != pb.Control.CONTROL_TEAM_ADMIT
                or not admission.HasField("team_admission")
            ):
                raise ValueError("team admission required")
            session, player_id, connection = await self.service.admit(
                admission.team_admission, websocket
            )
            async for raw in websocket:
                received_at = self.service.clock()  # capture before parsing/awaits
                if session.connections.get(player_id) is not connection:
                    return
                now = time.monotonic()
                if now - window_start >= 1:
                    window_start, count = now, 0
                count += 1
                if count > 240:
                    raise ValueError("game message rate exceeded")
                msg = self._decode(raw)
                try:
                    self.service.handle_control(
                        session, player_id, msg, received_at=received_at
                    )
                except ValueError as exc:
                    # Stale/replayed controls do not evict a legitimate resumed
                    # slot. A snapshot allows the client to reconcile its state.
                    connection.send(
                        pb.Control(
                            type=pb.Control.CONTROL_TEAM_ERROR,
                            error_code="invalid_control",
                            error_message=str(exc),
                        )
                    )
                    connection.send(self.service._state(session, player_id))
        except (ValueError, DecodeError) as exc:
            try:
                await websocket.send(
                    pb.Control(
                        type=pb.Control.CONTROL_TEAM_ERROR,
                        error_code="admission_rejected",
                        error_message=str(exc),
                    ).SerializeToString()
                )
                await websocket.close(
                    code=1008, reason="invalid team admission or frame"
                )
            except ConnectionClosed:
                pass
        except ConnectionClosed:
            pass
        finally:
            if connection:
                self.service.disconnect(session, player_id, connection)

    @staticmethod
    def _decode(raw):
        if not isinstance(raw, bytes) or len(raw) > 8192:
            raise ValueError("binary gameplay frame required")
        msg = pb.Control()
        msg.ParseFromString(raw)
        validate_control(msg)
        return msg
