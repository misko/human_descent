"""Strict version-one JSON validation, independent of gameplay/protobuf."""

import json
from dataclasses import dataclass

from hudes.match_settings import validate_settings


@dataclass
class LobbyError(Exception):
    code: str
    message: str


COMMAND_FIELDS = {
    "hello": ({"name"}, {"resumeToken"}),
    "snapshot": (set(), set()),
    "chat.send": ({"text"}, set()),
    "team.create": ({"name"}, set()),
    "team.rename": ({"name"}, set()),
    "team.invite": ({"playerId"}, set()),
    "invite.accept": ({"inviteId"}, set()),
    "invite.decline": ({"inviteId"}, set()),
    "team.leave": (set(), set()),
    "team.start_speedrun": (set(), {"settings"}),
    "challenge.create": ({"teamId"}, {"settings"}),
    "challenge.accept": ({"challengeId"}, set()),
    "challenge.decline": ({"challengeId"}, set()),
    "challenge.cancel": ({"challengeId"}, set()),
    "match.ready": ({"matchId", "ready"}, set()),
    "match.cancel": ({"matchId"}, set()),
}


def decode_frame(frame, max_bytes=16384):
    if not isinstance(frame, str):
        raise LobbyError("invalid_request", "Lobby frames must be JSON text")
    if len(frame.encode("utf-8")) > max_bytes:
        raise LobbyError("invalid_request", "Message is too large")

    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    try:
        value = json.loads(
            frame,
            object_pairs_hook=unique_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
        pending = [(value, 0)]
        while pending:
            item, depth = pending.pop()
            if depth > 4:
                raise ValueError("excessive nesting")
            if isinstance(item, dict):
                pending.extend((child, depth + 1) for child in item.values())
            elif isinstance(item, list):
                pending.extend((child, depth + 1) for child in item)
        return value
    except (ValueError, RecursionError) as exc:
        raise LobbyError("invalid_request", "Malformed JSON") from exc


def validate_request(request):
    if not isinstance(request, dict) or set(request) != {
        "v",
        "type",
        "requestId",
        "payload",
    }:
        raise LobbyError("invalid_request", "Expected v, type, requestId and payload")
    if type(request["v"]) is not int or request["v"] != 1:
        raise LobbyError("invalid_request", "Unsupported protocol version")
    request_id = request["requestId"]
    if not isinstance(request_id, str) or not 1 <= len(request_id) <= 80:
        raise LobbyError("invalid_request", "Invalid requestId")
    command = request["type"]
    if not isinstance(command, str) or command not in COMMAND_FIELDS:
        raise LobbyError("invalid_request", "Unknown command")
    payload = request["payload"]
    required, optional = COMMAND_FIELDS[command]
    if (
        not isinstance(payload, dict)
        or not required <= set(payload) <= required | optional
    ):
        raise LobbyError("invalid_request", "Invalid command payload")
    for key, value in payload.items():
        if key == "settings":
            try:
                validate_settings(value)
            except ValueError as exc:
                raise LobbyError("invalid_request", str(exc)) from exc
        elif key == "ready":
            if type(value) is not bool:
                raise LobbyError("invalid_request", "ready must be a boolean")
        elif (
            not isinstance(value, str)
            or not value
            or len(value) > (1000 if key == "text" else 128)
        ):
            raise LobbyError("invalid_request", f"Invalid {key}")
    return command, request_id, payload


def response(request_id, payload=None, error=None):
    result = {"v": 1, "type": "response", "requestId": request_id, "ok": error is None}
    if error is None:
        result["payload"] = payload or {}
    else:
        result["error"] = {"code": error.code, "message": error.message}
    return result


def fingerprint(request):
    return json.dumps(
        {"type": request["type"], "payload": request["payload"]},
        sort_keys=True,
        separators=(",", ":"),
    )
