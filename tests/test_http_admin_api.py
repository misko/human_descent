import asyncio
from queue import Queue

import pytest

from hudes.high_scores import get_all_scores, init_db, insert_high_score
from hudes.websocket_server import run_server


@pytest.fixture
async def api_request(monkeypatch, tmp_path, unused_tcp_port):
    monkeypatch.setenv("HIGH_SCORES_PATH", str(tmp_path / "scores.sqlite3"))
    monkeypatch.setenv("HUDES_ADMIN_TOKEN", "isolated-http-test-secret")
    monkeypatch.setenv("HUDES_HEALTH_PORT", str(unused_tcp_port))
    stop = asyncio.get_running_loop().create_future()
    task = asyncio.create_task(run_server(stop, Queue(), 0, None))

    async def request(method, path, headers=""):
        for _ in range(100):
            try:
                reader, writer = await asyncio.open_connection(
                    "127.0.0.1", unused_tcp_port
                )
                break
            except ConnectionRefusedError:
                await asyncio.sleep(0.01)
        else:
            pytest.fail("HTTP server did not start")
        try:
            writer.write(
                f"{method} {path} HTTP/1.1\r\nHost: localhost\r\n{headers}\r\n".encode()
            )
            await writer.drain()
            return await asyncio.wait_for(reader.read(), 5)
        finally:
            writer.close()
            await writer.wait_closed()

    try:
        yield request
    finally:
        stop.set_result(True)
        await task


@pytest.mark.asyncio
async def test_api_preflight_has_no_body_or_framing_headers(api_request):
    response = await api_request(
        "OPTIONS",
        "/api/highscores",
        "Origin: https://client.example\r\n"
        "Access-Control-Request-Method: GET\r\n"
        "Access-Control-Request-Headers: x-client-version\r\n",
    )
    head, boundary, body = response.partition(b"\r\n\r\n")
    assert boundary == b"\r\n\r\n"
    assert body == b""
    status, *header_lines = head.split(b"\r\n")
    assert status == b"HTTP/1.1 204 No Content"
    headers = {
        name.lower(): value.strip()
        for name, value in (line.split(b":", 1) for line in header_lines)
    }
    assert b"content-length" not in headers
    assert b"transfer-encoding" not in headers
    assert headers[b"access-control-allow-origin"] == b"*"
    assert headers[b"access-control-allow-methods"] == b"GET, OPTIONS, DELETE"
    assert headers[b"access-control-allow-headers"] == b"*"


@pytest.mark.asyncio
async def test_delete_requires_bearer_on_actual_http_server(api_request):
    init_db()
    insert_high_score("TEST", 1.0, 1.0, 10, 0, b"")
    path = f"/api/highscores?id={get_all_scores()[0][0]}"

    response = await api_request("DELETE", path)
    assert response.startswith(b"HTTP/1.1 403")
    assert len(get_all_scores()) == 1
    response = await api_request(
        "DELETE", path, "Authorization: Bearer isolated-http-test-secret\r\n"
    )
    assert response.startswith(b"HTTP/1.1 200")
    assert get_all_scores() == []
