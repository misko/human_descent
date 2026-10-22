import asyncio
from queue import Queue

import pytest

from hudes.high_scores import get_all_scores, init_db, insert_high_score
from hudes.websocket_server import run_server


@pytest.mark.asyncio
async def test_delete_requires_bearer_on_actual_http_server(monkeypatch, tmp_path):
    monkeypatch.setenv("HIGH_SCORES_PATH", str(tmp_path / "scores.sqlite3"))
    monkeypatch.setenv("HUDES_ADMIN_TOKEN", "isolated-http-test-secret")
    monkeypatch.setenv("HUDES_HEALTH_PORT", "18998")
    init_db()
    insert_high_score("TEST", 1.0, 1.0, 10, 0, b"")
    row_id = get_all_scores()[0][0]
    stop = asyncio.get_running_loop().create_future()
    task = asyncio.create_task(run_server(stop, Queue(), 18997, None))

    async def request(headers):
        reader, writer = await asyncio.open_connection("127.0.0.1", 18998)
        writer.write(
            f"DELETE /api/highscores?id={row_id} HTTP/1.1\r\nHost: localhost\r\n{headers}\r\n".encode()
        )
        await writer.drain()
        response = await asyncio.wait_for(reader.read(), 5)
        writer.close()
        await writer.wait_closed()
        return response

    try:
        for _ in range(100):
            try:
                response = await request("")
                break
            except ConnectionRefusedError:
                await asyncio.sleep(0.01)
        else:
            pytest.fail("HTTP server did not start")
        assert response.startswith(b"HTTP/1.1 403")
        assert len(get_all_scores()) == 1
        response = await request("Authorization: Bearer isolated-http-test-secret\r\n")
        assert response.startswith(b"HTTP/1.1 200")
        assert get_all_scores() == []
    finally:
        stop.set_result(True)
        await task
