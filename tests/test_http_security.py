import pytest

from hudes.http_security import admin_request_authorized


@pytest.mark.parametrize(
    "headers",
    [
        b"",
        b"Authorization: Bearer invalid\r\n",
        b"Authorization: Basic audit-secret\r\n",
        b"Authorization: Bearer audit-secret\r\nAuthorization: Bearer audit-secret\r\n",
    ],
)
def test_public_clients_cannot_authorize_deletion(monkeypatch, headers):
    monkeypatch.setenv("HUDES_ADMIN_TOKEN", "audit-secret")
    request = b"DELETE /api/highscores?id=1 HTTP/1.1\r\n" + headers + b"\r\n"
    assert not admin_request_authorized(request)


def test_admin_operations_disabled_without_configuration(monkeypatch):
    monkeypatch.delenv("HUDES_ADMIN_TOKEN", raising=False)
    request = b"DELETE /api/highscores?id=1 HTTP/1.1\r\nAuthorization: Bearer audit-secret\r\n\r\n"
    assert not admin_request_authorized(request)


def test_explicit_admin_bearer_is_accepted(monkeypatch):
    monkeypatch.setenv("HUDES_ADMIN_TOKEN", "audit-secret")
    request = b"DELETE /api/highscores?id=1 HTTP/1.1\r\nauthorization: Bearer audit-secret\r\n\r\n"
    assert admin_request_authorized(request)
