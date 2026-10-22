"""Authorization for destructive HTTP operations on the public game API."""

import os
import secrets


def admin_request_authorized(request_headers: bytes) -> bool:
    """Disable admin operations unless an explicit bearer secret is configured."""
    expected = os.environ.get("HUDES_ADMIN_TOKEN", "")
    if not expected:
        return False
    credentials = []
    for line in request_headers.split(b"\r\n")[1:]:
        name, separator, value = line.partition(b":")
        if separator and name.strip().lower() == b"authorization":
            credentials.append(value.strip())
    if len(credentials) != 1:
        return False
    scheme, separator, token = credentials[0].partition(b" ")
    return (
        bool(separator)
        and scheme.lower() == b"bearer"
        and secrets.compare_digest(token, expected.encode("utf-8"))
    )
