"""Plain helpers shared by test modules (fixtures live in conftest.py)."""

from __future__ import annotations
from unittest.mock import MagicMock

import botocore.exceptions


def client_error(code: str, message: str = "test", operation: str = "Op") -> botocore.exceptions.ClientError:
    """A botocore ``ClientError`` with this error code, as a failing AWS API call raises it."""
    return botocore.exceptions.ClientError({"Error": {"Code": code, "Message": message}}, operation)


def session_with_clients(clients_by_region: dict[str, MagicMock]) -> MagicMock:
    """A boto3 Session stand-in whose ``client(service, region_name=...)`` returns that region's client."""
    session = MagicMock()
    session.client.side_effect = lambda _service, region_name=None, **_kw: clients_by_region[region_name]
    return session
