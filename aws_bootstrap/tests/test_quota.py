"""Tests for quota module (Service Quotas API operations)."""

from __future__ import annotations
from datetime import UTC, datetime
from unittest.mock import MagicMock

import botocore.exceptions
import pytest

from aws_bootstrap.ec2 import CLIError, instance_type_to_family
from aws_bootstrap.quota import (
    QUOTA_CODE_ON_DEMAND,
    QUOTA_CODE_SPOT,
    QUOTA_FAMILIES,
    QUOTA_FAMILY_LABELS,
    get_family_quotas,
    get_quota,
    get_quota_request_history,
    request_quota_increase,
)


def _make_client_error(code: str, message: str = "error") -> botocore.exceptions.ClientError:
    return botocore.exceptions.ClientError(
        {"Error": {"Code": code, "Message": message}},
        "ServiceQuotas",
    )


@pytest.fixture
def sq():
    """A bare mocked Service Quotas client."""
    return MagicMock()


def _requested(request_id, status, desired, created=None, case_id=""):
    """A RequestedQuota record as the Service Quotas API returns it."""
    record = {
        "Id": request_id,
        "Status": status,
        "QuotaCode": QUOTA_CODE_SPOT,
        "QuotaName": "All G and VT Spot Instance Requests",
        "DesiredValue": desired,
        "CaseId": case_id,
    }
    if created:
        record["Created"] = created
    return record


# ---------------------------------------------------------------------------
# get_quota
# ---------------------------------------------------------------------------


def test_get_quota_happy_path(sq):
    sq.get_service_quota.return_value = {
        "Quota": {"QuotaCode": QUOTA_CODE_SPOT, "QuotaName": "All G and VT Spot Instance Requests", "Value": 4.0}
    }
    result = get_quota(sq, QUOTA_CODE_SPOT)
    assert result["quota_code"] == QUOTA_CODE_SPOT
    assert result["quota_name"] == "All G and VT Spot Instance Requests"
    assert result["value"] == 4.0
    sq.get_service_quota.assert_called_once_with(ServiceCode="ec2", QuotaCode=QUOTA_CODE_SPOT)


# ---------------------------------------------------------------------------
# request_quota_increase
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case_id,expected_case_id",
    [("", None), ("case-789", "case-789")],
    ids=["no-case-omitted", "case-opened"],
)
def test_request_quota_increase_success(sq, case_id, expected_case_id):
    status = "CASE_OPENED" if case_id else "PENDING"
    sq.request_service_quota_increase.return_value = {
        "RequestedQuota": _requested("req-123", status, 8.0, case_id=case_id)
    }
    result = request_quota_increase(sq, QUOTA_CODE_SPOT, 8.0)
    assert result["request_id"] == "req-123"
    assert result["status"] == status
    assert result["desired_value"] == 8.0
    if expected_case_id is None:
        assert "case_id" not in result  # an empty CaseId is omitted
    else:
        assert result["case_id"] == expected_case_id


# ---------------------------------------------------------------------------
# API error mapping (every call maps the same codes)
# ---------------------------------------------------------------------------

_CALLS = {
    "get_quota": ("get_service_quota", lambda sq: get_quota(sq, QUOTA_CODE_SPOT)),
    "request": ("request_service_quota_increase", lambda sq: request_quota_increase(sq, QUOTA_CODE_SPOT, 8.0)),
    "history": (
        "list_requested_service_quota_change_history_by_quota",
        lambda sq: get_quota_request_history(sq, QUOTA_CODE_SPOT),
    ),
}


@pytest.mark.parametrize(
    "call,code,raises,match",
    [
        ("get_quota", "NoSuchResourceException", CLIError, "not found"),
        ("get_quota", "InternalServiceException", botocore.exceptions.ClientError, None),
        ("request", "NoSuchResourceException", CLIError, "not found"),
        ("request", "ResourceAlreadyExistsException", CLIError, "already pending"),
        ("request", "IllegalArgumentException", CLIError, "Invalid request"),
        ("request", "InternalServiceException", botocore.exceptions.ClientError, None),
        ("history", "NoSuchResourceException", CLIError, "not found"),
    ],
)
def test_api_errors_are_mapped(sq, call, code, raises, match):
    method, invoke = _CALLS[call]
    getattr(sq, method).side_effect = _make_client_error(code)
    with pytest.raises(raises, match=match):
        invoke(sq)


# ---------------------------------------------------------------------------
# get_quota_request_history
# ---------------------------------------------------------------------------


def test_get_quota_request_history_sorted(sq):
    sq.list_requested_service_quota_change_history_by_quota.return_value = {
        "RequestedQuotas": [
            _requested("req-old", "APPROVED", 4.0, created=datetime(2025, 1, 1, tzinfo=UTC)),
            _requested("req-new", "PENDING", 8.0, created=datetime(2025, 6, 1, tzinfo=UTC)),
        ]
    }
    results = get_quota_request_history(sq, QUOTA_CODE_SPOT)
    assert [r["request_id"] for r in results] == ["req-new", "req-old"]  # newest first


@pytest.mark.parametrize("status_filter", ["APPROVED", None])
def test_get_quota_request_history_status_filter(sq, status_filter):
    sq.list_requested_service_quota_change_history_by_quota.return_value = {"RequestedQuotas": []}
    assert get_quota_request_history(sq, QUOTA_CODE_SPOT, status_filter=status_filter) == []
    call_kwargs = sq.list_requested_service_quota_change_history_by_quota.call_args[1]
    if status_filter is None:
        assert "Status" not in call_kwargs
    else:
        assert call_kwargs["Status"] == status_filter


# ---------------------------------------------------------------------------
# get_family_quotas / multi-family support
# ---------------------------------------------------------------------------


def test_get_family_quotas_gvt(sq):

    def fake_get(ServiceCode, QuotaCode):
        names = {
            QUOTA_CODE_SPOT: "All G and VT Spot Instance Requests",
            QUOTA_CODE_ON_DEMAND: "Running On-Demand G and VT instances",
        }
        return {
            "Quota": {
                "QuotaCode": QuotaCode,
                "QuotaName": names[QuotaCode],
                "Value": 8.0 if QuotaCode == QUOTA_CODE_SPOT else 0.0,
            }
        }

    sq.get_service_quota.side_effect = fake_get
    results = get_family_quotas(sq, "gvt")
    assert len(results) == 2
    assert all(r["family"] == "gvt" for r in results)
    types = {r["quota_type"] for r in results}
    assert types == {"spot", "on-demand"}


def test_quota_families_has_expected_keys():
    """QUOTA_FAMILIES should have all GPU families with spot and on-demand each."""
    for key in ("gvt", "p", "dl"):
        assert key in QUOTA_FAMILIES
    for family in QUOTA_FAMILIES.values():
        assert "spot" in family
        assert "on-demand" in family


def test_quota_family_labels_derive_from_the_classifier():
    """Labels come from the same prefix map as instance_type_to_family, so they can't drift.

    The hand-kept family lists they replace went stale with each new AWS GPU family
    (still listing retired g3/p2/p3/dl1, then missing p6e-gb300).
    """
    assert QUOTA_FAMILY_LABELS == {"gvt": "G and VT (g*, vt*)", "p": "P (p*)", "dl": "DL (dl*)"}
    assert set(QUOTA_FAMILY_LABELS) == set(QUOTA_FAMILIES)


# ---------------------------------------------------------------------------
# instance_type_to_family
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "instance_type,expected",
    [
        ("g4dn.xlarge", "gvt"),
        ("g5.2xlarge", "gvt"),
        ("g6e.xlarge", "gvt"),
        ("g3s.xlarge", "gvt"),
        ("vt1.3xlarge", "gvt"),
        ("p5.48xlarge", "p"),
        ("p5e.48xlarge", "p"),
        ("p5en.48xlarge", "p"),
        ("p6.48xlarge", "p"),
        ("p4d.24xlarge", "p"),
        ("p4de.24xlarge", "p"),
        ("p3.2xlarge", "p"),
        ("p2.xlarge", "p"),
        ("dl1.24xlarge", "dl"),
        ("dl2q.24xlarge", "dl"),
        ("t3.medium", None),
        ("m5.large", None),
        ("c6i.xlarge", None),
    ],
)
def test_instance_type_to_family(instance_type: str, expected: str | None):
    assert instance_type_to_family(instance_type) == expected
