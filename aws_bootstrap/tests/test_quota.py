"""Tests for quota module (Service Quotas API operations)."""

from __future__ import annotations
from datetime import UTC, datetime
from unittest.mock import MagicMock

import botocore.exceptions
import pytest

from aws_bootstrap.ec2 import CLIError
from aws_bootstrap.quota import (
    QUOTA_CODE_ON_DEMAND,
    QUOTA_CODE_SPOT,
    QUOTA_FAMILIES,
    get_all_gvt_quotas,
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


# ---------------------------------------------------------------------------
# get_quota
# ---------------------------------------------------------------------------


def test_get_quota_happy_path():
    sq = MagicMock()
    sq.get_service_quota.return_value = {
        "Quota": {
            "QuotaCode": QUOTA_CODE_SPOT,
            "QuotaName": "All G and VT Spot Instance Requests",
            "Value": 4.0,
        }
    }
    result = get_quota(sq, QUOTA_CODE_SPOT)
    assert result["quota_code"] == QUOTA_CODE_SPOT
    assert result["quota_name"] == "All G and VT Spot Instance Requests"
    assert result["value"] == 4.0
    sq.get_service_quota.assert_called_once_with(ServiceCode="ec2", QuotaCode=QUOTA_CODE_SPOT)


def test_get_quota_not_found_raises_cli_error():
    sq = MagicMock()
    sq.get_service_quota.side_effect = _make_client_error("NoSuchResourceException")
    with pytest.raises(CLIError, match="not found"):
        get_quota(sq, QUOTA_CODE_SPOT)


def test_get_quota_unknown_error_propagates():
    sq = MagicMock()
    sq.get_service_quota.side_effect = _make_client_error("InternalServiceException")
    with pytest.raises(botocore.exceptions.ClientError):
        get_quota(sq, QUOTA_CODE_SPOT)


# ---------------------------------------------------------------------------
# get_all_gvt_quotas
# ---------------------------------------------------------------------------


def test_get_all_gvt_quotas_returns_both_types():
    sq = MagicMock()

    def fake_get(ServiceCode, QuotaCode):
        names = {
            QUOTA_CODE_SPOT: "All G and VT Spot Instance Requests",
            QUOTA_CODE_ON_DEMAND: "Running On-Demand G and VT instances",
        }
        return {
            "Quota": {
                "QuotaCode": QuotaCode,
                "QuotaName": names[QuotaCode],
                "Value": 4.0 if QuotaCode == QUOTA_CODE_SPOT else 0.0,
            }
        }

    sq.get_service_quota.side_effect = fake_get
    results = get_all_gvt_quotas(sq)
    assert len(results) == 2
    types = {r["quota_type"] for r in results}
    assert types == {"spot", "on-demand"}
    assert sq.get_service_quota.call_count == 2


# ---------------------------------------------------------------------------
# request_quota_increase
# ---------------------------------------------------------------------------


def test_request_quota_increase_success():
    sq = MagicMock()
    sq.request_service_quota_increase.return_value = {
        "RequestedQuota": {
            "Id": "req-123",
            "Status": "PENDING",
            "QuotaCode": QUOTA_CODE_SPOT,
            "QuotaName": "All G and VT Spot Instance Requests",
            "DesiredValue": 8.0,
            "CaseId": "",
        }
    }
    result = request_quota_increase(sq, QUOTA_CODE_SPOT, 8.0)
    assert result["request_id"] == "req-123"
    assert result["status"] == "PENDING"
    assert result["desired_value"] == 8.0
    assert "case_id" not in result  # empty CaseId should be omitted


def test_request_quota_increase_with_case_id():
    sq = MagicMock()
    sq.request_service_quota_increase.return_value = {
        "RequestedQuota": {
            "Id": "req-456",
            "Status": "CASE_OPENED",
            "QuotaCode": QUOTA_CODE_SPOT,
            "QuotaName": "All G and VT Spot Instance Requests",
            "DesiredValue": 16.0,
            "CaseId": "case-789",
        }
    }
    result = request_quota_increase(sq, QUOTA_CODE_SPOT, 16.0)
    assert result["case_id"] == "case-789"


def test_request_quota_increase_not_found():
    sq = MagicMock()
    sq.request_service_quota_increase.side_effect = _make_client_error("NoSuchResourceException")
    with pytest.raises(CLIError, match="not found"):
        request_quota_increase(sq, QUOTA_CODE_SPOT, 8.0)


def test_request_quota_increase_duplicate():
    sq = MagicMock()
    sq.request_service_quota_increase.side_effect = _make_client_error("ResourceAlreadyExistsException")
    with pytest.raises(CLIError, match="already pending"):
        request_quota_increase(sq, QUOTA_CODE_SPOT, 8.0)


def test_request_quota_increase_invalid_argument():
    sq = MagicMock()
    sq.request_service_quota_increase.side_effect = _make_client_error("IllegalArgumentException", "bad value")
    with pytest.raises(CLIError, match="Invalid request"):
        request_quota_increase(sq, QUOTA_CODE_SPOT, -1.0)


def test_request_quota_increase_unknown_error_propagates():
    sq = MagicMock()
    sq.request_service_quota_increase.side_effect = _make_client_error("InternalServiceException")
    with pytest.raises(botocore.exceptions.ClientError):
        request_quota_increase(sq, QUOTA_CODE_SPOT, 8.0)


# ---------------------------------------------------------------------------
# get_quota_request_history
# ---------------------------------------------------------------------------


def test_get_quota_request_history_sorted():
    sq = MagicMock()
    sq.list_requested_service_quota_change_history_by_quota.return_value = {
        "RequestedQuotas": [
            {
                "Id": "req-old",
                "Status": "APPROVED",
                "QuotaCode": QUOTA_CODE_SPOT,
                "QuotaName": "All G and VT Spot Instance Requests",
                "DesiredValue": 4.0,
                "Created": datetime(2025, 1, 1, tzinfo=UTC),
                "CaseId": "",
            },
            {
                "Id": "req-new",
                "Status": "PENDING",
                "QuotaCode": QUOTA_CODE_SPOT,
                "QuotaName": "All G and VT Spot Instance Requests",
                "DesiredValue": 8.0,
                "Created": datetime(2025, 6, 1, tzinfo=UTC),
                "CaseId": "",
            },
        ]
    }
    results = get_quota_request_history(sq, QUOTA_CODE_SPOT)
    assert len(results) == 2
    assert results[0]["request_id"] == "req-new"  # newest first
    assert results[1]["request_id"] == "req-old"


def test_get_quota_request_history_status_filter():
    sq = MagicMock()
    sq.list_requested_service_quota_change_history_by_quota.return_value = {"RequestedQuotas": []}
    get_quota_request_history(sq, QUOTA_CODE_SPOT, status_filter="APPROVED")
    call_kwargs = sq.list_requested_service_quota_change_history_by_quota.call_args[1]
    assert call_kwargs["Status"] == "APPROVED"


def test_get_quota_request_history_no_status_filter():
    sq = MagicMock()
    sq.list_requested_service_quota_change_history_by_quota.return_value = {"RequestedQuotas": []}
    get_quota_request_history(sq, QUOTA_CODE_SPOT)
    call_kwargs = sq.list_requested_service_quota_change_history_by_quota.call_args[1]
    assert "Status" not in call_kwargs


def test_get_quota_request_history_empty():
    sq = MagicMock()
    sq.list_requested_service_quota_change_history_by_quota.return_value = {"RequestedQuotas": []}
    results = get_quota_request_history(sq, QUOTA_CODE_SPOT)
    assert results == []


def test_get_quota_request_history_not_found():
    sq = MagicMock()
    sq.list_requested_service_quota_change_history_by_quota.side_effect = _make_client_error("NoSuchResourceException")
    with pytest.raises(CLIError, match="not found"):
        get_quota_request_history(sq, QUOTA_CODE_SPOT)


# ---------------------------------------------------------------------------
# get_family_quotas / multi-family support
# ---------------------------------------------------------------------------


def test_get_family_quotas_gvt():
    sq = MagicMock()

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


def test_get_family_quotas_p5():
    sq = MagicMock()
    p5_spot = QUOTA_FAMILIES["p5"]["spot"]
    p5_on_demand = QUOTA_FAMILIES["p5"]["on-demand"]

    def fake_get(ServiceCode, QuotaCode):
        names = {
            p5_spot: "All P5 Spot Instance Requests",
            p5_on_demand: "Running On-Demand P instances",
        }
        return {
            "Quota": {
                "QuotaCode": QuotaCode,
                "QuotaName": names[QuotaCode],
                "Value": 0.0,
            }
        }

    sq.get_service_quota.side_effect = fake_get
    results = get_family_quotas(sq, "p5")
    assert len(results) == 2
    assert all(r["family"] == "p5" for r in results)
    codes = {r["quota_code"] for r in results}
    assert p5_spot in codes
    assert p5_on_demand in codes


def test_get_all_gvt_quotas_delegates_to_get_family_quotas():
    """get_all_gvt_quotas is a convenience wrapper for get_family_quotas('gvt')."""
    sq = MagicMock()

    def fake_get(ServiceCode, QuotaCode):
        return {
            "Quota": {
                "QuotaCode": QuotaCode,
                "QuotaName": "Test",
                "Value": 4.0,
            }
        }

    sq.get_service_quota.side_effect = fake_get
    results = get_all_gvt_quotas(sq)
    assert len(results) == 2
    assert all(r["family"] == "gvt" for r in results)


def test_quota_families_has_expected_keys():
    """QUOTA_FAMILIES should have all GPU families with spot and on-demand each."""
    for key in ("gvt", "p5", "p", "dl"):
        assert key in QUOTA_FAMILIES
    for family in QUOTA_FAMILIES.values():
        assert "spot" in family
        assert "on-demand" in family


def test_p_and_p5_share_on_demand_code():
    """P4/P3/P2 and P5 families share the same on-demand quota code."""
    assert QUOTA_FAMILIES["p"]["on-demand"] == QUOTA_FAMILIES["p5"]["on-demand"]
    assert QUOTA_FAMILIES["p"]["spot"] != QUOTA_FAMILIES["p5"]["spot"]
