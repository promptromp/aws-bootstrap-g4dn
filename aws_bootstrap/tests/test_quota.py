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
    vcpu_fit,
)
from aws_bootstrap.tests.helpers import client_error


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


def test_request_quota_increase_success(sq):
    sq.request_service_quota_increase.return_value = {"RequestedQuota": _requested("req-123", "PENDING", 8.0)}
    result = request_quota_increase(sq, QUOTA_CODE_SPOT, 8.0)
    assert result["request_id"] == "req-123"
    assert result["status"] == "PENDING"
    assert result["desired_value"] == 8.0
    assert "case_id" not in result  # an empty CaseId is omitted


def test_request_quota_increase_with_case_id(sq):
    sq.request_service_quota_increase.return_value = {
        "RequestedQuota": _requested("req-456", "CASE_OPENED", 16.0, case_id="case-789")
    }
    assert request_quota_increase(sq, QUOTA_CODE_SPOT, 16.0)["case_id"] == "case-789"


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
    ids=[
        "get-quota-not-found",
        "get-quota-unknown-propagates",
        "request-not-found",
        "request-already-pending",
        "request-invalid-argument",
        "request-unknown-propagates",
        "history-not-found",
    ],
)
def test_api_errors_are_mapped(sq, call, code, raises, match):
    method, invoke = _CALLS[call]
    getattr(sq, method).side_effect = client_error(code)
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


def test_get_quota_request_history_status_filter(sq):
    sq.list_requested_service_quota_change_history_by_quota.return_value = {"RequestedQuotas": []}
    get_quota_request_history(sq, QUOTA_CODE_SPOT, status_filter="APPROVED")
    assert sq.list_requested_service_quota_change_history_by_quota.call_args[1]["Status"] == "APPROVED"


def test_get_quota_request_history_no_status_filter_by_default(sq):
    sq.list_requested_service_quota_change_history_by_quota.return_value = {"RequestedQuotas": []}
    assert get_quota_request_history(sq, QUOTA_CODE_SPOT) == []
    assert "Status" not in sq.list_requested_service_quota_change_history_by_quota.call_args[1]


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


# ---------------------------------------------------------------------------
# vcpu_fit: will one more instance fit each market's vCPU quota?
# ---------------------------------------------------------------------------


_DEFAULT_VCPUS = {"g6.2xlarge": 8, "g6.xlarge": 4, "p5.4xlarge": 16}


def _running(*instances: tuple[str, str | None]) -> MagicMock:
    """An EC2 client listing (type, lifecycle) instances, each running with SMT off (half its vCPUs)."""
    ec2 = MagicMock()
    ec2.get_paginator.return_value.paginate.return_value = [
        {
            "Reservations": [
                {
                    "Instances": [
                        {
                            "InstanceType": itype,
                            **({"InstanceLifecycle": lifecycle} if lifecycle else {}),
                            "CpuOptions": {"CoreCount": _DEFAULT_VCPUS.get(itype, 2) // 2, "ThreadsPerCore": 1},
                        }
                    ]
                }
                for itype, lifecycle in instances
            ]
        }
    ]
    ec2.describe_instance_types.side_effect = lambda InstanceTypes: {
        "InstanceTypes": [{"InstanceType": t, "VCpuInfo": {"DefaultVCpus": _DEFAULT_VCPUS[t]}} for t in InstanceTypes]
    }
    return ec2


def _quotas(spot: float, on_demand: float) -> MagicMock:
    sq = MagicMock()
    by_code = {QUOTA_FAMILIES["gvt"]["spot"]: spot, QUOTA_FAMILIES["gvt"]["on-demand"]: on_demand}
    sq.get_service_quota.side_effect = lambda ServiceCode, QuotaCode: {
        "Quota": {"QuotaCode": QuotaCode, "QuotaName": "x", "Value": by_code[QuotaCode]}
    }
    return sq


def test_vcpu_fit_counts_running_vcpus_per_market_within_the_family():
    """Quota usage is each type's default vCPUs, whatever CPU options the instance runs with."""
    ec2 = _running(
        ("g6.2xlarge", "spot"),  # 8 spot vCPUs (though running with 4)
        ("g6.xlarge", None),  # 4 on-demand vCPUs
        ("g6.2xlarge", "capacity-block"),  # Capacity Blocks don't count against the quotas
        ("g6.xlarge", "scheduled"),  # nor do (legacy) Scheduled Instances
        ("p5.4xlarge", "spot"),  # another family: not counted
        ("t3.large", None),  # not a GPU family: not counted
    )
    fit = vcpu_fit(_quotas(spot=16, on_demand=8), ec2, "g7e.2xlarge", 8)
    assert fit == {
        "family": "gvt",
        "spot": {"quota": 16, "in_use": 8, "needed": 8, "fits": True},  # exactly at the quota still fits
        "on-demand": {"quota": 8, "in_use": 4, "needed": 8, "fits": False},
    }
    assert ec2.get_paginator.return_value.paginate.call_args.kwargs == {
        "Filters": [{"Name": "instance-state-name", "Values": ["pending", "running"]}]
    }


@pytest.mark.parametrize(
    "sq,ec2,instance_type",
    [
        pytest.param(_quotas(8, 8), _running(), "t3.medium", id="not-a-gpu-family"),
        pytest.param(
            MagicMock(**{"get_service_quota.side_effect": client_error("AccessDeniedException")}),
            _running(),
            "g7e.2xlarge",
            id="quota-denied",
        ),
        pytest.param(
            _quotas(8, 8),
            MagicMock(**{"get_paginator.return_value.paginate.side_effect": client_error("UnauthorizedOperation")}),
            "g7e.2xlarge",
            id="usage-denied",
        ),
    ],
)
def test_vcpu_fit_is_none_when_it_cannot_tell(sq, ec2, instance_type):
    assert vcpu_fit(sq, ec2, instance_type, 8) is None
