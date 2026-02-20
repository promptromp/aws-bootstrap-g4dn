"""Service Quotas API operations for EC2 GPU vCPU quotas."""

from __future__ import annotations

import botocore.exceptions

from .ec2 import CLIError


SERVICE_CODE = "ec2"

# G/VT family (g4dn, g5, g6, vt1) — default for this tool
QUOTA_CODE_SPOT = "L-3819A6DF"
QUOTA_CODE_ON_DEMAND = "L-DB2E81BA"

# Multi-family quota mapping: family -> {type -> quota code}
QUOTA_FAMILIES: dict[str, dict[str, str]] = {
    "gvt": {
        "spot": QUOTA_CODE_SPOT,
        "on-demand": QUOTA_CODE_ON_DEMAND,
    },
    "p": {
        "spot": "L-7212CCBC",
        "on-demand": "L-417A185B",
    },
    "dl": {
        "spot": "L-85EED4F7",
        "on-demand": "L-6E869C2A",
    },
}

QUOTA_FAMILY_LABELS: dict[str, str] = {
    "gvt": "G and VT (g3, g4dn, g5, g5g, g6, g6e, vt1)",
    "p": "P (p2, p3, p4d, p4de, p5, p5e, p5en, p6)",
    "dl": "DL (dl1, dl2q)",
}

DEFAULT_FAMILY = "gvt"

# Legacy convenience mapping for the default (gvt) family
QUOTA_TYPES: dict[str, str] = QUOTA_FAMILIES[DEFAULT_FAMILY]


def get_quota(sq_client, quota_code: str) -> dict:
    """Get current value for a single service quota.

    Returns a dict with quota_code, quota_name, and value.
    """
    try:
        response = sq_client.get_service_quota(ServiceCode=SERVICE_CODE, QuotaCode=quota_code)
    except botocore.exceptions.ClientError as e:
        code = e.response["Error"]["Code"]
        if code == "NoSuchResourceException":
            raise CLIError(f"Quota {quota_code} not found. Check your region and quota code.") from None
        raise
    quota = response["Quota"]
    return {
        "quota_code": quota["QuotaCode"],
        "quota_name": quota["QuotaName"],
        "value": quota["Value"],
    }


def get_family_quotas(sq_client, family: str) -> list[dict]:
    """Get spot and on-demand quotas for a given instance family.

    Returns a list of quota dicts, each with added quota_type and family keys.
    """
    codes = QUOTA_FAMILIES[family]
    results = []
    for quota_type, quota_code in codes.items():
        q = get_quota(sq_client, quota_code)
        q["quota_type"] = quota_type
        q["family"] = family
        results.append(q)
    return results


def get_all_gvt_quotas(sq_client) -> list[dict]:
    """Get both spot and on-demand G/VT vCPU quotas (convenience wrapper)."""
    return get_family_quotas(sq_client, DEFAULT_FAMILY)


def request_quota_increase(sq_client, quota_code: str, desired_value: float) -> dict:
    """Request a quota increase for a G/VT vCPU quota.

    Returns a dict with request_id, status, quota_code, quota_name, desired_value,
    and optionally case_id.
    """
    try:
        response = sq_client.request_service_quota_increase(
            ServiceCode=SERVICE_CODE,
            QuotaCode=quota_code,
            DesiredValue=desired_value,
        )
    except botocore.exceptions.ClientError as e:
        code = e.response["Error"]["Code"]
        if code == "NoSuchResourceException":
            raise CLIError(f"Quota {quota_code} not found. Check your region and quota code.") from None
        if code == "ResourceAlreadyExistsException":
            raise CLIError(
                "A quota increase request for this quota is already pending.\n\n"
                "  Check the status with: aws-bootstrap quota history"
            ) from None
        if code == "IllegalArgumentException":
            raise CLIError(f"Invalid request: {e.response['Error']['Message']}") from None
        raise
    req = response["RequestedQuota"]
    result: dict = {
        "request_id": req["Id"],
        "status": req["Status"],
        "quota_code": req["QuotaCode"],
        "quota_name": req["QuotaName"],
        "desired_value": req["DesiredValue"],
    }
    if req.get("CaseId"):
        result["case_id"] = req["CaseId"]
    return result


def get_quota_request_history(sq_client, quota_code: str, status_filter: str | None = None) -> list[dict]:
    """Get history of quota increase requests for a quota code.

    Returns a list of request dicts sorted newest-first.
    """
    params: dict = {
        "ServiceCode": SERVICE_CODE,
        "QuotaCode": quota_code,
    }
    if status_filter:
        params["Status"] = status_filter
    try:
        response = sq_client.list_requested_service_quota_change_history_by_quota(**params)
    except botocore.exceptions.ClientError as e:
        code = e.response["Error"]["Code"]
        if code == "NoSuchResourceException":
            raise CLIError(f"Quota {quota_code} not found. Check your region and quota code.") from None
        raise
    requests = []
    for req in response.get("RequestedQuotas", []):
        item: dict = {
            "request_id": req["Id"],
            "status": req["Status"],
            "quota_code": req["QuotaCode"],
            "quota_name": req["QuotaName"],
            "desired_value": req["DesiredValue"],
            "created": req["Created"],
        }
        if req.get("CaseId"):
            item["case_id"] = req["CaseId"]
        requests.append(item)
    # Sort newest-first by created date
    requests.sort(key=lambda r: r["created"], reverse=True)
    return requests
