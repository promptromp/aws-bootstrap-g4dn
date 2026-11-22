"""Service Quotas API operations for EC2 GPU vCPU quotas."""

from __future__ import annotations

import botocore.exceptions

from .ec2 import _FAMILY_PREFIXES, CLIError, default_vcpus, instance_type_to_family


SERVICE_CODE = "ec2"


def _client_region(sq_client) -> str | None:
    """Region the client is bound to (None for non-str, e.g. mocked clients)."""
    region = getattr(getattr(sq_client, "meta", None), "region_name", None)
    return region if isinstance(region, str) else None


def _not_found_msg(sq_client, quota_code: str) -> str:
    region = _client_region(sq_client)
    where = f" in region {region}" if region else ""
    return f"Quota {quota_code} not found{where}. Quotas are per-region — check the --region and quota code."


# G/VT family (g4dn, g5, g6, g7, vt1, …) — default for this tool.
# Quota codes verified against the live Service Quotas API (`list-service-quotas
# --service-code ec2`): "All G and VT Spot Instance Requests" / "Running
# On-Demand G and VT instances".
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

_QUOTA_FAMILY_NAMES: dict[str, str] = {"gvt": "G and VT", "p": "P", "dl": "DL"}

# Display labels, e.g. "G and VT (g*, vt*)". The instance-type patterns come from
# the same prefix map instance_type_to_family uses, so the label can never drift
# from the classification — unlike a hand-kept list of families, which went stale
# with every new AWS GPU family. `list instance-types` shows the per-type mapping.
QUOTA_FAMILY_LABELS: dict[str, str] = {
    family: f"{_QUOTA_FAMILY_NAMES[family]} ({', '.join(p + '*' for p in prefixes)})"
    for prefixes, family in _FAMILY_PREFIXES
}

DEFAULT_FAMILY = "gvt"


def get_quota(sq_client, quota_code: str) -> dict:
    """Get current value for a single service quota.

    Returns a dict with quota_code, quota_name, and value.
    """
    try:
        response = sq_client.get_service_quota(ServiceCode=SERVICE_CODE, QuotaCode=quota_code)
    except botocore.exceptions.ClientError as e:
        code = e.response["Error"]["Code"]
        if code == "NoSuchResourceException":
            raise CLIError(_not_found_msg(sq_client, quota_code)) from None
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


# Lifecycles that don't draw on the spot or on-demand vCPU quotas: Capacity Block
# instances (AWS: they "don't count against your On-Demand Instances limits") and the
# legacy Scheduled Instances.
_QUOTA_EXEMPT_LIFECYCLES = frozenset({"capacity-block", "scheduled"})


def _running_vcpus(ec2_client, family: str) -> dict[str, int]:
    """vCPUs of the pending/running instances of a quota family in the client's region, by market.

    All instances count, not just this tool's: the quota is the account's. Each counts
    its type's *default* vCPUs, which is how AWS counts quota usage whatever CPU options
    (cores, threads) the instance runs with. Spot instances carry
    ``InstanceLifecycle: spot``; Capacity Block (and Scheduled) instances are exempt;
    the rest are on-demand.
    """
    instances: list[tuple[str, str]] = []  # (type, market)
    paginator = ec2_client.get_paginator("describe_instances")
    for page in paginator.paginate(Filters=[{"Name": "instance-state-name", "Values": ["pending", "running"]}]):
        for reservation in page["Reservations"]:
            for inst in reservation["Instances"]:
                lifecycle = inst.get("InstanceLifecycle")
                if lifecycle in _QUOTA_EXEMPT_LIFECYCLES or instance_type_to_family(inst["InstanceType"]) != family:
                    continue
                instances.append((inst["InstanceType"], "spot" if lifecycle == "spot" else "on-demand"))
    vcpus = default_vcpus(ec2_client, (t for t, _ in instances))
    used = {"spot": 0, "on-demand": 0}
    for itype, market in instances:
        used[market] += vcpus[itype]  # KeyError if EC2 doesn't know the type: vcpu_fit says "unknown"
    return used


def vcpu_fit(sq_client, ec2_client, instance_type: str, vcpus: int) -> dict | None:
    """Whether one more *instance_type* (*vcpus*) fits the region's spot and on-demand vCPU quotas.

    ``{"family": ..., "spot": {quota, in_use, needed, fits}, "on-demand": {...}}``, or
    ``None`` for a type outside the GPU quota families or when a lookup fails. An
    estimate for display that never blocks a launch: open spot requests without an
    instance, and unused On-Demand Capacity Reservations (which do count toward the
    on-demand limit), aren't counted.
    """
    family = instance_type_to_family(instance_type)
    if family is None:
        return None
    try:
        quotas = {q["quota_type"]: q["value"] for q in get_family_quotas(sq_client, family)}
        used = _running_vcpus(ec2_client, family)
    except Exception:  # display only: never raise
        return None
    fit: dict = {"family": family}
    for market, quota in quotas.items():
        fit[market] = {
            "quota": int(quota),
            "in_use": used[market],
            "needed": vcpus,
            "fits": used[market] + vcpus <= quota,
        }
    return fit


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
            raise CLIError(_not_found_msg(sq_client, quota_code)) from None
        if code == "ResourceAlreadyExistsException":
            region = _client_region(sq_client)
            region_flag = f" --region {region}" if region else ""
            raise CLIError(
                "A quota increase request for this quota is already pending.\n\n"
                f"  Check the status with: aws-bootstrap quota history{region_flag}"
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
            raise CLIError(_not_found_msg(sq_client, quota_code)) from None
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
