"""EC2 instance provisioning: AMI lookup, security groups, and instance launch."""

from __future__ import annotations
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

import botocore.exceptions
import click

from .config import LaunchConfig
from .constants import (
    AMI_OWNER_AMAZON,
    AMI_OWNER_CANONICAL,
    AMI_OWNER_RHEL,
    EBS_DEVICE_NAME,
    EBS_VOLUME_WAITER,
    INSTANCE_RUNNING_WAITER,
    INSTANCE_STATUS_OK_WAITER,
    RES_INSTANCE,
    RES_SECURITY_GROUP,
    RES_VOLUME,
    ROOT_DEVICE_NAME,
    SSH_INGRESS_CIDR,
    SSH_PORT_DEFAULT,
    TAG_BOOTSTRAP_INSTANCE,
    TAG_CREATED_BY,
    TAG_NAME,
    VOLUME_TYPE,
)
from .output import echo, is_text, secho
from .retry import backoff_sleep_seconds


class CLIError(click.ClickException):
    """A ClickException that displays the error message in red."""

    def show(self, file=None):  # type: ignore[no-untyped-def]
        if file is None:
            file = click.get_text_stream("stderr")
        click.secho(f"Error: {self.format_message()}", file=file, fg="red")


# Well-known AMI owners by name prefix
_OWNER_HINTS = {
    "Deep Learning": [AMI_OWNER_AMAZON],
    "ubuntu": [AMI_OWNER_CANONICAL],
    "Ubuntu": [AMI_OWNER_CANONICAL],
    "RHEL": [AMI_OWNER_RHEL],
    "al20": [AMI_OWNER_AMAZON],  # Amazon Linux
}


def get_latest_ami(ec2_client, ami_filter: str) -> dict:
    """Find the latest AMI matching the filter pattern.

    Infers the owner from the filter prefix when possible,
    otherwise searches all public AMIs.
    """
    owners = None
    for prefix, owner_ids in _OWNER_HINTS.items():
        if ami_filter.startswith(prefix):
            owners = owner_ids
            break

    params: dict = {
        "Filters": [
            {"Name": "name", "Values": [ami_filter]},
            {"Name": "state", "Values": ["available"]},
            {"Name": "architecture", "Values": ["x86_64"]},
        ],
    }
    if owners:
        params["Owners"] = owners

    response = ec2_client.describe_images(**params)
    images = response["Images"]
    if not images:
        raise CLIError(f"No AMI found matching filter: {ami_filter}\nTry adjusting --ami-filter or check the region.")

    images.sort(key=lambda x: x["CreationDate"], reverse=True)
    return images[0]


def ensure_security_group(ec2_client, name: str, tag_value: str, ssh_port: int = SSH_PORT_DEFAULT) -> str:
    """Find or create a security group with SSH ingress in the default VPC."""
    # Find default VPC
    vpcs = ec2_client.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])
    if not vpcs["Vpcs"]:
        raise CLIError("No default VPC found. Create one or specify a VPC.")
    vpc_id = vpcs["Vpcs"][0]["VpcId"]

    # Check if SG already exists
    existing = ec2_client.describe_security_groups(
        Filters=[
            {"Name": "group-name", "Values": [name]},
            {"Name": "vpc-id", "Values": [vpc_id]},
        ]
    )
    if existing["SecurityGroups"]:
        sg_id = existing["SecurityGroups"][0]["GroupId"]
        msg = "  Security group " + click.style(f"'{name}'", fg="bright_white")
        echo(msg + f" already exists ({sg_id}), reusing.")
        return sg_id

    # Create new SG
    sg = ec2_client.create_security_group(
        GroupName=name,
        Description="SSH access for aws-bootstrap-g4dn instances",
        VpcId=vpc_id,
        TagSpecifications=[
            {
                "ResourceType": RES_SECURITY_GROUP,
                "Tags": [
                    {"Key": TAG_CREATED_BY, "Value": tag_value},
                    {"Key": TAG_NAME, "Value": name},
                ],
            }
        ],
    )
    sg_id = sg["GroupId"]

    # Add SSH ingress
    ec2_client.authorize_security_group_ingress(
        GroupId=sg_id,
        IpPermissions=[
            {
                "IpProtocol": "tcp",
                "FromPort": ssh_port,
                "ToPort": ssh_port,
                "IpRanges": [{"CidrIp": SSH_INGRESS_CIDR, "Description": "SSH access"}],
            }
        ],
    )
    secho(f"  Created security group '{name}' ({sg_id}) with SSH ingress.", fg="green")
    return sg_id


class CapacityError(Exception):
    """Retryable: AWS has no capacity for the requested type in this region/AZ.

    Eligible for both next-region fallthrough and ``--wait`` backoff retries,
    unlike :class:`RegionFatalError` (quota / spot price) which only falls
    through to the next region.
    """

    def __init__(self, region: str, market: str, message: str) -> None:
        super().__init__(message)
        self.region = region
        self.market = market


class RegionFatalError(Exception):
    """A region can't satisfy this launch and waiting won't help (quota / spot price).

    Unlike :class:`CapacityError` it never justifies a ``--wait`` sleep, but in
    multi-region mode the launcher still moves on to the next ``--region`` (a
    different region may have quota/price headroom). ``kind`` is ``"quota"`` or
    ``"price"``; ``message`` is the full user-facing remediation text.
    """

    def __init__(self, region: str, kind: str, message: str) -> None:
        super().__init__(message)
        self.region = region
        self.kind = kind
        self.message = message


@dataclass
class RegionContext:
    """Region-scoped launch prerequisites, prepared once and reused across retries."""

    region: str
    ec2_client: object
    ami: dict
    sg_id: str


@dataclass
class RegionLaunch:
    """Result of a successful launch: which region/market won and its context."""

    region: str
    context: RegionContext
    instance: dict
    pricing: str


def _build_launch_params(config: LaunchConfig, ami_id: str, sg_id: str, spot: bool) -> dict:
    params: dict = {
        "ImageId": ami_id,
        "InstanceType": config.instance_type,
        "KeyName": config.key_name,
        "SecurityGroupIds": [sg_id],
        "MinCount": 1,
        "MaxCount": 1,
        "BlockDeviceMappings": [
            {
                "DeviceName": ROOT_DEVICE_NAME,
                "Ebs": {
                    "VolumeSize": config.volume_size,
                    "VolumeType": VOLUME_TYPE,
                    "DeleteOnTermination": True,
                },
            }
        ],
        "TagSpecifications": [
            {
                "ResourceType": RES_INSTANCE,
                "Tags": [
                    {"Key": TAG_NAME, "Value": f"aws-bootstrap-{config.instance_type}"},
                    {"Key": TAG_CREATED_BY, "Value": config.tag_value},
                ],
            }
        ],
    }
    if spot:
        params["InstanceMarketOptions"] = {
            "MarketType": "spot",
            "SpotOptions": {
                "SpotInstanceType": "one-time",
                "InstanceInterruptionBehavior": "terminate",
            },
        }
    return params


def _run_instances(ec2_client, config: LaunchConfig, ami_id: str, sg_id: str, region: str, spot: bool) -> dict:
    """Single ``run_instances`` call.

    Raises :class:`CapacityError` on ``InsufficientInstanceCapacity`` (retryable
    by next-region fallthrough and ``--wait``), or :class:`RegionFatalError` on
    quota / ``SpotMaxPriceTooLow`` (next-region fallthrough only — never waited).
    """
    market = "spot" if spot else "on-demand"
    try:
        response = ec2_client.run_instances(**_build_launch_params(config, ami_id, sg_id, spot))
    except botocore.exceptions.ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("MaxSpotInstanceCountExceeded", "VcpuLimitExceeded"):
            raise RegionFatalError(region, "quota", _quota_error_message(code, config, region)) from None
        if code == "InsufficientInstanceCapacity":
            raise CapacityError(
                region,
                market,
                f"Insufficient {market} capacity for {config.instance_type} in {region}.",
            ) from None
        if code == "SpotMaxPriceTooLow":
            raise RegionFatalError(
                region,
                "price",
                f"Spot price for {config.instance_type} in {region} exceeds the default maximum.\n\n"
                "  Waiting will not help — retry with --on-demand or a different instance type.",
            ) from None
        raise
    return response["Instances"][0]


def launch_instance(ec2_client, config: LaunchConfig, ami_id: str, sg_id: str) -> dict:
    """Launch a single instance in one region (spot, with interactive on-demand fallback).

    Back-compat single-region primitive. Multi-region and ``--wait`` retries
    go through :func:`launch_with_retry`.
    """
    region = config.region
    try:
        try:
            return _run_instances(ec2_client, config, ami_id, sg_id, region, config.spot)
        except CapacityError:
            if not config.spot:
                raise CLIError(
                    f"Insufficient capacity for {config.instance_type} in {region}.\n\n"
                    "  The requested instance type is not currently available.\n"
                    "  Try a different region, availability zone, or instance type."
                ) from None
            secho(f"\n  Spot request failed: insufficient capacity in {region}.", fg="yellow")
            if not is_text() or click.confirm("  Retry as on-demand instance?"):
                try:
                    return _run_instances(ec2_client, config, ami_id, sg_id, region, spot=False)
                except CapacityError:
                    raise CLIError(
                        f"Insufficient capacity for {config.instance_type} (on-demand) in {region}.\n\n"
                        "  Neither spot nor on-demand capacity is currently available.\n"
                        "  Try a different region, availability zone, or instance type."
                    ) from None
            raise CLIError("Launch cancelled.") from None
    except RegionFatalError as e:
        # Single-region path: quota / spot-price problems are terminal.
        raise CLIError(e.message) from None


def _describe_failures(regions: tuple[str, ...], failures: dict[str, tuple[str, str]], market: str) -> str:
    """One '- region: reason' line per region, in attempt order."""
    reason = {
        "capacity": f"no {market} capacity",
        "quota": f"{market} quota exceeded",
        "price": "spot price exceeds the default maximum",
    }
    lines = []
    for region in regions:
        if region in failures:
            kind = failures[region][0]
            lines.append(f"    - {region}: {reason.get(kind, kind)}")
    return "\n".join(lines)


def _aggregated_error(
    config: LaunchConfig,
    regions: tuple[str, ...],
    failures: dict[str, tuple[str, str]],
    market: str,
    *,
    suffix: str = "",
) -> CLIError:
    """Hard-fail message: per-region reasons + full hint for every quota/price region."""
    header = f"Could not launch {config.instance_type} ({market}) in any region{suffix}:\n\n"
    body = _describe_failures(regions, failures, market)
    hints: list[str] = []
    for region in regions:
        if region in failures and failures[region][0] in ("quota", "price"):
            msg = failures[region][1]
            if msg not in hints:
                hints.append(msg)
    if hints:
        tail = "\n\n" + "\n\n".join(f"  {h}" for h in hints)
    else:
        tail = "\n\n  Try --wait, more --region values, --on-demand, or a different instance type."
    return CLIError(header + body + tail)


def launch_with_retry(
    config: LaunchConfig,
    prepare_region: Callable[[str], RegionContext],
    *,
    on_attempt: Callable[[str, str, int], None] | None = None,
    on_wait: Callable[[int, float, float], None] | None = None,
    on_region_fatal: Callable[[str, str, str], None] | None = None,
    confirm_on_demand: Callable[[], bool] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    rng=None,
) -> RegionLaunch:
    """Launch across ``config.regions``, spot-first, with optional bounded-backoff wait.

    Each sweep tries the primary market (spot, unless ``--on-demand``) in every
    region in order. ``InsufficientInstanceCapacity`` moves on to the next
    region (and, with ``--wait``, is retried after backoff). A quota or
    spot-price problem (:class:`RegionFatalError`) also moves on to the next
    region — a different region may have headroom — but that region is then
    dropped (waiting/re-sweeping can't fix it) and ``on_region_fatal`` is fired
    so the caller can warn immediately. If every region is exhausted the command
    fails hard with an aggregated message that includes the remediation hint for
    each quota/price region. Without ``--wait``, a fully-exhausted spot sweep
    offers the on-demand fallback (across all regions).

    ``prepare_region`` builds region-scoped prerequisites (client, AMI, SG); it
    is invoked at most once per region and cached. ``on_attempt(region, market,
    attempt)``'s ``attempt`` is the 0-based ``--wait`` backoff-cycle index; it
    stays ``0`` outside the wait loop (no-wait pass and on-demand fallback).
    """
    regions = config.regions
    primary_spot = config.spot
    cache: dict[str, RegionContext] = {}
    start = clock()
    deadline = start + config.wait_timeout if config.wait else None
    attempt = 0

    def ctx_for(region: str) -> RegionContext:
        if region not in cache:
            cache[region] = prepare_region(region)
        return cache[region]

    def sweep(
        spot: bool,
        candidates: list[str],
        failures: dict[str, tuple[str, str]],
        fatal: set[str],
    ) -> RegionLaunch | None:
        mkt = "spot" if spot else "on-demand"
        for region in candidates:
            ctx = ctx_for(region)
            if on_attempt:
                on_attempt(region, mkt, attempt)
            try:
                inst = _run_instances(ctx.ec2_client, config, ctx.ami["ImageId"], ctx.sg_id, region, spot)
            except CapacityError as e:
                failures[region] = ("capacity", str(e))
                continue
            except RegionFatalError as e:
                failures[region] = (e.kind, e.message)
                if region not in fatal:
                    fatal.add(region)
                    if on_region_fatal:
                        on_region_fatal(region, e.kind, e.message)
                continue
            return RegionLaunch(region, ctx, inst, mkt)
        return None

    # --- Primary market (spot unless --on-demand), with optional wait loop ---
    market = "spot" if primary_spot else "on-demand"
    failures: dict[str, tuple[str, str]] = {}
    fatal: set[str] = set()

    while True:
        candidates = [r for r in regions if r not in fatal]
        result = sweep(primary_spot, candidates, failures, fatal) if candidates else None
        if result is not None:
            return result

        capacity_left = [r for r in regions if r not in fatal and failures.get(r, ("", ""))[0] == "capacity"]

        if config.wait and capacity_left:
            now = clock()
            assert deadline is not None
            if now >= deadline:
                raise _aggregated_error(config, regions, failures, market, suffix=f" within {config.wait_timeout}s")
            sleep_s = min(backoff_sleep_seconds(attempt, rng=rng), max(0.0, deadline - now))
            if on_wait:
                on_wait(attempt + 1, sleep_s, now - start)
            sleeper(sleep_s)
            attempt += 1
            continue
        break

    # --- No-wait on-demand fallback (spot was the primary market) ---
    if primary_spot:
        secho(
            f"\n  No spot capacity for {config.instance_type} in {', '.join(regions)}.",
            fg="yellow",
        )
        confirmed = (
            confirm_on_demand()
            if confirm_on_demand
            else (not is_text() or click.confirm("  Retry as on-demand instance?"))
        )
        if not confirmed:
            raise CLIError("Launch cancelled.") from None
        # Spot quota/price is independent of on-demand quota — try every region.
        od_failures: dict[str, tuple[str, str]] = {}
        od_fatal: set[str] = set()
        result = sweep(False, list(regions), od_failures, od_fatal)
        if result is not None:
            return result
        raise _aggregated_error(config, regions, od_failures, "on-demand")

    raise _aggregated_error(config, regions, failures, "on-demand")


_UBUNTU_AMI = "ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"

# Prefix patterns for mapping instance types to GPU quota families.
_FAMILY_PREFIXES: list[tuple[tuple[str, ...], str]] = [
    (("g", "vt"), "gvt"),
    (("p",), "p"),
    (("dl",), "dl"),
]


def instance_type_to_family(instance_type: str) -> str | None:
    """Map an EC2 instance type (e.g. 'g4dn.xlarge') to its GPU quota family.

    Returns None for non-GPU instance types (e.g. 't3.medium').
    """
    prefix = instance_type.split(".", maxsplit=1)[0]
    for prefixes, family in _FAMILY_PREFIXES:
        if prefix.startswith(prefixes):
            return family
    return None


def _quota_hint(quota_type: str, family: str, region: str | None = None) -> str:
    # Quotas are per-region — pin the suggested commands to the region that
    # actually failed, otherwise the user inspects the wrong region's limits.
    region_flag = f" --region {region}" if region else ""
    return (
        "Check your current quotas with:\n"
        f"    aws-bootstrap quota show --family {family}{region_flag}\n\n"
        "  Request an increase with:\n"
        f"    aws-bootstrap quota request --family {family} --type {quota_type} --desired-value 4{region_flag}\n\n"
        "  To test the flow without GPU quotas, try:\n"
        f'    aws-bootstrap launch --instance-type t3.medium --ami-filter "{_UBUNTU_AMI}"{region_flag}'
    )


def _quota_error_message(code: str, config: LaunchConfig, region: str | None = None) -> str:
    """Full, region-pinned quota-exceeded message + remediation hint."""
    if code == "MaxSpotInstanceCountExceeded":
        quota_type = "spot"
        label = "Spot instance"
    else:
        # VcpuLimitExceeded is always an on-demand quota error
        quota_type = "on-demand"
        label = "On-demand vCPU"
    family = instance_type_to_family(config.instance_type) or "gvt"
    failed_region = region or config.region
    hint = _quota_hint(quota_type, family, failed_region)
    return (
        f"{label} quota exceeded for {config.instance_type} in {failed_region}.\n\n"
        f"  Your account's {quota_type} vCPU limit for this instance family is too low.\n"
        f"  {hint}"
    )


def _raise_quota_error(code: str, config: LaunchConfig, region: str | None = None) -> None:
    raise CLIError(_quota_error_message(code, config, region))


def find_tagged_instances(ec2_client, tag_value: str) -> list[dict]:
    """Find all non-terminated instances with the created-by tag."""
    response = ec2_client.describe_instances(
        Filters=[
            {"Name": f"tag:{TAG_CREATED_BY}", "Values": [tag_value]},
            {
                "Name": "instance-state-name",
                "Values": ["pending", "running", "stopping", "stopped", "shutting-down"],
            },
        ]
    )
    instances = []
    for reservation in response["Reservations"]:
        for inst in reservation["Instances"]:
            name = next((tag["Value"] for tag in inst.get("Tags", []) if tag["Key"] == TAG_NAME), "")
            instances.append(
                {
                    "InstanceId": inst["InstanceId"],
                    "Name": name,
                    "State": inst["State"]["Name"],
                    "InstanceType": inst["InstanceType"],
                    "PublicIp": inst.get("PublicIpAddress", ""),
                    "LaunchTime": inst["LaunchTime"],
                    "Lifecycle": inst.get("InstanceLifecycle", "on-demand"),
                    "AvailabilityZone": inst["Placement"]["AvailabilityZone"],
                }
            )
    return instances


def get_spot_price(ec2_client, instance_type: str, availability_zone: str) -> float | None:
    """Get the current spot price for an instance type in a given AZ.

    Returns the hourly price as a float, or None if unavailable.
    """
    response = ec2_client.describe_spot_price_history(
        InstanceTypes=[instance_type],
        ProductDescriptions=["Linux/UNIX"],
        AvailabilityZone=availability_zone,
        StartTime=datetime.now(UTC),
        MaxResults=1,
    )
    prices = response.get("SpotPriceHistory", [])
    if not prices:
        return None
    return float(prices[0]["SpotPrice"])


def list_instance_types(ec2_client, name_prefix: str = "g4dn") -> list[dict]:
    """List EC2 instance types matching a name prefix (e.g. 'g4dn', 'p3').

    Returns a list of dicts with InstanceType, vCPUs, MemoryMiB, and GPUs info,
    sorted by instance type name.
    """
    paginator = ec2_client.get_paginator("describe_instance_types")
    pages = paginator.paginate(
        Filters=[{"Name": "instance-type", "Values": [f"{name_prefix}.*"]}],
    )
    results = []
    for page in pages:
        for it in page["InstanceTypes"]:
            gpus = it.get("GpuInfo", {}).get("Gpus", [])
            gpu_summary = ""
            if gpus:
                g = gpus[0]
                mem = g.get("MemoryInfo", {}).get("SizeInMiB", 0)
                gpu_summary = f"{g.get('Count', '?')}x {g.get('Name', 'GPU')} ({mem} MiB)"
            results.append(
                {
                    "InstanceType": it["InstanceType"],
                    "VCpuCount": it["VCpuInfo"]["DefaultVCpus"],
                    "MemoryMiB": it["MemoryInfo"]["SizeInMiB"],
                    "GpuSummary": gpu_summary,
                }
            )
    results.sort(key=lambda x: x["InstanceType"])
    return results


def list_amis(ec2_client, ami_filter: str) -> list[dict]:
    """List available AMIs matching a name filter pattern.

    Returns a list of dicts with ImageId, Name, CreationDate, and Architecture,
    sorted by creation date (newest first). Limited to the 20 most recent.
    """
    owners = None
    for prefix, owner_ids in _OWNER_HINTS.items():
        if ami_filter.startswith(prefix):
            owners = owner_ids
            break

    params: dict = {
        "Filters": [
            {"Name": "name", "Values": [ami_filter]},
            {"Name": "state", "Values": ["available"]},
            {"Name": "architecture", "Values": ["x86_64"]},
        ],
    }
    if owners:
        params["Owners"] = owners

    response = ec2_client.describe_images(**params)
    images = response["Images"]
    images.sort(key=lambda x: x["CreationDate"], reverse=True)
    return [
        {
            "ImageId": img["ImageId"],
            "Name": img["Name"],
            "CreationDate": img["CreationDate"],
            "Architecture": img.get("Architecture", ""),
        }
        for img in images[:20]
    ]


def terminate_tagged_instances(ec2_client, instance_ids: list[str]) -> list[dict]:
    """Terminate instances by ID. Returns the state changes."""
    response = ec2_client.terminate_instances(InstanceIds=instance_ids)
    return response["TerminatingInstances"]


def wait_instance_ready(ec2_client, instance_id: str) -> dict:
    """Wait for the instance to be running and pass status checks."""
    echo("  Waiting for instance " + click.style(instance_id, fg="bright_white") + " to enter 'running' state...")
    waiter = ec2_client.get_waiter("instance_running")
    waiter.wait(InstanceIds=[instance_id], WaiterConfig=INSTANCE_RUNNING_WAITER)
    secho("  Instance running.", fg="green")

    echo("  Waiting for instance status checks to pass...")
    waiter = ec2_client.get_waiter("instance_status_ok")
    waiter.wait(InstanceIds=[instance_id], WaiterConfig=INSTANCE_STATUS_OK_WAITER)
    secho("  Status checks passed.", fg="green")

    # Refresh instance info to get public IP
    desc = ec2_client.describe_instances(InstanceIds=[instance_id])
    instance = desc["Reservations"][0]["Instances"][0]
    return instance


# ---------------------------------------------------------------------------
# EBS data volume operations
# ---------------------------------------------------------------------------


def create_ebs_volume(ec2_client, size_gb: int, availability_zone: str, tag_value: str, instance_id: str) -> str:
    """Create a gp3 EBS volume and wait for it to become available.

    Returns the volume ID.
    """
    response = ec2_client.create_volume(
        AvailabilityZone=availability_zone,
        Size=size_gb,
        VolumeType=VOLUME_TYPE,
        TagSpecifications=[
            {
                "ResourceType": RES_VOLUME,
                "Tags": [
                    {"Key": TAG_CREATED_BY, "Value": tag_value},
                    {"Key": TAG_NAME, "Value": f"aws-bootstrap-data-{instance_id}"},
                    {"Key": TAG_BOOTSTRAP_INSTANCE, "Value": instance_id},
                ],
            }
        ],
    )
    volume_id = response["VolumeId"]

    waiter = ec2_client.get_waiter("volume_available")
    waiter.wait(VolumeIds=[volume_id], WaiterConfig=EBS_VOLUME_WAITER)
    return volume_id


def validate_ebs_volume(ec2_client, volume_id: str, availability_zone: str) -> dict:
    """Validate that an existing EBS volume can be attached.

    Checks that the volume exists, is available (not in-use), and is in the
    correct availability zone. Returns the volume description dict.

    Raises CLIError for validation failures.
    """
    # AZ -> region (e.g. "us-east-1a" -> "us-east-1"); EBS volumes are
    # region/AZ-scoped, so name the region the lookup actually used.
    region = availability_zone[:-1] if availability_zone else None
    not_found = (
        f"EBS volume not found: {volume_id}"
        + (f" in {region} (the region this launch landed in)." if region else ".")
        + "\n  EBS volumes are region-scoped — pass --ebs-volume-id only when launching"
        " in the volume's region."
    )
    try:
        response = ec2_client.describe_volumes(VolumeIds=[volume_id])
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "InvalidVolume.NotFound":
            raise CLIError(not_found) from None
        raise

    volumes = response["Volumes"]
    if not volumes:
        raise CLIError(not_found)

    vol = volumes[0]

    if vol["State"] != "available":
        raise CLIError(
            f"EBS volume {volume_id} is currently '{vol['State']}' (must be 'available').\n"
            "  Detach it from its current instance first."
        )

    if vol["AvailabilityZone"] != availability_zone:
        raise CLIError(
            f"EBS volume {volume_id} is in {vol['AvailabilityZone']} "
            f"but the instance is in {availability_zone}.\n"
            "  EBS volumes must be in the same availability zone as the instance."
        )

    return vol


def attach_ebs_volume(ec2_client, volume_id: str, instance_id: str, device_name: str = EBS_DEVICE_NAME) -> None:
    """Attach an EBS volume to an instance and wait for it to be in-use."""
    ec2_client.attach_volume(
        VolumeId=volume_id,
        InstanceId=instance_id,
        Device=device_name,
    )
    waiter = ec2_client.get_waiter("volume_in_use")
    waiter.wait(VolumeIds=[volume_id], WaiterConfig=EBS_VOLUME_WAITER)


def detach_ebs_volume(ec2_client, volume_id: str) -> None:
    """Detach an EBS volume and wait for it to become available."""
    ec2_client.detach_volume(VolumeId=volume_id)
    waiter = ec2_client.get_waiter("volume_available")
    waiter.wait(VolumeIds=[volume_id], WaiterConfig=EBS_VOLUME_WAITER)


def delete_ebs_volume(ec2_client, volume_id: str) -> None:
    """Delete an EBS volume."""
    ec2_client.delete_volume(VolumeId=volume_id)


def find_ebs_volumes_for_instance(ec2_client, instance_id: str, tag_value: str) -> list[dict]:
    """Find EBS data volumes associated with an instance via tags.

    Returns a list of dicts with VolumeId, Size, Device, and State.
    Excludes root volumes (only returns volumes tagged by aws-bootstrap).
    """
    try:
        response = ec2_client.describe_volumes(
            Filters=[
                {"Name": f"tag:{TAG_BOOTSTRAP_INSTANCE}", "Values": [instance_id]},
                {"Name": f"tag:{TAG_CREATED_BY}", "Values": [tag_value]},
            ]
        )
    except botocore.exceptions.ClientError:
        return []

    volumes = []
    for vol in response.get("Volumes", []):
        device = ""
        if vol.get("Attachments"):
            device = vol["Attachments"][0].get("Device", "")
        volumes.append(
            {
                "VolumeId": vol["VolumeId"],
                "Size": vol["Size"],
                "Device": device,
                "State": vol["State"],
            }
        )
    return volumes


def find_orphan_ebs_volumes(ec2_client, tag_value: str, live_instance_ids: set[str]) -> list[dict]:
    """Find aws-bootstrap EBS volumes whose linked instance no longer exists.

    Only returns volumes in ``available`` state (not attached to any instance).
    Volumes that are ``in-use`` are never considered orphans, even if their
    tagged instance ID is not in *live_instance_ids*.

    Returns a list of dicts with VolumeId, Size, State, and InstanceId
    (the instance ID from the ``aws-bootstrap-instance`` tag).
    """
    try:
        response = ec2_client.describe_volumes(
            Filters=[
                {"Name": f"tag:{TAG_CREATED_BY}", "Values": [tag_value]},
                {"Name": "status", "Values": ["available"]},
            ]
        )
    except botocore.exceptions.ClientError:
        return []

    orphans = []
    for vol in response.get("Volumes", []):
        tags = {t["Key"]: t["Value"] for t in vol.get("Tags", [])}
        linked_instance = tags.get(TAG_BOOTSTRAP_INSTANCE, "")
        if linked_instance and linked_instance not in live_instance_ids:
            orphans.append(
                {
                    "VolumeId": vol["VolumeId"],
                    "Size": vol["Size"],
                    "State": vol["State"],
                    "InstanceId": linked_instance,
                }
            )
    return orphans
