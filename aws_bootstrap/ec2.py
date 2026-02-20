"""EC2 instance provisioning: AMI lookup, security groups, and instance launch."""

from __future__ import annotations
from datetime import UTC, datetime

import botocore.exceptions
import click

from .config import LaunchConfig
from .output import echo, is_text, secho


EBS_DEVICE_NAME = "/dev/sdf"
EBS_MOUNT_POINT = "/data"


class CLIError(click.ClickException):
    """A ClickException that displays the error message in red."""

    def show(self, file=None):  # type: ignore[no-untyped-def]
        if file is None:
            file = click.get_text_stream("stderr")
        click.secho(f"Error: {self.format_message()}", file=file, fg="red")


# Well-known AMI owners by name prefix
_OWNER_HINTS = {
    "Deep Learning": ["amazon"],
    "ubuntu": ["099720109477"],  # Canonical
    "Ubuntu": ["099720109477"],
    "RHEL": ["309956199498"],
    "al20": ["amazon"],  # Amazon Linux
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


def ensure_security_group(ec2_client, name: str, tag_value: str, ssh_port: int = 22) -> str:
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
                "ResourceType": "security-group",
                "Tags": [
                    {"Key": "created-by", "Value": tag_value},
                    {"Key": "Name", "Value": name},
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
                "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "SSH access"}],
            }
        ],
    )
    secho(f"  Created security group '{name}' ({sg_id}) with SSH ingress.", fg="green")
    return sg_id


def launch_instance(ec2_client, config: LaunchConfig, ami_id: str, sg_id: str) -> dict:
    """Launch an EC2 instance (spot or on-demand)."""
    launch_params = {
        "ImageId": ami_id,
        "InstanceType": config.instance_type,
        "KeyName": config.key_name,
        "SecurityGroupIds": [sg_id],
        "MinCount": 1,
        "MaxCount": 1,
        "BlockDeviceMappings": [
            {
                "DeviceName": "/dev/sda1",
                "Ebs": {
                    "VolumeSize": config.volume_size,
                    "VolumeType": "gp3",
                    "DeleteOnTermination": True,
                },
            }
        ],
        "TagSpecifications": [
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name", "Value": f"aws-bootstrap-{config.instance_type}"},
                    {"Key": "created-by", "Value": config.tag_value},
                ],
            }
        ],
    }

    if config.spot:
        launch_params["InstanceMarketOptions"] = {
            "MarketType": "spot",
            "SpotOptions": {
                "SpotInstanceType": "one-time",
                "InstanceInterruptionBehavior": "terminate",
            },
        }

    try:
        response = ec2_client.run_instances(**launch_params)
    except botocore.exceptions.ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("MaxSpotInstanceCountExceeded", "VcpuLimitExceeded"):
            _raise_quota_error(code, config)
        elif code in ("InsufficientInstanceCapacity", "SpotMaxPriceTooLow") and config.spot:
            secho(f"\n  Spot request failed: {e.response['Error']['Message']}", fg="yellow")
            if not is_text() or click.confirm("  Retry as on-demand instance?"):
                launch_params.pop("InstanceMarketOptions", None)
                try:
                    response = ec2_client.run_instances(**launch_params)
                except botocore.exceptions.ClientError as retry_e:
                    retry_code = retry_e.response["Error"]["Code"]
                    if retry_code in ("MaxSpotInstanceCountExceeded", "VcpuLimitExceeded"):
                        _raise_quota_error(retry_code, config)
                    if retry_code == "InsufficientInstanceCapacity":
                        raise CLIError(
                            f"Insufficient capacity for {config.instance_type} (on-demand) in {config.region}.\n\n"
                            "  Neither spot nor on-demand capacity is currently available.\n"
                            "  Try a different region, availability zone, or instance type."
                        ) from None
                    raise
            else:
                raise CLIError("Launch cancelled.") from None
        elif code == "InsufficientInstanceCapacity":
            raise CLIError(
                f"Insufficient capacity for {config.instance_type} in {config.region}.\n\n"
                "  The requested instance type is not currently available.\n"
                "  Try a different region, availability zone, or instance type."
            ) from None
        else:
            raise

    return response["Instances"][0]


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


def _quota_hint(quota_type: str, family: str) -> str:
    return (
        "Check your current quotas with:\n"
        f"    aws-bootstrap quota show --family {family}\n\n"
        "  Request an increase with:\n"
        f"    aws-bootstrap quota request --family {family} --type {quota_type} --desired-value 4\n\n"
        "  To test the flow without GPU quotas, try:\n"
        f'    aws-bootstrap launch --instance-type t3.medium --ami-filter "{_UBUNTU_AMI}"'
    )


def _raise_quota_error(code: str, config: LaunchConfig) -> None:
    if code == "MaxSpotInstanceCountExceeded":
        quota_type = "spot"
        label = "Spot instance"
    else:
        # VcpuLimitExceeded is always an on-demand quota error
        quota_type = "on-demand"
        label = "On-demand vCPU"
    family = instance_type_to_family(config.instance_type) or "gvt"
    hint = _quota_hint(quota_type, family)
    msg = (
        f"{label} quota exceeded for {config.instance_type} in {config.region}.\n\n"
        f"  Your account's {quota_type} vCPU limit for this instance family is too low.\n"
        f"  {hint}"
    )
    raise CLIError(msg)


def find_tagged_instances(ec2_client, tag_value: str) -> list[dict]:
    """Find all non-terminated instances with the created-by tag."""
    response = ec2_client.describe_instances(
        Filters=[
            {"Name": "tag:created-by", "Values": [tag_value]},
            {
                "Name": "instance-state-name",
                "Values": ["pending", "running", "stopping", "stopped", "shutting-down"],
            },
        ]
    )
    instances = []
    for reservation in response["Reservations"]:
        for inst in reservation["Instances"]:
            name = next((tag["Value"] for tag in inst.get("Tags", []) if tag["Key"] == "Name"), "")
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
    waiter.wait(InstanceIds=[instance_id], WaiterConfig={"Delay": 10, "MaxAttempts": 60})
    secho("  Instance running.", fg="green")

    echo("  Waiting for instance status checks to pass...")
    waiter = ec2_client.get_waiter("instance_status_ok")
    waiter.wait(InstanceIds=[instance_id], WaiterConfig={"Delay": 15, "MaxAttempts": 60})
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
        VolumeType="gp3",
        TagSpecifications=[
            {
                "ResourceType": "volume",
                "Tags": [
                    {"Key": "created-by", "Value": tag_value},
                    {"Key": "Name", "Value": f"aws-bootstrap-data-{instance_id}"},
                    {"Key": "aws-bootstrap-instance", "Value": instance_id},
                ],
            }
        ],
    )
    volume_id = response["VolumeId"]

    waiter = ec2_client.get_waiter("volume_available")
    waiter.wait(VolumeIds=[volume_id], WaiterConfig={"Delay": 5, "MaxAttempts": 24})
    return volume_id


def validate_ebs_volume(ec2_client, volume_id: str, availability_zone: str) -> dict:
    """Validate that an existing EBS volume can be attached.

    Checks that the volume exists, is available (not in-use), and is in the
    correct availability zone. Returns the volume description dict.

    Raises CLIError for validation failures.
    """
    try:
        response = ec2_client.describe_volumes(VolumeIds=[volume_id])
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "InvalidVolume.NotFound":
            raise CLIError(f"EBS volume not found: {volume_id}") from None
        raise

    volumes = response["Volumes"]
    if not volumes:
        raise CLIError(f"EBS volume not found: {volume_id}")

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
    waiter.wait(VolumeIds=[volume_id], WaiterConfig={"Delay": 5, "MaxAttempts": 24})


def detach_ebs_volume(ec2_client, volume_id: str) -> None:
    """Detach an EBS volume and wait for it to become available."""
    ec2_client.detach_volume(VolumeId=volume_id)
    waiter = ec2_client.get_waiter("volume_available")
    waiter.wait(VolumeIds=[volume_id], WaiterConfig={"Delay": 5, "MaxAttempts": 24})


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
                {"Name": "tag:aws-bootstrap-instance", "Values": [instance_id]},
                {"Name": "tag:created-by", "Values": [tag_value]},
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
                {"Name": "tag:created-by", "Values": [tag_value]},
                {"Name": "status", "Values": ["available"]},
            ]
        )
    except botocore.exceptions.ClientError:
        return []

    orphans = []
    for vol in response.get("Volumes", []):
        tags = {t["Key"]: t["Value"] for t in vol.get("Tags", [])}
        linked_instance = tags.get("aws-bootstrap-instance", "")
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
