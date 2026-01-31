"""EC2 instance provisioning: AMI lookup, security groups, and instance launch."""

from __future__ import annotations
from datetime import UTC, datetime

import botocore.exceptions
import click

from .config import LaunchConfig


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
        click.echo(msg + f" already exists ({sg_id}), reusing.")
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
    click.secho(f"  Created security group '{name}' ({sg_id}) with SSH ingress.", fg="green")
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
            click.secho(f"\n  Spot request failed: {e.response['Error']['Message']}", fg="yellow")
            if click.confirm("  Retry as on-demand instance?"):
                launch_params.pop("InstanceMarketOptions", None)
                try:
                    response = ec2_client.run_instances(**launch_params)
                except botocore.exceptions.ClientError as retry_e:
                    retry_code = retry_e.response["Error"]["Code"]
                    if retry_code in ("MaxSpotInstanceCountExceeded", "VcpuLimitExceeded"):
                        _raise_quota_error(retry_code, config)
                    raise
            else:
                raise CLIError("Launch cancelled.") from None
        else:
            raise

    return response["Instances"][0]


_UBUNTU_AMI = "ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"

QUOTA_HINT = (
    "See the 'EC2 vCPU Quotas' section in README.md for instructions on\n"
    "  checking and requesting quota increases.\n\n"
    "  To test the flow without GPU quotas, try:\n"
    f'    aws-bootstrap launch --instance-type t3.medium --ami-filter "{_UBUNTU_AMI}"'
)


def _raise_quota_error(code: str, config: LaunchConfig) -> None:
    if code == "MaxSpotInstanceCountExceeded":
        pricing = "spot"
        label = "Spot instance"
    else:
        pricing = "spot" if config.spot else "on-demand"
        label = "On-demand vCPU"
    msg = (
        f"{label} quota exceeded for {config.instance_type} in {config.region}.\n\n"
        f"  Your account's {pricing} vCPU limit for this instance family is too low.\n"
        f"  {QUOTA_HINT}"
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
    click.echo("  Waiting for instance " + click.style(instance_id, fg="bright_white") + " to enter 'running' state...")
    waiter = ec2_client.get_waiter("instance_running")
    waiter.wait(InstanceIds=[instance_id], WaiterConfig={"Delay": 10, "MaxAttempts": 60})
    click.secho("  Instance running.", fg="green")

    click.echo("  Waiting for instance status checks to pass...")
    waiter = ec2_client.get_waiter("instance_status_ok")
    waiter.wait(InstanceIds=[instance_id], WaiterConfig={"Delay": 15, "MaxAttempts": 60})
    click.secho("  Status checks passed.", fg="green")

    # Refresh instance info to get public IP
    desc = ec2_client.describe_instances(InstanceIds=[instance_id])
    instance = desc["Reservations"][0]["Instances"][0]
    return instance
