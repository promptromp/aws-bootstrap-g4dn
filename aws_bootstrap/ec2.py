"""EC2 instance provisioning: AMI lookup, security groups, and instance launch."""

from __future__ import annotations

import botocore.exceptions
import click

from .config import LaunchConfig


def get_latest_dl_ami(ec2_client, ami_filter: str) -> dict:
    """Find the latest Deep Learning AMI matching the filter pattern."""
    response = ec2_client.describe_images(
        Owners=["amazon"],
        Filters=[
            {"Name": "name", "Values": [ami_filter]},
            {"Name": "state", "Values": ["available"]},
            {"Name": "architecture", "Values": ["x86_64"]},
        ],
    )
    images = response["Images"]
    if not images:
        raise click.ClickException(
            f"No AMI found matching filter: {ami_filter}\nTry adjusting --ami-filter or check the region."
        )

    images.sort(key=lambda x: x["CreationDate"], reverse=True)
    ami = images[0]
    return ami


def ensure_security_group(ec2_client, name: str, tag_value: str) -> str:
    """Find or create a security group with SSH ingress in the default VPC."""
    # Find default VPC
    vpcs = ec2_client.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])
    if not vpcs["Vpcs"]:
        raise click.ClickException("No default VPC found. Create one or specify a VPC.")
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
                "FromPort": 22,
                "ToPort": 22,
                "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "SSH access"}],
            }
        ],
    )
    click.secho(f"  Created security group '{name}' ({sg_id}) with SSH ingress.", fg="green")
    return sg_id


def launch_instance(ec2_client, config: LaunchConfig, ami_id: str, sg_id: str) -> dict:
    """Launch an EC2 instance (spot or on-demand)."""
    launch_params = dict(
        ImageId=ami_id,
        InstanceType=config.instance_type,
        KeyName=config.key_name,
        SecurityGroupIds=[sg_id],
        MinCount=1,
        MaxCount=1,
        BlockDeviceMappings=[
            {
                "DeviceName": "/dev/sda1",
                "Ebs": {
                    "VolumeSize": config.volume_size,
                    "VolumeType": "gp3",
                    "DeleteOnTermination": True,
                },
            }
        ],
        TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name", "Value": f"aws-bootstrap-{config.instance_type}"},
                    {"Key": "created-by", "Value": config.tag_value},
                ],
            }
        ],
    )

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
                raise click.ClickException("Launch cancelled.") from None
        else:
            raise

    instance = response["Instances"][0]
    return instance


QUOTA_HINT = (
    "See the 'EC2 vCPU Quotas' section in README.md for instructions on\n"
    "  checking and requesting quota increases.\n\n"
    "  To test the flow without GPU quotas, try:\n"
    '    aws-bootstrap launch --instance-type t3.medium --ami-filter "Ubuntu Server 24.04*"'
)


def _raise_quota_error(code: str, config: LaunchConfig) -> None:
    pricing = "spot" if config.spot else "on-demand"
    if code == "MaxSpotInstanceCountExceeded":
        msg = (
            f"Spot instance quota exceeded for {config.instance_type} in {config.region}.\n\n"
            f"  Your account's spot vCPU limit for this instance family is too low.\n"
            f"  {QUOTA_HINT}"
        )
    else:
        msg = (
            f"On-demand vCPU quota exceeded for {config.instance_type} in {config.region}.\n\n"
            f"  Your account's {pricing} vCPU limit for this instance family is too low.\n"
            f"  {QUOTA_HINT}"
        )
    raise click.ClickException(msg)


def find_tagged_instances(ec2_client, tag_value: str) -> list[dict]:
    """Find all running/pending instances with the created-by tag."""
    response = ec2_client.describe_instances(
        Filters=[
            {"Name": "tag:created-by", "Values": [tag_value]},
            {"Name": "instance-state-name", "Values": ["pending", "running", "stopping", "stopped"]},
        ]
    )
    instances = []
    for reservation in response["Reservations"]:
        for inst in reservation["Instances"]:
            name = ""
            for tag in inst.get("Tags", []):
                if tag["Key"] == "Name":
                    name = tag["Value"]
            instances.append(
                {
                    "InstanceId": inst["InstanceId"],
                    "Name": name,
                    "State": inst["State"]["Name"],
                    "InstanceType": inst["InstanceType"],
                    "PublicIp": inst.get("PublicIpAddress", ""),
                    "LaunchTime": inst["LaunchTime"],
                }
            )
    return instances


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
