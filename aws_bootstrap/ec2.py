"""EC2 instance provisioning: AMI lookup, security groups, and instance launch."""

from __future__ import annotations

import click
import botocore.exceptions

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
            f"No AMI found matching filter: {ami_filter}\n"
            "Try adjusting --ami-filter or check the region."
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
        click.echo("  Security group " + click.style(f"'{name}'", fg="bright_white") + f" already exists ({sg_id}), reusing.")
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
        if code in ("InsufficientInstanceCapacity", "SpotMaxPriceTooLow") and config.spot:
            click.secho(f"\n  Spot request failed: {e.response['Error']['Message']}", fg="yellow")
            if click.confirm("  Retry as on-demand instance?"):
                launch_params.pop("InstanceMarketOptions", None)
                response = ec2_client.run_instances(**launch_params)
            else:
                raise click.ClickException("Launch cancelled.")
        else:
            raise

    instance = response["Instances"][0]
    return instance


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
