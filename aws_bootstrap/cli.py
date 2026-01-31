"""CLI entry point for aws-bootstrap-g4dn."""

from __future__ import annotations
from pathlib import Path

import boto3
import click

from .config import LaunchConfig
from .ec2 import (
    ensure_security_group,
    find_tagged_instances,
    get_latest_ami,
    launch_instance,
    terminate_tagged_instances,
    wait_instance_ready,
)
from .ssh import import_key_pair, private_key_path, run_remote_setup, wait_for_ssh


SETUP_SCRIPT = Path(__file__).parent / "resources" / "remote_setup.sh"


def step(number: int, total: int, msg: str) -> None:
    click.secho(f"\n[{number}/{total}] {msg}", bold=True, fg="cyan")


def info(msg: str) -> None:
    click.echo(f"  {msg}")


def val(label: str, value: str) -> None:
    click.echo(f"  {label}: " + click.style(str(value), fg="bright_white"))


def success(msg: str) -> None:
    click.secho(f"  {msg}", fg="green")


def warn(msg: str) -> None:
    click.secho(f"  WARNING: {msg}", fg="yellow", err=True)


@click.group()
@click.version_option(package_name="aws-bootstrap-g4dn")
def main():
    """Bootstrap AWS EC2 GPU instances for hybrid local-remote development."""


@main.command()
@click.option("--instance-type", default="g4dn.xlarge", show_default=True, help="EC2 instance type.")
@click.option("--ami-filter", default=None, help="AMI name pattern filter (auto-detected if omitted).")
@click.option("--spot/--on-demand", default=True, show_default=True, help="Use spot or on-demand pricing.")
@click.option(
    "--key-path",
    default="~/.ssh/id_ed25519.pub",
    show_default=True,
    type=click.Path(),
    help="Path to local SSH public key.",
)
@click.option("--key-name", default="aws-bootstrap-key", show_default=True, help="AWS key pair name.")
@click.option("--region", default="us-west-2", show_default=True, help="AWS region.")
@click.option("--security-group", default="aws-bootstrap-ssh", show_default=True, help="Security group name.")
@click.option("--volume-size", default=100, show_default=True, type=int, help="Root EBS volume size in GB (gp3).")
@click.option("--no-setup", is_flag=True, default=False, help="Skip running the remote setup script.")
@click.option("--dry-run", is_flag=True, default=False, help="Show what would be done without executing.")
@click.option("--profile", default=None, help="AWS profile override (defaults to AWS_PROFILE env var).")
def launch(
    instance_type,
    ami_filter,
    spot,
    key_path,
    key_name,
    region,
    security_group,
    volume_size,
    no_setup,
    dry_run,
    profile,
):
    """Launch a GPU-accelerated EC2 instance."""
    config = LaunchConfig(
        instance_type=instance_type,
        spot=spot,
        key_path=Path(key_path).expanduser(),
        key_name=key_name,
        region=region,
        security_group=security_group,
        volume_size=volume_size,
        run_setup=not no_setup,
        dry_run=dry_run,
    )
    if ami_filter:
        config.ami_filter = ami_filter
    if profile:
        config.profile = profile

    # Validate key path
    if not config.key_path.exists():
        raise click.ClickException(f"SSH public key not found: {config.key_path}")

    # Build boto3 session
    session = boto3.Session(profile_name=config.profile, region_name=config.region)
    ec2 = session.client("ec2")

    # Step 1: AMI lookup
    step(1, 6, "Looking up AMI...")
    ami = get_latest_ami(ec2, config.ami_filter)
    info(f"Found: {ami['Name']}")
    val("AMI ID", ami["ImageId"])

    # Step 2: SSH key pair
    step(2, 6, "Importing SSH key pair...")
    import_key_pair(ec2, config.key_name, config.key_path)

    # Step 3: Security group
    step(3, 6, "Ensuring security group...")
    sg_id = ensure_security_group(ec2, config.security_group, config.tag_value)

    pricing = "spot" if config.spot else "on-demand"

    if config.dry_run:
        click.echo()
        click.secho("--- Dry Run Summary ---", bold=True, fg="yellow")
        val("Instance type", config.instance_type)
        val("AMI", f"{ami['ImageId']} ({ami['Name']})")
        val("Pricing", pricing)
        val("Key pair", config.key_name)
        val("Security group", sg_id)
        val("Volume", f"{config.volume_size} GB gp3")
        val("Region", config.region)
        val("Remote setup", "yes" if config.run_setup else "no")
        click.echo()
        click.secho("No resources launched (dry-run mode).", fg="yellow")
        return

    # Step 4: Launch instance
    step(4, 6, f"Launching {config.instance_type} instance ({pricing})...")
    instance = launch_instance(ec2, config, ami["ImageId"], sg_id)
    instance_id = instance["InstanceId"]
    val("Instance ID", instance_id)

    # Step 5: Wait for ready
    step(5, 6, "Waiting for instance to be ready...")
    instance = wait_instance_ready(ec2, instance_id)
    public_ip = instance.get("PublicIpAddress")
    if not public_ip:
        warn(f"No public IP assigned. Instance ID: {instance_id}")
        info("You may need to assign an Elastic IP or check your VPC settings.")
        return

    val("Public IP", public_ip)

    # Step 6: SSH and remote setup
    step(6, 6, "Waiting for SSH access...")
    private_key = private_key_path(config.key_path)
    if not wait_for_ssh(public_ip, config.ssh_user, config.key_path):
        warn("SSH did not become available within the timeout.")
        info(f"Instance is running — try connecting manually: ssh -i {private_key} {config.ssh_user}@{public_ip}")
        return

    if config.run_setup:
        if not SETUP_SCRIPT.exists():
            warn(f"Setup script not found at {SETUP_SCRIPT}, skipping.")
        else:
            info("Running remote setup...")
            if run_remote_setup(public_ip, config.ssh_user, config.key_path, SETUP_SCRIPT):
                success("Remote setup completed successfully.")
            else:
                warn("Remote setup failed. Instance is still running.")

    # Print connection info
    click.echo()
    click.secho("=" * 60, fg="green")
    click.secho("  Instance ready!", bold=True, fg="green")
    click.secho("=" * 60, fg="green")
    click.echo()
    val("Instance ID", instance_id)
    val("Public IP", public_ip)
    val("Instance", config.instance_type)
    val("Pricing", pricing)

    click.echo()
    click.secho("  SSH:", fg="cyan")
    click.secho(f"    ssh -i {private_key} {config.ssh_user}@{public_ip}", bold=True)

    click.echo()
    click.secho("  Jupyter (via SSH tunnel):", fg="cyan")
    click.secho(f"    ssh -i {private_key} -NL 8888:localhost:8888 {config.ssh_user}@{public_ip}", bold=True)
    info("Then open: http://localhost:8888")

    click.echo()
    click.secho("  Terminate:", fg="cyan")
    click.secho(f"    aws ec2 terminate-instances --instance-ids {instance_id} --region {config.region}", bold=True)
    click.secho(f"    aws-bootstrap terminate --region {config.region}", bold=True)
    click.echo()


@main.command()
@click.option("--region", default="us-west-2", show_default=True, help="AWS region.")
@click.option("--profile", default=None, help="AWS profile override.")
def status(region, profile):
    """Show running instances created by aws-bootstrap."""
    session = boto3.Session(profile_name=profile, region_name=region)
    ec2 = session.client("ec2")

    instances = find_tagged_instances(ec2, "aws-bootstrap-g4dn")
    if not instances:
        click.secho("No active aws-bootstrap instances found.", fg="yellow")
        return

    click.secho(f"\n  Found {len(instances)} instance(s):\n", bold=True, fg="cyan")
    for inst in instances:
        state = inst["State"]
        state_color = {"running": "green", "pending": "yellow", "stopping": "yellow", "stopped": "red"}.get(
            state, "white"
        )
        click.echo(
            "  " + click.style(inst["InstanceId"], fg="bright_white") + "  " + click.style(state, fg=state_color)
        )
        val("    Type", inst["InstanceType"])
        if inst["PublicIp"]:
            val("    IP", inst["PublicIp"])
        val("    Launched", str(inst["LaunchTime"]))
    click.echo()
    first_id = instances[0]["InstanceId"]
    click.echo("  To terminate:  " + click.style(f"aws-bootstrap terminate {first_id}", bold=True))
    click.echo()


@main.command()
@click.option("--region", default="us-west-2", show_default=True, help="AWS region.")
@click.option("--profile", default=None, help="AWS profile override.")
@click.option("--yes", "-y", is_flag=True, default=False, help="Skip confirmation prompt.")
@click.argument("instance_ids", nargs=-1)
def terminate(region, profile, yes, instance_ids):
    """Terminate instances created by aws-bootstrap.

    Pass specific instance IDs to terminate, or omit to terminate all
    aws-bootstrap instances in the region.
    """
    session = boto3.Session(profile_name=profile, region_name=region)
    ec2 = session.client("ec2")

    if instance_ids:
        targets = list(instance_ids)
    else:
        instances = find_tagged_instances(ec2, "aws-bootstrap-g4dn")
        if not instances:
            click.secho("No active aws-bootstrap instances found.", fg="yellow")
            return
        targets = [inst["InstanceId"] for inst in instances]
        click.secho(f"\n  Found {len(targets)} instance(s) to terminate:\n", bold=True, fg="cyan")
        for inst in instances:
            iid = click.style(inst["InstanceId"], fg="bright_white")
            click.echo(f"  {iid}  {inst['State']}  {inst['InstanceType']}")

    if not yes:
        click.echo()
        if not click.confirm(f"  Terminate {len(targets)} instance(s)?"):
            click.secho("  Cancelled.", fg="yellow")
            return

    changes = terminate_tagged_instances(ec2, targets)
    click.echo()
    for change in changes:
        prev = change["PreviousState"]["Name"]
        curr = change["CurrentState"]["Name"]
        click.echo(
            "  " + click.style(change["InstanceId"], fg="bright_white") + f"  {prev} -> " + click.style(curr, fg="red")
        )
    click.echo()
    success(f"Terminated {len(changes)} instance(s).")
