"""CLI entry point for aws-bootstrap-g4dn."""

from __future__ import annotations
from datetime import UTC, datetime
from pathlib import Path

import boto3
import botocore.exceptions
import click

from .config import LaunchConfig
from .ec2 import (
    CLIError,
    ensure_security_group,
    find_tagged_instances,
    get_latest_ami,
    get_spot_price,
    launch_instance,
    list_amis,
    list_instance_types,
    terminate_tagged_instances,
    wait_instance_ready,
)
from .ssh import (
    add_ssh_host,
    get_ssh_host_details,
    import_key_pair,
    list_ssh_hosts,
    private_key_path,
    query_gpu_info,
    remove_ssh_host,
    run_remote_setup,
    wait_for_ssh,
)


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


class _AWSGroup(click.Group):
    """Click group that catches common AWS credential/auth errors."""

    def invoke(self, ctx):
        try:
            return super().invoke(ctx)
        except botocore.exceptions.NoCredentialsError:
            raise CLIError(
                "Unable to locate AWS credentials.\n\n"
                "  Make sure you have configured AWS credentials using one of:\n"
                "    - Set the AWS_PROFILE environment variable:  export AWS_PROFILE=<profile-name>\n"
                "    - Pass --profile to the command:  aws-bootstrap <command> --profile <profile-name>\n"
                "    - Configure a default profile:  aws configure\n\n"
                "  See: https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html"
            ) from None
        except botocore.exceptions.ProfileNotFound as e:
            raise CLIError(f"{e}\n\n  List available profiles with:  aws configure list-profiles") from None
        except botocore.exceptions.PartialCredentialsError as e:
            raise CLIError(
                f"Incomplete AWS credentials: {e}\n\n  Check your AWS configuration with:  aws configure list"
            ) from None
        except botocore.exceptions.ClientError as e:
            code = e.response["Error"]["Code"]
            if code in ("AuthFailure", "UnauthorizedOperation", "ExpiredTokenException", "ExpiredToken"):
                raise CLIError(
                    f"AWS authorization failed: {e.response['Error']['Message']}\n\n"
                    "  Your credentials may be expired or lack the required permissions.\n"
                    "  Check your AWS configuration with:  aws configure list"
                ) from None
            raise


@click.group(cls=_AWSGroup)
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
@click.option(
    "--python-version",
    default=None,
    help="Python version for the remote venv (e.g. 3.13, 3.14.2). Passed to uv during setup.",
)
@click.option("--ssh-port", default=22, show_default=True, type=int, help="SSH port on the remote instance.")
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
    python_version,
    ssh_port,
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
        ssh_port=ssh_port,
        python_version=python_version,
    )
    if ami_filter:
        config.ami_filter = ami_filter
    if profile:
        config.profile = profile

    # Validate key path
    if not config.key_path.exists():
        raise CLIError(f"SSH public key not found: {config.key_path}")

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
    sg_id = ensure_security_group(ec2, config.security_group, config.tag_value, ssh_port=config.ssh_port)

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
        if config.ssh_port != 22:
            val("SSH port", str(config.ssh_port))
        if config.python_version:
            val("Python version", config.python_version)
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
    if not wait_for_ssh(public_ip, config.ssh_user, config.key_path, port=config.ssh_port):
        warn("SSH did not become available within the timeout.")
        port_flag = f" -p {config.ssh_port}" if config.ssh_port != 22 else ""
        info(
            f"Instance is running — try connecting manually:"
            f" ssh -i {private_key}{port_flag} {config.ssh_user}@{public_ip}"
        )
        return

    if config.run_setup:
        if not SETUP_SCRIPT.exists():
            warn(f"Setup script not found at {SETUP_SCRIPT}, skipping.")
        else:
            info("Running remote setup...")
            if run_remote_setup(
                public_ip, config.ssh_user, config.key_path, SETUP_SCRIPT, config.python_version, port=config.ssh_port
            ):
                success("Remote setup completed successfully.")
            else:
                warn("Remote setup failed. Instance is still running.")

    # Add SSH config alias
    alias = add_ssh_host(
        instance_id=instance_id,
        hostname=public_ip,
        user=config.ssh_user,
        key_path=config.key_path,
        alias_prefix=config.alias_prefix,
        port=config.ssh_port,
    )
    success(f"Added SSH config alias: {alias}")

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
    val("SSH alias", alias)

    port_flag = f" -p {config.ssh_port}" if config.ssh_port != 22 else ""

    click.echo()
    click.secho("  SSH:", fg="cyan")
    click.secho(f"    ssh{port_flag} {alias}", bold=True)
    info(f"or: ssh -i {private_key}{port_flag} {config.ssh_user}@{public_ip}")

    click.echo()
    click.secho("  Jupyter (via SSH tunnel):", fg="cyan")
    click.secho(f"    ssh -NL 8888:localhost:8888{port_flag} {alias}", bold=True)
    info(f"or: ssh -i {private_key} -NL 8888:localhost:8888{port_flag} {config.ssh_user}@{public_ip}")
    info("Then open: http://localhost:8888")
    info("Notebook: ~/gpu_smoke_test.ipynb (GPU smoke test)")

    click.echo()
    click.secho("  VSCode Remote SSH:", fg="cyan")
    click.secho(
        f"    code --folder-uri vscode-remote://ssh-remote+{alias}/home/{config.ssh_user}/workspace",
        bold=True,
    )

    click.echo()
    click.secho("  GPU Benchmark:", fg="cyan")
    click.secho(f"    ssh {alias} 'python ~/gpu_benchmark.py'", bold=True)
    info("Runs CNN (MNIST) and Transformer benchmarks with tqdm progress")

    click.echo()
    click.secho("  Terminate:", fg="cyan")
    click.secho(f"    aws-bootstrap terminate {instance_id} --region {config.region}", bold=True)
    click.echo()


@main.command()
@click.option("--region", default="us-west-2", show_default=True, help="AWS region.")
@click.option("--profile", default=None, help="AWS profile override.")
@click.option("--gpu", is_flag=True, default=False, help="Query GPU info (CUDA, driver) via SSH.")
@click.option(
    "--instructions/--no-instructions",
    "-I",
    default=True,
    show_default=True,
    help="Show connection commands (SSH, Jupyter, VSCode) for each running instance.",
)
def status(region, profile, gpu, instructions):
    """Show running instances created by aws-bootstrap."""
    session = boto3.Session(profile_name=profile, region_name=region)
    ec2 = session.client("ec2")

    instances = find_tagged_instances(ec2, "aws-bootstrap-g4dn")
    if not instances:
        click.secho("No active aws-bootstrap instances found.", fg="yellow")
        return

    ssh_hosts = list_ssh_hosts()

    click.secho(f"\n  Found {len(instances)} instance(s):\n", bold=True, fg="cyan")
    if gpu:
        click.echo("  " + click.style("Querying GPU info via SSH...", dim=True))
        click.echo()

    for inst in instances:
        state = inst["State"]
        state_color = {
            "running": "green",
            "pending": "yellow",
            "stopping": "yellow",
            "stopped": "red",
            "shutting-down": "red",
        }.get(state, "white")
        alias = ssh_hosts.get(inst["InstanceId"])
        alias_str = f" ({alias})" if alias else ""
        click.echo(
            "  "
            + click.style(inst["InstanceId"], fg="bright_white")
            + click.style(alias_str, fg="cyan")
            + "  "
            + click.style(state, fg=state_color)
        )
        val("    Type", inst["InstanceType"])
        if inst["PublicIp"]:
            val("    IP", inst["PublicIp"])

        # Look up SSH config details once (used by --gpu and --with-instructions)
        details = None
        if (gpu or instructions) and state == "running" and inst["PublicIp"]:
            details = get_ssh_host_details(inst["InstanceId"])

        # GPU info (opt-in, only for running instances with a public IP)
        if gpu and state == "running" and inst["PublicIp"]:
            if details:
                gpu_info = query_gpu_info(details.hostname, details.user, details.identity_file, port=details.port)
            else:
                gpu_info = query_gpu_info(
                    inst["PublicIp"],
                    "ubuntu",
                    Path("~/.ssh/id_ed25519").expanduser(),
                )
            if gpu_info:
                val("    GPU", f"{gpu_info.gpu_name} ({gpu_info.architecture})")
                if gpu_info.cuda_toolkit_version:
                    cuda_str = gpu_info.cuda_toolkit_version
                    if gpu_info.cuda_driver_version != gpu_info.cuda_toolkit_version:
                        cuda_str += f" (driver supports up to {gpu_info.cuda_driver_version})"
                else:
                    cuda_str = f"{gpu_info.cuda_driver_version} (driver max, toolkit unknown)"
                val("    CUDA", cuda_str)
                val("    Driver", gpu_info.driver_version)
            else:
                click.echo("    GPU: " + click.style("unavailable", dim=True))

        lifecycle = inst["Lifecycle"]
        is_spot = lifecycle == "spot"

        if is_spot:
            spot_price = get_spot_price(ec2, inst["InstanceType"], inst["AvailabilityZone"])
            if spot_price is not None:
                val("    Pricing", f"spot (${spot_price:.4f}/hr)")
            else:
                val("    Pricing", "spot")
        else:
            val("    Pricing", "on-demand")

        if state == "running" and is_spot:
            uptime = datetime.now(UTC) - inst["LaunchTime"]
            total_seconds = int(uptime.total_seconds())
            hours, remainder = divmod(total_seconds, 3600)
            minutes = remainder // 60
            val("    Uptime", f"{hours}h {minutes:02d}m")
            if spot_price is not None:
                uptime_hours = uptime.total_seconds() / 3600
                est_cost = uptime_hours * spot_price
                val("    Est. cost", f"~${est_cost:.4f}")

        val("    Launched", str(inst["LaunchTime"]))

        # Connection instructions (opt-in, only for running instances with a public IP and alias)
        if instructions and state == "running" and inst["PublicIp"] and alias:
            user = details.user if details else "ubuntu"
            port = details.port if details else 22
            port_flag = f" -p {port}" if port != 22 else ""

            click.echo()
            click.secho("    SSH:", fg="cyan")
            click.secho(f"      ssh{port_flag} {alias}", bold=True)

            click.secho("    Jupyter (via SSH tunnel):", fg="cyan")
            click.secho(f"      ssh -NL 8888:localhost:8888{port_flag} {alias}", bold=True)

            click.secho("    VSCode Remote SSH:", fg="cyan")
            click.secho(
                f"      code --folder-uri vscode-remote://ssh-remote+{alias}/home/{user}/workspace",
                bold=True,
            )

            click.secho("    GPU Benchmark:", fg="cyan")
            click.secho(f"      ssh {alias} 'python ~/gpu_benchmark.py'", bold=True)

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
        removed_alias = remove_ssh_host(change["InstanceId"])
        if removed_alias:
            info(f"Removed SSH config alias: {removed_alias}")
    click.echo()
    success(f"Terminated {len(changes)} instance(s).")


# ---------------------------------------------------------------------------
# list command group
# ---------------------------------------------------------------------------

DEFAULT_AMI_PREFIX = "Deep Learning Base OSS Nvidia Driver GPU AMI*"


@main.group(name="list")
def list_cmd():
    """List AWS resources (instance types, AMIs)."""


@list_cmd.command(name="instance-types")
@click.option("--prefix", default="g4dn", show_default=True, help="Instance type family prefix to filter on.")
@click.option("--region", default="us-west-2", show_default=True, help="AWS region.")
@click.option("--profile", default=None, help="AWS profile override.")
def list_instance_types_cmd(prefix, region, profile):
    """List EC2 instance types matching a family prefix (e.g. g4dn, p3, g5)."""
    session = boto3.Session(profile_name=profile, region_name=region)
    ec2 = session.client("ec2")

    types = list_instance_types(ec2, prefix)
    if not types:
        click.secho(f"No instance types found matching '{prefix}.*'", fg="yellow")
        return

    click.secho(f"\n  {len(types)} instance type(s) matching '{prefix}.*':\n", bold=True, fg="cyan")

    # Header
    click.echo(
        "  " + click.style(f"{'Instance Type':<24}{'vCPUs':>6}{'Memory (MiB)':>14}  GPU", fg="bright_white", bold=True)
    )
    click.echo("  " + "-" * 72)

    for t in types:
        gpu = t["GpuSummary"] or "-"
        click.echo(f"  {t['InstanceType']:<24}{t['VCpuCount']:>6}{t['MemoryMiB']:>14}  {gpu}")

    click.echo()


@list_cmd.command(name="amis")
@click.option("--filter", "ami_filter", default=DEFAULT_AMI_PREFIX, show_default=True, help="AMI name pattern.")
@click.option("--region", default="us-west-2", show_default=True, help="AWS region.")
@click.option("--profile", default=None, help="AWS profile override.")
def list_amis_cmd(ami_filter, region, profile):
    """List available AMIs matching a name pattern."""
    session = boto3.Session(profile_name=profile, region_name=region)
    ec2 = session.client("ec2")

    amis = list_amis(ec2, ami_filter)
    if not amis:
        click.secho(f"No AMIs found matching '{ami_filter}'", fg="yellow")
        return

    click.secho(f"\n  {len(amis)} AMI(s) matching '{ami_filter}' (newest first):\n", bold=True, fg="cyan")

    for ami in amis:
        click.echo("  " + click.style(ami["ImageId"], fg="bright_white") + "  " + ami["CreationDate"][:10])
        click.echo(f"    {ami['Name']}")

    click.echo()
