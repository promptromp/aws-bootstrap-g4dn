"""CLI entry point for aws-bootstrap-g4dn."""

from __future__ import annotations
from datetime import UTC, datetime
from pathlib import Path

import boto3
import botocore.exceptions
import click

from .config import LaunchConfig
from .ec2 import (
    EBS_MOUNT_POINT,
    CLIError,
    attach_ebs_volume,
    create_ebs_volume,
    delete_ebs_volume,
    ensure_security_group,
    find_ebs_volumes_for_instance,
    find_orphan_ebs_volumes,
    find_tagged_instances,
    get_latest_ami,
    get_spot_price,
    launch_instance,
    list_amis,
    list_instance_types,
    terminate_tagged_instances,
    validate_ebs_volume,
    wait_instance_ready,
)
from .output import OutputFormat, emit, is_text
from .quota import (
    QUOTA_FAMILIES,
    QUOTA_FAMILY_LABELS,
    get_family_quotas,
    get_quota,
    get_quota_request_history,
    request_quota_increase,
)
from .ssh import (
    add_ssh_host,
    cleanup_stale_ssh_hosts,
    find_stale_ssh_hosts,
    get_ssh_host_details,
    import_key_pair,
    list_ssh_hosts,
    mount_ebs_volume,
    private_key_path,
    query_gpu_info,
    remove_ssh_host,
    resolve_instance_id,
    run_remote_setup,
    wait_for_ssh,
)


SETUP_SCRIPT = Path(__file__).parent / "resources" / "remote_setup.sh"


def step(number: int, total: int, msg: str) -> None:
    if not is_text():
        return
    click.secho(f"\n[{number}/{total}] {msg}", bold=True, fg="cyan")


def info(msg: str) -> None:
    if not is_text():
        return
    click.echo(f"  {msg}")


def val(label: str, value: str) -> None:
    if not is_text():
        return
    click.echo(f"  {label}: " + click.style(str(value), fg="bright_white"))


def success(msg: str) -> None:
    if not is_text():
        return
    click.secho(f"  {msg}", fg="green")


def warn(msg: str) -> None:
    if not is_text():
        return
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
@click.option(
    "--output",
    "-o",
    type=click.Choice(["text", "json", "yaml", "table"], case_sensitive=False),
    default="text",
    show_default=True,
    help="Output format.",
)
@click.pass_context
def main(ctx, output):
    """Bootstrap AWS EC2 GPU instances for hybrid local-remote development."""
    ctx.ensure_object(dict)
    ctx.obj["output_format"] = OutputFormat(output)


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
@click.option(
    "--ebs-storage",
    default=None,
    type=int,
    help="Create and attach a new EBS data volume (size in GB, gp3). Mounted at /data.",
)
@click.option(
    "--ebs-volume-id",
    default=None,
    type=str,
    help="Attach an existing EBS volume by ID (e.g. vol-0abc123). Mounted at /data.",
)
@click.pass_context
def launch(
    ctx,
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
    ebs_storage,
    ebs_volume_id,
):
    """Launch a GPU-accelerated EC2 instance."""
    if ebs_storage is not None and ebs_volume_id is not None:
        raise CLIError("--ebs-storage and --ebs-volume-id are mutually exclusive.")

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
        ebs_storage=ebs_storage,
        ebs_volume_id=ebs_volume_id,
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

    has_ebs = config.ebs_storage is not None or config.ebs_volume_id is not None
    total_steps = 7 if has_ebs else 6

    # Step 1: AMI lookup
    step(1, total_steps, "Looking up AMI...")
    ami = get_latest_ami(ec2, config.ami_filter)
    info(f"Found: {ami['Name']}")
    val("AMI ID", ami["ImageId"])

    # Step 2: SSH key pair
    step(2, total_steps, "Importing SSH key pair...")
    import_key_pair(ec2, config.key_name, config.key_path)

    # Step 3: Security group
    step(3, total_steps, "Ensuring security group...")
    sg_id = ensure_security_group(ec2, config.security_group, config.tag_value, ssh_port=config.ssh_port)

    pricing = "spot" if config.spot else "on-demand"

    if config.dry_run:
        if is_text(ctx):
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
            if config.ebs_storage:
                val("EBS data volume", f"{config.ebs_storage} GB gp3 (new, mounted at {EBS_MOUNT_POINT})")
            if config.ebs_volume_id:
                val("EBS data volume", f"{config.ebs_volume_id} (existing, mounted at {EBS_MOUNT_POINT})")
            click.echo()
            click.secho("No resources launched (dry-run mode).", fg="yellow")
        else:
            result: dict = {
                "dry_run": True,
                "instance_type": config.instance_type,
                "ami_id": ami["ImageId"],
                "ami_name": ami["Name"],
                "pricing": pricing,
                "key_name": config.key_name,
                "security_group": sg_id,
                "volume_size_gb": config.volume_size,
                "region": config.region,
            }
            if config.ssh_port != 22:
                result["ssh_port"] = config.ssh_port
            if config.python_version:
                result["python_version"] = config.python_version
            if config.ebs_storage:
                result["ebs_storage_gb"] = config.ebs_storage
            if config.ebs_volume_id:
                result["ebs_volume_id"] = config.ebs_volume_id
            emit(result, ctx=ctx)
        return

    # Step 4: Launch instance
    step(4, total_steps, f"Launching {config.instance_type} instance ({pricing})...")
    instance = launch_instance(ec2, config, ami["ImageId"], sg_id)
    instance_id = instance["InstanceId"]
    val("Instance ID", instance_id)

    # Step 5: Wait for ready
    step(5, total_steps, "Waiting for instance to be ready...")
    instance = wait_instance_ready(ec2, instance_id)
    public_ip = instance.get("PublicIpAddress")
    if not public_ip:
        warn(f"No public IP assigned. Instance ID: {instance_id}")
        info("You may need to assign an Elastic IP or check your VPC settings.")
        return

    val("Public IP", public_ip)
    az = instance["Placement"]["AvailabilityZone"]

    # Step 5.5 (optional): EBS data volume
    ebs_volume_attached = None
    ebs_format = False
    if has_ebs:
        step(6, total_steps, "Setting up EBS data volume...")
        if config.ebs_storage:
            info(f"Creating {config.ebs_storage} GB gp3 volume in {az}...")
            ebs_volume_attached = create_ebs_volume(ec2, config.ebs_storage, az, config.tag_value, instance_id)
            val("Volume ID", ebs_volume_attached)
            ebs_format = True
        elif config.ebs_volume_id:
            info(f"Validating volume {config.ebs_volume_id}...")
            validate_ebs_volume(ec2, config.ebs_volume_id, az)
            ebs_volume_attached = config.ebs_volume_id
            # Tag the existing volume for discovery
            ec2.create_tags(
                Resources=[ebs_volume_attached],
                Tags=[
                    {"Key": "aws-bootstrap-instance", "Value": instance_id},
                    {"Key": "created-by", "Value": config.tag_value},
                ],
            )
            ebs_format = False

        info(f"Attaching {ebs_volume_attached} to {instance_id}...")
        attach_ebs_volume(ec2, ebs_volume_attached, instance_id)
        success("EBS volume attached.")

    # SSH and remote setup step
    ssh_step = 7 if has_ebs else 6
    step(ssh_step, total_steps, "Waiting for SSH access...")
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

    # Mount EBS volume via SSH (after setup so the instance is fully ready)
    if ebs_volume_attached:
        info(f"Mounting EBS volume at {EBS_MOUNT_POINT}...")
        if mount_ebs_volume(
            public_ip,
            config.ssh_user,
            config.key_path,
            ebs_volume_attached,
            mount_point=EBS_MOUNT_POINT,
            format_volume=ebs_format,
            port=config.ssh_port,
        ):
            success(f"EBS volume mounted at {EBS_MOUNT_POINT}.")
        else:
            warn(f"Failed to mount EBS volume at {EBS_MOUNT_POINT}. You may need to mount it manually.")

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

    # Structured output for non-text modes
    if not is_text(ctx):
        result_data: dict = {
            "instance_id": instance_id,
            "public_ip": public_ip,
            "instance_type": config.instance_type,
            "availability_zone": az,
            "ami_id": ami["ImageId"],
            "pricing": pricing,
            "region": config.region,
            "ssh_alias": alias,
        }
        if ebs_volume_attached:
            ebs_info: dict = {
                "volume_id": ebs_volume_attached,
                "mount_point": EBS_MOUNT_POINT,
            }
            if config.ebs_storage:
                ebs_info["size_gb"] = config.ebs_storage
            result_data["ebs_volume"] = ebs_info
        emit(result_data, ctx=ctx)
        return

    # Print connection info (text mode)
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
    if ebs_volume_attached:
        if config.ebs_storage:
            ebs_label = f"{ebs_volume_attached} ({config.ebs_storage} GB, {EBS_MOUNT_POINT})"
        else:
            ebs_label = f"{ebs_volume_attached} ({EBS_MOUNT_POINT})"
        val("EBS data volume", ebs_label)

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
    click.secho(f"    aws-bootstrap terminate {alias} --region {config.region}", bold=True)
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
@click.pass_context
def status(ctx, region, profile, gpu, instructions):
    """Show running instances created by aws-bootstrap."""
    session = boto3.Session(profile_name=profile, region_name=region)
    ec2 = session.client("ec2")

    instances = find_tagged_instances(ec2, "aws-bootstrap-g4dn")
    if not instances:
        if is_text(ctx):
            click.secho("No active aws-bootstrap instances found.", fg="yellow")
        else:
            emit({"instances": []}, ctx=ctx)
        return

    ssh_hosts = list_ssh_hosts()

    if is_text(ctx):
        click.secho(f"\n  Found {len(instances)} instance(s):\n", bold=True, fg="cyan")
        if gpu:
            click.echo("  " + click.style("Querying GPU info via SSH...", dim=True))
            click.echo()

    structured_instances = []

    for inst in instances:
        state = inst["State"]
        alias = ssh_hosts.get(inst["InstanceId"])

        # Text mode: inline display
        if is_text(ctx):
            state_color = {
                "running": "green",
                "pending": "yellow",
                "stopping": "yellow",
                "stopped": "red",
                "shutting-down": "red",
            }.get(state, "white")
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

        # Build structured record
        inst_data: dict = {
            "instance_id": inst["InstanceId"],
            "state": state,
            "instance_type": inst["InstanceType"],
            "public_ip": inst["PublicIp"] or None,
            "ssh_alias": alias,
            "lifecycle": inst["Lifecycle"],
            "availability_zone": inst["AvailabilityZone"],
            "launch_time": inst["LaunchTime"],
        }

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
                if is_text(ctx):
                    val("    GPU", f"{gpu_info.gpu_name} ({gpu_info.architecture})")
                    if gpu_info.cuda_toolkit_version:
                        cuda_str = gpu_info.cuda_toolkit_version
                        if gpu_info.cuda_driver_version != gpu_info.cuda_toolkit_version:
                            cuda_str += f" (driver supports up to {gpu_info.cuda_driver_version})"
                    else:
                        cuda_str = f"{gpu_info.cuda_driver_version} (driver max, toolkit unknown)"
                    val("    CUDA", cuda_str)
                    val("    Driver", gpu_info.driver_version)
                inst_data["gpu"] = {
                    "name": gpu_info.gpu_name,
                    "architecture": gpu_info.architecture,
                    "cuda_toolkit": gpu_info.cuda_toolkit_version,
                    "cuda_driver_max": gpu_info.cuda_driver_version,
                    "driver": gpu_info.driver_version,
                }
            else:
                if is_text(ctx):
                    click.echo("    GPU: " + click.style("unavailable", dim=True))

        # EBS data volumes
        ebs_volumes = find_ebs_volumes_for_instance(ec2, inst["InstanceId"], "aws-bootstrap-g4dn")
        if ebs_volumes:
            if is_text(ctx):
                for vol in ebs_volumes:
                    vol_state = f", {vol['State']}" if vol["State"] != "in-use" else ""
                    val("    EBS", f"{vol['VolumeId']} ({vol['Size']} GB, {EBS_MOUNT_POINT}{vol_state})")
            inst_data["ebs_volumes"] = [
                {
                    "volume_id": vol["VolumeId"],
                    "size_gb": vol["Size"],
                    "mount_point": EBS_MOUNT_POINT,
                    "state": vol["State"],
                }
                for vol in ebs_volumes
            ]

        lifecycle = inst["Lifecycle"]
        is_spot = lifecycle == "spot"
        spot_price = None

        if is_spot:
            spot_price = get_spot_price(ec2, inst["InstanceType"], inst["AvailabilityZone"])
            if is_text(ctx):
                if spot_price is not None:
                    val("    Pricing", f"spot (${spot_price:.4f}/hr)")
                else:
                    val("    Pricing", "spot")
            if spot_price is not None:
                inst_data["spot_price_per_hour"] = spot_price
        else:
            if is_text(ctx):
                val("    Pricing", "on-demand")

        if state == "running" and is_spot:
            uptime = datetime.now(UTC) - inst["LaunchTime"]
            total_seconds = int(uptime.total_seconds())
            inst_data["uptime_seconds"] = total_seconds
            if is_text(ctx):
                hours, remainder = divmod(total_seconds, 3600)
                minutes = remainder // 60
                val("    Uptime", f"{hours}h {minutes:02d}m")
            if spot_price is not None:
                uptime_hours = uptime.total_seconds() / 3600
                est_cost = uptime_hours * spot_price
                inst_data["estimated_cost"] = round(est_cost, 4)
                if is_text(ctx):
                    val("    Est. cost", f"~${est_cost:.4f}")

        if is_text(ctx):
            val("    Launched", str(inst["LaunchTime"]))

        # Connection instructions (opt-in, only for running instances with a public IP and alias)
        if is_text(ctx) and instructions and state == "running" and inst["PublicIp"] and alias:
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

        structured_instances.append(inst_data)

    if not is_text(ctx):
        emit(
            {"instances": structured_instances},
            headers={
                "instance_id": "Instance ID",
                "state": "State",
                "instance_type": "Type",
                "public_ip": "IP",
                "ssh_alias": "Alias",
                "lifecycle": "Pricing",
                "uptime_seconds": "Uptime (s)",
            },
            ctx=ctx,
        )
        return

    click.echo()
    first_id = instances[0]["InstanceId"]
    first_ref = ssh_hosts.get(first_id, first_id)
    click.echo("  To terminate:  " + click.style(f"aws-bootstrap terminate {first_ref}", bold=True))
    click.echo()


@main.command()
@click.option("--region", default="us-west-2", show_default=True, help="AWS region.")
@click.option("--profile", default=None, help="AWS profile override.")
@click.option("--yes", "-y", is_flag=True, default=False, help="Skip confirmation prompt.")
@click.option("--keep-ebs", is_flag=True, default=False, help="Preserve EBS data volumes instead of deleting them.")
@click.argument("instance_ids", nargs=-1, metavar="[INSTANCE_ID_OR_ALIAS]...")
@click.pass_context
def terminate(ctx, region, profile, yes, keep_ebs, instance_ids):
    """Terminate instances created by aws-bootstrap.

    Pass specific instance IDs or SSH aliases (e.g. aws-gpu1) to terminate,
    or omit to terminate all aws-bootstrap instances in the region.
    """
    session = boto3.Session(profile_name=profile, region_name=region)
    ec2 = session.client("ec2")

    # In structured output modes, require --yes (prompts would corrupt output)
    if not is_text(ctx) and not yes:
        raise CLIError("--yes is required when using structured output (--output json/yaml/table).")

    if instance_ids:
        targets = []
        for value in instance_ids:
            resolved = resolve_instance_id(value)
            if resolved is None:
                raise CLIError(
                    f"Could not resolve '{value}' to an instance ID.\n\n"
                    "  It is not a valid instance ID or a known SSH alias."
                )
            if resolved != value:
                info(f"Resolved alias '{value}' -> {resolved}")
            targets.append(resolved)
    else:
        instances = find_tagged_instances(ec2, "aws-bootstrap-g4dn")
        if not instances:
            if is_text(ctx):
                click.secho("No active aws-bootstrap instances found.", fg="yellow")
            else:
                emit({"terminated": []}, ctx=ctx)
            return
        targets = [inst["InstanceId"] for inst in instances]
        if is_text(ctx):
            click.secho(f"\n  Found {len(targets)} instance(s) to terminate:\n", bold=True, fg="cyan")
            for inst in instances:
                iid = click.style(inst["InstanceId"], fg="bright_white")
                click.echo(f"  {iid}  {inst['State']}  {inst['InstanceType']}")

    if not yes:
        click.echo()
        if not click.confirm(f"  Terminate {len(targets)} instance(s)?"):
            click.secho("  Cancelled.", fg="yellow")
            return

    # Discover EBS volumes before termination (while instances still exist)
    ebs_by_instance: dict[str, list[dict]] = {}
    for target in targets:
        volumes = find_ebs_volumes_for_instance(ec2, target, "aws-bootstrap-g4dn")
        if volumes:
            ebs_by_instance[target] = volumes

    changes = terminate_tagged_instances(ec2, targets)

    terminated_results = []

    if is_text(ctx):
        click.echo()
    for change in changes:
        prev = change["PreviousState"]["Name"]
        curr = change["CurrentState"]["Name"]
        iid = change["InstanceId"]
        if is_text(ctx):
            click.echo("  " + click.style(iid, fg="bright_white") + f"  {prev} -> " + click.style(curr, fg="red"))
        removed_alias = remove_ssh_host(iid)
        if removed_alias:
            info(f"Removed SSH config alias: {removed_alias}")

        change_data: dict = {
            "instance_id": iid,
            "previous_state": prev,
            "current_state": curr,
        }
        if removed_alias:
            change_data["ssh_alias_removed"] = removed_alias
        terminated_results.append(change_data)

    # Handle EBS volume cleanup
    for _iid, volumes in ebs_by_instance.items():
        for vol in volumes:
            vid = vol["VolumeId"]
            if keep_ebs:
                if is_text(ctx):
                    click.echo()
                info(f"Preserving EBS volume: {vid} ({vol['Size']} GB)")
                info(f"Reattach with: aws-bootstrap launch --ebs-volume-id {vid}")
            else:
                if is_text(ctx):
                    click.echo()
                info(f"Waiting for EBS volume {vid} to detach...")
                try:
                    waiter = ec2.get_waiter("volume_available")
                    waiter.wait(VolumeIds=[vid], WaiterConfig={"Delay": 10, "MaxAttempts": 30})
                    delete_ebs_volume(ec2, vid)
                    success(f"Deleted EBS volume: {vid}")
                    # Record deleted volume in the corresponding terminated result
                    for tr in terminated_results:
                        if tr["instance_id"] == _iid:
                            tr.setdefault("ebs_volumes_deleted", []).append(vid)
                except Exception as e:
                    warn(f"Failed to delete EBS volume {vid}: {e}")

    if not is_text(ctx):
        emit({"terminated": terminated_results}, ctx=ctx)
        return

    click.echo()
    success(f"Terminated {len(changes)} instance(s).")


@main.command()
@click.option("--dry-run", is_flag=True, default=False, help="Show what would be removed without removing.")
@click.option("--yes", "-y", is_flag=True, default=False, help="Skip confirmation prompt.")
@click.option("--include-ebs", is_flag=True, default=False, help="Also find and delete orphan EBS data volumes.")
@click.option("--region", default="us-west-2", show_default=True, help="AWS region.")
@click.option("--profile", default=None, help="AWS profile override.")
@click.pass_context
def cleanup(ctx, dry_run, yes, include_ebs, region, profile):
    """Remove stale SSH config entries for terminated instances."""
    session = boto3.Session(profile_name=profile, region_name=region)
    ec2 = session.client("ec2")

    # In structured output modes, require --yes for non-dry-run (prompts would corrupt output)
    if not is_text(ctx) and not yes and not dry_run:
        raise CLIError("--yes is required when using structured output (--output json/yaml/table).")

    live_instances = find_tagged_instances(ec2, "aws-bootstrap-g4dn")
    live_ids = {inst["InstanceId"] for inst in live_instances}

    stale = find_stale_ssh_hosts(live_ids)

    # Orphan EBS discovery
    orphan_volumes: list[dict] = []
    if include_ebs:
        orphan_volumes = find_orphan_ebs_volumes(ec2, "aws-bootstrap-g4dn", live_ids)

    if not stale and not orphan_volumes:
        if is_text(ctx):
            msg = "No stale SSH config entries found."
            if include_ebs:
                msg = "No stale SSH config entries or orphan EBS volumes found."
            click.secho(msg, fg="green")
        else:
            result_key = "stale" if dry_run else "cleaned"
            result: dict = {result_key: []}
            if include_ebs:
                ebs_key = "orphan_volumes" if dry_run else "deleted_volumes"
                result[ebs_key] = []
            emit(result, ctx=ctx)
        return

    if is_text(ctx):
        if stale:
            click.secho(f"\n  Found {len(stale)} stale SSH config entry(ies):\n", bold=True, fg="cyan")
            for iid, alias in stale:
                click.echo("  " + click.style(alias, fg="bright_white") + f"  ({iid})")
        if orphan_volumes:
            click.secho(f"\n  Found {len(orphan_volumes)} orphan EBS volume(s):\n", bold=True, fg="cyan")
            for vol in orphan_volumes:
                click.echo(
                    "  "
                    + click.style(vol["VolumeId"], fg="bright_white")
                    + f"  ({vol['Size']} GB, was {vol['InstanceId']})"
                )

    if dry_run:
        if is_text(ctx):
            click.echo()
            for iid, alias in stale:
                info(f"Would remove {alias} ({iid})")
            for vol in orphan_volumes:
                info(f"Would delete {vol['VolumeId']} ({vol['Size']} GB)")
        else:
            result = {
                "stale": [{"instance_id": iid, "alias": alias} for iid, alias in stale],
                "dry_run": True,
            }
            if include_ebs:
                result["orphan_volumes"] = [
                    {
                        "volume_id": vol["VolumeId"],
                        "size_gb": vol["Size"],
                        "instance_id": vol["InstanceId"],
                    }
                    for vol in orphan_volumes
                ]
            emit(result, ctx=ctx)
        return

    if not yes:
        click.echo()
        parts = []
        if stale:
            parts.append(f"{len(stale)} stale SSH entry(ies)")
        if orphan_volumes:
            parts.append(f"{len(orphan_volumes)} orphan EBS volume(s)")
        if not click.confirm(f"  Remove {' and '.join(parts)}?"):
            click.secho("  Cancelled.", fg="yellow")
            return

    ssh_results = cleanup_stale_ssh_hosts(live_ids) if stale else []

    # Delete orphan EBS volumes
    deleted_volumes: list[dict] = []
    for vol in orphan_volumes:
        try:
            delete_ebs_volume(ec2, vol["VolumeId"])
            deleted_volumes.append({"volume_id": vol["VolumeId"], "size_gb": vol["Size"], "deleted": True})
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as exc:
            if is_text(ctx):
                warn(f"Failed to delete {vol['VolumeId']}: {exc}")
            deleted_volumes.append({"volume_id": vol["VolumeId"], "size_gb": vol["Size"], "deleted": False})

    if not is_text(ctx):
        result = {
            "cleaned": [{"instance_id": r.instance_id, "alias": r.alias, "removed": r.removed} for r in ssh_results],
        }
        if include_ebs:
            result["deleted_volumes"] = deleted_volumes
        emit(result, ctx=ctx)
        return

    click.echo()
    for r in ssh_results:
        success(f"Removed {r.alias} ({r.instance_id})")
    for vol in deleted_volumes:
        if vol["deleted"]:
            success(f"Deleted {vol['volume_id']} ({vol['size_gb']} GB)")

    click.echo()
    parts = []
    if ssh_results:
        parts.append(f"{len(ssh_results)} stale entry(ies)")
    if deleted_volumes:
        ok_count = sum(1 for v in deleted_volumes if v["deleted"])
        parts.append(f"{ok_count} orphan volume(s)")
    success(f"Cleaned up {' and '.join(parts)}.")


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
@click.pass_context
def list_instance_types_cmd(ctx, prefix, region, profile):
    """List EC2 instance types matching a family prefix (e.g. g4dn, p3, g5)."""
    session = boto3.Session(profile_name=profile, region_name=region)
    ec2 = session.client("ec2")

    types = list_instance_types(ec2, prefix)
    if not types:
        if is_text(ctx):
            click.secho(f"No instance types found matching '{prefix}.*'", fg="yellow")
        else:
            emit([], ctx=ctx)
        return

    if not is_text(ctx):
        structured = [
            {
                "instance_type": t["InstanceType"],
                "vcpus": t["VCpuCount"],
                "memory_mib": t["MemoryMiB"],
                "gpu": t["GpuSummary"] or None,
            }
            for t in types
        ]
        emit(
            structured,
            headers={
                "instance_type": "Instance Type",
                "vcpus": "vCPUs",
                "memory_mib": "Memory (MiB)",
                "gpu": "GPU",
            },
            ctx=ctx,
        )
        return

    click.secho(f"\n  {len(types)} instance type(s) matching '{prefix}.*':\n", bold=True, fg="cyan")

    # Header
    click.echo(
        "  " + click.style(f"{'Instance Type':<24}{'vCPUs':>6}{'Memory (MiB)':>14}  GPU", fg="bright_white", bold=True)
    )
    click.echo("  " + "-" * 72)

    for t in types:
        gpu_str = t["GpuSummary"] or "-"
        click.echo(f"  {t['InstanceType']:<24}{t['VCpuCount']:>6}{t['MemoryMiB']:>14}  {gpu_str}")

    click.echo()
    click.echo(
        "  "
        + click.style("Tip: ", fg="bright_black")
        + click.style("use --prefix to list other families (e.g. --prefix p5, --prefix g5)", fg="bright_black")
    )
    click.echo()


@list_cmd.command(name="amis")
@click.option("--filter", "ami_filter", default=DEFAULT_AMI_PREFIX, show_default=True, help="AMI name pattern.")
@click.option("--region", default="us-west-2", show_default=True, help="AWS region.")
@click.option("--profile", default=None, help="AWS profile override.")
@click.pass_context
def list_amis_cmd(ctx, ami_filter, region, profile):
    """List available AMIs matching a name pattern."""
    session = boto3.Session(profile_name=profile, region_name=region)
    ec2 = session.client("ec2")

    amis = list_amis(ec2, ami_filter)
    if not amis:
        if is_text(ctx):
            click.secho(f"No AMIs found matching '{ami_filter}'", fg="yellow")
        else:
            emit([], ctx=ctx)
        return

    if not is_text(ctx):
        structured = [
            {
                "image_id": ami["ImageId"],
                "name": ami["Name"],
                "creation_date": ami["CreationDate"][:10],
                "architecture": ami["Architecture"],
            }
            for ami in amis
        ]
        emit(
            structured,
            headers={
                "image_id": "Image ID",
                "name": "Name",
                "creation_date": "Created",
                "architecture": "Arch",
            },
            ctx=ctx,
        )
        return

    click.secho(f"\n  {len(amis)} AMI(s) matching '{ami_filter}' (newest first):\n", bold=True, fg="cyan")

    for ami in amis:
        click.echo("  " + click.style(ami["ImageId"], fg="bright_white") + "  " + ami["CreationDate"][:10])
        click.echo(f"    {ami['Name']}")

    click.echo()


# ---------------------------------------------------------------------------
# quota command group
# ---------------------------------------------------------------------------


@main.group()
def quota():
    """Manage EC2 GPU vCPU service quotas."""


@quota.command(name="show")
@click.option(
    "--family",
    default=None,
    type=click.Choice(list(QUOTA_FAMILIES.keys()), case_sensitive=False),
    help="Instance family to show (default: all).",
)
@click.option("--region", default="us-west-2", show_default=True, help="AWS region.")
@click.option("--profile", default=None, help="AWS profile override.")
@click.pass_context
def quota_show(ctx, family, region, profile):
    """Show current spot and on-demand vCPU quota values."""
    session = boto3.Session(profile_name=profile, region_name=region)
    sq = session.client("service-quotas")

    families = [family] if family else list(QUOTA_FAMILIES.keys())
    all_quotas: list[dict] = []
    for fam in families:
        all_quotas.extend(get_family_quotas(sq, fam))

    if not is_text(ctx):
        emit(
            {"quotas": all_quotas},
            headers={
                "family": "Family",
                "quota_type": "Type",
                "quota_code": "Quota Code",
                "quota_name": "Name",
                "value": "Value (vCPUs)",
            },
            ctx=ctx,
        )
        return

    click.secho("\n  EC2 GPU vCPU Quotas:\n", bold=True, fg="cyan")
    for fam in families:
        fam_quotas = [q for q in all_quotas if q["family"] == fam]
        if len(families) > 1:
            click.secho(f"  {QUOTA_FAMILY_LABELS[fam]}:", bold=True)
        for q in fam_quotas:
            label = q["quota_type"].capitalize()
            click.echo(
                "  "
                + click.style(f"{label:<12}", fg="bright_white")
                + click.style(f"{q['value']:.0f}", fg="green", bold=True)
                + " vCPUs"
            )
            click.echo(f"    {q['quota_name']}")
            click.echo(f"    Code: {q['quota_code']}")
            click.echo()

    example_family = family or "gvt"
    click.echo(
        "  " + click.style("Tip: ", fg="bright_black") + click.style("g4dn.xlarge requires 4 vCPUs", fg="bright_black")
    )
    click.echo(
        "  "
        + click.style("To request an increase: ", fg="bright_black")
        + click.style(
            f"aws-bootstrap quota request --family {example_family} --type spot --desired-value 4", fg="bright_black"
        )
    )
    click.echo()


@quota.command(name="request")
@click.option(
    "--family",
    default="gvt",
    show_default=True,
    type=click.Choice(list(QUOTA_FAMILIES.keys()), case_sensitive=False),
    help="Instance family.",
)
@click.option(
    "--type",
    "quota_type",
    required=True,
    type=click.Choice(["spot", "on-demand"], case_sensitive=False),
    help="Quota type to increase.",
)
@click.option("--desired-value", required=True, type=float, help="Desired quota value (vCPUs).")
@click.option("--region", default="us-west-2", show_default=True, help="AWS region.")
@click.option("--profile", default=None, help="AWS profile override.")
@click.option("--yes", "-y", is_flag=True, default=False, help="Skip confirmation prompt.")
@click.pass_context
def quota_request(ctx, family, quota_type, desired_value, region, profile, yes):
    """Request a vCPU quota increase."""
    # In structured output modes, require --yes
    if not is_text(ctx) and not yes:
        raise CLIError("--yes is required when using structured output (--output json/yaml/table).")

    session = boto3.Session(profile_name=profile, region_name=region)
    sq = session.client("service-quotas")

    quota_code = QUOTA_FAMILIES[family][quota_type]

    # Show current value
    current = get_quota(sq, quota_code)

    if desired_value <= current["value"]:
        raise CLIError(
            f"Desired value ({desired_value:.0f}) must be greater than current value ({current['value']:.0f})."
        )

    if is_text(ctx):
        click.echo()
        val("Quota", current["quota_name"])
        val("Family", QUOTA_FAMILY_LABELS[family])
        val("Current value", f"{current['value']:.0f} vCPUs")
        val("Requested value", f"{desired_value:.0f} vCPUs")
        click.echo()

    if not yes and not click.confirm(f"  Request increase from {current['value']:.0f} to {desired_value:.0f} vCPUs?"):
        click.secho("  Cancelled.", fg="yellow")
        return

    result = request_quota_increase(sq, quota_code, desired_value)
    result["quota_type"] = quota_type
    result["family"] = family

    if not is_text(ctx):
        emit(result, ctx=ctx)
        return

    click.echo()
    success("Quota increase request submitted.")
    val("Request ID", result["request_id"])
    val("Status", result["status"])
    if result.get("case_id"):
        val("Support case", result["case_id"])
    click.echo()
    info("Track status with: aws-bootstrap quota history")
    click.echo()


@quota.command(name="history")
@click.option(
    "--family",
    default=None,
    type=click.Choice(list(QUOTA_FAMILIES.keys()), case_sensitive=False),
    help="Filter by instance family (default: all).",
)
@click.option(
    "--type",
    "quota_type",
    default=None,
    type=click.Choice(["spot", "on-demand"], case_sensitive=False),
    help="Filter by quota type (spot or on-demand).",
)
@click.option(
    "--status",
    "status_filter",
    default=None,
    type=click.Choice(
        ["PENDING", "CASE_OPENED", "APPROVED", "DENIED", "CASE_CLOSED", "NOT_APPROVED"],
        case_sensitive=False,
    ),
    help="Filter by request status.",
)
@click.option("--region", default="us-west-2", show_default=True, help="AWS region.")
@click.option("--profile", default=None, help="AWS profile override.")
@click.pass_context
def quota_history(ctx, family, quota_type, status_filter, region, profile):
    """Show history of vCPU quota increase requests."""
    session = boto3.Session(profile_name=profile, region_name=region)
    sq = session.client("service-quotas")

    # Normalize status filter to uppercase for the API
    resolved_status = status_filter.upper() if status_filter else None

    # Determine which families and types to query
    families = [family] if family else list(QUOTA_FAMILIES.keys())

    all_requests: list[dict] = []
    for fam in families:
        codes = QUOTA_FAMILIES[fam]
        for qt, qc in codes.items():
            if quota_type and qt != quota_type:
                continue
            requests = get_quota_request_history(sq, qc, status_filter=resolved_status)
            for r in requests:
                r["quota_type"] = qt
                r["family"] = fam
            all_requests.extend(requests)

    # Sort merged results newest-first
    all_requests.sort(key=lambda r: r["created"], reverse=True)

    if not is_text(ctx):
        emit(
            {"requests": all_requests},
            headers={
                "request_id": "Request ID",
                "family": "Family",
                "quota_type": "Type",
                "status": "Status",
                "desired_value": "Requested vCPUs",
                "created": "Created",
            },
            ctx=ctx,
        )
        return

    if not all_requests:
        click.secho("No quota increase requests found.", fg="yellow")
        return

    click.secho(f"\n  {len(all_requests)} quota increase request(s):\n", bold=True, fg="cyan")

    for r in all_requests:
        status_color = {
            "APPROVED": "green",
            "PENDING": "yellow",
            "CASE_OPENED": "yellow",
            "DENIED": "red",
            "NOT_APPROVED": "red",
            "CASE_CLOSED": "bright_black",
        }.get(r["status"], "white")

        click.echo(
            "  " + click.style(r["request_id"], fg="bright_white") + "  " + click.style(r["status"], fg=status_color)
        )
        val("    Family", r["family"])
        val("    Type", r["quota_type"])
        val("    Requested", f"{r['desired_value']:.0f} vCPUs")
        val("    Created", str(r["created"]))
        if r.get("case_id"):
            val("    Support case", r["case_id"])
        click.echo()

    click.echo()
