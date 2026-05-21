"""CLI entry point for aws-bootstrap-g4dn."""

from __future__ import annotations
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import boto3
import botocore.exceptions
import click

from . import cluster as cluster_mod
from .config import LaunchConfig
from .constants import (
    DEFAULT_REGION,
    DEFAULT_WAIT_TIMEOUT,
    EBS_DETACH_WAITER,
    EBS_MOUNT_POINT,
    JUPYTER_PORT,
    SSH_PORT_DEFAULT,
    TAG_BOOTSTRAP_INSTANCE,
    TAG_CLUSTER_ID,
    TAG_CLUSTER_RANK,
    TAG_CREATED_BY,
    TAG_VALUE,
)
from .ec2 import (
    CLIError,
    RegionContext,
    attach_ebs_volume,
    create_ebs_volume,
    delete_cluster_placement_group,
    delete_ebs_volume,
    ensure_cluster_placement_group,
    ensure_cluster_security_group_rule,
    ensure_security_group,
    find_cluster_instances,
    find_ebs_volumes_for_instance,
    find_orphan_ebs_volumes,
    find_tagged_instances,
    find_tagged_instances_in_regions,
    get_latest_ami,
    get_spot_price,
    instance_type_to_family,
    launch_with_retry,
    list_amis,
    list_clusters,
    list_enabled_regions,
    list_instance_types,
    resolve_ebs_placement_az,
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
from .retry import parse_duration, resolve_regions
from .ssh import (
    add_ssh_host,
    cleanup_stale_ssh_hosts,
    find_stale_ssh_hosts,
    generate_ssh_keypair,
    get_ssh_host_details,
    import_key_pair,
    list_ssh_hosts,
    mount_ebs_volume,
    private_key_path,
    query_cuda_version,
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


# --- Multi-region launch color scheme ---------------------------------------
# Region-scoped output is hard to scan during retries/quota fallthrough.
# Consistent hues let the eye group by region and spot actionable commands.


def _cmd(s: str) -> str:
    """Style a copy-paste runnable command consistently (bold bright-cyan)."""
    return click.style(s, fg="bright_cyan", bold=True)


def _rtag(region: str) -> str:
    """Region marker, same hue everywhere so attempts group visually."""
    return click.style(f"[{region}]", fg="bright_blue", bold=True)


def _region_rule(region: str) -> None:
    """Separator printed before a region's attempt block (multi-region, text only)."""
    if not is_text():
        return
    bar = "┄" * max(4, 44 - len(region))
    click.secho(f"\n┄┄ {region} {bar}", fg="bright_blue", bold=True)


def _emit_region_fatal(region: str, kind: str, message: str, more_regions: bool) -> None:
    """Render a quota/price skip: bold-yellow verdict, copy-paste commands in
    bright-cyan so they stand out of the warning block."""
    if not is_text():
        return
    label = "spot quota exceeded" if kind == "quota" else "spot price too low"
    click.secho(f"\n  ✗ {region}: {label} — skipping", fg="yellow", bold=True, err=True)
    for line in message.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("aws-bootstrap"):
            click.secho(f"      {s}", fg="bright_cyan", bold=True, err=True)
        else:
            click.secho(f"    {s}", fg="yellow", err=True)
    if more_regions:
        click.secho("  → trying the next region", fg="yellow", dim=True, err=True)


def _session_region(profile: str | None) -> str | None:
    """Region configured via AWS_DEFAULT_REGION or the active profile (no override)."""
    region = boto3.Session(profile_name=profile).region_name
    # boto3 returns a str or None; guard against non-str (e.g. mocked sessions).
    return region if isinstance(region, str) else None


def resolve_region_list(regions: tuple[str, ...], profile: str | None) -> tuple[str, ...]:
    """Ordered region list: explicit --region > profile/env region > default."""
    return resolve_regions(regions, _session_region(profile))


def resolve_single_region(region: str | None, profile: str | None) -> str:
    """Single region for non-launch commands, honoring the same precedence."""
    explicit = (region,) if region else ()
    return resolve_region_list(explicit, profile)[0]


def _region_block_header(region: str, multi: bool, summary: str) -> None:
    """Print a per-region section header in text mode.

    For a single region, labels the active region then the summary; for multiple
    regions, prefixes the summary with the region so blocks are distinguishable.
    """
    if not is_text():
        return
    if multi:
        click.secho(f"\n  {region} — {summary}", bold=True, fg="cyan")
    else:
        val("Region", region)
        click.secho(f"\n  {summary}", bold=True, fg="cyan")


class _Duration(click.ParamType):
    """Click type accepting durations like '30m', '90s', '1h', or bare seconds."""

    name = "duration"

    def convert(self, value, param, ctx):  # type: ignore[no-untyped-def]
        if isinstance(value, int):
            return value
        try:
            return parse_duration(str(value))
        except ValueError as e:
            self.fail(str(e), param, ctx)


DURATION = _Duration()


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
@click.option(
    "--region",
    "regions",
    multiple=True,
    metavar="REGION",
    help=(
        "AWS region. Repeatable: regions are attempted one at a time in the "
        "given order and the first with capacity is used (NOT one instance per "
        "region) — e.g. --region us-west-2 --region us-east-1. "
        "Defaults to AWS_DEFAULT_REGION / profile region, then us-west-2."
    ),
)
@click.option(
    "--wait",
    is_flag=True,
    default=False,
    help=(
        "On insufficient spot capacity, keep retrying: re-sweep all --region "
        "values (bounded exponential backoff between sweeps) until --wait-timeout, "
        "then fail."
    ),
)
@click.option(
    "--wait-timeout",
    type=DURATION,
    default=DEFAULT_WAIT_TIMEOUT,
    show_default="30m",
    help="Max time to keep retrying when --wait is set (e.g. 30m, 90s, 1h).",
)
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
    regions,
    wait,
    wait_timeout,
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

    resolved_regions = resolve_region_list(regions, profile)

    config = LaunchConfig(
        instance_type=instance_type,
        spot=spot,
        key_path=Path(key_path).expanduser(),
        key_name=key_name,
        regions=resolved_regions,
        security_group=security_group,
        volume_size=volume_size,
        run_setup=not no_setup,
        dry_run=dry_run,
        ssh_port=ssh_port,
        python_version=python_version,
        ebs_storage=ebs_storage,
        ebs_volume_id=ebs_volume_id,
        wait=wait,
        wait_timeout=wait_timeout,
    )
    if ami_filter:
        config.ami_filter = ami_filter
    if profile:
        config.profile = profile

    # Validate key path — auto-generate an ed25519 key pair if absent so we
    # never abort a real launch (or leave the user without a usable key).
    # In --dry-run we MUST NOT touch the filesystem (and the notice would be
    # invisible under --output json), so report intent and stop instead.
    if not config.key_path.exists():
        if config.dry_run:
            raise CLIError(
                f"SSH public key not found at {config.key_path}.\n"
                "  It would be auto-generated on a real launch — re-run without --dry-run,\n"
                f"  or create one first: ssh-keygen -t ed25519 -f {private_key_path(config.key_path)}"
            )
        info(f"SSH public key not found at {config.key_path} — generating a new ed25519 key pair…")
        try:
            generate_ssh_keypair(config.key_path)
        except (OSError, subprocess.CalledProcessError) as e:
            raise CLIError(
                f"SSH public key not found and could not be generated at {config.key_path}: {e}\n"
                "  Create one with: ssh-keygen -t ed25519, or pass --key-path to an existing key."
            ) from None
        success(f"Generated {private_key_path(config.key_path)} (and .pub)")

    multi_region = len(config.regions) > 1
    # "us-west-2" for one region; "us-west-2 → us-east-1 → eu-west-1" for many,
    # to make the in-order, one-at-a-time attempt sequence unambiguous.
    regions_label = " → ".join(config.regions) if multi_region else config.regions[0]
    pricing = "spot" if config.spot else "on-demand"

    has_ebs = config.ebs_storage is not None or config.ebs_volume_id is not None
    total_steps = 4 if has_ebs else 3

    def prepare_region(region: str) -> RegionContext:
        """Build region-scoped prerequisites: client, AMI, key pair, security group.

        Setup failures (no matching AMI, no default VPC, …) are region-specific —
        prefix them with the region so the message is unambiguous in
        multi-region mode.
        """
        ec2r = boto3.Session(profile_name=config.profile, region_name=region).client("ec2")
        if multi_region:
            _region_rule(region)
        try:
            if is_text():
                click.echo(f"  {_rtag(region)} " + click.style("looking up AMI…", dim=True))
            ami_r = get_latest_ami(ec2r, config.ami_filter)
            if is_text():
                click.echo(f"  {_rtag(region)} " + click.style(f"AMI {ami_r['ImageId']}", dim=True))
            effective_key = import_key_pair(ec2r, config.key_name, config.key_path)
            sg_id_r = ensure_security_group(ec2r, config.security_group, config.tag_value, ssh_port=config.ssh_port)
        except CLIError as e:
            raise CLIError(f"[{region}] {e.format_message()}") from None
        # Pin the instance to an existing data volume's AZ (EBS is AZ-scoped, so
        # a random AZ would fail to attach). resolve_ebs_placement_az raises a
        # region-named CLIError if the volume isn't in this region.
        placement_az = resolve_ebs_placement_az(ec2r, config.ebs_volume_id, region) if config.ebs_volume_id else None
        return RegionContext(
            region=region, ec2_client=ec2r, ami=ami_r, sg_id=sg_id_r, key_name=effective_key, placement_az=placement_az
        )

    if config.dry_run:
        ctx0 = prepare_region(config.regions[0])
        ami = ctx0.ami
        if is_text(ctx):
            click.echo()
            click.secho("--- Dry Run Summary ---", bold=True, fg="yellow")
            val("Instance type", config.instance_type)
            val("AMI", f"{ami['ImageId']} ({ami['Name']})")
            val("Pricing", pricing)
            val("Key pair", ctx0.key_name)
            val("Security group", ctx0.sg_id)
            val("Volume", f"{config.volume_size} GB gp3")
            val("Region(s)", regions_label)
            if config.wait:
                val("Wait", f"yes (timeout {config.wait_timeout}s)")
            val("Remote setup", "yes" if config.run_setup else "no")
            if config.ssh_port != SSH_PORT_DEFAULT:
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
                "key_name": ctx0.key_name,
                "security_group": ctx0.sg_id,
                "volume_size_gb": config.volume_size,
                "regions": list(config.regions),
                "wait": config.wait,
                "wait_timeout_seconds": config.wait_timeout,
            }
            if config.ssh_port != SSH_PORT_DEFAULT:
                result["ssh_port"] = config.ssh_port
            if config.python_version:
                result["python_version"] = config.python_version
            if config.ebs_storage:
                result["ebs_storage_gb"] = config.ebs_storage
            if config.ebs_volume_id:
                result["ebs_volume_id"] = config.ebs_volume_id
            emit(result, ctx=ctx)
        return

    # Step 1: Launch instance (prepare region prerequisites + run, spot-first across regions)
    if multi_region:
        caption = (
            f"Launching {config.instance_type} ({pricing}) — trying regions in order, one at a time: {regions_label}..."
        )
    else:
        caption = f"Launching {config.instance_type} ({pricing}) in {regions_label}..."
    step(1, total_steps, caption)

    def on_attempt(region: str, market: str, attempt: int) -> None:
        if is_text():
            verb = click.style(f"requesting {market}", fg="cyan", bold=True)
            click.echo(f"  {_rtag(region)} {verb} {config.instance_type}…")

    def on_wait(cycle: int, sleep_s: float, elapsed: float, retrying: list[str], skipped: list[str]) -> None:
        if is_text():
            nxt = click.style(f"{sleep_s:.0f}s", fg="yellow", bold=True)
            retry_label = " → ".join(retrying) if len(retrying) > 1 else retrying[0]
            click.secho(
                f"\n  ⏳ wait cycle {cycle}: no {pricing} capacity in {retry_label} — next sweep in {nxt}",
                fg="yellow",
            )
            if skipped:
                click.secho(
                    f"     (not retried — quota/price blocked: {', '.join(skipped)}; "
                    "fix quota then re-run to include them)",
                    fg="yellow",
                    dim=True,
                )
            click.secho(f"     (elapsed {elapsed:.0f}s of {config.wait_timeout}s budget)", fg="yellow", dim=True)

    def on_region_fatal(region: str, kind: str, message: str) -> None:
        # Quota / spot-price problem in this region: warn with the full
        # remediation hint (commands highlighted), then move on.
        _emit_region_fatal(region, kind, message, more_regions=len(config.regions) > 1)

    launched = launch_with_retry(
        config,
        prepare_region,
        on_attempt=on_attempt,
        on_wait=on_wait,
        on_region_fatal=on_region_fatal,
    )
    ec2 = launched.context.ec2_client
    ami = launched.context.ami
    active_region = launched.region
    pricing = launched.pricing
    instance = launched.instance
    instance_id = instance["InstanceId"]
    val("Instance ID", instance_id)
    val("Region", active_region)
    val("Pricing", pricing)

    # Step 2: Wait for ready
    step(2, total_steps, "Waiting for instance to be ready...")
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
        step(3, total_steps, "Setting up EBS data volume...")
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
                    {"Key": TAG_BOOTSTRAP_INSTANCE, "Value": instance_id},
                    {"Key": TAG_CREATED_BY, "Value": config.tag_value},
                ],
            )
            ebs_format = False

        info(f"Attaching {ebs_volume_attached} to {instance_id}...")
        attach_ebs_volume(ec2, ebs_volume_attached, instance_id)
        success("EBS volume attached.")

    # SSH and remote setup step
    ssh_step = 4 if has_ebs else 3
    step(ssh_step, total_steps, "Waiting for SSH access...")
    private_key = private_key_path(config.key_path)
    if not wait_for_ssh(public_ip, config.ssh_user, config.key_path, port=config.ssh_port):
        warn("SSH did not become available within the timeout.")
        port_flag = f" -p {config.ssh_port}" if config.ssh_port != SSH_PORT_DEFAULT else ""
        info(
            f"Instance is running — try connecting manually:"
            f" ssh -i {private_key}{port_flag} {config.ssh_user}@{public_ip}"
        )
        return

    cuda_version: str | None = None
    if config.run_setup:
        if not SETUP_SCRIPT.exists():
            warn(f"Setup script not found at {SETUP_SCRIPT}, skipping.")
        else:
            info("Running remote setup...")
            if run_remote_setup(
                public_ip, config.ssh_user, config.key_path, SETUP_SCRIPT, config.python_version, port=config.ssh_port
            ):
                success("Remote setup completed successfully.")
                cuda_version = query_cuda_version(public_ip, config.ssh_user, config.key_path, port=config.ssh_port)
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
            "region": active_region,
            "regions_tried": list(config.regions),
            "ssh_alias": alias,
        }
        if cuda_version:
            result_data["cuda_version"] = cuda_version
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
    val("Region", active_region)
    val("Pricing", pricing)
    if cuda_version:
        val("CUDA Version", cuda_version)
    val("SSH alias", alias)
    if ebs_volume_attached:
        if config.ebs_storage:
            ebs_label = f"{ebs_volume_attached} ({config.ebs_storage} GB, {EBS_MOUNT_POINT})"
        else:
            ebs_label = f"{ebs_volume_attached} ({EBS_MOUNT_POINT})"
        val("EBS data volume", ebs_label)

    port_flag = f" -p {config.ssh_port}" if config.ssh_port != SSH_PORT_DEFAULT else ""

    click.echo()
    click.secho("  SSH:", fg="cyan")
    click.echo("    " + _cmd(f"ssh{port_flag} {alias}"))
    info(f"or: ssh -i {private_key}{port_flag} {config.ssh_user}@{public_ip}")

    click.echo()
    click.secho("  Jupyter (via SSH tunnel):", fg="cyan")
    click.echo("    " + _cmd(f"ssh -NL {JUPYTER_PORT}:localhost:{JUPYTER_PORT}{port_flag} {alias}"))
    info(
        f"or: ssh -i {private_key} -NL {JUPYTER_PORT}:localhost:{JUPYTER_PORT}{port_flag} {config.ssh_user}@{public_ip}"
    )
    info(f"Then open: http://localhost:{JUPYTER_PORT}")
    info("Notebook: ~/gpu_smoke_test.ipynb (GPU smoke test)")

    click.echo()
    click.secho("  VSCode Remote SSH:", fg="cyan")
    click.echo("    " + _cmd(f"code --folder-uri vscode-remote://ssh-remote+{alias}/home/{config.ssh_user}/workspace"))

    click.echo()
    click.secho("  GPU Benchmark:", fg="cyan")
    click.echo("    " + _cmd(f"ssh {alias} '~/venv/bin/python ~/gpu_benchmark.py'"))
    info("Runs CNN (MNIST) and Transformer benchmarks with tqdm progress")

    click.echo()
    click.secho("  Terminate:", fg="cyan")
    click.echo("    " + _cmd(f"aws-bootstrap terminate {alias} --region {active_region}"))
    click.echo()


@main.command()
@click.option(
    "--region",
    "-r",
    multiple=True,
    metavar="REGION",
    help="AWS region to query (repeatable). If omitted, queries all enabled regions.",
)
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
    """Show running instances created by aws-bootstrap.

    With no --region, queries all regions enabled for the account and labels
    each instance with its region. Pass one or more --region values to restrict
    the query to those regions.
    """
    session = boto3.Session(profile_name=profile)

    selected = bool(region)
    if selected:
        regions = list(region)
    else:
        discovery_region = _session_region(profile) or DEFAULT_REGION
        regions = list_enabled_regions(session.client("ec2", region_name=discovery_region))

    if is_text(ctx):
        if selected:
            click.secho(f"\n  Showing status for selected region(s): {', '.join(regions)}\n", bold=True, fg="cyan")
        else:
            shown = ", ".join(regions[:12])
            extra = f" (+{len(regions) - 12} more)" if len(regions) > 12 else ""
            click.secho(f"\n  Querying {len(regions)} enabled region(s): {shown}{extra}\n", bold=True, fg="cyan")

    instances, failures = find_tagged_instances_in_regions(session, TAG_VALUE, regions)

    for failure in failures:
        warn(f"Skipped region {failure['region']}: {failure['error']}")

    if not instances:
        if is_text(ctx):
            click.secho("No active aws-bootstrap instances found.", fg="yellow")
        else:
            result: dict = {"instances": [], "regions_queried": regions}
            if failures:
                result["regions_failed"] = failures
            emit(result, ctx=ctx)
        return

    ssh_hosts = list_ssh_hosts()

    # Cache one EC2 client per region for per-instance follow-up calls.
    region_clients: dict[str, object] = {}

    def region_client(region_name: str):
        if region_name not in region_clients:
            region_clients[region_name] = session.client("ec2", region_name=region_name)
        return region_clients[region_name]

    if is_text(ctx):
        click.secho(f"  Found {len(instances)} instance(s):\n", bold=True, fg="cyan")
        if gpu:
            click.echo("  " + click.style("Querying GPU info via SSH...", dim=True))
            click.echo()

    structured_instances = []

    for inst in instances:
        state = inst["State"]
        alias = ssh_hosts.get(inst["InstanceId"])
        ec2 = region_client(inst["Region"])

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
            val("    Region", inst["Region"])
            val("    Type", inst["InstanceType"])
            if inst["PublicIp"]:
                val("    IP", inst["PublicIp"])

        # Build structured record
        inst_data: dict = {
            "instance_id": inst["InstanceId"],
            "region": inst["Region"],
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
        ebs_volumes = find_ebs_volumes_for_instance(ec2, inst["InstanceId"], TAG_VALUE)
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
            port_flag = f" -p {port}" if port != SSH_PORT_DEFAULT else ""

            click.echo()
            click.secho("    SSH:", fg="cyan")
            click.echo("      " + _cmd(f"ssh{port_flag} {alias}"))

            click.secho("    Jupyter (via SSH tunnel):", fg="cyan")
            click.echo("      " + _cmd(f"ssh -NL {JUPYTER_PORT}:localhost:{JUPYTER_PORT}{port_flag} {alias}"))

            click.secho("    VSCode Remote SSH:", fg="cyan")
            click.echo("      " + _cmd(f"code --folder-uri vscode-remote://ssh-remote+{alias}/home/{user}/workspace"))

            click.secho("    GPU Benchmark:", fg="cyan")
            click.echo("      " + _cmd(f"ssh {alias} '~/venv/bin/python ~/gpu_benchmark.py'"))

        structured_instances.append(inst_data)

    if not is_text(ctx):
        result = {"instances": structured_instances, "regions_queried": regions}
        if failures:
            result["regions_failed"] = failures
        emit(
            result,
            headers={
                "instance_id": "Instance ID",
                "region": "Region",
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
    first = instances[0]
    first_ref = ssh_hosts.get(first["InstanceId"], first["InstanceId"])
    click.echo("  To terminate:  " + _cmd(f"aws-bootstrap terminate {first_ref} --region {first['Region']}"))
    click.echo()


@main.command()
@click.option(
    "--region",
    default=None,
    metavar="REGION",
    help="AWS region. Defaults to AWS_DEFAULT_REGION / profile region, then us-west-2.",
)
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
    region = resolve_single_region(region, profile)
    session = boto3.Session(profile_name=profile, region_name=region)
    ec2 = session.client("ec2")

    # In structured output modes, require --yes (prompts would corrupt output)
    if not is_text(ctx) and not yes:
        raise CLIError("--yes is required when using structured output (--output json/yaml/table).")

    if is_text(ctx):
        val("Region", region)

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
        instances = find_tagged_instances(ec2, TAG_VALUE)
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
        volumes = find_ebs_volumes_for_instance(ec2, target, TAG_VALUE)
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
                if is_text(ctx):
                    click.echo(
                        "  Reattach with: " + _cmd(f"aws-bootstrap launch --ebs-volume-id {vid} --region {region}")
                    )
            else:
                if is_text(ctx):
                    click.echo()
                info(f"Waiting for EBS volume {vid} to detach...")
                try:
                    waiter = ec2.get_waiter("volume_available")
                    waiter.wait(VolumeIds=[vid], WaiterConfig=EBS_DETACH_WAITER)
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
@click.option(
    "--region",
    default=None,
    metavar="REGION",
    help="AWS region. Defaults to AWS_DEFAULT_REGION / profile region, then us-west-2.",
)
@click.option("--profile", default=None, help="AWS profile override.")
@click.pass_context
def cleanup(ctx, dry_run, yes, include_ebs, region, profile):
    """Remove stale SSH config entries for terminated instances."""
    region = resolve_single_region(region, profile)
    session = boto3.Session(profile_name=profile, region_name=region)
    ec2 = session.client("ec2")

    # In structured output modes, require --yes for non-dry-run (prompts would corrupt output)
    if not is_text(ctx) and not yes and not dry_run:
        raise CLIError("--yes is required when using structured output (--output json/yaml/table).")

    if is_text(ctx):
        val("Region", region)

    live_instances = find_tagged_instances(ec2, TAG_VALUE)
    live_ids = {inst["InstanceId"] for inst in live_instances}

    stale = find_stale_ssh_hosts(live_ids)

    # Orphan EBS discovery
    orphan_volumes: list[dict] = []
    if include_ebs:
        orphan_volumes = find_orphan_ebs_volumes(ec2, TAG_VALUE, live_ids)

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
@click.option(
    "--region",
    "-r",
    multiple=True,
    metavar="REGION",
    help="AWS region(s) to query (repeatable). Defaults to AWS_DEFAULT_REGION / profile region, then us-west-2.",
)
@click.option("--profile", default=None, help="AWS profile override.")
@click.pass_context
def list_instance_types_cmd(ctx, prefix, region, profile):
    """List EC2 instance types matching a family prefix (e.g. g4dn, p3, g5)."""
    regions = resolve_region_list(region, profile)
    session = boto3.Session(profile_name=profile)

    per_region = [(r, list_instance_types(session.client("ec2", region_name=r), prefix)) for r in regions]

    if not is_text(ctx):
        structured = [
            {
                "region": r,
                "instance_type": t["InstanceType"],
                "vcpus": t["VCpuCount"],
                "memory_mib": t["MemoryMiB"],
                # vCPU quota family this type draws from (gvt/p/dl), or None for non-GPU.
                "quota_family": instance_type_to_family(t["InstanceType"]),
                "gpu": t["GpuSummary"] or None,
            }
            for r, types in per_region
            for t in types
        ]
        emit(
            structured,
            headers={
                "region": "Region",
                "instance_type": "Instance Type",
                "vcpus": "vCPUs",
                "memory_mib": "Memory (MiB)",
                "quota_family": "Quota Family",
                "gpu": "GPU",
            },
            ctx=ctx,
        )
        return

    if not any(types for _, types in per_region):
        click.secho(f"No instance types found matching '{prefix}.*' in {', '.join(regions)}", fg="yellow")
        return

    multi = len(regions) > 1
    for r, types in per_region:
        _region_block_header(r, multi, f"{len(types)} instance type(s) matching '{prefix}.*':")
        if not types:
            click.echo("  " + click.style("(none)", dim=True))
            continue
        click.echo(
            "  "
            + click.style(
                f"{'Instance Type':<20}{'vCPUs':>7}{'Memory (MiB)':>15}  {'Quota Family':<14}GPU",
                fg="bright_white",
                bold=True,
            )
        )
        click.echo("  " + "-" * 80)
        for t in types:
            gpu_str = t["GpuSummary"] or "-"
            fam_str = instance_type_to_family(t["InstanceType"]) or "-"
            click.echo(f"  {t['InstanceType']:<20}{t['VCpuCount']:>7}{t['MemoryMiB']:>15}  {fam_str:<14}{gpu_str}")

    click.echo()
    click.echo(
        "  "
        + click.style("Tip: ", fg="bright_black")
        + click.style("use --prefix to list other families (e.g. --prefix p5, --prefix g5)", fg="bright_black")
    )

    # Suggested next steps: check/raise the GPU vCPU quota for this family. AWS
    # groups instance types into vCPU quota families (e.g. all G/VT types share
    # the "gvt" quota), which is why the suggested --family may not look like the
    # --prefix. The "Quota Family" column above makes the mapping explicit.
    fam = instance_type_to_family(prefix)
    if fam:
        hint_region = regions[0]
        click.echo()
        click.secho(f"  Next steps — {prefix}.* draws from the '{fam}' vCPU quota family:", bold=True, fg="cyan")
        click.echo("    Check your vCPU quota:")
        click.echo("      " + _cmd(f"aws-bootstrap quota show --family {fam} --region {hint_region}"))
        click.echo("    Request a quota increase (adjust --desired-value to your needs):")
        click.echo(
            "      "
            + _cmd(f"aws-bootstrap quota request --family {fam} --type spot --desired-value 8 --region {hint_region}")
        )
    click.echo()


@list_cmd.command(name="amis")
@click.option("--filter", "ami_filter", default=DEFAULT_AMI_PREFIX, show_default=True, help="AMI name pattern.")
@click.option(
    "--region",
    "-r",
    multiple=True,
    metavar="REGION",
    help="AWS region(s) to query (repeatable). Defaults to AWS_DEFAULT_REGION / profile region, then us-west-2.",
)
@click.option("--profile", default=None, help="AWS profile override.")
@click.pass_context
def list_amis_cmd(ctx, ami_filter, region, profile):
    """List available AMIs matching a name pattern.

    AMI IDs are region-specific, so each result is labelled with its region.
    """
    regions = resolve_region_list(region, profile)
    session = boto3.Session(profile_name=profile)

    per_region = [(r, list_amis(session.client("ec2", region_name=r), ami_filter)) for r in regions]

    if not is_text(ctx):
        structured = [
            {
                "region": r,
                "image_id": ami["ImageId"],
                "name": ami["Name"],
                "creation_date": ami["CreationDate"][:10],
                "architecture": ami["Architecture"],
            }
            for r, amis in per_region
            for ami in amis
        ]
        emit(
            structured,
            headers={
                "region": "Region",
                "image_id": "Image ID",
                "name": "Name",
                "creation_date": "Created",
                "architecture": "Arch",
            },
            ctx=ctx,
        )
        return

    if not any(amis for _, amis in per_region):
        click.secho(f"No AMIs found matching '{ami_filter}' in {', '.join(regions)}", fg="yellow")
        return

    multi = len(regions) > 1
    for r, amis in per_region:
        _region_block_header(r, multi, f"{len(amis)} AMI(s) matching '{ami_filter}' (newest first):")
        if not amis:
            click.echo("  " + click.style("(none)", dim=True))
            continue
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
@click.option(
    "--region",
    "-r",
    multiple=True,
    metavar="REGION",
    help="AWS region(s) to query (repeatable). Defaults to AWS_DEFAULT_REGION / profile region, then us-west-2.",
)
@click.option("--profile", default=None, help="AWS profile override.")
@click.pass_context
def quota_show(ctx, family, region, profile):
    """Show current spot and on-demand vCPU quota values."""
    regions = resolve_region_list(region, profile)
    session = boto3.Session(profile_name=profile)
    families = [family] if family else list(QUOTA_FAMILIES.keys())

    all_quotas: list[dict] = []
    per_region: list[tuple[str, list[dict]]] = []
    for r in regions:
        sq = session.client("service-quotas", region_name=r)
        region_quotas: list[dict] = []
        for fam in families:
            region_quotas.extend(get_family_quotas(sq, fam))
        per_region.append((r, region_quotas))
        all_quotas.extend({"region": r, **q} for q in region_quotas)

    if not is_text(ctx):
        emit(
            {"quotas": all_quotas},
            headers={
                "region": "Region",
                "family": "Family",
                "quota_type": "Type",
                "quota_code": "Quota Code",
                "quota_name": "Name",
                "value": "Value (vCPUs)",
            },
            ctx=ctx,
        )
        return

    multi = len(regions) > 1
    for r, region_quotas in per_region:
        _region_block_header(r, multi, "EC2 GPU vCPU Quotas:")
        click.echo()
        for fam in families:
            fam_quotas = [q for q in region_quotas if q["family"] == fam]
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
        # AWS rejects a desired value that is not strictly greater than the current
        # quota, so base the suggestion on the family's current spot value rather
        # than a fixed 4 (which fails outright when the quota is already >= 4).
        current_spot = next(
            (q["value"] for q in region_quotas if q["family"] == example_family and q["quota_type"] == "spot"),
            0.0,
        )
        suggested = max(8, int(current_spot) + 4)
        click.echo(
            "  "
            + click.style("Tip: ", fg="bright_black")
            + click.style("g4dn.xlarge requires 4 vCPUs", fg="bright_black")
        )
        click.echo(
            "  "
            + click.style("To request an increase: ", fg="bright_black")
            + _cmd(
                f"aws-bootstrap quota request --family {example_family} --type spot "
                f"--desired-value {suggested} --region {r}"
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
@click.option(
    "--region",
    "-r",
    multiple=True,
    metavar="REGION",
    help="AWS region(s) to submit the request in (repeatable). "
    "Defaults to AWS_DEFAULT_REGION / profile region, then us-west-2.",
)
@click.option("--profile", default=None, help="AWS profile override.")
@click.option("--yes", "-y", is_flag=True, default=False, help="Skip confirmation prompt.")
@click.pass_context
def quota_request(ctx, family, quota_type, desired_value, region, profile, yes):
    """Request a vCPU quota increase (in one or more regions)."""
    # In structured output modes, require --yes
    if not is_text(ctx) and not yes:
        raise CLIError("--yes is required when using structured output (--output json/yaml/table).")

    regions = resolve_region_list(region, profile)
    session = boto3.Session(profile_name=profile)
    quota_code = QUOTA_FAMILIES[family][quota_type]

    # Fetch the current value in every target region first so we can validate the
    # whole request up front and never submit a partial set.
    clients: dict[str, object] = {}
    current_by_region: dict[str, dict] = {}
    for r in regions:
        sq = session.client("service-quotas", region_name=r)
        clients[r] = sq
        current_by_region[r] = get_quota(sq, quota_code)

    too_low = [r for r in regions if desired_value <= current_by_region[r]["value"]]
    if too_low:
        details = ", ".join(f"{r} (current {current_by_region[r]['value']:.0f})" for r in too_low)
        raise CLIError(
            f"Desired value ({desired_value:.0f}) must be greater than the current value in: {details}.\n"
            "  No requests were submitted."
        )

    quota_name = next(iter(current_by_region.values()))["quota_name"]
    if is_text(ctx):
        click.echo()
        val("Quota", quota_name)
        val("Family", QUOTA_FAMILY_LABELS[family])
        val("Region(s)", ", ".join(regions))
        val("Requested value", f"{desired_value:.0f} vCPUs")
        click.echo()

    if not yes and not click.confirm(
        f"  Request increase to {desired_value:.0f} vCPUs in {len(regions)} region(s) ({', '.join(regions)})?"
    ):
        click.secho("  Cancelled.", fg="yellow")
        return

    results = []
    for r in regions:
        result = request_quota_increase(clients[r], quota_code, desired_value)
        result["region"] = r
        result["quota_type"] = quota_type
        result["family"] = family
        results.append(result)

    if not is_text(ctx):
        emit({"requests": results}, ctx=ctx)
        return

    click.echo()
    success(f"{len(results)} quota increase request(s) submitted.")
    for result in results:
        click.echo()
        val("Region", result["region"])
        val("  Request ID", result["request_id"])
        val("  Status", result["status"])
        if result.get("case_id"):
            val("  Support case", result["case_id"])
    click.echo()
    click.echo("  Track status with: " + _cmd(f"aws-bootstrap quota history --region {' --region '.join(regions)}"))
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
@click.option(
    "--region",
    "-r",
    multiple=True,
    metavar="REGION",
    help="AWS region(s) to query (repeatable). Defaults to AWS_DEFAULT_REGION / profile region, then us-west-2.",
)
@click.option("--profile", default=None, help="AWS profile override.")
@click.pass_context
def quota_history(ctx, family, quota_type, status_filter, region, profile):
    """Show history of vCPU quota increase requests."""
    regions = resolve_region_list(region, profile)
    session = boto3.Session(profile_name=profile)

    # Normalize status filter to uppercase for the API
    resolved_status = status_filter.upper() if status_filter else None

    # Determine which families and types to query
    families = [family] if family else list(QUOTA_FAMILIES.keys())

    all_requests: list[dict] = []
    for region_name in regions:
        sq = session.client("service-quotas", region_name=region_name)
        for fam in families:
            codes = QUOTA_FAMILIES[fam]
            for qt, qc in codes.items():
                if quota_type and qt != quota_type:
                    continue
                requests = get_quota_request_history(sq, qc, status_filter=resolved_status)
                for r in requests:
                    r["region"] = region_name
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
                "region": "Region",
                "family": "Family",
                "quota_type": "Type",
                "status": "Status",
                "desired_value": "Requested vCPUs",
                "created": "Created",
            },
            ctx=ctx,
        )
        return

    if is_text(ctx):
        val("Region(s)", ", ".join(regions))

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
        val("    Region", r["region"])
        val("    Family", r["family"])
        val("    Type", r["quota_type"])
        val("    Requested", f"{r['desired_value']:.0f} vCPUs")
        val("    Created", str(r["created"]))
        if r.get("case_id"):
            val("    Support case", r["case_id"])
        click.echo()

    click.echo()


# ---------------------------------------------------------------------------
# cluster: multi-node training clusters (Phase 1 — launch / status / terminate)
# ---------------------------------------------------------------------------


@main.group(name="cluster")
def cluster_cmd():
    """Manage multi-node training clusters (launch, status, terminate)."""


@cluster_cmd.command(name="launch")
@click.option("--cluster-id", required=True, help="Cluster identifier (used as an EC2 tag).")
@click.option("--nodes", default=2, show_default=True, type=int, help="Target number of nodes.")
@click.option("--instance-type", default="g4dn.xlarge", show_default=True, help="EC2 instance type.")
@click.option("--spot/--on-demand", default=True, show_default=True, help="Use spot or on-demand pricing.")
@click.option(
    "--key-path", default="~/.ssh/id_ed25519.pub", show_default=True, type=click.Path(), help="Local SSH public key."
)
@click.option("--key-name", default="aws-bootstrap-key", show_default=True, help="AWS key pair name.")
@click.option("--region", "regions", multiple=True, metavar="REGION", help="AWS region (single AZ chosen within it).")
@click.option("--security-group", default="aws-bootstrap-ssh", show_default=True, help="Security group name.")
@click.option("--volume-size", default=100, show_default=True, type=int, help="Root EBS volume size in GB (gp3).")
@click.option("--no-setup", is_flag=True, default=False, help="Skip running the remote setup script.")
@click.option("--ssh-port", default=22, show_default=True, type=int, help="SSH port on the remote instances.")
@click.option("--python-version", default=None, help="Python version for the remote venv.")
@click.option("--profile", default=None, help="AWS profile override.")
@click.pass_context
def cluster_launch(
    ctx,
    cluster_id,
    nodes,
    instance_type,
    spot,
    key_path,
    key_name,
    regions,
    security_group,
    volume_size,
    no_setup,
    ssh_port,
    python_version,
    profile,
):
    """Launch (or grow) a training cluster to N nodes in one AZ + placement group."""
    region = resolve_single_region(regions[0] if regions else None, profile)
    ec2 = boto3.Session(profile_name=profile, region_name=region).client("ec2")

    existing = find_cluster_instances(ec2, cluster_id)
    to_add = cluster_mod.nodes_to_add(len(existing), nodes)
    start_rank = max((n["Rank"] for n in existing if n["Rank"] is not None), default=-1) + 1

    if to_add == 0:
        success(f"Cluster '{cluster_id}' already has {len(existing)} node(s); nothing to add.")
        emit({"cluster_id": cluster_id, "nodes_added": 0, "node_count": len(existing)}, ctx=ctx)
        return

    key = Path(key_path).expanduser()
    if not key.exists():
        generate_ssh_keypair(key)

    pg_name = cluster_mod.placement_group_name(cluster_id)
    ensure_cluster_placement_group(ec2, pg_name, TAG_VALUE)
    sg_id = ensure_security_group(ec2, security_group, TAG_VALUE, ssh_port=ssh_port)
    ensure_cluster_security_group_rule(ec2, sg_id)

    config = LaunchConfig(
        instance_type=instance_type,
        spot=spot,
        key_path=key,
        key_name=key_name,
        regions=(region,),
        security_group=security_group,
        volume_size=volume_size,
        run_setup=not no_setup,
        ssh_port=ssh_port,
        python_version=python_version,
    )
    if profile:
        config.profile = profile

    # AZ is captured from the first launched node and pinned for the rest;
    # AMI/key are looked up once and reused across nodes.
    shared: dict = {"az": existing[0]["AvailabilityZone"] if existing else None, "ami": None, "key": None}

    def prepare_region(r: str) -> RegionContext:
        if shared["ami"] is None:
            shared["ami"] = get_latest_ami(ec2, config.ami_filter)
        if shared["key"] is None:
            shared["key"] = import_key_pair(ec2, config.key_name, config.key_path)
        return RegionContext(
            region=r,
            ec2_client=ec2,
            ami=shared["ami"],
            sg_id=sg_id,
            key_name=shared["key"],
            placement_az=shared["az"],
            placement_group=pg_name,
        )

    added: list[dict] = []

    def on_node(rank: int, launch) -> None:
        instance_id = launch.instance["InstanceId"]
        inst = wait_instance_ready(ec2, instance_id)
        if shared["az"] is None:
            shared["az"] = inst["Placement"]["AvailabilityZone"]
        ec2.create_tags(
            Resources=[instance_id],
            Tags=[
                {"Key": TAG_CLUSTER_ID, "Value": cluster_id},
                {"Key": TAG_CLUSTER_RANK, "Value": str(rank)},
            ],
        )
        public_ip = inst.get("PublicIpAddress", "")
        alias = cluster_mod.node_alias(cluster_id, rank)
        if public_ip:
            add_ssh_host(instance_id, public_ip, config.ssh_user, config.key_path, port=ssh_port, alias=alias)
        added.append({"rank": rank, "instance_id": instance_id, "public_ip": public_ip, "alias": alias})

    step(1, 1, f"Launching {to_add} node(s) for cluster '{cluster_id}' in {region} (placement group {pg_name})...")
    cluster_mod.launch_cluster_nodes(
        config, prepare_region, to_add, start_rank, launch_fn=launch_with_retry, on_node=on_node
    )

    success(f"Cluster '{cluster_id}' now has {len(existing) + to_add} node(s).")
    if is_text(ctx):
        info(f"Next: aws-bootstrap cluster status --cluster-id {cluster_id} --region {region}")
    emit(
        {
            "cluster_id": cluster_id,
            "region": region,
            "availability_zone": shared["az"],
            "placement_group": pg_name,
            "nodes_added": to_add,
            "node_count": len(existing) + to_add,
            "nodes": added,
        },
        ctx=ctx,
    )


@cluster_cmd.command(name="status")
@click.option("--cluster-id", default=None, help="Cluster id (omit to list all clusters).")
@click.option("--region", default=None, help="AWS region.")
@click.option("--profile", default=None, help="AWS profile override.")
@click.pass_context
def cluster_status(ctx, cluster_id, region, profile):
    """Show cluster node membership and readiness."""
    resolved = resolve_single_region(region, profile)
    ec2 = boto3.Session(profile_name=profile, region_name=resolved).client("ec2")

    if cluster_id:
        nodes = find_cluster_instances(ec2, cluster_id)
        if is_text(ctx):
            val("Cluster", cluster_id)
            val("Region", resolved)
            for n in nodes:
                click.echo(
                    f"  rank {n['Rank']}  {n['InstanceId']}  {n['State']}  "
                    f"{n['InstanceType']}  {n['AvailabilityZone']}  {n['PublicIp']}"
                )
            if not nodes:
                warn(f"No nodes found for cluster '{cluster_id}' in {resolved}.")
        emit(
            {
                "cluster_id": cluster_id,
                "region": resolved,
                "nodes": [
                    {
                        "rank": n["Rank"],
                        "instance_id": n["InstanceId"],
                        "state": n["State"],
                        "instance_type": n["InstanceType"],
                        "az": n["AvailabilityZone"],
                        "public_ip": n["PublicIp"],
                        "private_ip": n["PrivateIp"],
                    }
                    for n in nodes
                ],
            },
            ctx=ctx,
        )
        return

    clusters = list_clusters(ec2, TAG_VALUE)
    if is_text(ctx):
        val("Region", resolved)
        for cid, nodes in sorted(clusters.items()):
            click.echo(f"  {cid}: {len(nodes)} node(s)")
        if not clusters:
            warn(f"No clusters found in {resolved}.")
    emit(
        {
            "region": resolved,
            "clusters": [{"cluster_id": cid, "node_count": len(nodes)} for cid, nodes in sorted(clusters.items())],
        },
        ctx=ctx,
    )


@cluster_cmd.command(name="terminate")
@click.option("--cluster-id", required=True, help="Cluster id to terminate.")
@click.option("--region", default=None, help="AWS region.")
@click.option("--profile", default=None, help="AWS profile override.")
@click.option("--yes", is_flag=True, default=False, help="Skip the confirmation prompt.")
@click.option("--keep-ebs", is_flag=True, default=False, help="Preserve EBS data volumes.")
@click.pass_context
def cluster_terminate(ctx, cluster_id, region, profile, yes, keep_ebs):
    """Terminate all nodes of a cluster and delete its placement group."""
    resolved = resolve_single_region(region, profile)
    ec2 = boto3.Session(profile_name=profile, region_name=resolved).client("ec2")
    nodes = find_cluster_instances(ec2, cluster_id)

    if not nodes:
        warn(f"No nodes found for cluster '{cluster_id}' in {resolved}.")
        emit({"cluster_id": cluster_id, "terminated": []}, ctx=ctx)
        return

    if not yes:
        if not is_text(ctx):
            raise CLIError("Refusing to terminate without --yes in structured output mode.")
        click.confirm(f"Terminate {len(nodes)} node(s) of cluster '{cluster_id}'?", abort=True)

    instance_ids = [n["InstanceId"] for n in nodes]
    terminate_tagged_instances(ec2, instance_ids)
    for iid in instance_ids:
        remove_ssh_host(iid)
    delete_cluster_placement_group(ec2, cluster_mod.placement_group_name(cluster_id))

    success(f"Terminated {len(instance_ids)} node(s) of cluster '{cluster_id}'.")
    if keep_ebs and is_text(ctx):
        info("EBS data volumes preserved (per-node).")
    emit({"cluster_id": cluster_id, "region": resolved, "terminated": instance_ids}, ctx=ctx)
