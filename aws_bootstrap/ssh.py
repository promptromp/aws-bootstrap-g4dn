"""SSH key pair management and SSH config management for EC2 instances."""

from __future__ import annotations
import hashlib
import os
import re
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import click

from .constants import (
    ALIAS_PREFIX,
    EBS_MOUNT_POINT,
    RES_KEY_PAIR,
    SSH_CONNECT_TIMEOUT,
    SSH_PORT_DEFAULT,
    TAG_CREATED_BY,
    TAG_VALUE,
)
from .ec2 import CLIError
from .gpu import _GPU_ARCHITECTURES, GpuInfo
from .output import echo, secho


def _pubkey_blob(pub_text: str) -> str:
    """The base64 key material from an OpenSSH public key line.

    Identity of the key independent of the trailing comment, so two keys
    compare equal iff they are the same key regardless of host/user comment.
    """
    parts = pub_text.split()
    return parts[1] if len(parts) >= 2 else pub_text.strip()


def _derived_key_name(base: str, pub_blob: str) -> str:
    """Deterministic, collision-free key-pair name for a given local key."""
    fp8 = hashlib.sha256(pub_blob.encode()).hexdigest()[:8]
    return f"{base}-{fp8}"


def _aws_key_pub_blob(ec2_client, key_name: str) -> str | None:
    """Public-key blob of an existing AWS key pair, or None if AWS has no
    public key on record for it (older imported pairs). Raises ClientError
    (InvalidKeyPair.NotFound) if the key pair does not exist."""
    resp = ec2_client.describe_key_pairs(KeyNames=[key_name], IncludePublicKey=True)
    pub = resp["KeyPairs"][0].get("PublicKey")
    return _pubkey_blob(pub) if pub else None


def generate_ssh_keypair(pub_path: Path) -> None:
    """Ensure an SSH key pair exists at ``pub_path`` (and its private
    counterpart). If the private key already exists (only the ``.pub`` is
    missing), re-derive the public key from it rather than overwriting the
    private key. Raises OSError / CalledProcessError on failure."""
    priv = private_key_path(pub_path)
    priv.parent.mkdir(parents=True, exist_ok=True)
    if priv.exists():
        # Private key present, public key missing: regenerate the .pub only.
        result = subprocess.run(
            ["ssh-keygen", "-y", "-f", str(priv)],
            check=True,
            capture_output=True,
            text=True,
        )
        pub_path.write_text(result.stdout)
        return
    subprocess.run(
        ["ssh-keygen", "-t", "ed25519", "-N", "", "-f", str(priv)],
        check=True,
        capture_output=True,
        text=True,
    )


# ---------------------------------------------------------------------------
# SSH config markers
# ---------------------------------------------------------------------------

_BEGIN_MARKER = "# >>> aws-bootstrap [{instance_id}] >>>"
_END_MARKER = "# <<< aws-bootstrap [{instance_id}] <<<"
_BEGIN_RE = re.compile(r"^# >>> aws-bootstrap \[(?P<iid>i-[a-f0-9]+)\] >>>$")
_END_RE = re.compile(r"^# <<< aws-bootstrap \[(?P<iid>i-[a-f0-9]+)\] <<<$")

_DEFAULT_SSH_CONFIG = Path.home() / ".ssh" / "config"


def private_key_path(key_path: Path) -> Path:
    """Derive the private key path from a public key path (strips .pub suffix)."""
    return key_path.with_suffix("") if key_path.suffix == ".pub" else key_path


def _ssh_opts(key_path: Path) -> list[str]:
    """Build common SSH/SCP options: suppress host-key checking and specify identity."""
    return [
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-i",
        str(private_key_path(key_path)),
    ]


def import_key_pair(ec2_client, key_name: str, key_path: Path) -> str:
    """Import a local SSH public key to AWS, reusing if it already exists.

    Returns the key pair name.
    """
    local_blob = _pubkey_blob(key_path.read_text())

    def _import(name: str) -> str:
        ec2_client.import_key_pair(
            KeyName=name,
            PublicKeyMaterial=key_path.read_bytes(),
            TagSpecifications=[
                {
                    "ResourceType": RES_KEY_PAIR,
                    "Tags": [{"Key": TAG_CREATED_BY, "Value": TAG_VALUE}],
                }
            ],
        )
        secho(f"  Imported key pair '{name}' from {key_path}", fg="green")
        return name

    def _lookup(name: str) -> str | None | bool:
        """Returns the pub blob (str), None (exists but AWS has no pubkey),
        or False (does not exist)."""
        try:
            return _aws_key_pub_blob(ec2_client, name)
        except ec2_client.exceptions.ClientError as e:
            if "InvalidKeyPair.NotFound" not in str(e):
                raise
            return False

    existing = _lookup(key_name)
    if existing is False:
        return _import(key_name)
    if existing is None or existing == local_blob:
        # Matches our local key (or AWS has no pubkey on record for an older
        # pair created from this same material) -> safe to reuse.
        echo("  Key pair " + click.style(f"'{key_name}'", fg="bright_white") + " already exists, reusing.")
        return key_name

    # MISMATCH: a different key already owns this name. Never modify/delete it
    # (another machine may rely on it). Use a deterministic, collision-free
    # name derived from *our* local key so the launched instance is reachable.
    derived = _derived_key_name(key_name, local_blob)
    secho(
        f"  WARNING: AWS key pair '{key_name}' is a different key than {key_path};",
        fg="yellow",
        err=True,
    )
    secho(
        f"  leaving it untouched and using '{derived}' (your local key) instead.",
        fg="yellow",
        err=True,
    )
    derived_existing = _lookup(derived)
    if derived_existing is False:
        return _import(derived)
    if derived_existing is None or derived_existing == local_blob:
        echo("  Key pair " + click.style(f"'{derived}'", fg="bright_white") + " already exists, reusing.")
        return derived
    raise CLIError(
        f"Key pair '{derived}' already exists in this region with a different key. Pass a distinct --key-name."
    )


def _classify_ssh_failure(stderr: str) -> str:
    """Classify an ``ssh`` failure as ``"auth"`` (fatal — retrying can never
    help: wrong key, host-key change, too many auth failures) or
    ``"transient"`` (sshd not up yet, timeout, connection reset)."""
    s = stderr or ""
    fatal_markers = (
        "Permission denied",
        "Too many authentication failures",
        "REMOTE HOST IDENTIFICATION HAS CHANGED",
        "no matching host key",
        "Host key verification failed",
    )
    return "auth" if any(m in s for m in fatal_markers) else "transient"


def wait_for_ssh(
    host: str, user: str, key_path: Path, retries: int = 30, delay: int = 10, port: int = SSH_PORT_DEFAULT
) -> bool:
    """Wait for SSH to become available on the instance.

    Tries a TCP connection to the SSH port first, then an actual SSH command.
    Fails fast (no further retries) on a fatal authentication / host-key
    error — retrying a wrong-key instance for the full budget only hides the
    real cause — and always surfaces the underlying ``ssh`` stderr instead of
    an opaque "not ready".
    """
    base_opts = _ssh_opts(key_path)
    port_opts = ["-p", str(port)] if port != SSH_PORT_DEFAULT else []
    last_err = ""

    for attempt in range(1, retries + 1):
        # First check if the SSH port is open
        try:
            sock = socket.create_connection((host, port), timeout=5)
            sock.close()
        except (TimeoutError, ConnectionRefusedError, OSError) as e:
            last_err = f"port {port} unreachable: {e}"
            echo("  SSH not ready " + click.style(f"(attempt {attempt}/{retries})", dim=True) + ", waiting...")
            time.sleep(delay)
            continue

        # Port is open, try actual SSH
        cmd = [
            "ssh",
            *base_opts,
            *port_opts,
            "-o",
            f"ConnectTimeout={SSH_CONNECT_TIMEOUT}",
            "-o",
            "BatchMode=yes",
            f"{user}@{host}",
            "echo ok",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            secho("  SSH connection established.", fg="green")
            return True

        err = (result.stderr or "").strip()
        if err:
            last_err = err
        detail = err.splitlines()[-1] if err else "ssh exited non-zero"

        if _classify_ssh_failure(err) == "auth":
            # Not retryable — the instance's key pair does not match the
            # local private key. Surface the real error immediately.
            secho(f"  SSH authentication failed (not retrying): {detail}", fg="red", err=True)
            secho(
                f"  The instance's key pair does not match {private_key_path(key_path)}. "
                "Re-run with the matching --key-name/--key-path, or terminate and relaunch.",
                fg="red",
                err=True,
            )
            return False

        echo("  SSH not ready " + click.style(f"(attempt {attempt}/{retries}: {detail})", dim=True) + ", waiting...")
        time.sleep(delay)

    if last_err:
        secho(f"  SSH never became available. Last error: {last_err.splitlines()[-1]}", fg="yellow", err=True)
    return False


def run_remote_setup(
    host: str,
    user: str,
    key_path: Path,
    script_path: Path,
    python_version: str | None = None,
    port: int = SSH_PORT_DEFAULT,
) -> bool:
    """SCP the setup script and requirements.txt to the instance and execute."""
    ssh_opts = _ssh_opts(key_path)
    scp_port_opts = ["-P", str(port)] if port != SSH_PORT_DEFAULT else []
    ssh_port_opts = ["-p", str(port)] if port != SSH_PORT_DEFAULT else []
    requirements_path = script_path.parent / "requirements.txt"

    # SCP the requirements file
    echo("  Uploading requirements.txt...")
    req_result = subprocess.run(
        ["scp", *ssh_opts, *scp_port_opts, str(requirements_path), f"{user}@{host}:/tmp/requirements.txt"],
        capture_output=True,
        text=True,
    )
    if req_result.returncode != 0:
        secho(f"  SCP failed: {req_result.stderr}", fg="red", err=True)
        return False

    # SCP the GPU benchmark script
    benchmark_path = script_path.parent / "gpu_benchmark.py"
    echo("  Uploading gpu_benchmark.py...")
    bench_result = subprocess.run(
        ["scp", *ssh_opts, *scp_port_opts, str(benchmark_path), f"{user}@{host}:/tmp/gpu_benchmark.py"],
        capture_output=True,
        text=True,
    )
    if bench_result.returncode != 0:
        secho(f"  SCP failed: {bench_result.stderr}", fg="red", err=True)
        return False

    # SCP the GPU smoke test notebook
    notebook_path = script_path.parent / "gpu_smoke_test.ipynb"
    echo("  Uploading gpu_smoke_test.ipynb...")
    nb_result = subprocess.run(
        ["scp", *ssh_opts, *scp_port_opts, str(notebook_path), f"{user}@{host}:/tmp/gpu_smoke_test.ipynb"],
        capture_output=True,
        text=True,
    )
    if nb_result.returncode != 0:
        secho(f"  SCP failed: {nb_result.stderr}", fg="red", err=True)
        return False

    # SCP the CUDA example source
    saxpy_path = script_path.parent / "saxpy.cu"
    echo("  Uploading saxpy.cu...")
    saxpy_result = subprocess.run(
        ["scp", *ssh_opts, *scp_port_opts, str(saxpy_path), f"{user}@{host}:/tmp/saxpy.cu"],
        capture_output=True,
        text=True,
    )
    if saxpy_result.returncode != 0:
        secho(f"  SCP failed: {saxpy_result.stderr}", fg="red", err=True)
        return False

    # SCP the Triton vector add example
    triton_path = script_path.parent / "triton_vector_add.py"
    echo("  Uploading triton_vector_add.py...")
    triton_result = subprocess.run(
        ["scp", *ssh_opts, *scp_port_opts, str(triton_path), f"{user}@{host}:/tmp/triton_vector_add.py"],
        capture_output=True,
        text=True,
    )
    if triton_result.returncode != 0:
        secho(f"  SCP failed: {triton_result.stderr}", fg="red", err=True)
        return False

    # SCP the VSCode launch.json
    launch_json_path = script_path.parent / "launch.json"
    echo("  Uploading launch.json...")
    launch_result = subprocess.run(
        ["scp", *ssh_opts, *scp_port_opts, str(launch_json_path), f"{user}@{host}:/tmp/launch.json"],
        capture_output=True,
        text=True,
    )
    if launch_result.returncode != 0:
        secho(f"  SCP failed: {launch_result.stderr}", fg="red", err=True)
        return False

    # SCP the VSCode tasks.json
    tasks_json_path = script_path.parent / "tasks.json"
    echo("  Uploading tasks.json...")
    tasks_result = subprocess.run(
        ["scp", *ssh_opts, *scp_port_opts, str(tasks_json_path), f"{user}@{host}:/tmp/tasks.json"],
        capture_output=True,
        text=True,
    )
    if tasks_result.returncode != 0:
        secho(f"  SCP failed: {tasks_result.stderr}", fg="red", err=True)
        return False

    # SCP the script
    echo("  Uploading remote_setup.sh...")
    scp_result = subprocess.run(
        ["scp", *ssh_opts, *scp_port_opts, str(script_path), f"{user}@{host}:/tmp/remote_setup.sh"],
        capture_output=True,
        text=True,
    )
    if scp_result.returncode != 0:
        secho(f"  SCP failed: {scp_result.stderr}", fg="red", err=True)
        return False

    # Execute the script, passing PYTHON_VERSION as an inline env var if specified
    echo("  Running remote_setup.sh on instance...")
    remote_cmd = "chmod +x /tmp/remote_setup.sh && "
    if python_version:
        remote_cmd += f"PYTHON_VERSION={python_version} "
    remote_cmd += "/tmp/remote_setup.sh"
    ssh_result = subprocess.run(
        ["ssh", *ssh_opts, *ssh_port_opts, f"{user}@{host}", remote_cmd],
        capture_output=False,
    )
    return ssh_result.returncode == 0


# ---------------------------------------------------------------------------
# SSH config management
# ---------------------------------------------------------------------------


def _read_ssh_config(config_path: Path) -> str:
    """Read SSH config content. Returns ``""`` if file doesn't exist."""
    if config_path.exists():
        return config_path.read_text()
    return ""


def _write_ssh_config(config_path: Path, content: str) -> None:
    """Atomically write *content* to *config_path*.

    Creates ``~/.ssh/`` (mode 0700) and the file (mode 0600) if needed.
    """
    ssh_dir = config_path.parent
    ssh_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

    fd, tmp = tempfile.mkstemp(dir=str(ssh_dir), prefix=".ssh_config_tmp_")
    try:
        os.write(fd, content.encode())
        os.close(fd)
        os.chmod(tmp, 0o600)
        os.replace(tmp, str(config_path))
    except BaseException:
        os.close(fd) if not os.get_inheritable(fd) else None  # noqa: B018
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _next_alias(content: str, prefix: str = ALIAS_PREFIX) -> str:
    """Return the next sequential alias like ``aws-gpu3``.

    Only considers aliases inside aws-bootstrap marker blocks so that
    user-defined hosts with coincidentally matching names are ignored.
    """
    max_n = 0
    in_block = False
    for line in content.splitlines():
        if _BEGIN_RE.match(line):
            in_block = True
            continue
        if _END_RE.match(line):
            in_block = False
            continue
        if in_block and line.strip().startswith("Host "):
            alias = line.strip().removeprefix("Host ").strip()
            if alias.startswith(prefix):
                suffix = alias[len(prefix) :]
                if suffix.isdigit():
                    max_n = max(max_n, int(suffix))
    return f"{prefix}{max_n + 1}"


def _build_stanza(
    instance_id: str, alias: str, hostname: str, user: str, key_path: Path, port: int = SSH_PORT_DEFAULT
) -> str:
    """Build a complete SSH config stanza with markers."""
    priv_key = private_key_path(key_path)
    port_line = f"    Port {port}\n" if port != SSH_PORT_DEFAULT else ""
    return (
        f"{_BEGIN_MARKER.format(instance_id=instance_id)}\n"
        f"Host {alias}\n"
        f"    HostName {hostname}\n"
        f"    User {user}\n"
        f"    IdentityFile {priv_key}\n"
        f"{port_line}"
        f"    StrictHostKeyChecking no\n"
        f"    UserKnownHostsFile /dev/null\n"
        f"{_END_MARKER.format(instance_id=instance_id)}\n"
    )


def add_ssh_host(
    instance_id: str,
    hostname: str,
    user: str,
    key_path: Path,
    config_path: Path | None = None,
    alias_prefix: str = ALIAS_PREFIX,
    port: int = SSH_PORT_DEFAULT,
) -> str:
    """Add (or update) an SSH host stanza for *instance_id*.

    Returns the alias that was created (e.g. ``aws-gpu1``).
    """
    config_path = config_path or _DEFAULT_SSH_CONFIG
    content = _read_ssh_config(config_path)

    # Idempotent: if this instance already has a stanza, remember its alias
    existing_alias = _find_alias_in_content(content, instance_id)
    content = _remove_block(content, instance_id)

    alias = existing_alias or _next_alias(content, alias_prefix)
    stanza = _build_stanza(instance_id, alias, hostname, user, key_path, port=port)

    # Ensure a blank line before our block if file has content
    if content and not content.endswith("\n\n") and not content.endswith("\n"):
        content += "\n\n"
    elif content and not content.endswith("\n") or content and content.endswith("\n") and not content.endswith("\n\n"):
        content += "\n"

    content += stanza
    _write_ssh_config(config_path, content)
    return alias


def remove_ssh_host(instance_id: str, config_path: Path | None = None) -> str | None:
    """Remove the SSH host stanza for *instance_id*.

    Returns the alias that was removed, or ``None`` if not found.
    """
    config_path = config_path or _DEFAULT_SSH_CONFIG
    content = _read_ssh_config(config_path)
    if not content:
        return None

    alias = _find_alias_in_content(content, instance_id)
    if alias is None:
        return None

    content = _remove_block(content, instance_id)
    _write_ssh_config(config_path, content)
    return alias


def find_ssh_alias(instance_id: str, config_path: Path | None = None) -> str | None:
    """Read-only lookup of alias for a given instance ID."""
    config_path = config_path or _DEFAULT_SSH_CONFIG
    content = _read_ssh_config(config_path)
    return _find_alias_in_content(content, instance_id)


def list_ssh_hosts(config_path: Path | None = None) -> dict[str, str]:
    """Return ``{instance_id: alias}`` for all aws-bootstrap-managed hosts."""
    config_path = config_path or _DEFAULT_SSH_CONFIG
    content = _read_ssh_config(config_path)
    result: dict[str, str] = {}
    current_iid: str | None = None
    for line in content.splitlines():
        begin = _BEGIN_RE.match(line)
        if begin:
            current_iid = begin.group("iid")
            continue
        end = _END_RE.match(line)
        if end:
            current_iid = None
            continue
        if current_iid and line.strip().startswith("Host "):
            alias = line.strip().removeprefix("Host ").strip()
            result[current_iid] = alias
    return result


def find_stale_ssh_hosts(live_instance_ids: set[str], config_path: Path | None = None) -> list[tuple[str, str]]:
    """Identify SSH config entries whose instances no longer exist.

    Returns ``[(instance_id, alias), ...]`` for entries where the instance ID
    is **not** in *live_instance_ids*, sorted by alias.
    """
    hosts = list_ssh_hosts(config_path)
    stale = [(iid, alias) for iid, alias in hosts.items() if iid not in live_instance_ids]
    stale.sort(key=lambda t: t[1])
    return stale


def cleanup_stale_ssh_hosts(
    live_instance_ids: set[str],
    config_path: Path | None = None,
    dry_run: bool = False,
) -> list[CleanupResult]:
    """Remove SSH config entries for terminated/non-existent instances.

    If *dry_run* is ``True``, entries are identified but not removed.
    Returns a list of :class:`CleanupResult` objects.
    """
    stale = find_stale_ssh_hosts(live_instance_ids, config_path)
    results: list[CleanupResult] = []
    for iid, alias in stale:
        if not dry_run:
            remove_ssh_host(iid, config_path)
        results.append(CleanupResult(instance_id=iid, alias=alias, removed=not dry_run))
    return results


_INSTANCE_ID_RE = re.compile(r"^i-[0-9a-f]{8,17}$")


def _is_instance_id(value: str) -> bool:
    """Return ``True`` if *value* looks like an EC2 instance ID (``i-`` + hex)."""
    return _INSTANCE_ID_RE.match(value) is not None


def resolve_instance_id(value: str, config_path: Path | None = None) -> str | None:
    """Resolve *value* to an EC2 instance ID.

    If *value* already looks like an instance ID (``i-`` prefix followed by hex
    digits) it is returned as-is.  Otherwise it is treated as an SSH host alias
    and looked up in the managed SSH config blocks.

    Returns the instance ID on success, or ``None`` if the alias was not found.
    """
    if _is_instance_id(value):
        return value

    hosts = list_ssh_hosts(config_path)
    # Reverse lookup: alias -> instance_id
    for iid, alias in hosts.items():
        if alias == value:
            return iid
    return None


@dataclass
class CleanupResult:
    """Result of cleaning up a single stale SSH config entry."""

    instance_id: str
    alias: str
    removed: bool


@dataclass
class SSHHostDetails:
    """Connection details parsed from an SSH config stanza."""

    hostname: str
    user: str
    identity_file: Path
    port: int = SSH_PORT_DEFAULT


def get_ssh_host_details(instance_id: str, config_path: Path | None = None) -> SSHHostDetails | None:
    """Parse the managed SSH config block for *instance_id*.

    Returns ``SSHHostDetails`` with HostName, User, and IdentityFile,
    or ``None`` if no complete managed block is found.
    """
    config_path = config_path or _DEFAULT_SSH_CONFIG
    content = _read_ssh_config(config_path)
    if not content:
        return None

    begin_marker = _BEGIN_MARKER.format(instance_id=instance_id)
    end_marker = _END_MARKER.format(instance_id=instance_id)

    in_block = False
    hostname: str | None = None
    user: str | None = None
    identity_file: str | None = None
    port: int = SSH_PORT_DEFAULT

    for line in content.splitlines():
        if line == begin_marker:
            in_block = True
            continue
        if line == end_marker and in_block:
            if hostname and user and identity_file:
                return SSHHostDetails(hostname=hostname, user=user, identity_file=Path(identity_file), port=port)
            return None
        if in_block:
            stripped = line.strip()
            if stripped.startswith("HostName "):
                hostname = stripped.removeprefix("HostName ").strip()
            elif stripped.startswith("User "):
                user = stripped.removeprefix("User ").strip()
            elif stripped.startswith("IdentityFile "):
                identity_file = stripped.removeprefix("IdentityFile ").strip()
            elif stripped.startswith("Port "):
                port = int(stripped.removeprefix("Port ").strip())

    return None


def query_gpu_info(
    host: str, user: str, key_path: Path, timeout: int = SSH_CONNECT_TIMEOUT, port: int = SSH_PORT_DEFAULT
) -> GpuInfo | None:
    """SSH into a host and query GPU info via ``nvidia-smi``.

    Returns ``GpuInfo`` on success, or ``None`` if the SSH connection fails,
    ``nvidia-smi`` is unavailable, or the output is malformed.
    """
    ssh_opts = _ssh_opts(key_path)
    port_opts = ["-p", str(port)] if port != SSH_PORT_DEFAULT else []
    remote_cmd = (
        "nvidia-smi --query-gpu=driver_version,name,compute_cap --format=csv,noheader,nounits"
        " && nvidia-smi | grep -oP 'CUDA Version: \\K[\\d.]+'"
        " && (nvcc --version 2>/dev/null | grep -oP 'release \\K[\\d.]+' || echo 'N/A')"
    )
    cmd = [
        "ssh",
        *ssh_opts,
        *port_opts,
        "-o",
        f"ConnectTimeout={timeout}",
        "-o",
        "BatchMode=yes",
        f"{user}@{host}",
        remote_cmd,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 5)
    except subprocess.TimeoutExpired:
        secho(f"  GPU query timed out after {timeout + 5}s ({host})", fg="yellow", dim=True, err=True)
        return None

    if result.returncode != 0:
        # Surface the reason instead of a bare "unavailable" (text-mode only;
        # secho is silent in structured output).
        err = (result.stderr or "").strip().splitlines()
        if err:
            secho(f"  GPU query failed ({host}): {err[-1]}", fg="yellow", dim=True, err=True)
        return None

    lines = result.stdout.strip().splitlines()
    if len(lines) < 2:
        return None

    try:
        parts = [p.strip() for p in lines[0].split(",")]
        if len(parts) != 3:
            return None
        driver_version, gpu_name, compute_cap = parts
        cuda_driver_version = lines[1].strip()
        cuda_toolkit_version: str | None = None
        if len(lines) >= 3:
            toolkit_line = lines[2].strip()
            if toolkit_line and toolkit_line != "N/A":
                cuda_toolkit_version = toolkit_line
        architecture = _GPU_ARCHITECTURES.get(compute_cap, f"Unknown ({compute_cap})")
        return GpuInfo(
            driver_version=driver_version,
            cuda_driver_version=cuda_driver_version,
            cuda_toolkit_version=cuda_toolkit_version,
            gpu_name=gpu_name,
            compute_capability=compute_cap,
            architecture=architecture,
        )
    except (ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# EBS volume mount
# ---------------------------------------------------------------------------


def mount_ebs_volume(
    host: str,
    user: str,
    key_path: Path,
    volume_id: str,
    mount_point: str = EBS_MOUNT_POINT,
    format_volume: bool = True,
    port: int = SSH_PORT_DEFAULT,
) -> bool:
    """Mount an EBS volume on the remote instance via SSH.

    Detects the NVMe device by volume ID serial, formats if requested,
    mounts at *mount_point*, and adds an fstab entry for persistence.

    Returns True on success, False on failure.
    """
    ssh_opts = _ssh_opts(key_path)
    port_opts = ["-p", str(port)] if port != SSH_PORT_DEFAULT else []

    # Strip the vol- prefix and hyphen for NVMe serial matching
    vol_serial = volume_id.replace("-", "")

    format_cmd = ""
    if format_volume:
        format_cmd = (
            '  if ! sudo blkid "$DEVICE" > /dev/null 2>&1; then\n'
            '    echo "Formatting $DEVICE as ext4..."\n'
            '    sudo mkfs.ext4 "$DEVICE"\n'
            "  fi\n"
        )

    remote_script = (
        "set -e\n"
        "# Detect EBS device by NVMe serial (Nitro instances)\n"
        f'SERIAL="{vol_serial}"\n'
        "DEVICE=$(lsblk -o NAME,SERIAL -dpn 2>/dev/null | "
        "awk -v s=\"$SERIAL\" '$2 == s {print $1}' | head -1)\n"
        "# Fallback to common device paths\n"
        'if [ -z "$DEVICE" ]; then\n'
        "  for dev in /dev/nvme1n1 /dev/xvdf /dev/sdf; do\n"
        '    if [ -b "$dev" ]; then DEVICE="$dev"; break; fi\n'
        "  done\n"
        "fi\n"
        'if [ -z "$DEVICE" ]; then\n'
        '  echo "ERROR: Could not find EBS device" >&2\n'
        "  exit 1\n"
        "fi\n"
        'echo "Found EBS device: $DEVICE"\n'
        f"{format_cmd}"
        f"sudo mkdir -p {mount_point}\n"
        f'sudo mount "$DEVICE" {mount_point}\n'
        f"sudo chown {user}:{user} {mount_point}\n"
        "# Add fstab entry for reboot persistence\n"
        'UUID=$(sudo blkid -s UUID -o value "$DEVICE")\n'
        'if [ -n "$UUID" ]; then\n'
        f'  if ! grep -q "$UUID" /etc/fstab; then\n'
        f'    echo "UUID=$UUID {mount_point} ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab > /dev/null\n'
        "  fi\n"
        "fi\n"
        f'echo "Mounted $DEVICE at {mount_point}"'
    )

    cmd = [
        "ssh",
        *ssh_opts,
        *port_opts,
        "-o",
        f"ConnectTimeout={SSH_CONNECT_TIMEOUT}",
        f"{user}@{host}",
        remote_script,
    ]

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_alias_in_content(content: str, instance_id: str) -> str | None:
    """Extract the alias from a managed block for *instance_id*, or ``None``.

    Only returns an alias when both begin and end markers are present (safety).
    """
    in_block = False
    alias: str | None = None
    begin_marker = _BEGIN_MARKER.format(instance_id=instance_id)
    end_marker = _END_MARKER.format(instance_id=instance_id)
    for line in content.splitlines():
        if line == begin_marker:
            in_block = True
            alias = None
            continue
        if line == end_marker and in_block:
            return alias  # complete block found
        if in_block and alias is None and line.strip().startswith("Host "):
            alias = line.strip().removeprefix("Host ").strip()
    return None  # no complete block found


def _remove_block(content: str, instance_id: str) -> str:
    """Remove the marker block for *instance_id* from *content*.

    If begin marker is found without matching end marker, content is returned
    unchanged (safety measure).
    """
    begin_marker = _BEGIN_MARKER.format(instance_id=instance_id)
    end_marker = _END_MARKER.format(instance_id=instance_id)

    lines = content.splitlines(keepends=True)
    begin_idx: int | None = None
    end_idx: int | None = None

    for i, line in enumerate(lines):
        if line.rstrip("\n") == begin_marker:
            begin_idx = i
        elif line.rstrip("\n") == end_marker and begin_idx is not None:
            end_idx = i
            break

    if begin_idx is None or end_idx is None:
        return content

    # Remove block lines
    del lines[begin_idx : end_idx + 1]

    # Clean up extra blank lines at removal site
    while begin_idx < len(lines) and lines[begin_idx].strip() == "":
        if begin_idx > 0 and lines[begin_idx - 1].strip() == "":
            del lines[begin_idx]
        else:
            break

    return "".join(lines)
