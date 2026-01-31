"""SSH key pair management and SSH config management for EC2 instances."""

from __future__ import annotations
import os
import re
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import click

from .gpu import _GPU_ARCHITECTURES, GpuInfo


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
    pub_key = key_path.read_bytes()

    # Check if key pair already exists
    try:
        existing = ec2_client.describe_key_pairs(KeyNames=[key_name])
        click.echo("  Key pair " + click.style(f"'{key_name}'", fg="bright_white") + " already exists, reusing.")
        return existing["KeyPairs"][0]["KeyName"]
    except ec2_client.exceptions.ClientError as e:
        if "InvalidKeyPair.NotFound" not in str(e):
            raise

    ec2_client.import_key_pair(
        KeyName=key_name,
        PublicKeyMaterial=pub_key,
        TagSpecifications=[
            {
                "ResourceType": "key-pair",
                "Tags": [{"Key": "created-by", "Value": "aws-bootstrap-g4dn"}],
            }
        ],
    )
    click.secho(f"  Imported key pair '{key_name}' from {key_path}", fg="green")
    return key_name


def wait_for_ssh(host: str, user: str, key_path: Path, retries: int = 30, delay: int = 10, port: int = 22) -> bool:
    """Wait for SSH to become available on the instance.

    Tries a TCP connection to the SSH port first, then an actual SSH command.
    """
    base_opts = _ssh_opts(key_path)
    port_opts = ["-p", str(port)] if port != 22 else []

    for attempt in range(1, retries + 1):
        # First check if the SSH port is open
        try:
            sock = socket.create_connection((host, port), timeout=5)
            sock.close()
        except (TimeoutError, ConnectionRefusedError, OSError):
            click.echo("  SSH not ready " + click.style(f"(attempt {attempt}/{retries})", dim=True) + ", waiting...")
            time.sleep(delay)
            continue

        # Port is open, try actual SSH
        cmd = [
            "ssh",
            *base_opts,
            *port_opts,
            "-o",
            "ConnectTimeout=10",
            "-o",
            "BatchMode=yes",
            f"{user}@{host}",
            "echo ok",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            click.secho("  SSH connection established.", fg="green")
            return True

        click.echo("  SSH not ready " + click.style(f"(attempt {attempt}/{retries})", dim=True) + ", waiting...")
        time.sleep(delay)

    return False


def run_remote_setup(
    host: str, user: str, key_path: Path, script_path: Path, python_version: str | None = None, port: int = 22
) -> bool:
    """SCP the setup script and requirements.txt to the instance and execute."""
    ssh_opts = _ssh_opts(key_path)
    scp_port_opts = ["-P", str(port)] if port != 22 else []
    ssh_port_opts = ["-p", str(port)] if port != 22 else []
    requirements_path = script_path.parent / "requirements.txt"

    # SCP the requirements file
    click.echo("  Uploading requirements.txt...")
    req_result = subprocess.run(
        ["scp", *ssh_opts, *scp_port_opts, str(requirements_path), f"{user}@{host}:/tmp/requirements.txt"],
        capture_output=True,
        text=True,
    )
    if req_result.returncode != 0:
        click.secho(f"  SCP failed: {req_result.stderr}", fg="red", err=True)
        return False

    # SCP the GPU benchmark script
    benchmark_path = script_path.parent / "gpu_benchmark.py"
    click.echo("  Uploading gpu_benchmark.py...")
    bench_result = subprocess.run(
        ["scp", *ssh_opts, *scp_port_opts, str(benchmark_path), f"{user}@{host}:/tmp/gpu_benchmark.py"],
        capture_output=True,
        text=True,
    )
    if bench_result.returncode != 0:
        click.secho(f"  SCP failed: {bench_result.stderr}", fg="red", err=True)
        return False

    # SCP the GPU smoke test notebook
    notebook_path = script_path.parent / "gpu_smoke_test.ipynb"
    click.echo("  Uploading gpu_smoke_test.ipynb...")
    nb_result = subprocess.run(
        ["scp", *ssh_opts, *scp_port_opts, str(notebook_path), f"{user}@{host}:/tmp/gpu_smoke_test.ipynb"],
        capture_output=True,
        text=True,
    )
    if nb_result.returncode != 0:
        click.secho(f"  SCP failed: {nb_result.stderr}", fg="red", err=True)
        return False

    # SCP the script
    click.echo("  Uploading remote_setup.sh...")
    scp_result = subprocess.run(
        ["scp", *ssh_opts, *scp_port_opts, str(script_path), f"{user}@{host}:/tmp/remote_setup.sh"],
        capture_output=True,
        text=True,
    )
    if scp_result.returncode != 0:
        click.secho(f"  SCP failed: {scp_result.stderr}", fg="red", err=True)
        return False

    # Execute the script, passing PYTHON_VERSION as an inline env var if specified
    click.echo("  Running remote_setup.sh on instance...")
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


def _next_alias(content: str, prefix: str = "aws-gpu") -> str:
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


def _build_stanza(instance_id: str, alias: str, hostname: str, user: str, key_path: Path, port: int = 22) -> str:
    """Build a complete SSH config stanza with markers."""
    priv_key = private_key_path(key_path)
    port_line = f"    Port {port}\n" if port != 22 else ""
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
    alias_prefix: str = "aws-gpu",
    port: int = 22,
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


@dataclass
class SSHHostDetails:
    """Connection details parsed from an SSH config stanza."""

    hostname: str
    user: str
    identity_file: Path
    port: int = 22


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
    port: int = 22

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


def query_gpu_info(host: str, user: str, key_path: Path, timeout: int = 10, port: int = 22) -> GpuInfo | None:
    """SSH into a host and query GPU info via ``nvidia-smi``.

    Returns ``GpuInfo`` on success, or ``None`` if the SSH connection fails,
    ``nvidia-smi`` is unavailable, or the output is malformed.
    """
    ssh_opts = _ssh_opts(key_path)
    port_opts = ["-p", str(port)] if port != 22 else []
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
        return None

    if result.returncode != 0:
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
