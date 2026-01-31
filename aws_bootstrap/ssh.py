"""SSH key pair management for EC2 instances."""

from __future__ import annotations

import subprocess
import socket
import time
from pathlib import Path

import click


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


def wait_for_ssh(host: str, user: str, key_path: Path, retries: int = 30, delay: int = 10) -> bool:
    """Wait for SSH to become available on the instance.

    Tries a TCP connection to port 22 first, then an actual SSH command.
    """
    # Strip .pub to get the private key path
    private_key = key_path.with_suffix("") if key_path.suffix == ".pub" else key_path

    for attempt in range(1, retries + 1):
        # First check if port 22 is open
        try:
            sock = socket.create_connection((host, 22), timeout=5)
            sock.close()
        except (socket.timeout, ConnectionRefusedError, OSError):
            click.echo("  SSH not ready " + click.style(f"(attempt {attempt}/{retries})", dim=True) + ", waiting...")
            time.sleep(delay)
            continue

        # Port is open, try actual SSH
        result = subprocess.run(
            [
                "ssh",
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "ConnectTimeout=10",
                "-o", "BatchMode=yes",
                "-i", str(private_key),
                f"{user}@{host}",
                "echo ok",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            click.secho("  SSH connection established.", fg="green")
            return True

        click.echo("  SSH not ready " + click.style(f"(attempt {attempt}/{retries})", dim=True) + ", waiting...")
        time.sleep(delay)

    return False


def run_remote_setup(host: str, user: str, key_path: Path, script_path: Path) -> bool:
    """SCP the setup script to the instance and execute it."""
    private_key = key_path.with_suffix("") if key_path.suffix == ".pub" else key_path
    ssh_opts = [
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-i", str(private_key),
    ]

    # SCP the script
    click.echo("  Uploading remote_setup.sh...")
    scp_result = subprocess.run(
        ["scp", *ssh_opts, str(script_path), f"{user}@{host}:/tmp/remote_setup.sh"],
        capture_output=True,
        text=True,
    )
    if scp_result.returncode != 0:
        click.secho(f"  SCP failed: {scp_result.stderr}", fg="red", err=True)
        return False

    # Execute the script
    click.echo("  Running remote_setup.sh on instance...")
    ssh_result = subprocess.run(
        ["ssh", *ssh_opts, f"{user}@{host}", "chmod +x /tmp/remote_setup.sh && /tmp/remote_setup.sh"],
        capture_output=False,
    )
    return ssh_result.returncode == 0
