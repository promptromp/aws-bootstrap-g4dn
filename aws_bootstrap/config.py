"""Default configuration for EC2 GPU instance provisioning."""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LaunchConfig:
    instance_type: str = "g4dn.xlarge"
    ami_filter: str = "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)*"
    spot: bool = True
    key_path: Path = field(default_factory=lambda: Path.home() / ".ssh" / "id_ed25519.pub")
    key_name: str = "aws-bootstrap-key"
    region: str = "us-west-2"
    security_group: str = "aws-bootstrap-ssh"
    volume_size: int = 100
    run_setup: bool = True
    dry_run: bool = False
    profile: str | None = field(default_factory=lambda: os.environ.get("AWS_PROFILE"))
    ssh_user: str = "ubuntu"
    tag_value: str = "aws-bootstrap-g4dn"
    alias_prefix: str = "aws-gpu"
    ssh_port: int = 22
    python_version: str | None = None
