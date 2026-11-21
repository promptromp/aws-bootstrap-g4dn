"""Default configuration for EC2 GPU instance provisioning."""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_REGION = "us-west-2"
DEFAULT_WAIT_TIMEOUT = 1800  # seconds (30m)
# Value of the `created-by` tag on every resource this tool creates; the basis
# for status/terminate/cleanup discovery. (Defined here, not in constants.py,
# to avoid a config<->constants import cycle; re-exported as constants.TAG_VALUE.)
DEFAULT_TAG_VALUE = "aws-bootstrap-g4dn"
DEFAULT_SSH_PORT = 22
DEFAULT_ALIAS_PREFIX = "aws-gpu"

# AWS Deep Learning Base AMI (OSS NVIDIA driver), as an EC2 image-name pattern.
# AWS names the builds per architecture — "Deep Learning Base OSS …" for x86_64,
# "Deep Learning ARM64 Base OSS …" for arm64 — so the "*" after "Deep Learning"
# spans that infix, and get_latest_ami's architecture filter (taken from EC2 for
# the instance type) picks the right build. One pattern then serves every
# architecture, and resolves to the same image as AWS's public SSM pointer
# /aws/service/deeplearning/ami/<arch>/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id.
DLAMI_BASE_PATTERN = "Deep Learning*Base OSS Nvidia Driver GPU AMI"
DEFAULT_AMI_OS = "Ubuntu 24.04"
DEFAULT_AMI_FILTER = f"{DLAMI_BASE_PATTERN} ({DEFAULT_AMI_OS})*"


@dataclass
class LaunchConfig:
    instance_type: str = "g4dn.xlarge"
    ami_filter: str = DEFAULT_AMI_FILTER
    spot: bool = True
    key_path: Path = field(default_factory=lambda: Path.home() / ".ssh" / "id_ed25519.pub")
    key_name: str = "aws-bootstrap-key"
    regions: tuple[str, ...] = (DEFAULT_REGION,)
    security_group: str = "aws-bootstrap-ssh"
    volume_size: int = 100
    run_setup: bool = True
    dry_run: bool = False
    profile: str | None = field(default_factory=lambda: os.environ.get("AWS_PROFILE"))
    ssh_user: str = "ubuntu"
    tag_value: str = DEFAULT_TAG_VALUE
    alias_prefix: str = DEFAULT_ALIAS_PREFIX
    ssh_port: int = DEFAULT_SSH_PORT
    python_version: str | None = None
    ebs_storage: int | None = None
    ebs_volume_id: str | None = None
    wait: bool = False
    wait_timeout: int = DEFAULT_WAIT_TIMEOUT
    # --availability-zone values (each implies its region; tried in the order given),
    # and --cheapest-az (rank each region's AZs by current spot price, cheapest first).
    availability_zones: tuple[str, ...] = ()
    cheapest_az: bool = False

    @property
    def region(self) -> str:
        """Primary region (first in the ordered list).

        Kept for single-region callers and display fallback; the launch
        retry loop iterates ``regions`` explicitly. Falls back to
        ``DEFAULT_REGION`` if ``regions`` is empty so callers get a sane
        value instead of an opaque ``IndexError``.
        """
        return self.regions[0] if self.regions else DEFAULT_REGION
