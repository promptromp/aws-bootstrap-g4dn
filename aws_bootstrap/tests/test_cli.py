"""Tests for the CLI commands (launch, status, terminate, cleanup, list, quota)."""

from __future__ import annotations
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import botocore.exceptions
import pytest
import yaml

from aws_bootstrap.cli import main
from aws_bootstrap.ec2 import CLIError
from aws_bootstrap.gpu import GpuInfo
from aws_bootstrap.ssh import SSHHostDetails, add_ssh_host
from aws_bootstrap.tests.helpers import client_error


def _instance(instance_id="i-abc123", **overrides):
    """An instance as find_tagged_instances reports it (running spot g4dn in us-west-2a)."""
    return {
        "InstanceId": instance_id,
        "Name": "aws-bootstrap-g4dn.xlarge",
        "State": "running",
        "InstanceType": "g4dn.xlarge",
        "PublicIp": "1.2.3.4",
        "LaunchTime": datetime(2025, 1, 1, tzinfo=UTC),
        "Lifecycle": "spot",
        "AvailabilityZone": "us-west-2a",
        "IamInstanceProfile": "",
        **overrides,
    }


def _region_instance(instance_id="i-east", region="us-east-1"):
    return _instance(instance_id, AvailabilityZone=f"{region}a", Region=region)


# ---------------------------------------------------------------------------
# help / version
# ---------------------------------------------------------------------------


_HELP_CASES = [
    (["--help"], ["Bootstrap AWS EC2 GPU instances", "launch", "status", "terminate", "list", "--output", "-o"]),
    (
        ["launch", "--help"],
        [
            "--instance-type",
            "--spot",
            "--dry-run",
            "--key-path",
            "--wait",
            "--wait-timeout",
            "--ssh-port",
            "--ebs-storage",
            "--ebs-volume-id",
            "--iam-instance-profile",
        ],
    ),
    (
        ["status", "--help"],
        ["--region", "--profile", "repeatable", "all enabled regions", "--gpu", "--instructions"]
        + ["--no-instructions", "-I"],
    ),
    (["terminate", "--help"], ["--region", "--yes", "--keep-ebs"]),
    (["list", "--help"], ["instance-types", "amis"]),
    (["list", "instance-types", "--help"], ["--prefix", "--region"]),
    (["list", "amis", "--help"], ["--filter", "--region"]),
    (["quota", "--help"], ["show", "request", "history"]),
    (["quota", "show", "--help"], ["--region", "--profile", "--family"]),
    (["quota", "request", "--help"], ["--type", "--desired-value", "--yes", "--family"]),
    (["quota", "history", "--help"], ["--type", "--status", "--family"]),
]


@pytest.mark.parametrize("argv,expected", _HELP_CASES, ids=[" ".join(argv[:-1]) or "main" for argv, _ in _HELP_CASES])
def test_help_lists_options(runner, argv, expected):
    result = runner.invoke(main, argv)
    assert result.exit_code == 0
    for text in expected:
        assert text in result.output


def test_launch_help_explains_multi_region_semantics(runner):
    result = runner.invoke(main, ["launch", "--help"])
    # Help text is line-wrapped by Click; normalize whitespace before matching.
    flat = " ".join(result.output.split())
    assert "attempted one at a time in the given order" in flat
    assert "NOT one instance per region" in flat


def test_cleanup_help(runner):
    result = runner.invoke(main, ["cleanup", "--help"])
    assert result.exit_code == 0
    for opt in ("--dry-run", "--yes", "--sync", "--key-path"):
        assert opt in result.output
    assert "--region" not in result.output  # cleanup is always all-regions now


def test_version(runner):
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "version" in result.output


# ---------------------------------------------------------------------------
# launch: key handling
# ---------------------------------------------------------------------------


def test_launch_missing_key_ungeneratable_path_errors_clearly(runner):
    # Missing key now triggers auto-generation; an ungeneratable path
    # (cannot create /nonexistent) must fail with a clear message.
    result = runner.invoke(main, ["launch", "--key-path", "/nonexistent/key.pub"])
    assert result.exit_code != 0
    assert "could not be generated" in result.output


@patch("aws_bootstrap.cli.generate_ssh_keypair")
def test_launch_missing_key_dry_run_does_not_generate(mock_gen, runner, tmp_path):
    # --dry-run must NOT touch the filesystem (and the notice would be
    # invisible under --output json) — report intent and stop instead.
    key = tmp_path / "id_ed25519.pub"
    result = runner.invoke(main, ["launch", "--key-path", str(key), "--dry-run", "--region", "us-west-2"])
    assert result.exit_code != 0
    mock_gen.assert_not_called()
    assert not key.exists()
    assert "re-run without --dry-run" in result.output


@patch("aws_bootstrap.cli.generate_ssh_keypair")
def test_launch_missing_key_autogenerates_on_real_launch(mock_gen, runner, launched, tmp_path):
    key = tmp_path / "id_ed25519.pub"
    mock_gen.side_effect = lambda pub_path: Path(pub_path).write_text("ssh-ed25519 AAAAGEN generated\n")
    result = runner.invoke(main, ["launch", "--key-path", str(key), "--no-setup", "--region", "us-west-2"])
    assert result.exit_code == 0
    mock_gen.assert_called_once()
    assert "generating a new ed25519 key pair" in result.output
    launched.setup.assert_not_called()  # --no-setup


# ---------------------------------------------------------------------------
# launch: argument validation and --dry-run
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "args,message",
    [
        (["--dry-run", "--wait-timeout", "bogus"], "Invalid duration"),
        (["--ebs-storage", "96", "--ebs-volume-id", "vol-abc"], "mutually exclusive"),
    ],
    ids=["bad-wait-timeout", "ebs-flags-exclusive"],
)
def test_launch_rejects_invalid_arguments(runner, key_path, args, message):
    result = runner.invoke(main, ["launch", "--key-path", str(key_path), *args])
    assert result.exit_code != 0
    assert message in result.output


@pytest.mark.parametrize(
    "args,present,absent",
    [
        ([], [], ["Python version", "SSH port"]),
        (["--python-version", "3.14.2"], ["3.14.2", "Python version"], []),
        (["--ssh-port", "2222"], ["2222"], []),
        (["--ebs-storage", "96"], ["96 GB gp3", "/data"], []),
        (["--ebs-volume-id", "vol-abc"], ["vol-abc", "/data"], []),
    ],
    ids=["defaults", "python-version", "ssh-port", "ebs-storage", "ebs-volume-id"],
)
def test_launch_dry_run_text_summary(runner, launch_aws, key_path, args, present, absent):
    with patch("aws_bootstrap.cli.add_ssh_host") as mock_add_ssh:
        result = runner.invoke(main, ["launch", "--key-path", str(key_path), "--dry-run", *args])
    assert result.exit_code == 0
    for text in present:
        assert text in result.output
    for text in absent:
        assert text not in result.output
    mock_add_ssh.assert_not_called()  # a dry run never touches ~/.ssh/config


@pytest.mark.parametrize(
    "args,session_region,expected",
    [
        pytest.param(
            [],
            None,
            {
                "dry_run": True,
                "instance_type": "g4dn.xlarge",
                "ami_id": "ami-123",
                "pricing": "spot",
                "regions": ["us-west-2"],
                "wait": False,
                "wait_timeout_seconds": 1800,
            },
            id="defaults",
        ),
        pytest.param(
            ["--region", "us-east-1", "--region", "us-west-2", "--wait", "--wait-timeout", "90s"],
            None,
            {"regions": ["us-east-1", "us-west-2"], "wait": True, "wait_timeout_seconds": 90},
            id="multi-region-and-wait",
        ),
        # With no --region, the profile/env region is used (not hardcoded us-west-2).
        pytest.param([], "ap-south-1", {"regions": ["ap-south-1"]}, id="profile-region"),
    ],
)
def test_launch_dry_run_json(runner, launch_aws, key_path, args, session_region, expected):
    launch_aws.session.return_value.region_name = session_region
    result = runner.invoke(main, ["-o", "json", "launch", "--key-path", str(key_path), "--dry-run", *args])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert {k: data[k] for k in expected} == expected


@pytest.mark.parametrize("instance_type,arch", [("g4dn.xlarge", "x86_64"), ("g5g.xlarge", "arm64")])
def test_launch_looks_up_ami_for_the_instance_type_architecture(runner, launch_aws, key_path, instance_type, arch):
    """g5g is arm64: the AMI lookup must be told, or the launch picks an x86_64 image."""
    launch_aws.arch.return_value = arch
    args = ["launch", "--key-path", str(key_path), "--instance-type", instance_type, "--dry-run"]
    result = runner.invoke(main, args)
    assert result.exit_code == 0
    assert launch_aws.arch.call_args.args[1] == instance_type
    assert launch_aws.ami.call_args.kwargs["architecture"] == arch


# ---------------------------------------------------------------------------
# launch: real launch (mocked end to end)
# ---------------------------------------------------------------------------


def test_launch_output_shows_ssh_alias(runner, launched, key_path):
    result = runner.invoke(main, ["launch", "--key-path", str(key_path), "--no-setup"])
    assert result.exit_code == 0
    assert "ssh aws-gpu1" in result.output
    assert "SSH alias: aws-gpu1" in result.output
    launched.add_ssh.assert_called_once()
    # --no-setup skips remote setup (and the CUDA query that follows it); the SSH
    # readiness wait still runs, since the alias is only useful once sshd answers.
    launched.setup.assert_not_called()
    launched.cuda.assert_not_called()
    launched.ssh_ready.assert_called_once()


def test_launch_benchmark_hint_uses_venv_python(runner, launched, key_path):
    """`ssh alias 'cmd'` is non-interactive (no ~/.bashrc / venv activation)
    and Ubuntu 24.04 has no unversioned `python` — the GPU-benchmark hint
    must invoke the venv interpreter by absolute path."""
    result = runner.invoke(main, ["launch", "--key-path", str(key_path), "--no-setup"])
    assert result.exit_code == 0
    assert "~/venv/bin/python ~/gpu_benchmark.py" in result.output
    assert "'python ~/gpu_benchmark.py'" not in result.output
    launched.setup.assert_not_called()


def test_launch_text_output_shows_resolved_region(runner, launched, region_launch, key_path):
    launched.launch.return_value = region_launch(region="us-east-1")
    launched.ready.return_value = {"PublicIpAddress": "1.2.3.4", "Placement": {"AvailabilityZone": "us-east-1a"}}
    args = ["launch", "--key-path", str(key_path), "--no-setup", "--region", "us-west-2", "--region", "us-east-1"]
    result = runner.invoke(main, args)
    assert result.exit_code == 0
    assert "Region: us-east-1" in result.output
    launched.setup.assert_not_called()


def test_launch_python_version_passed_to_setup(runner, launched, key_path):
    result = runner.invoke(main, ["launch", "--key-path", str(key_path), "--python-version", "3.13"])
    assert result.exit_code == 0
    launched.setup.assert_called_once()
    assert launched.setup.call_args[0][4] == "3.13"


def test_launch_success_is_status_ready(runner, launched, key_path):
    """The success record carries the same discriminator, so one key answers
    'did this work?' across every outcome."""
    result = runner.invoke(main, ["-o", "json", "launch", "--key-path", str(key_path)])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "ready"
    assert payload["ssh_alias"] == "aws-gpu1"


@pytest.fixture
def ebs_ops():
    with (
        patch("aws_bootstrap.cli.create_ebs_volume", return_value="vol-new123") as create,
        patch("aws_bootstrap.cli.validate_ebs_volume", return_value={"VolumeId": "vol-existing", "Size": 200}) as val,
        patch("aws_bootstrap.cli.attach_ebs_volume") as attach,
        patch("aws_bootstrap.cli.mount_ebs_volume", return_value=True) as mount,
    ):
        yield SimpleNamespace(create=create, validate=val, attach=attach, mount=mount)


def test_launch_with_new_ebs_volume_creates_attaches_and_formats(runner, launched, ebs_ops, key_path):
    result = runner.invoke(main, ["launch", "--key-path", str(key_path), "--no-setup", "--ebs-storage", "96"])
    assert result.exit_code == 0
    assert "vol-new123" in result.output
    ebs_ops.create.assert_called_once()
    ebs_ops.validate.assert_not_called()
    ebs_ops.attach.assert_called_once()
    ebs_ops.mount.assert_called_once()
    assert ebs_ops.mount.call_args[1]["format_volume"] is True
    launched.setup.assert_not_called()


def test_launch_with_existing_ebs_volume_validates_and_mounts_as_is(runner, launched, ebs_ops, key_path):
    args = ["launch", "--key-path", str(key_path), "--no-setup", "--ebs-volume-id", "vol-existing"]
    result = runner.invoke(main, args)
    assert result.exit_code == 0
    ebs_ops.validate.assert_called_once()
    ebs_ops.create.assert_not_called()
    ebs_ops.attach.assert_called_once()
    ebs_ops.mount.assert_called_once()
    assert ebs_ops.mount.call_args[1]["format_volume"] is False  # never reformat existing data
    launched.setup.assert_not_called()


def test_launch_ebs_mount_failure_warns(runner, launched, ebs_ops, key_path):
    ebs_ops.mount.return_value = False
    result = runner.invoke(main, ["launch", "--key-path", str(key_path), "--ebs-storage", "96", "--no-setup"])
    # Should succeed despite mount failure (just a warning)
    assert result.exit_code == 0
    assert "WARNING" in result.output or "Failed to mount" in result.output
    launched.setup.assert_not_called()


# ---------------------------------------------------------------------------
# launch: an incomplete launch must still report the instance it left running
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["json", "yaml"])
def test_launch_emits_a_record_when_ssh_never_comes_up(runner, launched, key_path, fmt):
    """Regression: this path returned early with empty stdout and exit 0, so an
    agent saw success while a billable instance was left running and unnamed."""
    launched.ssh_ready.return_value = False
    result = runner.invoke(main, ["-o", fmt, "launch", "--key-path", str(key_path)])
    assert result.exit_code != 0
    payload = json.loads(result.output) if fmt == "json" else yaml.safe_load(result.output)
    assert payload["status"] == "ssh-unavailable"
    assert payload["instance_id"] == "i-test123"
    assert payload["public_ip"] == "1.2.3.4"
    assert payload["region"] == "us-west-2"
    assert payload["error"]
    assert "ebs_volume" not in payload  # no EBS requested -> no volume key


def test_launch_emits_a_record_when_no_public_ip_is_assigned(runner, launched, key_path):
    launched.ready.return_value = {"Placement": {"AvailabilityZone": "us-west-2a"}}
    result = runner.invoke(main, ["-o", "json", "launch", "--key-path", str(key_path)])
    assert result.exit_code != 0
    payload = json.loads(result.output)
    assert payload["status"] == "no-public-ip"
    assert payload["instance_id"] == "i-test123"
    assert payload["public_ip"] is None


def test_launch_incomplete_text_mode_names_the_instance_to_terminate(runner, launched, key_path):
    launched.ssh_ready.return_value = False
    result = runner.invoke(main, ["launch", "--key-path", str(key_path)])
    assert result.exit_code != 0
    assert "i-test123" in result.output
    assert "terminate" in result.output


def test_incomplete_launch_record_names_the_ebs_volume_it_created(runner, launched, ebs_ops, key_path):
    """The volume is created and attached *before* the SSH wait, so an
    ssh-unavailable launch is already billing for it — a caller reclaiming the
    instance has to be able to see it in the same record."""
    launched.ssh_ready.return_value = False
    ebs_ops.create.return_value = "vol-0abc123"
    result = runner.invoke(main, ["-o", "json", "launch", "--key-path", str(key_path), "--ebs-storage", "200"])
    assert result.exit_code != 0
    payload = json.loads(result.output)
    assert payload["status"] == "ssh-unavailable"
    assert payload["ebs_volume"]["volume_id"] == "vol-0abc123"


def test_no_public_ip_record_is_emitted_before_any_ebs_work(runner, launched, key_path):
    """Regression guard: `ebs_volume_attached` is assigned below the EBS block but
    read by _incomplete above it — it must be bound, not UnboundLocalError."""
    launched.ready.return_value = {"Placement": {"AvailabilityZone": "us-west-2a"}}
    result = runner.invoke(main, ["-o", "json", "launch", "--key-path", str(key_path), "--ebs-storage", "200"])
    assert result.exit_code != 0
    payload = json.loads(result.output)
    assert payload["status"] == "no-public-ip"
    assert "ebs_volume" not in payload


# ---------------------------------------------------------------------------
# status (single region via --region; find_tagged_instances patched at the ec2 level)
# ---------------------------------------------------------------------------

STATUS = ["status", "--region", "us-west-2"]

_SAMPLE_GPU_INFO = GpuInfo(
    driver_version="560.35.03",
    cuda_driver_version="13.0",
    cuda_toolkit_version="12.8",
    gpu_name="Tesla T4",
    compute_capability="7.5",
    architecture="Turing",
)

_SSH_DETAILS = SSHHostDetails(hostname="1.2.3.4", user="ubuntu", identity_file=Path("/home/user/.ssh/id_ed25519"))

_ORPHAN_VOLUME = {
    "VolumeId": "vol-orphan1",
    "Size": 100,
    "State": "available",
    "InstanceId": "i-dead1",
    "Region": "us-west-2",
}


@pytest.fixture
def status_aws(cli_session):
    """Patch what `status` reads: no instances, $0.15/hr spot, an empty SSH config."""
    with (
        patch("aws_bootstrap.ec2.find_tagged_instances", return_value=[]) as find,
        patch("aws_bootstrap.cli.get_spot_price", return_value=0.15) as spot,
        patch("aws_bootstrap.cli.list_ssh_hosts", return_value={}) as hosts,
        patch("aws_bootstrap.cli.get_ssh_host_details", return_value=None) as details,
        patch("aws_bootstrap.cli.find_ebs_volumes_for_instance", return_value=[]) as ebs,
    ):
        yield SimpleNamespace(session=cli_session, find=find, spot=spot, hosts=hosts, details=details, ebs=ebs)


@pytest.fixture
def orphans():
    with patch("aws_bootstrap.cli.find_orphan_ebs_volumes_in_regions", return_value=[]) as mock_orphans:
        yield mock_orphans


@pytest.fixture
def gpu_query():
    with patch("aws_bootstrap.cli.query_gpu_info", return_value=_SAMPLE_GPU_INFO) as mock_gpu:
        yield mock_gpu


def test_status_no_instances(runner, status_aws):
    result = runner.invoke(main, STATUS)
    assert result.exit_code == 0
    assert "No active" in result.output


def test_status_no_instances_json(runner, status_aws):
    result = runner.invoke(main, ["-o", "json", *STATUS])
    assert result.exit_code == 0
    assert json.loads(result.output) == {"instances": [], "regions_queried": ["us-west-2"], "orphan_volumes": []}


def test_status_shows_instances(runner, status_aws):
    status_aws.find.return_value = [_instance()]
    status_aws.spot.return_value = 0.1578
    result = runner.invoke(main, STATUS)
    assert result.exit_code == 0
    assert "i-abc123" in result.output
    assert "1.2.3.4" in result.output
    assert "spot ($0.1578/hr)" in result.output
    assert "Uptime" in result.output
    assert "Est. cost" in result.output


def test_status_terminate_hint_pins_resolved_region(runner, status_aws):
    """'To terminate' hint must carry --region for the region status used."""
    status_aws.find.return_value = [_instance(AvailabilityZone="us-west-1a")]
    result = runner.invoke(main, ["status", "--region", "us-west-1"])
    assert result.exit_code == 0
    assert "aws-bootstrap terminate i-abc123 --region us-west-1" in result.output


def test_status_on_demand_no_cost(runner, status_aws):
    status_aws.find.return_value = [_instance("i-ondemand", PublicIp="5.6.7.8", Lifecycle="on-demand")]
    result = runner.invoke(main, STATUS)
    assert result.exit_code == 0
    assert "on-demand" in result.output
    assert "Uptime" not in result.output
    assert "Est. cost" not in result.output
    status_aws.spot.assert_not_called()


def test_status_shows_alias(runner, status_aws):
    status_aws.find.return_value = [_instance()]
    status_aws.hosts.return_value = {"i-abc123": "aws-gpu1"}
    result = runner.invoke(main, STATUS)
    assert result.exit_code == 0
    assert "aws-gpu1" in result.output


def test_status_shows_ebs_volumes(runner, status_aws):
    status_aws.find.return_value = [_instance()]
    status_aws.ebs.return_value = [{"VolumeId": "vol-data1", "Size": 96, "Device": "/dev/sdf", "State": "in-use"}]
    result = runner.invoke(main, STATUS)
    assert result.exit_code == 0
    assert "vol-data1" in result.output
    assert "96 GB" in result.output
    assert "/data" in result.output


def test_status_output_json(runner, status_aws):
    status_aws.find.return_value = [_instance()]
    status_aws.spot.return_value = 0.1578
    result = runner.invoke(main, ["-o", "json", *STATUS])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["regions_queried"] == ["us-west-2"]
    assert len(data["instances"]) == 1
    inst = data["instances"][0]
    assert inst["instance_id"] == "i-abc123"
    assert inst["region"] == "us-west-2"
    assert inst["state"] == "running"
    assert inst["instance_type"] == "g4dn.xlarge"
    assert inst["public_ip"] == "1.2.3.4"
    assert inst["lifecycle"] == "spot"
    assert inst["spot_price_per_hour"] == 0.1578
    assert "uptime_seconds" in inst
    assert "estimated_cost" in inst
    # No ANSI or progress text in structured output
    assert "\x1b[" not in result.output
    assert "Found" not in result.output


# --- --gpu ---------------------------------------------------------------------


def test_status_gpu_shows_info(runner, status_aws, gpu_query):
    status_aws.find.return_value = [_instance()]
    status_aws.hosts.return_value = {"i-abc123": "aws-gpu1"}
    status_aws.details.return_value = _SSH_DETAILS
    result = runner.invoke(main, [*STATUS, "--gpu"])
    assert result.exit_code == 0
    assert "Tesla T4 (Turing)" in result.output
    assert "12.8" in result.output
    assert "driver supports up to 13.0" in result.output
    assert "560.35.03" in result.output
    gpu_query.assert_called_once()


def test_status_gpu_ssh_fails_gracefully(runner, status_aws, gpu_query):
    status_aws.find.return_value = [_instance()]
    status_aws.details.return_value = _SSH_DETAILS
    gpu_query.return_value = None
    result = runner.invoke(main, [*STATUS, "--gpu"])
    assert result.exit_code == 0
    assert "unavailable" in result.output


def test_status_gpu_no_ssh_config_uses_defaults(runner, status_aws, gpu_query):
    status_aws.find.return_value = [_instance()]
    result = runner.invoke(main, [*STATUS, "--gpu"])
    assert result.exit_code == 0
    # Should have been called with the instance IP and default user/key
    gpu_query.assert_called_once()
    assert gpu_query.call_args[0][:2] == ("1.2.3.4", "ubuntu")


@pytest.mark.parametrize(
    "instance,args",
    [
        (_instance("i-stopped", State="stopped", PublicIp="", Lifecycle="on-demand"), ["--gpu"]),
        (_instance(), []),
    ],
    ids=["not-running", "no-gpu-flag"],
)
def test_status_skips_gpu_query(runner, status_aws, gpu_query, instance, args):
    status_aws.find.return_value = [instance]
    status_aws.hosts.return_value = {"i-abc123": "aws-gpu1"}
    result = runner.invoke(main, [*STATUS, *args])
    assert result.exit_code == 0
    gpu_query.assert_not_called()


# --- --instructions / --no-instructions ----------------------------------------


def test_status_instructions_shown_by_default(runner, status_aws):
    """Instructions are shown by default (no flag needed)."""
    status_aws.find.return_value = [_instance()]
    status_aws.hosts.return_value = {"i-abc123": "aws-gpu1"}
    status_aws.details.return_value = _SSH_DETAILS
    result = runner.invoke(main, STATUS)
    assert result.exit_code == 0
    assert "ssh aws-gpu1" in result.output
    assert "ssh -NL 8888:localhost:8888 aws-gpu1" in result.output
    assert "vscode-remote://ssh-remote+aws-gpu1/home/ubuntu/workspace" in result.output
    assert "python ~/gpu_benchmark.py" in result.output


def test_status_no_instructions_suppresses_commands(runner, status_aws):
    status_aws.find.return_value = [_instance()]
    status_aws.hosts.return_value = {"i-abc123": "aws-gpu1"}
    status_aws.details.return_value = _SSH_DETAILS
    result = runner.invoke(main, [*STATUS, "--no-instructions"])
    assert result.exit_code == 0
    assert "vscode-remote" not in result.output
    assert "Jupyter" not in result.output


def test_status_instructions_no_alias_skips(runner, status_aws):
    """Instances without an SSH alias don't get connection instructions."""
    status_aws.find.return_value = [_instance()]
    result = runner.invoke(main, STATUS)
    assert result.exit_code == 0
    assert "ssh aws-gpu" not in result.output
    assert "vscode-remote" not in result.output


def test_status_instructions_non_default_port(runner, status_aws):
    status_aws.find.return_value = [_instance()]
    status_aws.hosts.return_value = {"i-abc123": "aws-gpu1"}
    status_aws.details.return_value = SSHHostDetails(
        hostname="1.2.3.4", user="ubuntu", identity_file=Path("/home/user/.ssh/id_ed25519"), port=2222
    )
    result = runner.invoke(main, STATUS)
    assert result.exit_code == 0
    assert "ssh -p 2222 aws-gpu1" in result.output
    assert "ssh -NL 8888:localhost:8888 -p 2222 aws-gpu1" in result.output


# --- orphaned EBS volumes (detached, still billed) ------------------------------


def test_status_shows_orphan_volumes(runner, status_aws, orphans):
    """Detached tool-created volumes are surfaced with cost + remediation hints."""
    status_aws.find.return_value = [_instance()]
    orphans.return_value = [dict(_ORPHAN_VOLUME)]
    result = runner.invoke(main, STATUS)
    assert result.exit_code == 0, result.output
    assert "vol-orphan1" in result.output
    assert "100 GB" in result.output
    assert "$8.00/mo" in result.output  # 100 GB * $0.08/GB-month gp3
    assert "aws-bootstrap launch --ebs-volume-id vol-orphan1 --region us-west-2" in result.output
    assert "aws-bootstrap cleanup --include-ebs" in result.output


def test_status_no_instances_still_shows_orphans(runner, status_aws, orphans):
    """Zero instances is the worst case for orphans — the section must still render."""
    orphans.return_value = [dict(_ORPHAN_VOLUME)]
    result = runner.invoke(main, STATUS)
    assert result.exit_code == 0, result.output
    assert "No active" in result.output
    assert "vol-orphan1" in result.output
    assert "aws-bootstrap cleanup --include-ebs" in result.output


def test_status_no_orphans_no_section(runner, status_aws, orphans):
    status_aws.find.return_value = [_instance()]
    result = runner.invoke(main, STATUS)
    assert result.exit_code == 0, result.output
    assert "orphan" not in result.output.lower()


def test_status_json_orphan_volume_fields(runner, status_aws, orphans):
    orphans.return_value = [dict(_ORPHAN_VOLUME)]
    result = runner.invoke(main, ["-o", "json", *STATUS])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["instances"] == []
    assert data["orphan_volumes"] == [
        {
            "volume_id": "vol-orphan1",
            "region": "us-west-2",
            "size_gb": 100,
            "monthly_cost_usd": 8.0,
            "last_instance_id": "i-dead1",
        }
    ]


def test_status_json_no_orphans_empty_list(runner, status_aws, orphans):
    """orphan_volumes is always present in structured output so consumers can rely on it."""
    status_aws.find.return_value = [_instance()]
    result = runner.invoke(main, ["-o", "json", *STATUS])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["orphan_volumes"] == []


def test_status_structured_output_carries_instances_and_orphans(runner, status_aws, orphans, structured):
    """json/yaml keep both lists in the single result."""
    fmt, load = structured
    status_aws.find.return_value = [_instance()]
    orphans.return_value = [dict(_ORPHAN_VOLUME)]
    result = runner.invoke(main, ["-o", fmt, *STATUS])
    assert result.exit_code == 0, result.output
    data = load(result.output)
    assert data["instances"][0]["instance_id"] == "i-abc123"
    assert data["orphan_volumes"][0]["volume_id"] == "vol-orphan1"


@pytest.mark.parametrize("profile", ["app", ""], ids=["attached", "none"])
def test_status_structured_output_carries_the_iam_profile(runner, status_aws, structured, profile):
    fmt, load = structured
    status_aws.find.return_value = [_instance(IamInstanceProfile=profile)]
    result = runner.invoke(main, ["-o", fmt, *STATUS])
    assert result.exit_code == 0, result.output
    assert load(result.output)["instances"][0]["iam_instance_profile"] == (profile or None)


@pytest.mark.parametrize(
    "fmt,profile,present,absent",
    [
        ("text", "app", ["IAM profile", "app"], []),
        ("text", "", [], ["IAM profile"]),
        ("table", "app", ["IAM Profile", "app"], []),
        ("table", "", ["IAM Profile"], ["app"]),  # the column is always there
    ],
    ids=["text-attached", "text-none", "table-attached", "table-none"],
)
def test_status_renders_the_iam_profile(runner, status_aws, fmt, profile, present, absent):
    status_aws.find.return_value = [_instance(IamInstanceProfile=profile)]
    result = runner.invoke(main, ["-o", fmt, *STATUS])
    assert result.exit_code == 0, result.output
    for text in present:
        assert text in result.output
    for text in absent:
        assert text not in result.output


def test_status_table_renders_orphans_in_their_own_table(runner, status_aws, orphans):
    """table mode renders one list per emit, so the orphans get a second table."""
    status_aws.find.return_value = [_instance()]
    orphans.return_value = [dict(_ORPHAN_VOLUME)]
    result = runner.invoke(main, ["-o", "table", *STATUS])
    assert result.exit_code == 0, result.output
    assert "Instance ID" in result.output
    assert "i-abc123" in result.output
    assert "Volume ID" in result.output
    assert "vol-orphan1" in result.output


def test_status_table_without_orphans_has_no_volume_table(runner, status_aws, orphans):
    status_aws.find.return_value = [_instance()]
    result = runner.invoke(main, ["-o", "table", *STATUS])
    assert result.exit_code == 0, result.output
    assert "Instance ID" in result.output
    assert "i-abc123" in result.output
    assert "Volume ID" not in result.output


# --- all enabled regions (no --region) ------------------------------------------


@pytest.fixture
def multi_region_status(status_aws):
    """status across regions: find_tagged_instances_in_regions / list_enabled_regions patched."""
    with (
        patch("aws_bootstrap.cli.find_tagged_instances_in_regions", return_value=([], [])) as find,
        patch("aws_bootstrap.cli.list_enabled_regions", return_value=["us-east-1", "us-west-2"]) as regions,
    ):
        yield SimpleNamespace(find=find, regions=regions)


def test_status_no_region_queries_all_enabled(runner, multi_region_status):
    """Naked status discovers all enabled regions and queries them."""
    multi_region_status.regions.return_value = ["eu-west-1", "us-east-1", "us-west-2"]
    multi_region_status.find.return_value = ([_region_instance(region="us-east-1")], [])
    result = runner.invoke(main, ["status"])
    assert result.exit_code == 0
    multi_region_status.regions.assert_called_once()
    assert multi_region_status.find.call_args[0][2] == ["eu-west-1", "us-east-1", "us-west-2"]
    assert "Querying 3 enabled region(s)" in result.output
    assert "us-east-1" in result.output


def test_status_selected_regions_filter(runner, multi_region_status):
    """Explicit --region values skip discovery and restrict the query."""
    multi_region_status.find.return_value = ([_region_instance(region="us-east-1")], [])
    result = runner.invoke(main, ["status", "--region", "us-east-1", "--region", "eu-west-1"])
    assert result.exit_code == 0
    multi_region_status.regions.assert_not_called()
    assert multi_region_status.find.call_args[0][2] == ["us-east-1", "eu-west-1"]
    assert "Showing status for selected region(s): us-east-1, eu-west-1" in result.output


def test_status_region_failure_warns_and_continues(runner, multi_region_status):
    """A failed region produces a warning but does not abort the command."""
    multi_region_status.regions.return_value = ["us-east-1", "ap-south-1"]
    multi_region_status.find.return_value = (
        [_region_instance(region="us-east-1")],
        [{"region": "ap-south-1", "error": "AuthFailure: not authorized"}],
    )
    result = runner.invoke(main, ["status"])
    assert result.exit_code == 0
    assert "Skipped region ap-south-1" in result.output
    assert "i-east" in result.output


def test_status_shows_region_per_instance(runner, multi_region_status):
    """Each instance is labelled with its region, and the terminate hint pins it."""
    multi_region_status.find.return_value = (
        [_region_instance("i-east", "us-east-1"), _region_instance("i-west", "us-west-2")],
        [],
    )
    result = runner.invoke(main, ["status"])
    assert result.exit_code == 0
    assert "Region: us-east-1" in result.output
    assert "Region: us-west-2" in result.output
    assert "--region us-east-1" in result.output


def test_status_json_includes_regions_failed(runner, multi_region_status):
    """Structured output reports queried and failed regions."""
    multi_region_status.regions.return_value = ["us-east-1", "ap-south-1"]
    multi_region_status.find.return_value = ([], [{"region": "ap-south-1", "error": "AuthFailure: not authorized"}])
    result = runner.invoke(main, ["-o", "json", "status"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["instances"] == []
    assert data["regions_queried"] == ["us-east-1", "ap-south-1"]
    assert data["regions_failed"] == [{"region": "ap-south-1", "error": "AuthFailure: not authorized"}]


# ---------------------------------------------------------------------------
# AWS credential / auth error handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error,expected",
    [
        pytest.param(
            botocore.exceptions.NoCredentialsError(),
            ["Unable to locate AWS credentials", "AWS_PROFILE", "--profile", "aws configure"],
            id="no-credentials",
        ),
        pytest.param(
            botocore.exceptions.PartialCredentialsError(provider="env", cred_var="AWS_SECRET_ACCESS_KEY"),
            ["Incomplete AWS credentials", "aws configure list"],
            id="partial-credentials",
        ),
        pytest.param(
            client_error("ExpiredTokenException", "The security token is expired"),
            ["AWS authorization failed", "expired"],
            id="expired-token",
        ),
        pytest.param(
            client_error("AuthFailure", "credentials are invalid"), ["AWS authorization failed"], id="auth-failure"
        ),
    ],
)
def test_credential_errors_show_friendly_message(runner, cli_session, error, expected):
    """Credential/auth errors show a helpful message, not a raw traceback."""
    with patch("aws_bootstrap.cli.list_enabled_regions", side_effect=error):
        result = runner.invoke(main, ["status"])
    assert result.exit_code != 0
    for text in expected:
        assert text in result.output


def test_unhandled_client_error_propagates(runner, cli_session):
    """Non-auth ClientErrors should propagate without being caught."""
    with patch("aws_bootstrap.cli.list_enabled_regions", side_effect=client_error("UnknownError", "something")):
        result = runner.invoke(main, ["status"])
    assert result.exit_code != 0
    assert isinstance(result.exception, botocore.exceptions.ClientError)


def test_profile_not_found_shows_friendly_error(runner):
    """ProfileNotFound should show the missing profile name and list command."""
    with patch(
        "aws_bootstrap.cli.boto3.Session", side_effect=botocore.exceptions.ProfileNotFound(profile="nonexistent")
    ):
        result = runner.invoke(main, ["status", "--profile", "nonexistent"])
    assert result.exit_code != 0
    assert "nonexistent" in result.output
    assert "aws configure list-profiles" in result.output


@pytest.mark.parametrize(
    "argv,target",
    [
        (["terminate"], "aws_bootstrap.cli.find_tagged_instances"),
        (["list", "instance-types"], "aws_bootstrap.cli.list_instance_types"),
    ],
    ids=["terminate", "nested-list-subcommand"],
)
def test_no_credentials_caught_on_every_subcommand(runner, cli_session, argv, target):
    """Credential errors are caught for all subcommands (incl. nested ones), not just status."""
    with patch(target, side_effect=botocore.exceptions.NoCredentialsError()):
        result = runner.invoke(main, argv)
    assert result.exit_code != 0
    assert "Unable to locate AWS credentials" in result.output


# ---------------------------------------------------------------------------
# terminate (find_tagged_instances patched at the cli level)
# ---------------------------------------------------------------------------


def _state_changes(_ec2, ids):
    return [
        {"InstanceId": i, "PreviousState": {"Name": "running"}, "CurrentState": {"Name": "shutting-down"}} for i in ids
    ]


@pytest.fixture
def terminate_aws(cli_session):
    """One tagged instance (i-abc123) that terminates cleanly; explicit ids are owned and resolve to themselves."""
    with (
        patch("aws_bootstrap.cli.find_tagged_instances", return_value=[_instance(Name="test")]) as find,
        patch("aws_bootstrap.cli.terminate_tagged_instances", side_effect=_state_changes) as terminate,
        patch("aws_bootstrap.cli.remove_ssh_host", return_value=None) as remove,
        patch("aws_bootstrap.cli.find_unowned_instances", return_value=[]) as unowned,
        patch("aws_bootstrap.cli.resolve_instance_id", side_effect=lambda value: value) as resolve,
    ):
        yield SimpleNamespace(
            session=cli_session, find=find, terminate=terminate, remove=remove, unowned=unowned, resolve=resolve
        )


def test_terminate_no_instances_reports_and_never_prompts(runner, terminate_aws):
    terminate_aws.find.return_value = []
    result = runner.invoke(main, ["terminate"])  # no --yes and no stdin: a prompt would abort
    assert result.exit_code == 0
    assert "No active" in result.output
    assert "Terminate" not in result.output


def test_terminate_no_instances_json(runner, terminate_aws):
    terminate_aws.find.return_value = []
    result = runner.invoke(main, ["-o", "json", "terminate", "--yes"])
    assert result.exit_code == 0
    assert json.loads(result.output) == {"terminated": []}


def test_terminate_all_removes_aliases_and_skips_the_ownership_check(runner, terminate_aws):
    terminate_aws.remove.return_value = "aws-gpu1"
    result = runner.invoke(main, ["terminate", "--yes"])
    assert result.exit_code == 0
    assert "Terminated 1" in result.output
    terminate_aws.terminate.assert_called_once()
    assert terminate_aws.terminate.call_args[0][1] == ["i-abc123"]
    assert "Removed SSH config alias: aws-gpu1" in result.output
    terminate_aws.remove.assert_called_once_with("i-abc123")
    # Terminating everything already goes through tag discovery, so re-checking
    # ownership would be a pointless extra API call.
    terminate_aws.unowned.assert_not_called()


def test_terminate_output_json(runner, terminate_aws):
    terminate_aws.remove.return_value = "aws-gpu1"
    result = runner.invoke(main, ["-o", "json", "terminate", "--yes"])
    assert result.exit_code == 0
    assert json.loads(result.output)["terminated"] == [
        {
            "instance_id": "i-abc123",
            "previous_state": "running",
            "current_state": "shutting-down",
            "ssh_alias_removed": "aws-gpu1",
        }
    ]


def test_terminate_cancelled(runner, terminate_aws):
    result = runner.invoke(main, ["terminate"], input="n\n")
    assert result.exit_code == 0
    assert "Cancelled" in result.output
    terminate_aws.terminate.assert_not_called()


def test_terminate_json_requires_yes(runner, terminate_aws):
    """Structured output without --yes should error."""
    result = runner.invoke(main, ["-o", "json", "terminate"])
    assert result.exit_code != 0
    assert "--yes is required" in result.output


def test_terminate_by_alias_reports_the_resolution(runner, terminate_aws):
    terminate_aws.resolve.side_effect = lambda value: "i-abc123"
    result = runner.invoke(main, ["terminate", "--yes", "aws-gpu1"])
    assert result.exit_code == 0
    assert "Resolved alias 'aws-gpu1' -> i-abc123" in result.output
    assert "Terminated 1" in result.output
    terminate_aws.resolve.assert_called_once_with("aws-gpu1")
    assert terminate_aws.terminate.call_args[0][1] == ["i-abc123"]


def test_terminate_by_instance_id_passes_through_silently(runner, terminate_aws):
    result = runner.invoke(main, ["terminate", "--yes", "i-abc123"])
    assert result.exit_code == 0
    assert "Resolved alias" not in result.output
    assert "Terminated 1" in result.output
    terminate_aws.resolve.assert_called_once_with("i-abc123")
    assert terminate_aws.terminate.call_args[0][1] == ["i-abc123"]


def test_terminate_unknown_alias_errors(runner, terminate_aws):
    terminate_aws.resolve.side_effect = lambda value: None
    result = runner.invoke(main, ["terminate", "--yes", "aws-gpu99"])
    assert result.exit_code != 0
    assert "Could not resolve 'aws-gpu99'" in result.output


def test_terminate_refuses_an_instance_it_did_not_create(runner, terminate_aws):
    """Regression: an explicit id bypassed tag discovery entirely, so a typo (or
    an agent's wrong id) terminated an unrelated production instance."""
    terminate_aws.unowned.return_value = ["i-0deadbeefdeadbeef"]
    result = runner.invoke(main, ["terminate", "--yes", "i-0deadbeefdeadbeef"])
    assert result.exit_code != 0
    assert "not created by aws-bootstrap" in result.output
    assert "--force" in result.output
    terminate_aws.terminate.assert_not_called()


def test_terminate_force_overrides_the_ownership_guard(runner, terminate_aws):
    terminate_aws.unowned.return_value = ["i-0deadbeefdeadbeef"]
    result = runner.invoke(main, ["terminate", "--yes", "--force", "i-0deadbeefdeadbeef"])
    assert result.exit_code == 0
    terminate_aws.terminate.assert_called_once()
    terminate_aws.unowned.assert_not_called()


@pytest.fixture
def instance_volume():
    volume = {"VolumeId": "vol-data1", "Size": 96, "Device": "/dev/sdf", "State": "in-use"}
    with (
        patch("aws_bootstrap.cli.find_ebs_volumes_for_instance", return_value=[volume]),
        patch("aws_bootstrap.cli.delete_ebs_volume") as delete,
    ):
        yield delete


def test_terminate_deletes_ebs_by_default(runner, terminate_aws, instance_volume):
    result = runner.invoke(main, ["terminate", "--yes"])
    assert result.exit_code == 0
    instance_volume.assert_called_once_with(terminate_aws.session.return_value.client.return_value, "vol-data1")


def test_terminate_keep_ebs_preserves(runner, terminate_aws, instance_volume):
    result = runner.invoke(main, ["terminate", "--yes", "--keep-ebs"])
    assert result.exit_code == 0
    assert "Preserving EBS volume: vol-data1" in result.output
    assert "aws-bootstrap launch --ebs-volume-id vol-data1" in result.output
    instance_volume.assert_not_called()


# ---------------------------------------------------------------------------
# cleanup
# ---------------------------------------------------------------------------


def _live(instance_id, public_ip="1.2.3.4", region="us-east-1", cluster_id="", rank=None):
    return {"InstanceId": instance_id, "PublicIp": public_ip, "Region": region, "ClusterId": cluster_id, "Rank": rank}


@pytest.fixture
def cleanup_env(cli_session, tmp_path, key_path):
    """An empty ~/.ssh/config in tmp_path; one enabled region with nothing live in it.

    ``alias(id, ip, **kw)`` adds a managed stanza; ``sync`` is the ``--sync --yes`` argv.
    """
    cfg = tmp_path / "config"
    cfg.write_text("")
    with (
        patch("aws_bootstrap.cli.list_enabled_regions", return_value=["us-east-1"]) as regions,
        patch("aws_bootstrap.cli.find_tagged_instances_in_regions", return_value=([], [])) as find,
        patch("aws_bootstrap.cli._SSH_CONFIG_PATH", cfg),
    ):
        yield SimpleNamespace(
            cfg=cfg,
            regions=regions,
            find=find,
            alias=lambda iid, ip, **kw: add_ssh_host(iid, ip, "ubuntu", key_path, config_path=cfg, **kw),
            sync=["--sync", "--yes", "--key-path", str(key_path)],
        )


def test_cleanup_in_sync_reports_nothing_and_skips_the_orphan_scan(runner, cleanup_env):
    with patch("aws_bootstrap.cli.find_orphan_ebs_volumes_in_regions") as mock_orphans:
        result = runner.invoke(main, ["cleanup"])
    assert result.exit_code == 0
    assert "in sync" in result.output
    mock_orphans.assert_not_called()  # volumes are only checked with --include-ebs


def test_cleanup_keeps_alive_instance_in_other_region(runner, cleanup_env):
    cleanup_env.regions.return_value = ["us-east-1", "us-west-2"]
    cleanup_env.alias("i-east0001", "1.1.1.1")
    cleanup_env.find.return_value = ([_live("i-east0001", "1.1.1.1", "us-east-1")], [])
    result = runner.invoke(main, ["-o", "json", "cleanup", "--yes"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["cleaned"] == []
    assert "i-east0001" in cleanup_env.cfg.read_text()


def test_cleanup_removes_stale(runner, cleanup_env):
    cleanup_env.alias("i-dead0001", "1.1.1.1")  # not alive anywhere, no failures
    result = runner.invoke(main, ["-o", "json", "cleanup", "--yes"])
    assert result.exit_code == 0, result.output
    assert [c["instance_id"] for c in json.loads(result.output)["cleaned"]] == ["i-dead0001"]
    assert "i-dead0001" not in cleanup_env.cfg.read_text()


def test_cleanup_dry_run_does_not_modify(runner, cleanup_env):
    cleanup_env.alias("i-dead0001", "1.1.1.1")
    result = runner.invoke(main, ["cleanup", "--dry-run"])
    assert result.exit_code == 0
    assert "would remove" in result.output
    assert "i-dead0001" in cleanup_env.cfg.read_text()  # unchanged


def test_cleanup_conservative_guard_skips_removal_on_region_failure(runner, cleanup_env):
    cleanup_env.alias("i-gone0001", "1.1.1.1")
    # Not in the (partial) live list, but a region query FAILED -> must not remove.
    cleanup_env.find.return_value = ([], [{"region": "us-east-1", "error": "AuthFailure"}])
    result = runner.invoke(main, ["-o", "json", "cleanup", "--yes"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["cleaned"] == []
    assert data["regions_failed"]
    assert "i-gone0001" in cleanup_env.cfg.read_text()


def test_cleanup_sync_adds_missing_cluster_alias(runner, cleanup_env):
    cleanup_env.find.return_value = ([_live("i-c1000001", "1.2.3.4", cluster_id="ml1", rank=0)], [])
    with patch("aws_bootstrap.cli.add_ssh_host", return_value="aws-ml1-0") as mock_add:
        result = runner.invoke(main, ["-o", "json", "cleanup", *cleanup_env.sync])
    assert result.exit_code == 0, result.output
    assert any(a["instance_id"] == "i-c1000001" for a in json.loads(result.output)["added"])
    assert mock_add.call_args.kwargs.get("alias") == "aws-ml1-0"  # deterministic cluster alias


def test_cleanup_sync_repairs_drift(runner, cleanup_env):
    cleanup_env.alias("i-d41f7001", "1.1.1.1")  # old IP
    cleanup_env.find.return_value = ([_live("i-d41f7001", "9.9.9.9")], [])  # new IP
    result = runner.invoke(main, ["-o", "json", "cleanup", *cleanup_env.sync])
    assert result.exit_code == 0, result.output
    assert [u["instance_id"] for u in json.loads(result.output)["updated"]] == ["i-d41f7001"]
    assert "9.9.9.9" in cleanup_env.cfg.read_text()


def test_cleanup_sync_drift_preserves_port(runner, cleanup_env):
    cleanup_env.alias("i-0aaf7001", "1.1.1.1", port=2222)
    cleanup_env.find.return_value = ([_live("i-0aaf7001", "9.9.9.9")], [])
    result = runner.invoke(main, ["cleanup", *cleanup_env.sync])
    assert result.exit_code == 0, result.output
    content = cleanup_env.cfg.read_text()
    assert "HostName 9.9.9.9" in content  # IP repaired
    assert "Port 2222" in content  # non-standard port preserved, not reset to 22


def test_cleanup_requires_yes_in_structured(runner, cli_session):
    result = runner.invoke(main, ["-o", "json", "cleanup"])
    assert result.exit_code != 0


@pytest.mark.parametrize("fmt", ["json", "yaml", "table"])
def test_cleanup_renders_in_each_structured_format(runner, cleanup_env, fmt):
    cleanup_env.alias("i-dead0001", "1.1.1.1")
    result = runner.invoke(main, ["-o", fmt, "cleanup", "--yes"])
    assert result.exit_code == 0, result.output
    assert "i-dead0001" in result.output  # the removed alias's instance appears in output
    assert "'instance_id'" not in result.output  # no raw dict repr (table picks the right list)


@pytest.mark.parametrize("fmt", ["json", "yaml", "table"])
def test_cleanup_sync_renders_added_and_updated_in_each_format(runner, cleanup_env, fmt):
    # A mixed --sync run (remove + add + repair): every action must be visible in
    # ALL structured modes, not just json (table shows a unified action table).
    cleanup_env.alias("i-dead0001", "1.1.1.1")  # stale -> removed
    cleanup_env.alias("i-0d41f700", "2.2.2.2")  # drifted -> repaired
    cleanup_env.find.return_value = ([_live("i-0d41f700", "9.9.9.9"), _live("i-0aabbccd", "3.3.3.3")], [])
    result = runner.invoke(main, ["-o", fmt, "cleanup", *cleanup_env.sync])
    assert result.exit_code == 0, result.output
    for instance_id in ("i-dead0001", "i-0aabbccd", "i-0d41f700"):  # removed, added, repaired
        assert instance_id in result.output
    assert "'instance_id'" not in result.output


# --- --include-ebs (orphan volumes, scanned across regions) ----------------------


@pytest.fixture
def orphan_volume_ops():
    volume = {"VolumeId": "vol-orphan1", "Size": 50, "State": "available", "InstanceId": "i-x", "Region": "us-east-1"}
    with (
        patch("aws_bootstrap.cli.find_orphan_ebs_volumes_in_regions", return_value=[volume]),
        patch("aws_bootstrap.cli.delete_ebs_volume") as delete,
    ):
        yield delete


def test_cleanup_include_ebs_deletes_orphans(runner, cleanup_env, orphan_volume_ops):
    result = runner.invoke(main, ["-o", "json", "cleanup", "--include-ebs", "--yes"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["deleted_volumes"][0]["volume_id"] == "vol-orphan1"
    orphan_volume_ops.assert_called_once()


def test_cleanup_include_ebs_not_deleted_on_region_failure(runner, cleanup_env, orphan_volume_ops):
    cleanup_env.find.return_value = ([], [{"region": "us-east-1", "error": "AuthFailure"}])  # incomplete scan
    result = runner.invoke(main, ["-o", "json", "cleanup", "--include-ebs", "--yes"])
    assert result.exit_code == 0
    orphan_volume_ops.assert_not_called()  # conservative guard: don't delete on incomplete scan
    assert "orphan_volumes" in json.loads(result.output)  # reported as candidates, not deleted_volumes


# ---------------------------------------------------------------------------
# list instance-types / amis
# ---------------------------------------------------------------------------


def test_list_instance_types_output(runner, cli_session, instance_type_rows):
    with patch("aws_bootstrap.cli.list_instance_types", return_value=instance_type_rows):
        result = runner.invoke(main, ["list", "instance-types"])
    assert result.exit_code == 0
    assert "g4dn.xlarge" in result.output
    assert "16384" in result.output
    assert "T4" in result.output


def test_list_instance_types_json(runner, cli_session, instance_type_rows):
    with patch("aws_bootstrap.cli.list_instance_types", return_value=instance_type_rows):
        result = runner.invoke(main, ["-o", "json", "list", "instance-types"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data, list)
    assert data[0]["instance_type"] == "g4dn.xlarge"
    assert data[0]["vcpus"] == 4
    assert data[0]["memory_mib"] == 16384
    assert data[0]["gpu"] == "1x T4 (16384 MiB)"


def test_list_amis_output(runner, cli_session, ami_rows):
    with patch("aws_bootstrap.cli.list_amis", return_value=ami_rows):
        result = runner.invoke(main, ["list", "amis"])
    assert result.exit_code == 0
    assert "ami-abc123" in result.output
    assert "Deep Learning Base OSS Nvidia Driver GPU AMI" in result.output
    assert "2025-06-01" in result.output


def test_list_amis_json(runner, cli_session, ami_rows):
    with patch("aws_bootstrap.cli.list_amis", return_value=ami_rows):
        result = runner.invoke(main, ["-o", "json", "list", "amis"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data, list)
    assert data[0]["image_id"] == "ami-abc123"
    assert data[0]["name"] == "Deep Learning Base OSS Nvidia Driver GPU AMI"
    assert data[0]["creation_date"] == "2025-06-01"


@pytest.mark.parametrize(
    "argv,helper,message",
    [
        (["list", "instance-types", "--prefix", "zzz"], "list_instance_types", "No instance types found"),
        (["list", "amis", "--filter", "nonexistent*"], "list_amis", "No AMIs found"),
    ],
    ids=["instance-types", "amis"],
)
def test_list_empty(runner, cli_session, argv, helper, message):
    with patch(f"aws_bootstrap.cli.{helper}", return_value=[]):
        result = runner.invoke(main, argv)
    assert result.exit_code == 0
    assert message in result.output


def test_list_amis_default_filter_spans_architectures(runner, cli_session):
    """`list amis --arch arm64` must find the ARM64 builds without a hand-written filter.

    The old default ("Deep Learning Base OSS…*") could never match an ARM64 name
    ("Deep Learning ARM64 Base OSS…"), so --arch arm64 alone always came back empty.
    """
    with patch("aws_bootstrap.cli.list_amis", return_value=[]) as mock_list:
        result = runner.invoke(main, ["list", "amis", "--arch", "arm64", "--region", "us-west-2"])
    assert result.exit_code == 0
    assert mock_list.call_args.args[1] == "Deep Learning*Base OSS Nvidia Driver GPU AMI*"
    assert mock_list.call_args.kwargs["architecture"] == "arm64"


# ---------------------------------------------------------------------------
# quota
# ---------------------------------------------------------------------------


def _family_rows(family, spot, on_demand, codes=("L-3819A6DF", "L-DB2E81BA")):
    return [
        {"quota_code": codes[0], "quota_name": f"{family} Spot", "value": spot, "quota_type": "spot", "family": family},
        {
            "quota_code": codes[1],
            "quota_name": f"{family} On-Demand",
            "value": on_demand,
            "quota_type": "on-demand",
            "family": family,
        },
    ]


def test_quota_show_text(runner, cli_session):
    with patch("aws_bootstrap.cli.get_family_quotas", return_value=_family_rows("gvt", 4.0, 0.0)):
        result = runner.invoke(main, ["quota", "show", "--family", "gvt"])
    assert result.exit_code == 0
    assert "4" in result.output
    assert "Spot" in result.output
    assert "On-demand" in result.output


def test_quota_show_json(runner, cli_session):
    with patch("aws_bootstrap.cli.get_family_quotas", return_value=_family_rows("gvt", 4.0, 0.0)):
        result = runner.invoke(main, ["-o", "json", "quota", "show", "--family", "gvt"])
    assert result.exit_code == 0
    quotas = json.loads(result.output)["quotas"]
    assert len(quotas) == 2
    assert quotas[0]["quota_code"] == "L-3819A6DF"
    assert quotas[0]["value"] == 4.0


@pytest.mark.parametrize(
    "spot,region,desired",
    [
        # Reported regression: the hint must be pinned to the queried region, not the default.
        pytest.param(0.0, "us-west-1", 8, id="pins-region"),
        # AWS rejects a desired value <= current, so the suggestion is max(8, current + 4).
        pytest.param(32.0, "us-east-1", 36, id="above-current"),
    ],
)
def test_quota_show_request_hint(runner, cli_session, spot, region, desired):
    with patch("aws_bootstrap.cli.get_family_quotas", return_value=_family_rows("gvt", spot, 8.0)):
        result = runner.invoke(main, ["quota", "show", "--family", "gvt", "--region", region])
    assert result.exit_code == 0
    assert f"aws-bootstrap quota request --family gvt --type spot --desired-value {desired} --region {region}" in (
        result.output
    )


def test_quota_show_all_families(runner, cli_session):
    """Without --family, shows all families (gvt, p, dl)."""
    side_effect = [
        _family_rows("gvt", 4.0, 0.0),
        _family_rows("p", 0.0, 0.0, codes=("L-7212CCBC", "L-417A185B")),
        _family_rows("dl", 0.0, 0.0, codes=("L-85EED4F7", "L-6E869C2A")),
    ]
    with patch("aws_bootstrap.cli.get_family_quotas", side_effect=side_effect) as mock_quotas:
        result = runner.invoke(main, ["-o", "json", "quota", "show"])
    assert result.exit_code == 0
    assert len(json.loads(result.output)["quotas"]) == 6
    assert mock_quotas.call_count == 3


def test_quota_show_api_error(runner, cli_session):
    error = client_error("NoSuchResourceException", "not found")
    with patch("aws_bootstrap.cli.get_family_quotas", side_effect=error):
        result = runner.invoke(main, ["quota", "show", "--family", "gvt"])
    assert result.exit_code != 0


REQUEST = ["quota", "request", "--type", "spot", "--desired-value", "4"]


@pytest.fixture
def quota_request(cli_session):
    """Current spot quota 0; a request comes back as req-123 PENDING."""
    submitted = {
        "request_id": "req-123",
        "status": "PENDING",
        "quota_code": "L-3819A6DF",
        "quota_name": "Spot Quota",
        "desired_value": 4.0,
    }
    with (
        patch(
            "aws_bootstrap.cli.get_quota",
            return_value={"quota_code": "L-3819A6DF", "quota_name": "Spot Quota", "value": 0.0},
        ) as get,
        patch("aws_bootstrap.cli.request_quota_increase", return_value=submitted) as request,
    ):
        yield SimpleNamespace(get=get, request=request)


@pytest.mark.parametrize("args,stdin", [(["--yes"], None), ([], "y\n")], ids=["yes-flag", "confirmed"])
def test_quota_request_submits(runner, quota_request, args, stdin):
    result = runner.invoke(main, [*REQUEST, *args], input=stdin)
    assert result.exit_code == 0
    assert "submitted" in result.output
    assert "req-123" in result.output
    quota_request.request.assert_called_once()


def test_quota_request_cancelled(runner, quota_request):
    result = runner.invoke(main, REQUEST, input="n\n")
    assert result.exit_code == 0
    assert "Cancelled" in result.output
    quota_request.request.assert_not_called()


def test_quota_request_json(runner, quota_request):
    result = runner.invoke(main, ["-o", "json", *REQUEST, "--yes"])
    assert result.exit_code == 0
    requests = json.loads(result.output)["requests"]
    assert len(requests) == 1
    assert requests[0]["request_id"] == "req-123"
    assert requests[0]["status"] == "PENDING"
    assert requests[0]["region"] == "us-west-2"


def test_quota_request_json_requires_yes(runner, quota_request):
    result = runner.invoke(main, ["-o", "json", *REQUEST])
    assert result.exit_code != 0
    assert "--yes is required" in result.output


def test_quota_request_desired_le_current(runner, quota_request):
    quota_request.get.return_value = {"quota_code": "L-3819A6DF", "quota_name": "Spot Quota", "value": 4.0}
    result = runner.invoke(main, [*REQUEST, "--yes"])
    assert result.exit_code != 0
    assert "must be greater" in result.output


def test_quota_request_duplicate_pending(runner, quota_request):
    quota_request.request.side_effect = CLIError("already pending")
    result = runner.invoke(main, [*REQUEST, "--yes"])
    assert result.exit_code != 0
    assert "already pending" in result.output


HISTORY = ["quota", "history", "--family", "gvt", "--type", "spot"]


def test_quota_history_text(runner, cli_session, history_rows):
    with patch("aws_bootstrap.cli.get_quota_request_history", return_value=history_rows):
        result = runner.invoke(main, HISTORY)
    assert result.exit_code == 0
    assert "req-123" in result.output
    assert "APPROVED" in result.output


def test_quota_history_json(runner, cli_session, history_rows):
    with patch("aws_bootstrap.cli.get_quota_request_history", return_value=history_rows):
        result = runner.invoke(main, ["-o", "json", *HISTORY])
    assert result.exit_code == 0
    requests = json.loads(result.output)["requests"]
    assert len(requests) == 1
    assert requests[0]["request_id"] == "req-123"


def test_quota_history_empty_text(runner, cli_session):
    with patch("aws_bootstrap.cli.get_quota_request_history", return_value=[]):
        result = runner.invoke(main, HISTORY)
    assert result.exit_code == 0
    assert "No quota increase requests found" in result.output


def test_quota_history_empty_json(runner, cli_session):
    with patch("aws_bootstrap.cli.get_quota_request_history", return_value=[]):
        result = runner.invoke(main, ["-o", "json", *HISTORY])
    assert result.exit_code == 0
    assert json.loads(result.output) == {"requests": []}


def test_quota_history_all_families(runner, cli_session):
    """Without --type or --family, queries all families x all types."""
    with patch("aws_bootstrap.cli.get_quota_request_history", return_value=[]) as mock_history:
        result = runner.invoke(main, ["quota", "history"])
    assert result.exit_code == 0
    assert mock_history.call_count == 6  # 3 families x 2 types


def test_quota_history_filter_by_status(runner, cli_session):
    with patch("aws_bootstrap.cli.get_quota_request_history", return_value=[]) as mock_history:
        result = runner.invoke(main, [*HISTORY, "--status", "APPROVED"])
    assert result.exit_code == 0
    mock_history.assert_called_once()
    assert mock_history.call_args[1]["status_filter"] == "APPROVED"
