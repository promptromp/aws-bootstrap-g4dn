"""Region-aware behaviour of the ``quota`` and ``list`` CLI commands.

These exercise the repeatable ``--region`` support, region labelling, and the
per-region structured output added alongside multi-region ``status``. They use
the shared fixtures in ``conftest.py`` and ``pytest.mark.parametrize`` rather
than the decorator-stacking style of the older ``test_cli.py`` cases.
"""

from __future__ import annotations
import json
from unittest.mock import patch

import pytest

from aws_bootstrap.cli import main


MULTI_REGIONS = ["us-east-1", "us-west-2"]


def _r(regions):
    """Expand region names into repeatable ``-r <region>`` CLI flags."""
    flags: list[str] = []
    for region in regions:
        flags += ["-r", region]
    return flags


# ---------------------------------------------------------------------------
# repeatable --region appears in help for every region-aware command
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["quota", "show", "--help"],
        ["quota", "history", "--help"],
        ["quota", "request", "--help"],
        ["list", "instance-types", "--help"],
        ["list", "amis", "--help"],
    ],
)
def test_region_option_is_repeatable_in_help(runner, argv):
    result = runner.invoke(main, argv)
    assert result.exit_code == 0
    assert "-r, --region" in result.output
    assert "repeatable" in result.output


# ---------------------------------------------------------------------------
# quota show
# ---------------------------------------------------------------------------


def test_quota_show_labels_single_region(runner, cli_session, quota_rows):
    with patch("aws_bootstrap.cli.get_family_quotas", return_value=quota_rows):
        result = runner.invoke(main, ["quota", "show", "--family", "gvt"])
    assert result.exit_code == 0
    assert "Region: us-west-2" in result.output


def test_quota_show_multi_region_groups_and_queries_each(runner, cli_session, quota_rows):
    with patch("aws_bootstrap.cli.get_family_quotas", return_value=quota_rows) as mock_quotas:
        result = runner.invoke(main, ["quota", "show", "--family", "gvt", "-r", "us-east-1", "-r", "us-west-2"])
    assert result.exit_code == 0
    # one get_family_quotas call per region (single family)
    assert mock_quotas.call_count == 2
    assert "us-east-1 — EC2 GPU vCPU Quotas" in result.output
    assert "us-west-2 — EC2 GPU vCPU Quotas" in result.output
    # request hint is pinned to each region
    assert "--region us-east-1" in result.output
    assert "--region us-west-2" in result.output


def test_quota_show_json_tags_region(runner, cli_session, quota_rows):
    with patch("aws_bootstrap.cli.get_family_quotas", return_value=quota_rows):
        result = runner.invoke(
            main, ["-o", "json", "quota", "show", "--family", "gvt", "-r", "us-east-1", "-r", "eu-west-1"]
        )
    assert result.exit_code == 0
    data = json.loads(result.output)
    regions = {q["region"] for q in data["quotas"]}
    assert regions == {"us-east-1", "eu-west-1"}


# ---------------------------------------------------------------------------
# quota history
# ---------------------------------------------------------------------------


def test_quota_history_multi_region_tags_region(runner, cli_session, history_rows):
    # The real API returns fresh objects per call; mirror that so per-region
    # tagging is not aliased across calls.
    def fresh(*_args, **_kwargs):
        return [dict(r) for r in history_rows]

    argv = ["-o", "json", "quota", "history", "--family", "gvt", "--type", "spot", *_r(MULTI_REGIONS)]
    with patch("aws_bootstrap.cli.get_quota_request_history", side_effect=fresh):
        result = runner.invoke(main, argv)
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert {r["region"] for r in data["requests"]} == {"us-east-1", "us-west-2"}


def test_quota_history_text_shows_region(runner, cli_session, history_rows):
    with patch("aws_bootstrap.cli.get_quota_request_history", return_value=history_rows):
        result = runner.invoke(main, ["quota", "history", "--family", "gvt", "--type", "spot", "-r", "eu-west-1"])
    assert result.exit_code == 0
    assert "Region(s): eu-west-1" in result.output
    assert "Region: eu-west-1" in result.output


# ---------------------------------------------------------------------------
# quota request (multi-region writes)
# ---------------------------------------------------------------------------


def _fresh_request(_client, code, value):
    return {
        "request_id": "req-x",
        "status": "PENDING",
        "quota_code": code,
        "quota_name": "All G and VT Spot Instance Requests",
        "desired_value": value,
    }


def test_quota_request_submits_in_each_region(runner, cli_session):
    with (
        patch("aws_bootstrap.cli.get_quota", return_value={"quota_name": "Q", "value": 0.0}),
        patch("aws_bootstrap.cli.request_quota_increase", side_effect=_fresh_request) as mock_request,
    ):
        argv = ["-o", "json", "quota", "request", "--type", "spot", "--desired-value", "8", "--yes", *_r(MULTI_REGIONS)]
        result = runner.invoke(main, argv)
    assert result.exit_code == 0
    assert mock_request.call_count == 2
    data = json.loads(result.output)
    assert [r["region"] for r in data["requests"]] == ["us-east-1", "us-west-2"]


def test_quota_request_aborts_when_any_region_too_low(runner, cli_session):
    # us-east-1 current=0 (ok), us-west-2 current=8 (>= desired 8 -> abort all)
    currents = [{"quota_name": "Q", "value": 0.0}, {"quota_name": "Q", "value": 8.0}]
    with (
        patch("aws_bootstrap.cli.get_quota", side_effect=currents),
        patch("aws_bootstrap.cli.request_quota_increase") as mock_request,
    ):
        argv = ["quota", "request", "--type", "spot", "--desired-value", "8", "--yes", *_r(MULTI_REGIONS)]
        result = runner.invoke(main, argv)
    assert result.exit_code != 0
    assert "No requests were submitted" in result.output
    assert "us-west-2" in result.output
    mock_request.assert_not_called()


def test_quota_request_confirm_lists_all_regions(runner, cli_session):
    with (
        patch("aws_bootstrap.cli.get_quota", return_value={"quota_name": "Q", "value": 0.0}),
        patch("aws_bootstrap.cli.request_quota_increase", side_effect=_fresh_request),
    ):
        argv = ["quota", "request", "--type", "spot", "--desired-value", "8", *_r(MULTI_REGIONS)]
        result = runner.invoke(main, argv, input="y\n")
    assert result.exit_code == 0
    assert "in 2 region(s) (us-east-1, us-west-2)" in result.output


# ---------------------------------------------------------------------------
# list instance-types / amis
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command,helper,rows_fixture",
    [
        ("instance-types", "aws_bootstrap.cli.list_instance_types", "instance_type_rows"),
        ("amis", "aws_bootstrap.cli.list_amis", "ami_rows"),
    ],
)
def test_list_multi_region_queries_and_tags_each(runner, cli_session, request, command, helper, rows_fixture):
    rows = request.getfixturevalue(rows_fixture)
    with patch(helper, return_value=rows) as mock_helper:
        result = runner.invoke(main, ["-o", "json", "list", command, "-r", "us-east-1", "-r", "us-west-2"])
    assert result.exit_code == 0
    assert mock_helper.call_count == 2
    data = json.loads(result.output)
    assert {row["region"] for row in data} == {"us-east-1", "us-west-2"}


@pytest.mark.parametrize(
    "command,helper,rows_fixture",
    [
        ("instance-types", "aws_bootstrap.cli.list_instance_types", "instance_type_rows"),
        ("amis", "aws_bootstrap.cli.list_amis", "ami_rows"),
    ],
)
def test_list_single_region_labels_region(runner, cli_session, request, command, helper, rows_fixture):
    rows = request.getfixturevalue(rows_fixture)
    with patch(helper, return_value=rows):
        result = runner.invoke(main, ["list", command])
    assert result.exit_code == 0
    assert "Region: us-west-2" in result.output


@pytest.mark.parametrize(
    "prefix,family",
    [("g4dn", "gvt"), ("g5", "gvt"), ("p5", "p"), ("dl1", "dl")],
)
def test_list_instance_types_suggests_quota_commands(runner, cli_session, instance_type_rows, prefix, family):
    with patch("aws_bootstrap.cli.list_instance_types", return_value=instance_type_rows):
        result = runner.invoke(main, ["list", "instance-types", "--prefix", prefix])
    assert result.exit_code == 0
    # The next-steps header names the derived quota family so the gvt<->g5 mapping is explicit.
    assert f"{prefix}.* draws from the '{family}' vCPU quota family" in result.output
    assert f"aws-bootstrap quota show --family {family} --region us-west-2" in result.output
    request_hint = f"aws-bootstrap quota request --family {family} --type spot --desired-value 8 --region us-west-2"
    assert request_hint in result.output


def test_list_instance_types_no_quota_hint_for_non_gpu_family(runner, cli_session, instance_type_rows):
    with patch("aws_bootstrap.cli.list_instance_types", return_value=instance_type_rows):
        result = runner.invoke(main, ["list", "instance-types", "--prefix", "t3"])
    assert result.exit_code == 0
    assert "Next steps" not in result.output


def test_list_instance_types_shows_quota_family_column(runner, cli_session, instance_type_rows):
    with patch("aws_bootstrap.cli.list_instance_types", return_value=instance_type_rows):
        result = runner.invoke(main, ["list", "instance-types", "--prefix", "g5"])
    assert result.exit_code == 0
    assert "Quota Family" in result.output
    # g4dn.xlarge (the fixture row) belongs to the gvt quota family
    assert "gvt" in result.output


def test_list_instance_types_json_includes_quota_family(runner, cli_session, instance_type_rows):
    with patch("aws_bootstrap.cli.list_instance_types", return_value=instance_type_rows):
        result = runner.invoke(main, ["-o", "json", "list", "instance-types"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data[0]["quota_family"] == "gvt"
