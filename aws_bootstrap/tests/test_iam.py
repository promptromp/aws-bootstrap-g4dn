"""Tests for aws_bootstrap.iam: the RunInstances profile parameter and the pre-launch check."""

from __future__ import annotations
from unittest.mock import MagicMock

import botocore.exceptions
import pytest

from aws_bootstrap.errors import CLIError
from aws_bootstrap.iam import check_instance_profile, instance_profile_name, instance_profile_spec


ARN = "arn:aws:iam::123456789012:instance-profile/app"
PATH_ARN = "arn:aws:iam::123456789012:instance-profile/team/app"


def _client_error(code: str) -> botocore.exceptions.ClientError:
    error = {"Error": {"Code": code, "Message": f"{code} message"}}
    return botocore.exceptions.ClientError(error, "GetInstanceProfile")


def _iam(*, arn: str = ARN, roles=("app-role",), error: Exception | None = None) -> MagicMock:
    client = MagicMock()
    if error:
        client.get_instance_profile.side_effect = error
    else:
        client.get_instance_profile.return_value = {
            "InstanceProfile": {"Arn": arn, "Roles": [{"RoleName": r} for r in roles]}
        }
    return client


@pytest.mark.parametrize(
    "value,spec,name",
    [
        pytest.param("app", {"Name": "app"}, "app", id="name"),
        pytest.param(ARN, {"Arn": ARN}, "app", id="arn"),
        pytest.param(PATH_ARN, {"Arn": PATH_ARN}, "app", id="arn-with-path"),
    ],
)
def test_spec_and_name(value, spec, name):
    assert instance_profile_spec(value) == spec
    assert instance_profile_name(value) == name


@pytest.mark.parametrize(
    "value,arn",
    [pytest.param("app", ARN, id="name"), pytest.param(PATH_ARN, PATH_ARN, id="matching-arn")],
)
def test_check_returns_the_roles_and_looks_up_by_name(value, arn, capsys):
    iam = _iam(arn=arn, roles=("app-role", "other"))
    assert check_instance_profile(iam, value) == ["app-role", "other"]
    iam.get_instance_profile.assert_called_once_with(InstanceProfileName="app")
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(
    "iam,value,match",
    [
        pytest.param(_iam(error=_client_error("NoSuchEntity")), "app", "'app' not found", id="no-such-profile"),
        pytest.param(_iam(arn=PATH_ARN), ARN, "doesn't match this account's 'app'", id="arn-mismatch"),
    ],
)
def test_check_fails_on_a_definite_answer(iam, value, match):
    with pytest.raises(CLIError, match=match):
        check_instance_profile(iam, value)


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(_client_error("AccessDenied"), id="no-get-permission"),
        pytest.param(botocore.exceptions.EndpointConnectionError(endpoint_url="https://iam"), id="unreachable"),
    ],
)
def test_check_warns_and_continues_when_it_cannot_tell(error, capsys):
    """Launching needs iam:PassRole, not iam:GetInstanceProfile, and EC2 still checks the profile."""
    assert check_instance_profile(_iam(error=error), "app") is None
    assert "could not verify IAM instance profile 'app'" in capsys.readouterr().err


def test_check_warns_about_a_profile_without_a_role(capsys):
    assert check_instance_profile(_iam(roles=()), "app") == []
    assert "no role" in capsys.readouterr().err
