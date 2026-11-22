"""IAM instance profiles: the RunInstances parameter and a read-only pre-launch check.

aws-bootstrap only attaches a profile the user already has. It never creates or
changes roles, policies, or profiles.
"""

from __future__ import annotations

import botocore.exceptions
import click

from .errors import CLIError


# An unreachable or flaky endpoint (not credentials: those reach the CLI's own handler).
_UNREACHABLE = (botocore.exceptions.ConnectionError, botocore.exceptions.HTTPClientError)


class InstanceProfileError(CLIError):
    """EC2 refused the launch because of the instance profile: the same in every AZ, region and market."""


def instance_profile_spec(value: str) -> dict[str, str]:
    """RunInstances ``IamInstanceProfile``: ``{"Arn": …}`` for an ARN, else ``{"Name": …}``."""
    return {"Arn": value} if value.startswith("arn:") else {"Name": value}


def instance_profile_name(value: str) -> str:
    """The profile name: an ARN's last path segment (``…:instance-profile/path/name``), else *value*."""
    return value.rsplit("/", 1)[-1] if value.startswith("arn:") else value


def _warn(message: str) -> None:
    # stderr, unconditionally: structured-output users must see it too.
    click.echo(click.style("  WARNING: ", fg="yellow", bold=True) + message, err=True)


def check_instance_profile(iam_client, value: str) -> list[str] | None:
    """Check *value* (a name or ARN) names an instance profile; return its role names.

    Raises :class:`CLIError` only when the answer is definite: no such profile, or an
    ARN that isn't this account's profile of that name. When the profile can't be
    read (e.g. no ``iam:GetInstanceProfile`` — launching needs only ``iam:PassRole`` — or
    an unreachable endpoint), warns and returns ``None``: EC2 still rejects an invalid
    profile before it creates an instance. Missing credentials propagate to the CLI's
    handler. A role ARN passed by mistake is an error; a profile with no role gives the
    instance no credentials, so that warns.
    """
    if value.startswith("arn:") and ":role/" in value:
        raise CLIError(
            f"{value} is an IAM role ARN; --iam-instance-profile takes an instance profile (name or ARN).\n"
            "  A role created for EC2 in the console has a profile of the same name: pass that name."
        )
    name = instance_profile_name(value)
    try:
        profile = iam_client.get_instance_profile(InstanceProfileName=name)["InstanceProfile"]
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchEntity":
            raise CLIError(
                f"IAM instance profile '{name}' not found.\n"
                "  List yours with:  aws iam list-instance-profiles --query 'InstanceProfiles[].InstanceProfileName'\n"
                "  aws-bootstrap attaches an existing profile; it never creates one."
            ) from None
        _warn(f"could not verify IAM instance profile '{name}' ({e}); continuing (EC2 checks it at launch).")
        return None
    except _UNREACHABLE as e:
        _warn(f"could not verify IAM instance profile '{name}' ({e}); continuing (EC2 checks it at launch).")
        return None
    if value.startswith("arn:") and profile["Arn"] != value:
        raise CLIError(f"IAM instance profile ARN {value} doesn't match this account's '{name}': {profile['Arn']}.")
    roles = [r["RoleName"] for r in profile.get("Roles", [])]
    if not roles:
        _warn(f"IAM instance profile '{name}' has no role, so the instance gets no AWS credentials from it.")
    return roles
