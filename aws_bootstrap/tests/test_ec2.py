"""Tests for EC2 helper functions."""

from __future__ import annotations
from unittest.mock import MagicMock

import click
import pytest

from aws_bootstrap.ec2 import get_latest_dl_ami


def test_get_latest_dl_ami_picks_newest():
    ec2 = MagicMock()
    ec2.describe_images.return_value = {
        "Images": [
            {"ImageId": "ami-old", "Name": "DL AMI old", "CreationDate": "2024-01-01T00:00:00Z"},
            {"ImageId": "ami-new", "Name": "DL AMI new", "CreationDate": "2025-06-01T00:00:00Z"},
            {"ImageId": "ami-mid", "Name": "DL AMI mid", "CreationDate": "2025-01-01T00:00:00Z"},
        ]
    }
    ami = get_latest_dl_ami(ec2, "DL AMI*")
    assert ami["ImageId"] == "ami-new"


def test_get_latest_dl_ami_no_results():
    ec2 = MagicMock()
    ec2.describe_images.return_value = {"Images": []}
    with pytest.raises(click.ClickException, match="No AMI found"):
        get_latest_dl_ami(ec2, "nonexistent*")
