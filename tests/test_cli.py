"""Integration tests for the Swara CLI commands."""

from click.testing import CliRunner
import pytest

from swara.cli.main import cli


def test_cli_version():
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "Swara" in result.output


def test_cli_doctor():
    runner = CliRunner()
    result = runner.invoke(cli, ["doctor"])
    assert result.exit_code == 0
    assert "System & Runtime Checks" in result.output
    assert "Python Version" in result.output


def test_cli_models_list():
    runner = CliRunner()
    result = runner.invoke(cli, ["models", "list"])
    assert result.exit_code == 0
    assert "silero_vad" in result.output
    assert "wespeaker_resnet34" in result.output


def test_cli_speakers_list():
    runner = CliRunner()
    result = runner.invoke(cli, ["speakers", "list"])
    assert result.exit_code == 0
