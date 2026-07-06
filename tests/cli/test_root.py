# SPDX-License-Identifier: MIT
"""Back-compat tests for the root command: bare `sanzaru` must stay the MCP server."""

import pathlib
import subprocess
import sys

import pytest
from click.testing import CliRunner

from sanzaru.cli import cli


@pytest.mark.integration
def test_bare_invocation_starts_stdio_server(mocker):
    """`sanzaru` with no subcommand routes to run_server with historical defaults."""
    run_server = mocker.patch("sanzaru.server.run_server")

    result = CliRunner().invoke(cli, [])

    assert result.exit_code == 0
    run_server.assert_called_once_with(transport="stdio", host="127.0.0.1", port=8000)


@pytest.mark.integration
def test_bare_invocation_forwards_http_flags(mocker):
    """`sanzaru --transport http --host 0.0.0.0 --port 9000` keeps working."""
    run_server = mocker.patch("sanzaru.server.run_server")

    result = CliRunner().invoke(cli, ["--transport", "http", "--host", "0.0.0.0", "--port", "9000"])

    assert result.exit_code == 0
    run_server.assert_called_once_with(transport="http", host="0.0.0.0", port=9000)


@pytest.mark.integration
def test_serve_subcommand_forwards_flags(mocker):
    """`sanzaru serve` is the explicit alias for the server."""
    run_server = mocker.patch("sanzaru.server.run_server")

    result = CliRunner().invoke(cli, ["serve", "--transport", "http", "--port", "3000"])

    assert result.exit_code == 0
    run_server.assert_called_once_with(transport="http", host="127.0.0.1", port=3000)


@pytest.mark.integration
def test_server_main_argparse_path_still_routes(mocker):
    """Direct importers of sanzaru.server:main keep the argparse behavior."""
    import sanzaru.server as server

    run_server = mocker.patch("sanzaru.server.run_server")
    mocker.patch.object(sys, "argv", ["sanzaru", "--transport", "http", "--port", "9000"])

    server.main()

    run_server.assert_called_once_with(transport="http", host="127.0.0.1", port=9000)


@pytest.mark.integration
def test_help_lists_serve_and_exits_clean():
    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "serve" in result.output


@pytest.mark.integration
def test_console_script_points_at_cli_main():
    pyproject = pathlib.Path(__file__).parents[2] / "pyproject.toml"
    assert 'sanzaru = "sanzaru.cli:main"' in pyproject.read_text()


@pytest.mark.integration
def test_cli_import_is_lightweight():
    """`sanzaru <cmd> --help` latency guard: importing the CLI package must not
    pull the FastMCP server, openai, or pydantic into the process."""
    code = (
        "import sys; import sanzaru.cli; "
        "heavy = {'openai', 'sanzaru.server', 'pydantic'} & set(sys.modules); "
        "assert not heavy, f'heavy imports leaked: {heavy}'"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
