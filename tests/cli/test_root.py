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


# ==================== EXCEPTION GROUP UNWRAPPING ====================

if sys.version_info < (3, 11):  # pragma: no cover - 3.11+ has it as a builtin
    from exceptiongroup import ExceptionGroup


@pytest.mark.unit
def test_classify_unwraps_single_error_task_group():
    """Parallel tools raise ExceptionGroup; the real error must still classify.

    Regression: an ElevenLabs 429 inside the podcast fan-out surfaced as
    "internal: ExceptionGroup: unhandled errors in a TaskGroup", hiding the
    actionable "lower SANZARU_ELEVENLABS_MAX_CONCURRENCY" message.
    """
    from sanzaru.cli._runtime import _classify
    from sanzaru.exceptions import TTSAPIError

    group = ExceptionGroup("unhandled errors in a TaskGroup", [TTSAPIError("rate limit hit")])

    error = _classify(group)

    assert error.error_type == "api_error"
    assert "rate limit hit" in str(error)


@pytest.mark.unit
def test_classify_unwraps_nested_task_groups():
    """Segment fan-out nests inside chunk fan-out, so groups can nest."""
    from sanzaru.cli._runtime import _classify

    group = ExceptionGroup("outer", [ExceptionGroup("inner", [ValueError("bad speed")])])

    error = _classify(group)

    assert error.error_type == "usage"
    assert error.exit_code == 2
    assert "bad speed" in str(error)


@pytest.mark.unit
def test_classify_keeps_multi_error_groups_but_names_them():
    """Several distinct failures: picking one to represent the rest hides the others."""
    from sanzaru.cli._runtime import _classify

    group = ExceptionGroup("outer", [ValueError("first"), RuntimeError("second")])

    error = _classify(group)

    assert error.error_type == "internal"
    assert "2 parallel tasks failed" in str(error)
    assert "first" in str(error)
    assert "second" in str(error)
