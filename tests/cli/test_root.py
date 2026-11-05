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
        "heavy = {'openai', 'sanzaru.server', 'pydantic', 'httpx'} & set(sys.modules); "
        "assert not heavy, f'heavy imports leaked: {heavy}'"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


# The base install is the agent CLI; MCP lives behind the `mcp` extra. These guard
# both halves: nothing but the server may import the MCP stack, and a CLI-only
# install must say how to get the server rather than die on an ImportError.
_BLOCK_MCP = (
    "import sys\n"
    "class _Block:\n"
    "    def find_spec(self, name, path=None, target=None):\n"
    "        if name.split('.')[0] in {'mcp', 'mcp_types', 'starlette', 'uvicorn'}:\n"
    "            raise ImportError(f'{name} is blocked: the CLI install has no MCP stack')\n"
    "sys.meta_path.insert(0, _Block())\n"
)


@pytest.mark.integration
def test_everything_but_the_server_imports_without_the_mcp_stack():
    """Every module except the two MCP-facing ones imports with mcp/starlette/uvicorn absent."""
    code = (
        _BLOCK_MCP
        + "import importlib, pkgutil, sanzaru\n"
        + "skip = {'sanzaru.server', 'sanzaru.media_resources'}\n"
        + "for info in pkgutil.walk_packages(sanzaru.__path__, 'sanzaru.'):\n"
        + "    if info.name in skip or '.app.' in info.name:\n"
        + "        continue\n"
        + "    importlib.import_module(info.name)\n"
        + "import sanzaru.cli\n"
        + "sanzaru.cli.cli.main(['--help'], standalone_mode=False)\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


@pytest.mark.integration
@pytest.mark.parametrize("argv", [[], ["serve"]])
def test_server_without_the_mcp_extra_exits_3_with_the_install_command(argv):
    code = (
        "import sys; sys.modules['mcp'] = None\n"  # find_spec treats a None entry as not installed
        "import sanzaru.cli\n"
        f"sanzaru.cli.cli.main({argv!r}, standalone_mode=False)\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 3, proc.stderr
    assert "sanzaru[mcp]" in proc.stderr
    assert "Traceback" not in proc.stderr


# ==================== EXCEPTION GROUP UNWRAPPING ====================


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


@pytest.mark.unit
def test_classify_reports_a_required_user_context_as_config_not_internal():
    """`SANZARU_REQUIRE_USER_CONTEXT=1` with no identity is a deployment mismatch — the CLI
    never carries one — so it belongs beside a missing API key (exit 3), not "internal" (exit 1),
    which is where the original PermissionError landed."""
    from sanzaru.cli._runtime import _classify
    from sanzaru.user_context import UserContextRequiredError

    error = _classify(
        UserContextRequiredError("SANZARU_REQUIRE_USER_CONTEXT is set but this request carries no user identity")
    )

    assert error.error_type == "config"
    assert error.exit_code == 3
    assert "no user identity" in str(error)


@pytest.mark.unit
def test_classify_reports_a_garbage_require_user_context_value_as_config(monkeypatch):
    from sanzaru.cli._runtime import _classify
    from sanzaru.storage.databricks import REQUIRE_USER_CONTEXT_ENV, require_user_context

    monkeypatch.setenv(REQUIRE_USER_CONTEXT_ENV, "enabled")
    with pytest.raises(RuntimeError) as excinfo:
        require_user_context()

    error = _classify(excinfo.value)

    assert error.error_type == "config"
    assert error.exit_code == 3
    assert REQUIRE_USER_CONTEXT_ENV in str(error)
