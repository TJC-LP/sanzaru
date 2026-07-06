# SPDX-License-Identifier: MIT
"""sanzaru agent CLI.

Root command: with no subcommand, starts the MCP server exactly like the
historical entrypoint (existing .mcp.json / Claude Desktop configs keep
working byte-for-byte); with a subcommand, dispatches to the agent-facing
CLI (`sanzaru video create ...`, `sanzaru image generate ...`, ...).

Contract (details in docs/cli.md):
- stdout: exactly one JSON envelope per input (JSONL for fan-out) — nothing else
- stderr: progress, heartbeats, human-readable hints
- exit codes: 0 ok · 1 runtime/API · 2 usage · 3 config · 4 timeout (resumable)
  · 5 job failed · 6 partial batch · 130 interrupted
"""

from __future__ import annotations

import logging
import os

import click

from ._runtime import CLIState

WORKFLOW_HELP = """\
sanzaru: MCP server (no subcommand) or agent CLI (sanzaru <group> <verb>).

\b
Async job workflow (video, image create):
  create returns a job id in ~1s; wait/download are idempotent and resumable.
  One-shot: `create ... -o out.mp4` implies --download implies --wait.
  On exit 4 (timeout) the job keeps running — re-run the printed `resume`
  command to attach again.

\b
Output contract: stdout = one JSON envelope per input (JSONL for fan-out);
progress on stderr. Exit codes: 0 ok, 1 runtime/API, 2 usage, 3 config,
4 timeout (resumable), 5 job failed server-side, 6 partial batch, 130 SIGINT.
"""


@click.group(
    invoke_without_command=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    help=WORKFLOW_HELP,
)
@click.version_option(package_name="sanzaru", prog_name="sanzaru")
@click.option(
    "--transport",
    type=click.Choice(["stdio", "http"]),
    default="stdio",
    show_default=True,
    help="[server mode] Transport type",
)
@click.option("--host", default="127.0.0.1", show_default=True, help="[server mode] Host for HTTP transport")
@click.option("--port", type=int, default=8000, show_default=True, help="[server mode] Port for HTTP transport")
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Accepted for compatibility; stdout is always JSON.",
)
@click.option("-q", "--quiet", is_flag=True, help="Suppress stderr progress/log lines (errors still print).")
@click.option("-v", "--verbose", is_flag=True, help="Debug logging to stderr.")
@click.option(
    "--media-dir",
    type=click.Path(file_okay=False),
    default=None,
    help="Override SANZARU_MEDIA_PATH for this invocation.",
)
@click.pass_context
def cli(
    ctx: click.Context,
    transport: str,
    host: str,
    port: int,
    json_output: bool,
    quiet: bool,
    verbose: bool,
    media_dir: str | None,
) -> None:
    ctx.obj = CLIState(quiet=quiet, verbose=verbose)
    if verbose:
        os.environ["LOG_LEVEL"] = "DEBUG"
        logging.getLogger("sanzaru").setLevel(logging.DEBUG)
    elif quiet:
        logging.getLogger("sanzaru").setLevel(logging.ERROR)
    if media_dir:
        # Same semantics as exporting SANZARU_MEDIA_PATH (individual
        # VIDEO_PATH/IMAGE_PATH/AUDIO_PATH vars still take precedence).
        os.environ["SANZARU_MEDIA_PATH"] = media_dir
    if ctx.invoked_subcommand is None:
        from ..server import run_server  # lazy: keeps `sanzaru <cmd> --help` off the FastMCP import path

        run_server(transport="http" if transport == "http" else "stdio", host=host, port=port)


def _register_commands() -> None:
    from .serve import serve

    cli.add_command(serve)


_register_commands()


def main() -> None:
    # Optional .env for local development (parity with server.main()).
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    cli(prog_name="sanzaru")
