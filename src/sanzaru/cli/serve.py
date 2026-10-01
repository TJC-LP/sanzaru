# SPDX-License-Identifier: MIT
"""`sanzaru serve` — the MCP server, as an explicit subcommand."""

from __future__ import annotations

import importlib.util
import sys

import click

from ._output import EXIT_CONFIG

MCP_INSTALL_HINT = "uv tool install 'sanzaru[mcp]'  (or: uvx 'sanzaru[mcp]', pip install 'sanzaru[mcp]')"


def run_mcp_server(transport: str, host: str, port: int) -> None:
    """Start the MCP server, or exit 3 saying how to install it.

    The base install is the agent CLI and carries no MCP dependencies. Presence is
    checked with `find_spec` rather than by catching `ImportError` around the
    import: a broken `mcp` (the 0.8.0 failure) must surface as itself, not be
    relabelled "not installed".
    """
    if importlib.util.find_spec("mcp") is None:
        click.echo(f"sanzaru: the MCP server is not installed. Install it with:\n  {MCP_INSTALL_HINT}", err=True)
        sys.exit(EXIT_CONFIG)
    from ..server import run_server  # lazy: keeps FastMCP off the CLI import path

    run_server(transport="http" if transport == "http" else "stdio", host=host, port=port)


@click.command()
@click.option(
    "--transport",
    type=click.Choice(["stdio", "http"]),
    default="stdio",
    show_default=True,
    help="Transport type",
)
@click.option("--host", default="127.0.0.1", show_default=True, help="Host to bind for HTTP transport")
@click.option("--port", type=int, default=8000, show_default=True, help="Port to bind for HTTP transport")
def serve(transport: str, host: str, port: int) -> None:
    """Run the sanzaru MCP server (stdio for MCP clients, http for web).

    Equivalent to invoking bare `sanzaru` with the same flags; this explicit
    form exists so configs can be unambiguous about wanting the server. Requires
    the `mcp` extra (`sanzaru[mcp]`); the base install is CLI-only.
    """
    run_mcp_server(transport, host, port)
