# SPDX-License-Identifier: MIT
"""`sanzaru serve` — the MCP server, as an explicit subcommand."""

from __future__ import annotations

import click


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
    form exists so configs can be unambiguous about wanting the server.
    """
    from ..server import run_server  # lazy: keeps FastMCP off the CLI import path

    run_server(transport="http" if transport == "http" else "stdio", host=host, port=port)
