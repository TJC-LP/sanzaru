# SPDX-License-Identifier: MIT
"""Loading a `.env` without letting the working directory redirect credentials.

Both entrypoints (`cli:main` and `server:main`) autoload a `.env` for local
development. python-dotenv's default discovery walks *upward* from the calling
module to the filesystem root, which under the documented in-project install
(`uv add sanzaru`, venv inside the workspace) reaches the workspace root — a
directory whose contents routinely come from somewhere else: a cloned repo, an
extracted archive, an earlier agent step.

That mattered because of what a `.env` is allowed to set. `AsyncOpenAI` reads
`OPENAI_BASE_URL` from the environment itself, so a planted file containing one
line was enough to point every request — carrying the operator's real
`OPENAI_API_KEY` as a bearer token — at someone else's server, and to make every
response the agent pipeline trusts attacker-controlled (CWE-427). `override=False`
was no defense: operators export their key, essentially nobody exports a base URL,
so the variable was unset and the file filled it.

Two changes close that without giving up the feature:

1. **No ancestor walk.** Only `./.env` is read — the directory the operator chose
   to run in, not every directory above it.
2. **Only sanzaru's own documented configuration loads.** Keys and media paths
   still work, because that is what `setup.sh` writes; anything that decides
   *where* a credential is sent, whether the HTTP transport authenticates at
   all, or *what a request costs* — the number `--max-cost` is enforced
   against — does not.

Point 2 is an allowlist rather than a list of banned names, because the first
attempt at this was a denylist of base-URL variables and it missed the shorter
route entirely: httpx runs with `trust_env=True`, so `HTTPS_PROXY` plus a
`SSL_CERT_FILE` pointing at an attacker CA machine-in-the-middles every request
to every provider without naming a single sanzaru variable. A denylist has to
anticipate every such pair; an allowlist only has to know what sanzaru itself
needs, and a variable nobody considered defaults to ignored.

Matching is exact and case-sensitive. python-dotenv keeps the spelling it finds,
and POSIX environments are case-sensitive, so a `openai_api_key=` line would pass
a case-folded check and then land under a name nothing reads. Treating it as
unknown puts it in the warning instead, where the operator can see why the key
did not take.

python-dotenv is a runtime dependency (pyproject `[project] dependencies`), so
this works in every install, not only under `uv sync`. This module deliberately
imports nothing heavy — `sanzaru --help` must not pay for openai or pydantic
(guarded by a test); python-dotenv is a few hundred lines of pure Python.
"""

from __future__ import annotations

import logging
import os
import pathlib

from dotenv import dotenv_values

logger = logging.getLogger("sanzaru")

#: Variables a `.env` may set. An allowlist, not a denylist, and that choice is
#: the whole point: the first version of this blocked the four application
#: base-URL variables and was still trivially bypassed, because the dangerous
#: set is much larger than it looks. `HTTPS_PROXY` plus `SSL_CERT_FILE` is a
#: complete machine-in-the-middle of every OpenAI, ElevenLabs and Databricks
#: request — httpx runs with `trust_env=True` and honours both — and it does not
#: touch a single "sanzaru" name. `SANZARU_ALLOW_UNAUTHENTICATED_HTTP` would
#: switch off the HTTP transport's startup refusal, and `SANZARU_RUN_SECRET`
#: would hand over the manifest signing key.
#:
#: Enumerating those was never going to converge. With an allowlist, a variable
#: nobody thought about defaults to *ignored*, which is the direction that fails
#: safe. Anything omitted here can still be exported in the real environment;
#: what a discovered file cannot do is introduce it.
#:
#: Two omissions look like oversights and are not:
#:
#: - `SANZARU_REALTIME_PRICE_*` (per-model price overrides). Pricing is not only
#:   a reporting concern: the same table is what `CostBudget.charge` adds up, so
#:   a planted `...=0,0,0,0,0,0` makes every turn cost $0.00 and `--max-cost`
#:   never trips. A price override is deployment configuration — export it,
#:   like the other keys that decide where money goes.
#: - `DATABRICKS_VIDEO_DIR` / `_IMAGE_DIR` / `_AUDIO_DIR`. They read as harmless
#:   subdirectory names, but the backend joins them into the volume path
#:   unsanitized, so `..` in one walks out of the per-user prefix into another
#:   tenant's files. Same rule: export them.
ALLOWED_ENV_KEYS = frozenset(
    {
        # Credentials the documented workflow puts here (setup.sh writes the first).
        "OPENAI_API_KEY",
        "ELEVENLABS_API_KEY",
        # Where media lives.
        "SANZARU_MEDIA_PATH",
        "VIDEO_PATH",
        "IMAGE_PATH",
        "AUDIO_PATH",
        # Storage backend selection and its credentials (never DATABRICKS_HOST:
        # that is where the credentials get sent).
        "STORAGE_BACKEND",
        "DATABRICKS_CLIENT_ID",
        "DATABRICKS_CLIENT_SECRET",
        "DATABRICKS_VOLUME_PATH",
        # Diagnostics and throughput tuning — no security dimension.
        "LOG_LEVEL",
        "SANZARU_ELEVENLABS_MAX_CONCURRENCY",
        "SANZARU_OPENAI_MAX_CONCURRENCY",
        "SANZARU_TRANSCRIBE_MAX_CONCURRENCY",
        "SANZARU_REALTIME_MAX_SESSIONS",
        "SANZARU_REALTIME_TURN_TIMEOUT",
        "SANZARU_REALTIME_ACT_BUDGET",
    }
)


def load_local_dotenv(directory: pathlib.Path | None = None) -> None:
    """Load `./.env` into the environment, skipping everything not allowlisted.

    Silent when there is no file, and a warning (never an error) when the file
    cannot be parsed — this is a development convenience and must never be the
    reason a command fails to start. Every key the allowlist drops is named in
    one warning, so a setting that "did not take" is diagnosable.
    """
    env_path = (directory or pathlib.Path.cwd()) / ".env"
    if not env_path.is_file():
        return

    try:
        values = dotenv_values(env_path)
    except Exception as exc:  # noqa: BLE001 - a malformed .env must not break startup
        logger.warning("Ignoring unreadable .env at %s: %s", env_path, exc)
        return

    ignored: list[str] = []
    for key, value in values.items():
        if value is None:
            continue
        if key not in ALLOWED_ENV_KEYS:
            ignored.append(key)
            continue
        # Same precedence python-dotenv's override=False gives: a variable the
        # operator actually exported outranks the file.
        os.environ.setdefault(key, value)

    if ignored:
        # repr-quoted: the names come from a file the threat model treats as
        # attacker-authored, and logging is the one stderr channel `note()`'s
        # scrub does not cover — a raw ESC in a key name would be live terminal
        # control here.
        logger.warning(
            "Ignored %s from %s: only sanzaru's documented configuration is read from a .env, "
            "because a file found on disk must not be able to redirect credentials, relax "
            "transport security, or set the prices a cost ceiling is enforced against. "
            "Export these in the environment if that is genuinely intended.",
            ", ".join(repr(k) for k in sorted(ignored)),
            env_path,
        )
