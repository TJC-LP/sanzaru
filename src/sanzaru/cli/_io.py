# SPDX-License-Identifier: MIT
"""Path and content resolution for CLI commands.

The tool layer only accepts sandboxed bare filenames resolved against the
configured media directories. The CLI relaxes that for agents that bring
their own sandbox: path-form inputs and `-o` outputs are mapped onto
``LocalStorageBackend(path_overrides=...)`` via the factory override seam,
with the tool still receiving a plain basename (which
``validate_safe_path`` continues to sanitize).

Rules:
- Inputs given as paths (they contain a separator) override their path type's
  directory to the file's parent. A bare filename keeps default-backend
  resolution (media dir — or Databricks volume) unless a file of that name
  sits in cwd, which still captures it — but out loud, and never through a
  symlink (see `_check_bare_name_capture`).
- Outputs: ``-o`` file → write into its parent under its basename; ``-o``
  dir → auto-generated name inside. When the same path type is already
  claimed by an input from a different directory, the file is written next
  to the input under a temp name and moved to the target afterwards; when a
  bare-filename input pinned the type to the default backend, the artifact
  is written to the media library and then copied to the ``-o`` target.
- No ``-o`` and no configured media dir → cwd fallback with a stderr note
  (an agent-first tool never hard-fails when a usable default exists).
- Nothing here is ever written *through* a symlink. ``-o`` deliberately steps
  outside the media sandbox, so the tool layer's ``check_not_symlink`` never
  sees these paths, and every CLI-side relocation is a truncating write with
  the operator's privileges. A link at a final *file* path is refused; a
  ``-o`` directory that is itself a symlink is followed and resolved, because
  a directory is a parent of what gets written, not the write target (``-o
  /tmp`` must work on macOS, where /tmp is a link). The direct-write case
  (no relocation) is checked at plan time, because the write itself happens
  down in the storage backend.
"""

from __future__ import annotations

import errno
import os
import pathlib
import secrets
import shutil
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ._output import EXIT_USAGE, note
from ._runtime import CLIError

if TYPE_CHECKING:
    # Runtime import would chain storage → local → config → openai, defeating
    # the lightweight `sanzaru --help` path (guarded by a test).
    from ..storage.protocol import PathType


def read_content_arg(value: str, arg_name: str) -> str:
    """Resolve a long-content positional: inline string | ``@path`` | ``-`` (stdin).

    A literal leading ``@`` is escapable as ``@@``.
    """
    if value == "-":
        return sys.stdin.read()
    if value.startswith("@@"):
        return value[1:]
    if value.startswith("@"):
        path = pathlib.Path(value[1:]).expanduser()
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise CLIError("usage", f"{arg_name}: file not found: {path}", exit_code=EXIT_USAGE) from None
        except OSError as exc:
            raise CLIError("usage", f"{arg_name}: cannot read {path}: {exc}", exit_code=EXIT_USAGE) from None
    return value


# O_NOFOLLOW is POSIX; Windows has no symlink-following open to refuse.
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
# FreeBSD answers O_NOFOLLOW on a link with EMLINK where Linux/macOS say ELOOP.
_NOFOLLOW_ERRNOS = (errno.ELOOP, errno.EMLINK)


def _refuse_symlink(path: pathlib.Path, what: str) -> None:
    """Refuse a symlink at a path the CLI is about to write to or move.

    ``path.is_symlink()`` on its own — never ``path.exists() and
    path.is_symlink()``, which follows the link first and so misses the
    cheapest plant of all: a *dangling* symlink, whose target the write would
    then create.

    Only the leaf is checked. A symlinked parent directory is an ordinary thing
    for an operator to set up (``~/media`` pointing at another disk), and the
    caller named it; the leaf is the component the artifact is written through.
    """
    if path.is_symlink():
        raise CLIError(
            "usage",
            f"{what}: {path} is a symbolic link — refusing to write through it. "
            f"Pass the path it points at if that is what you meant.",
            exit_code=EXIT_USAGE,
        )


def _open_no_follow(path: pathlib.Path, flags: int, what: str) -> int:
    """Open `path` with the kernel, not a prior check, enforcing "not a symlink".

    The ``_refuse_symlink`` above is what produces a readable error; this is
    what still holds if the link appears between that check and this open.
    """
    try:
        return os.open(path, flags | _NOFOLLOW, 0o644)
    except OSError as exc:
        if exc.errno in _NOFOLLOW_ERRNOS:
            raise CLIError(
                "usage",
                f"{what}: {path} is a symbolic link (it appeared after the check) — refused",
                exit_code=EXIT_USAGE,
            ) from exc
        raise


def write_output_bytes(path: pathlib.Path, data: bytes) -> None:
    """Write an artifact to a final, caller-chosen path, never through a link."""
    _refuse_symlink(path, "-o target")
    with os.fdopen(_open_no_follow(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, "-o target"), "wb") as handle:
        handle.write(data)


def _relocate(source: pathlib.Path, target: pathlib.Path) -> None:
    """Move a staged artifact onto its final path without following symlinks.

    ``shutil.move`` — what this replaces — is unsafe at both ends: across
    devices it degrades to ``copy2``, which opens an existing destination and
    writes straight through a link planted there, and a symlinked *source*
    moves as the link rather than as the artifact. ``os.rename`` replaces a
    symlink at the destination instead of following it, which is exactly the
    semantics wanted, so the cross-device copy is the only part left to do by
    hand.
    """
    _refuse_symlink(source, "staged artifact")
    _refuse_symlink(target, "-o target")
    try:
        os.replace(source, target)
        return
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise

    read_fd = _open_no_follow(source, os.O_RDONLY, "staged artifact")
    with os.fdopen(read_fd, "rb") as src:
        write_fd = _open_no_follow(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, "-o target")
        with os.fdopen(write_fd, "wb") as dst:
            shutil.copyfileobj(src, dst)
    shutil.copystat(source, target)
    source.unlink()


@dataclass
class PathSession:
    """Accumulates per-path-type directory overrides for one invocation."""

    overrides: dict[str, pathlib.Path] = field(default_factory=dict)
    # Path types that must keep default-backend resolution (bare-filename inputs).
    default_locked: set[str] = field(default_factory=set)
    # (path_type, basename) → the directory that one input actually lives in.
    # Lets a batch span directories while every file is still validated
    # individually under its own parent (#38). `overrides` keeps the *first*
    # input's directory, which is what the output side anchors to.
    file_overrides: dict[tuple[str, str], pathlib.Path] = field(default_factory=dict)


@dataclass
class OutputPlan:
    """Where the artifact gets written, and where it must end up."""

    path_type: PathType
    filename: str | None  # basename handed to the tool (None → tool auto-generates)
    # Post-write relocation (set only when the write location differs from the target):
    final_dir: pathlib.Path | None = None
    final_name: str | None = None
    via_default_backend: bool = False  # finalize copies bytes out of the default backend


def _looks_like_path(value: str) -> bool:
    """True for inputs *written* as a path. Existence is deliberately not consulted.

    This used to also answer True when a file of that name happened to exist
    relative to cwd, which made routing a function of the working directory's
    contents. Agents run sanzaru inside workspaces full of material they did
    not write, so a planted ``episode.mp3`` silently stood in for the
    operator's media-library file of the same name — and since the envelope
    reports bare names, nothing in the output said so.

    Cwd capture is still honored (chained commands rely on it), but it is now
    a separate, announced decision: see `_check_bare_name_capture`.
    """
    return "/" in value or "\\" in value


def _media_library_twin(path_type: PathType, name: str) -> pathlib.Path | None:
    """The same basename in the configured media dir, if there is one.

    Local backends only: asking a remote default backend (Databricks) would
    cost a network round trip per input to decorate a warning.
    """
    if os.getenv("STORAGE_BACKEND", "local").strip().lower() != "local":
        return None

    from ..config import get_path, is_path_configured

    if not is_path_configured(path_type):
        return None
    try:
        twin = get_path(path_type) / name
    except (RuntimeError, OSError):
        # A misconfigured media dir is not this function's error to raise; the
        # bare name would have failed on its own further down.
        return None
    return twin if twin.is_file() else None


def _check_bare_name_capture(value: str, local: pathlib.Path, path_type: PathType, arg_name: str) -> None:
    """Guard a bare filename that a file in cwd has captured.

    Two ways that capture goes wrong, and neither used to make a sound:

    - the cwd entry is a symlink, so a bare name silently reads somewhere else
      entirely. Refused — a bare name has to mean one deterministic file, and
      writing ``./name`` says "the local one, links and all" on purpose.
    - a file of the same name also sits in the media library, so which one the
      command read depends on where it was run from. Both candidates are named
      on stderr; cwd still wins, because that is the behavior chained commands
      were built on.

    Not silenced by ``--quiet``: this is not progress, it is the CLI saying two
    different files answer to the name that was typed.
    """
    if local.is_symlink():
        raise CLIError(
            "usage",
            f"{arg_name}: {value!r} in the current directory is a symbolic link "
            f"(-> {os.readlink(local)}) — refusing to resolve a bare filename through it. "
            f"Write ./{value} to read the link's target deliberately.",
            exit_code=EXIT_USAGE,
        )
    twin = _media_library_twin(path_type, local.name)
    if twin is not None:
        note(
            f"warning: {arg_name} {value!r} names two files — reading {local.resolve()} "
            f"(current directory), NOT {twin} (media library). Pass a path to choose."
        )


def resolve_input(session: PathSession, value: str, path_type: PathType, arg_name: str) -> str:
    """Resolve one input argument to the bare filename the tool layer expects."""
    local = pathlib.Path(value).expanduser()
    bare = not _looks_like_path(value)
    # exists() follows links here on purpose: a *dangling* cwd symlink captures
    # nothing readable, so the media library is still the right answer for it.
    if bare and not local.exists():
        session.default_locked.add(path_type)
        if path_type in session.overrides:
            raise CLIError(
                "usage",
                f"{arg_name}: cannot mix bare media-dir filenames and explicit paths for the same media type",
                exit_code=EXIT_USAGE,
            )
        return value
    if bare:
        _check_bare_name_capture(value, local, path_type, arg_name)

    path = local.resolve()
    if not path.is_file():
        raise CLIError("usage", f"{arg_name}: input file not found: {path}", exit_code=EXIT_USAGE)
    if path_type in session.default_locked:
        raise CLIError(
            "usage",
            f"{arg_name}: cannot mix bare media-dir filenames and explicit paths for the same media type",
            exit_code=EXIT_USAGE,
        )
    # Inputs may span directories: each is anchored to its own parent and
    # validated there (#38). What cannot be resolved is two *different* files
    # with the same basename — the tool layer is handed bare names and the
    # envelope reports bare names, so one of them would be unaddressable.
    claimed = session.file_overrides.get((path_type, path.name))
    if claimed is not None and claimed != path.parent:
        raise CLIError(
            "usage",
            f"{arg_name}: two different {path_type} inputs are both named {path.name!r} "
            f"({claimed} and {path.parent}) — rename one, they are reported by basename",
            exit_code=EXIT_USAGE,
        )
    session.file_overrides[(path_type, path.name)] = path.parent
    # The first input's directory anchors the output side, which still works in
    # one directory per type.
    session.overrides.setdefault(path_type, path.parent)
    return path.name


def plan_output(session: PathSession, output: str | None, path_type: PathType, quiet: bool = False) -> OutputPlan:
    """Decide where the tool writes and whether a post-write move/copy is needed."""
    from ..config import is_path_configured

    if output is None:
        if path_type in session.overrides or path_type in session.default_locked:
            # Land next to the inputs / in the media library respectively.
            return OutputPlan(path_type=path_type, filename=None)
        if is_path_configured(path_type):
            return OutputPlan(path_type=path_type, filename=None)
        cwd = pathlib.Path.cwd()
        session.overrides[path_type] = cwd
        if not quiet:
            note(f"no media dir configured; writing to {cwd}")
        return OutputPlan(path_type=path_type, filename=None)

    raw = output
    target = pathlib.Path(raw).expanduser()
    # A symlink here is decided by what it points at, before is_dir() or
    # resolve() can follow it silently:
    #
    # - A link to a *directory* is followed, resolved out loud to the real
    #   path. A directory target is a parent of what gets written, not the
    #   write target itself, and `_refuse_symlink`'s own rule is that symlinked
    #   parents are an ordinary thing an operator sets up — refusing them broke
    #   `-o /tmp` on macOS, where /tmp is a symlink to /private/tmp.
    # - A link to a *file*, or a dangling one, stays refused: in the
    #   direct-write case the tool's write happens down in the storage backend,
    #   so plan time is the CLI's only chance to keep the artifact from going
    #   through the link (the fix is always just naming the real path).
    if target.is_symlink():
        if target.is_dir():  # follows the link: True only for a real directory
            target = target.resolve()
        else:
            _refuse_symlink(target, "-o")
    is_dir_target = raw.endswith(("/", "\\")) or target.is_dir()
    if is_dir_target:
        target_dir = target.resolve()
        target_name: str | None = None
    else:
        target_dir = target.parent.resolve()
        target_name = target.name
    target_dir.mkdir(parents=True, exist_ok=True)

    if path_type in session.default_locked:
        # Inputs pinned this type to the default backend (possibly remote):
        # write into the media library, then copy bytes to the target.
        return OutputPlan(
            path_type=path_type,
            filename=target_name,
            final_dir=target_dir,
            final_name=target_name,
            via_default_backend=True,
        )

    existing = session.overrides.get(path_type)
    if existing is not None and existing != target_dir:
        # Inputs claimed a different directory for this type: write next to
        # them under a collision-proof temp name, move to the target after.
        tmp_name = None
        if target_name is not None:
            # Unguessable, where this used to be `{stem}__sanzaru_tmp{suffix}`.
            # The name lands in a directory chosen by the *inputs*, which in an
            # agent workspace is shared with untrusted content: a predictable
            # one lets anything that can write there pre-plant a symlink for
            # the tool's write to follow, or swap the file between the write
            # and the move. Still a bare name with the real suffix — the tool
            # layer validates basenames and sniffs format from the extension.
            tmp_name = f"sanzaru_tmp_{secrets.token_hex(8)}{pathlib.Path(target_name).suffix}"
        return OutputPlan(path_type=path_type, filename=tmp_name, final_dir=target_dir, final_name=target_name)

    session.overrides[path_type] = target_dir
    return OutputPlan(path_type=path_type, filename=target_name)


def install_overrides(session: PathSession) -> None:
    """Activate the accumulated overrides for this invocation (if any)."""
    if session.overrides:
        from ..storage import set_storage_backend
        from ..storage.local import LocalStorageBackend

        set_storage_backend(
            LocalStorageBackend(
                path_overrides=dict(session.overrides),
                file_overrides=dict(session.file_overrides),
            )
        )


async def finalize_output(session: PathSession, plan: OutputPlan, written_filename: str) -> str:
    """Relocate the written artifact if needed; return its final absolute path/URI."""
    import anyio

    from ..storage import get_storage

    storage = get_storage()

    if plan.via_default_backend:
        # Copy bytes out of the (possibly remote) default backend to the target.
        # Buffers the full artifact in memory — same caveat as the Databricks
        # write_stream limitation documented in CLAUDE.md.
        assert plan.final_dir is not None
        data = await storage.read(plan.path_type, written_filename)
        final = plan.final_dir / (plan.final_name or written_filename)
        await anyio.to_thread.run_sync(write_output_bytes, final, data)
        return str(final)

    if plan.final_dir is not None:
        # A cross-device relocation degrades to a full copy — keep it off the loop.
        source_dir = session.overrides[plan.path_type]
        final = plan.final_dir / (plan.final_name or written_filename)
        await anyio.to_thread.run_sync(_relocate, source_dir / written_filename, final)
        return str(final)

    return storage.resolve_display_path(plan.path_type, written_filename)


#: Every spelling the tool layer uses for "the file I wrote". Each one is the
#: name as of the *write*, which `finalize_output` may since have changed.
_OUTPUT_NAME_KEYS = ("output_file", "output_filename", "filename")


def reconcile_output_name(payload: dict[str, object], final_path: str) -> None:
    """Point the envelope's name fields at the file that actually exists.

    Two ways the tool layer's name goes stale between the write and the
    envelope, both ending in a caller that cannot find the artifact (#54):

    - the tool auto-named the file because it takes no filename parameter, and
      `finalize_output` renamed it to what `-o` asked for;
    - `plan_output` handed down a `sanzaru_tmp_*` staging name, because the
      output directory differs from the one the inputs pinned, and
      `finalize_output` moved it to the real name afterwards.

    `file.path` was always right; this makes the obvious field agree with it
    rather than leaving the caller to pick between two answers.
    """
    for key in _OUTPUT_NAME_KEYS:
        if key in payload:
            payload[key] = pathlib.PurePath(final_path).name
