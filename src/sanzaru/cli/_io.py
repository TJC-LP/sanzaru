# SPDX-License-Identifier: MIT
"""Path and content resolution for CLI commands.

The tool layer only accepts sandboxed bare filenames resolved against the
configured media directories. The CLI relaxes that for agents that bring
their own sandbox: path-form inputs and `-o` outputs are mapped onto
``LocalStorageBackend(path_overrides=...)`` via the factory override seam,
with the tool still receiving a plain basename (which
``validate_safe_path`` continues to sanitize).

Rules:
- Inputs given as paths (contain a separator, or exist relative to cwd)
  override their path type's directory to the file's parent; bare filenames
  keep default-backend resolution (media dir — or Databricks volume).
- Outputs: ``-o`` file → write into its parent under its basename; ``-o``
  dir → auto-generated name inside. When the same path type is already
  claimed by an input from a different directory, the file is written next
  to the input under a temp name and moved to the target afterwards; when a
  bare-filename input pinned the type to the default backend, the artifact
  is written to the media library and then copied to the ``-o`` target.
- No ``-o`` and no configured media dir → cwd fallback with a stderr note
  (an agent-first tool never hard-fails when a usable default exists).
"""

from __future__ import annotations

import pathlib
import shutil
import sys
from dataclasses import dataclass, field

from ..storage.protocol import PathType
from ._output import EXIT_USAGE, note
from ._runtime import CLIError


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


@dataclass
class PathSession:
    """Accumulates per-path-type directory overrides for one invocation."""

    overrides: dict[str, pathlib.Path] = field(default_factory=dict)
    # Path types that must keep default-backend resolution (bare-filename inputs).
    default_locked: set[str] = field(default_factory=set)


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
    return "/" in value or "\\" in value or pathlib.Path(value).expanduser().exists()


def resolve_input(session: PathSession, value: str, path_type: PathType, arg_name: str) -> str:
    """Resolve one input argument to the bare filename the tool layer expects."""
    if not _looks_like_path(value):
        session.default_locked.add(path_type)
        if path_type in session.overrides:
            raise CLIError(
                "usage",
                f"{arg_name}: cannot mix bare media-dir filenames and explicit paths for the same media type",
                exit_code=EXIT_USAGE,
            )
        return value

    path = pathlib.Path(value).expanduser().resolve()
    if not path.is_file():
        raise CLIError("usage", f"{arg_name}: input file not found: {path}", exit_code=EXIT_USAGE)
    if path_type in session.default_locked:
        raise CLIError(
            "usage",
            f"{arg_name}: cannot mix bare media-dir filenames and explicit paths for the same media type",
            exit_code=EXIT_USAGE,
        )
    existing = session.overrides.get(path_type)
    if existing is not None and existing != path.parent:
        raise CLIError(
            "usage",
            f"{arg_name}: all {path_type} inputs must share one directory (got {existing} and {path.parent})",
            exit_code=EXIT_USAGE,
        )
    session.overrides[path_type] = path.parent
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
            stem, suffix = pathlib.Path(target_name).stem, pathlib.Path(target_name).suffix
            tmp_name = f"{stem}__sanzaru_tmp{suffix}"
        return OutputPlan(path_type=path_type, filename=tmp_name, final_dir=target_dir, final_name=target_name)

    session.overrides[path_type] = target_dir
    return OutputPlan(path_type=path_type, filename=target_name)


def install_overrides(session: PathSession) -> None:
    """Activate the accumulated overrides for this invocation (if any)."""
    if session.overrides:
        from ..storage import set_storage_backend
        from ..storage.local import LocalStorageBackend

        set_storage_backend(LocalStorageBackend(path_overrides=dict(session.overrides)))


async def finalize_output(session: PathSession, plan: OutputPlan, written_filename: str) -> str:
    """Relocate the written artifact if needed; return its final absolute path/URI."""
    from ..storage import get_storage

    storage = get_storage()

    if plan.via_default_backend:
        # Copy bytes out of the (possibly remote) default backend to the target.
        assert plan.final_dir is not None
        data = await storage.read(plan.path_type, written_filename)
        final = plan.final_dir / (plan.final_name or written_filename)
        final.write_bytes(data)
        return str(final)

    if plan.final_dir is not None:
        source_dir = session.overrides[plan.path_type]
        final = plan.final_dir / (plan.final_name or written_filename)
        shutil.move(str(source_dir / written_filename), str(final))
        return str(final)

    return storage.resolve_display_path(plan.path_type, written_filename)
