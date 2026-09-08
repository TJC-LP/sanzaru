# SPDX-License-Identifier: MIT
"""Per-request user context for multi-tenant deployments.

When running behind a proxy that injects user identity headers (e.g.,
Databricks Apps with ``x-forwarded-email``), this module provides a way
to thread user identity through to the storage layer without modifying
every tool function signature.

Usage::

    from sanzaru.user_context import UserContext, get_user_context, set_user_context

    # In middleware (before request):
    token = set_user_context(UserContext(email="user@example.com"))

    # In storage backend (during request):
    ctx = get_user_context()  # UserContext or None

    # After request:
    reset_user_context(token)
"""

from __future__ import annotations

import contextvars
import hashlib
import re

from pydantic import BaseModel, field_validator


class UserContext(BaseModel, frozen=True):
    """Identity of the user making the current request.

    Email is validated on construction to ensure it contains an ``@``
    with a non-empty local part.
    """

    email: str

    @field_validator("email")
    @classmethod
    def _validate_email(cls, v: str) -> str:
        if "@" not in v or not v.split("@")[0]:
            raise ValueError(f"Invalid email: {v!r}")
        return v


_user_context: contextvars.ContextVar[UserContext | None] = contextvars.ContextVar("sanzaru_user_context", default=None)


def get_user_context() -> UserContext | None:
    """Return the current user context, or ``None`` in single-tenant mode."""
    return _user_context.get()


def set_user_context(ctx: UserContext | None) -> contextvars.Token[UserContext | None]:
    """Set the user context for the current async task.

    Returns a token that can be passed to :func:`reset_user_context`
    to restore the previous value.
    """
    return _user_context.set(ctx)


def reset_user_context(token: contextvars.Token[UserContext | None]) -> None:
    """Restore the user context to its previous value."""
    _user_context.reset(token)


# ------------------------------------------------------------------
# Slug derivation
# ------------------------------------------------------------------

_SLUG_RE = re.compile(r"[^a-z0-9_]")

# Length of the hash half. Eight hex chars puts an even-odds collision at
# roughly 65k identities (birthday bound); twelve pushes that past 16M. Four
# extra characters in a path segment nobody types by hand is the cheapest
# headroom available, and the cost of being wrong here is one tenant reading
# another's files.
_HASH_CHARS = 12

# The readable half is drawn from [a-z0-9_], so a hyphen can never occur in it
# and the two halves stay unambiguously separable by eye.
_SEPARATOR = "-"


def user_slug(email: str) -> str:
    """Derive a filesystem-safe, per-identity path segment from an email address.

    Returns ``<readable>-<hash>``: the lowercased local part with every
    character outside ``[a-z0-9_]`` folded to ``_``, then a truncated SHA-256
    of the whole address.

    The hash is what makes this safe to isolate tenants with; the readable half
    alone is not injective, and collapses distinct people onto one namespace in
    two directions. It drops the domain, so ``bob@company-a.com`` and
    ``bob@company-b.com`` are the same string. It folds case and punctuation, so
    ``john.doe@x.com``, ``john-doe@x.com``, ``john_doe@x.com`` and
    ``John..Doe@x.com`` are too. Whoever arrives second would transparently read
    and overwrite the first's files (CWE-706). Hashing the full address restores
    injectivity while the prefix stays greppable in a directory listing or a log.

    Only the domain is case-folded before hashing. Local parts are case-sensitive
    per RFC 5321 even though most providers ignore that, so this errs toward two
    prefixes for one person rather than one prefix for two people.

    .. warning::

        This value **is** a storage path. Changing the algorithm, the hash
        length, the separator or the normalisation re-homes every existing
        user: their files stay where they were written, under a prefix nothing
        resolves to any more.

    Examples::

        >>> user_slug("RCaputo3@tjclp.com")
        'rcaputo3-058695b50073'
        >>> user_slug("Jane.Doe+work@example.com")
        'jane_doe_work-d4b6dd61c9c8'
        >>> user_slug("user@example.com")
        'user-b4c9a289323b'
    """
    local, _, domain = email.rpartition("@")
    readable = _SLUG_RE.sub("_", local.lower())
    # Collapse consecutive underscores and strip leading/trailing
    readable = re.sub(r"_+", "_", readable).strip("_")
    digest = hashlib.sha256(f"{local}@{domain.lower()}".encode()).hexdigest()[:_HASH_CHARS]
    if not readable:
        # A local part with nothing in [a-z0-9_] — "张三@example.com",
        # "+++@x.com" — leaves the readable half empty. `UserContext` accepts
        # those addresses, so raising here turned a legitimate user into a hard
        # failure deep in the storage layer. The hash alone is still injective,
        # which is the property isolation actually depends on; only the
        # human-readable convenience is lost.
        return f"user{_SEPARATOR}{digest}"
    return f"{readable}{_SEPARATOR}{digest}"
