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


def user_slug(email: str) -> str:
    """Derive a filesystem-safe slug from an email address.

    Takes the local part (before ``@``), lowercases it, and replaces
    any character that is not ``[a-z0-9_]`` with ``_``.

    Examples::

        >>> user_slug("RCaputo3@tjclp.com")
        'rcaputo3'
        >>> user_slug("Jane.Doe+work@example.com")
        'jane_doe_work'
        >>> user_slug("user@example.com")
        'user'
    """
    local = email.split("@")[0].lower()
    slug = _SLUG_RE.sub("_", local)
    # Collapse consecutive underscores and strip leading/trailing
    slug = re.sub(r"_+", "_", slug).strip("_")
    if not slug:
        raise ValueError(f"Cannot derive user slug from email: {email}")
    return slug
