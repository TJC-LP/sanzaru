# SPDX-License-Identifier: MIT
"""Unit tests for sanzaru.user_context."""

import pytest
from pydantic import ValidationError

from sanzaru.user_context import (
    UserContext,
    get_user_context,
    reset_user_context,
    set_user_context,
    user_slug,
)

# ------------------------------------------------------------------
# user_slug derivation
# ------------------------------------------------------------------


@pytest.mark.unit
class TestUserSlug:
    def test_simple_email(self):
        assert user_slug("user@example.com") == "user"

    def test_email_with_numbers(self):
        assert user_slug("rcaputo3@tjclp.com") == "rcaputo3"

    def test_dots_replaced(self):
        assert user_slug("jane.doe@example.com") == "jane_doe"

    def test_plus_replaced(self):
        assert user_slug("user+work@example.com") == "user_work"

    def test_mixed_special_chars(self):
        assert user_slug("Jane.Doe+work@example.com") == "jane_doe_work"

    def test_uppercase_lowered(self):
        assert user_slug("RCaputo3@TJCLP.COM") == "rcaputo3"

    def test_hyphens_replaced(self):
        assert user_slug("first-last@example.com") == "first_last"

    def test_consecutive_specials_collapsed(self):
        assert user_slug("a..b@example.com") == "a_b"

    def test_leading_trailing_specials_stripped(self):
        assert user_slug(".user.@example.com") == "user"

    def test_empty_local_part_raises(self):
        with pytest.raises(ValueError, match="Cannot derive user slug"):
            user_slug("@example.com")

    def test_all_special_chars_raises(self):
        with pytest.raises(ValueError, match="Cannot derive user slug"):
            user_slug("...@example.com")

    def test_underscores_preserved(self):
        assert user_slug("user_name@example.com") == "user_name"


# ------------------------------------------------------------------
# ContextVar get/set/reset
# ------------------------------------------------------------------


@pytest.mark.unit
class TestContextVar:
    def test_default_is_none(self):
        assert get_user_context() is None

    def test_set_and_get(self):
        ctx = UserContext(email="test@example.com")
        token = set_user_context(ctx)
        try:
            assert get_user_context() is ctx
            assert get_user_context().email == "test@example.com"
        finally:
            reset_user_context(token)

    def test_reset_restores_previous(self):
        assert get_user_context() is None
        token = set_user_context(UserContext(email="a@b.com"))
        try:
            assert get_user_context() is not None
        finally:
            reset_user_context(token)
        assert get_user_context() is None

    def test_set_none_clears(self):
        token1 = set_user_context(UserContext(email="a@b.com"))
        try:
            token2 = set_user_context(None)
            try:
                assert get_user_context() is None
            finally:
                reset_user_context(token2)
        finally:
            reset_user_context(token1)


# ------------------------------------------------------------------
# UserContext dataclass
# ------------------------------------------------------------------


@pytest.mark.unit
class TestUserContext:
    def test_frozen(self):
        ctx = UserContext(email="a@b.com")
        with pytest.raises(ValidationError):
            ctx.email = "c@d.com"  # type: ignore[misc]

    def test_equality(self):
        a = UserContext(email="a@b.com")
        b = UserContext(email="a@b.com")
        assert a == b

    def test_inequality(self):
        a = UserContext(email="a@b.com")
        b = UserContext(email="c@d.com")
        assert a != b

    def test_rejects_no_at_sign(self):
        with pytest.raises(ValidationError, match="Invalid email"):
            UserContext(email="notanemail")

    def test_rejects_empty_local_part(self):
        with pytest.raises(ValidationError, match="Invalid email"):
            UserContext(email="@example.com")

    def test_accepts_valid_email(self):
        ctx = UserContext(email="user@example.com")
        assert ctx.email == "user@example.com"
