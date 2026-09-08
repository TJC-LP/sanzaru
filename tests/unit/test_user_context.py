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


def _readable(email: str) -> str:
    """The human half of a slug — everything before the hash suffix."""
    return user_slug(email).rsplit("-", 1)[0]


@pytest.mark.unit
class TestUserSlug:
    """The readable half stays greppable; see TestUserSlugIsInjective for the rest."""

    def test_simple_email(self):
        assert _readable("user@example.com") == "user"

    def test_email_with_numbers(self):
        assert _readable("rcaputo3@tjclp.com") == "rcaputo3"

    def test_dots_replaced(self):
        assert _readable("jane.doe@example.com") == "jane_doe"

    def test_plus_replaced(self):
        assert _readable("user+work@example.com") == "user_work"

    def test_mixed_special_chars(self):
        assert _readable("Jane.Doe+work@example.com") == "jane_doe_work"

    def test_uppercase_lowered(self):
        assert _readable("RCaputo3@TJCLP.COM") == "rcaputo3"

    def test_hyphens_replaced(self):
        assert _readable("first-last@example.com") == "first_last"

    def test_consecutive_specials_collapsed(self):
        assert _readable("a..b@example.com") == "a_b"

    def test_leading_trailing_specials_stripped(self):
        assert _readable(".user.@example.com") == "user"

    @pytest.mark.parametrize("email", ["...@example.com", "+++@x.com", "张三@example.com", "@example.com"])
    def test_an_unreadable_local_part_still_yields_a_slug(self, email):
        """A local part with nothing in [a-z0-9_] must not be fatal.

        `UserContext` accepts these addresses, so raising here turned a
        legitimate user into a failure deep in the storage layer (and, over
        HTTP, a 500). The hash half alone is still injective, which is the
        property isolation actually rests on.
        """
        slug = user_slug(email)
        assert slug.startswith("user-")
        assert len(slug.rsplit("-", 1)[1]) >= 8

    def test_unreadable_local_parts_are_still_distinct_from_each_other(self):
        assert user_slug("...@example.com") != user_slug("+++@example.com")
        assert user_slug("张三@example.com") != user_slug("李四@example.com")

    def test_underscores_preserved(self):
        assert _readable("user_name@example.com") == "user_name"

    def test_hash_suffix_is_hex(self):
        suffix = user_slug("user@example.com").rsplit("-", 1)[1]
        assert len(suffix) >= 8
        assert set(suffix) <= set("0123456789abcdef")

    def test_slug_is_path_safe(self):
        """The slug becomes a Volumes path segment, so nothing may traverse or split it."""
        slug = user_slug("../../Jane.Doe+work@example.com")
        assert "/" not in slug
        assert ".." not in slug


@pytest.mark.unit
class TestUserSlugIsInjective:
    """Two distinct identities must never share a storage prefix (CWE-706)."""

    def test_punctuation_variants_differ(self):
        """The readable half folds ., - and _ together; the hash must not."""
        slugs = {
            user_slug("jane.doe@corp.com"),
            user_slug("jane-doe@corp.com"),
            user_slug("jane_doe@corp.com"),
            user_slug("Jane..Doe@corp.com"),
        }
        assert len(slugs) == 4

    def test_same_local_part_different_domain_differs(self):
        """The readable half drops the domain entirely — two companies, one bob."""
        assert user_slug("bob@company-a.com") != user_slug("bob@company-b.com")

    def test_domain_alone_distinguishes_identical_readable_halves(self):
        a = user_slug("jane.doe@corp.com")
        b = user_slug("jane.doe@other.com")
        assert a != b
        assert a.rsplit("-", 1)[0] == b.rsplit("-", 1)[0] == "jane_doe"

    def test_local_part_case_differs(self):
        """Local parts are case-sensitive per RFC 5321: err toward two prefixes."""
        assert user_slug("JaneDoe@corp.com") != user_slug("janedoe@corp.com")

    def test_domain_case_folded(self):
        """Domains are case-insensitive, so one identity keeps one prefix."""
        assert user_slug("jane@CORP.com") == user_slug("jane@corp.com")

    def test_stable_across_calls(self):
        """The slug is a storage path: the same identity must always resolve to it."""
        assert user_slug("jane.doe@corp.com") == user_slug("jane.doe@corp.com")


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
