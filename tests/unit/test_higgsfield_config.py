"""The Higgsfield client seam in config.py, and the feature flags that gate video tools."""

import os

import pytest

from sanzaru import config, features
from sanzaru.higgsfield.client import HiggsfieldClient

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_client(mocker):
    mocker.patch.object(config, "_higgsfield_override", None)
    mocker.patch.object(config, "_higgsfield_cached", None)


class TestGetHiggsfieldClient:
    def test_missing_key_raises(self, mocker):
        mocker.patch.dict(os.environ, {}, clear=True)
        with pytest.raises(RuntimeError, match="HF_KEY is not set"):
            config.get_higgsfield_client()

    @pytest.mark.parametrize("value", ["no-colon", ":secret", "kid:", "  :  "])
    def test_malformed_key_raises(self, mocker, value):
        mocker.patch.dict(os.environ, {"HF_KEY": value}, clear=True)
        with pytest.raises(RuntimeError, match="key_id:key_secret"):
            config.get_higgsfield_client()

    def test_splits_on_the_first_colon(self, mocker):
        """A secret may itself contain ':'; only the first colon separates the id."""
        mocker.patch.dict(os.environ, {"HF_KEY": "kid:sec:ret"}, clear=True)
        client = config.get_higgsfield_client()
        assert client._api.headers["Authorization"] == "Key kid:sec:ret"

    def test_client_is_cached_across_calls(self, mocker):
        mocker.patch.dict(os.environ, {"HF_KEY": "kid:secret"}, clear=True)
        assert config.get_higgsfield_client() is config.get_higgsfield_client()

    def test_override_wins_and_skips_construction(self, mocker):
        mocker.patch.dict(os.environ, {}, clear=True)
        sentinel = HiggsfieldClient("a", "b")
        config.set_higgsfield_client(sentinel)
        try:
            assert config.get_higgsfield_client() is sentinel
        finally:
            config.set_higgsfield_client(None)

    def test_set_clears_the_cache(self, mocker):
        mocker.patch.dict(os.environ, {"HF_KEY": "kid:secret"}, clear=True)
        first = config.get_higgsfield_client()
        config.set_higgsfield_client(None)
        assert config.get_higgsfield_client() is not first


class TestHiggsfieldBaseUrl:
    def test_unset_or_blank_uses_the_production_api(self, mocker):
        for env in ({}, {"HIGGSFIELD_BASE_URL": "   "}):
            mocker.patch.dict(os.environ, env, clear=True)
            assert config.higgsfield_base_url() == "https://api.higgsfield.ai"

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("https://proxy.example.com/", "https://proxy.example.com"),
            ("https://proxy.example.com/px/higgsfield", "https://proxy.example.com/px/higgsfield"),
            ("http://127.0.0.1:8103", "http://127.0.0.1:8103"),
            ("  http://localhost:8103/  ", "http://localhost:8103"),
            ("http://[::1]:8103", "http://[::1]:8103"),
        ],
    )
    def test_accepts_https_and_loopback_http(self, mocker, value, expected):
        mocker.patch.dict(os.environ, {"HIGGSFIELD_BASE_URL": value}, clear=True)
        assert config.higgsfield_base_url() == expected

    @pytest.mark.parametrize(
        "value",
        [
            "http://proxy.example.com",  # plaintext to a remote host
            "http://127.0.0.1.attacker.example:8103",
            "ftp://127.0.0.1",
            "https://user:pass@proxy.example.com",
            "https://proxy.example.com/?next=x",
            "https://proxy.example.com/#frag",
            "https://proxy.example.com:notaport",
            "127.0.0.1:8103",  # no scheme
        ],
    )
    def test_rejects_anything_else(self, mocker, value):
        mocker.patch.dict(os.environ, {"HIGGSFIELD_BASE_URL": value}, clear=True)
        with pytest.raises(RuntimeError, match="HIGGSFIELD_BASE_URL must be"):
            config.higgsfield_base_url()

    def test_client_sends_the_credential_to_the_override(self, mocker):
        mocker.patch.dict(
            os.environ, {"HF_KEY": "kid:secret", "HIGGSFIELD_BASE_URL": "http://127.0.0.1:8103"}, clear=True
        )
        client = config.get_higgsfield_client()
        assert str(client._api.base_url) == "http://127.0.0.1:8103"
        assert client._api.headers["Authorization"] == "Key kid:secret"
        # The credential-free client has no base URL: uploads and downloads use absolute URLs.
        assert str(client._bare.base_url) == ""

    def test_a_bad_override_fails_before_a_client_exists(self, mocker):
        mocker.patch.dict(
            os.environ, {"HF_KEY": "kid:secret", "HIGGSFIELD_BASE_URL": "http://evil.example"}, clear=True
        )
        with pytest.raises(RuntimeError, match="HIGGSFIELD_BASE_URL must be"):
            config.get_higgsfield_client()
        assert config._higgsfield_cached is None


@pytest.mark.anyio
class TestCloseHiggsfieldClient:
    async def test_closes_and_forgets_the_cached_client(self, mocker):
        mocker.patch.dict(os.environ, {"HF_KEY": "kid:secret"}, clear=True)
        client = config.get_higgsfield_client()
        await config.close_higgsfield_client()
        assert config._higgsfield_cached is None
        assert client._api.is_closed and client._bare.is_closed

    async def test_noop_when_never_built(self):
        await config.close_higgsfield_client()

    async def test_a_failing_close_never_raises(self, mocker):
        broken = mocker.MagicMock()
        broken.aclose = mocker.AsyncMock(side_effect=RuntimeError("boom"))
        mocker.patch.object(config, "_higgsfield_cached", broken)
        await config.close_higgsfield_client()
        assert config._higgsfield_cached is None


class TestFeatureGating:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [(None, False), ("", False), ("no-colon", False), (":s", False), ("k:", False), ("k:s", True)],
    )
    def test_higgsfield_available_needs_a_well_formed_key(self, mocker, value, expected):
        env = {} if value is None else {"HF_KEY": value}
        mocker.patch.dict(os.environ, env, clear=True)
        assert features.check_higgsfield_available() is expected
        assert features.get_video_providers() == {"higgsfield": expected}

    @pytest.mark.parametrize(
        ("path", "key", "expected"), [(True, True, True), (True, False, False), (False, True, False)]
    )
    def test_generation_needs_both_a_path_and_a_key(self, mocker, path, key, expected):
        mocker.patch.object(features, "check_video_available", return_value=path)
        mocker.patch.dict(os.environ, {"HF_KEY": "k:s"} if key else {}, clear=True)
        assert features.check_video_generation_available() is expected
