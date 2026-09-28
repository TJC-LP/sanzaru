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
