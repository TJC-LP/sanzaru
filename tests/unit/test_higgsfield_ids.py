"""Job ids and model slugs: the caller-supplied strings that reach a URL path."""

import uuid

import pytest

from sanzaru.higgsfield.ids import JOB_PREFIX, SORA_RETIRED, to_job_id, to_request_id, validate_slug

pytestmark = pytest.mark.unit

UUID = "d7e6c0f3-6699-4f6c-bb45-2ad7fd9158ff"


def test_round_trip():
    job = to_job_id(UUID)
    assert job == f"{JOB_PREFIX}{UUID}"
    assert to_request_id(job) == UUID


def test_job_ids_are_canonical_lower_case():
    assert to_job_id(UUID.upper()) == f"hf_{UUID}"
    assert to_request_id(f"hf_{UUID.upper()}") == UUID


def test_a_sora_id_is_explained_not_just_refused():
    with pytest.raises(ValueError) as excinfo:
        to_request_id("video_68d7512d07848190b3e45da0ecbebcde")
    assert str(excinfo.value) == SORA_RETIRED


@pytest.mark.parametrize("bad", ["hf_nope", "resp_abc", UUID, "hf_", f"hf_{UUID}x", "hf_../../models"])
def test_anything_but_hf_uuid_is_refused(bad):
    with pytest.raises(ValueError):
        to_request_id(bad)


def test_to_job_id_rejects_a_non_uuid():
    with pytest.raises(ValueError):
        to_job_id("not-a-uuid")


def test_to_request_id_returns_a_parseable_uuid():
    assert uuid.UUID(to_request_id(to_job_id(UUID))).version == 4


@pytest.mark.parametrize(
    "slug",
    [
        "bytedance/seedance-2.5/text-to-video",
        "kling-video/v3.0/std/image-to-video",
        "kling-video/v2.5-turbo/pro/text-to-video",
        "xai/grok-imagine-video/v1.5/reference-to-video",
    ],
)
def test_catalog_slugs_are_accepted(slug):
    assert validate_slug(slug) == slug


@pytest.mark.parametrize(
    "slug",
    [
        "requests/x/cancel",
        "requests/d7e6c0f3/status",
        "files/generate-upload-url",
        "estimate/bytedance/seedance-2.5/text-to-video",
        "models/x",
        "files/../x",
        "a/../b",
        "a//b",
        "../x",
        "/bytedance/seedance",
        "Bytedance/Seedance",
        "bytedance/%2e%2e/models",
        "bytedance/seedance?x=1",
        "single-segment",
        "",
    ],
)
def test_slugs_that_could_address_another_endpoint_are_refused(slug):
    with pytest.raises(ValueError):
        validate_slug(slug)
