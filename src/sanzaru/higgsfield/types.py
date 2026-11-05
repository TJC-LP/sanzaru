# SPDX-License-Identifier: MIT
"""Shapes of Higgsfield API responses, parsed defensively (no pydantic).

Only the fields sanzaru reads are declared. `RequestStatus` covers every
terminal state; `video` is present only on `completed`.
"""

from __future__ import annotations

from typing import Literal, NotRequired

from typing_extensions import TypedDict

RequestState = Literal["queued", "in_progress", "completed", "failed", "nsfw", "canceled"]
TERMINAL_STATES: frozenset[str] = frozenset({"completed", "failed", "nsfw", "canceled"})


class MediaOutput(TypedDict):
    url: str


class SubmitResponse(TypedDict):
    status: RequestState
    request_id: str


class RequestStatus(TypedDict):
    status: RequestState
    request_id: str
    error: NotRequired[str | None]
    video: NotRequired[MediaOutput]


class EstimateDiscount(TypedDict):
    percentage: str
    credits: str
    usd: str


class EstimateResponse(TypedDict):
    """Either a numeric estimate or, for some models (Seedance 2.5, Wan 3.0), prose only."""

    type: Literal["estimate", "description"]
    usd: NotRequired[str]
    credits: NotRequired[str]
    discount: NotRequired[EstimateDiscount | None]
    pricing_description: NotRequired[str]


class CatalogModel(TypedDict):
    slug: str
    title: str
    output_type: str
    operation_type: list[str]
