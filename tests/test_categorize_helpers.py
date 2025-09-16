"""Unit tests for helper utilities in :mod:`GptCategorize.categorize`."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import numpy as np
import pytest

from GptCategorize.categorize import (
    Chat,
    _detect_projectish,
    _first_user_prompt,
    cluster_text_cohesion,
    cosine_to_euclid_eps,
    extract_chats_from_json_blob,
    load_chats_from_conversations_json,
    temporal_cohesion,
    to_epoch,
)


def test_to_epoch_parses_various_inputs() -> None:
    """``to_epoch`` should gracefully parse diverse timestamp formats."""

    assert to_epoch(None) is None
    assert to_epoch(42) == pytest.approx(42.0)

    millis = 1_700_000_000_000
    assert to_epoch(millis) == pytest.approx(1_700_000_000.0)

    iso_text = "2024-01-01T12:34:56Z"
    expected = datetime(2024, 1, 1, 12, 34, 56, tzinfo=timezone.utc).timestamp()
    assert to_epoch(iso_text) == pytest.approx(expected)

    assert to_epoch("not-a-timestamp") is None


def test_detect_projectish_scans_nested_fields() -> None:
    """Project information is detected from direct and nested keys."""

    obj = {
        "metadata": {"folder_id": "folder-123"},
        "conversation": {"project": 7},
    }
    assert _detect_projectish(obj) == "folder-123"

    assert _detect_projectish({"conversation": {"project": 7}}) == "7"
    assert _detect_projectish({}) is None


def test_first_user_prompt_prefers_oldest_and_truncates() -> None:
    """The first user prompt should pick the earliest user message."""

    conversation = {
        "messages": [
            {
                "author": {"role": "assistant"},
                "content": {"parts": ["Assistant reply"]},
                "create_time": 150,
            },
            {
                "author": {"role": "user"},
                "content": {"parts": ["Later user message to ignore"]},
                "create_time": 200,
            },
        ],
        "mapping": {
            "node": {
                "message": {
                    "author": {"role": "user"},
                    "content": {
                        "parts": [
                            "Earliest user message with enough words to truncate",
                        ]
                    },
                    "create_time": "1970-01-01T00:00:50Z",
                }
            }
        },
    }

    result = _first_user_prompt(conversation, max_words=5)
    assert result == "Earliest user message with enough"


def test_extract_chats_from_json_blob_converts_nested_structures() -> None:
    """Conversations are detected whether direct or wrapped in another object."""

    data = [
        {
            "id": "chat-1",
            "title": "Direct Chat",
            "create_time": 1_700_000_000,
            "update_time": 1_700_000_360,
            "messages": [
                {
                    "author": {"role": "user"},
                    "content": {"parts": ["Hello direct chat prompt"]},
                }
            ],
            "metadata": {"workspace_id": "workspace-123"},
        },
        {
            "wrapper": {
                "title": "Nested Chat",
                "id": "chat-2",
                "messages": [
                    {
                        "author": {"role": "user"},
                        "content": {"parts": ["Nested user prompt"]},
                    }
                ],
                "created_at": "2024-01-02T00:00:00Z",
                "updated_at": "2024-01-02T01:00:00Z",
                "metadata": {"project": "proj-xyz"},
            }
        },
    ]

    chats = extract_chats_from_json_blob(data)
    assert {chat.id for chat in chats} == {"chat-1", "chat-2"}

    direct = next(chat for chat in chats if chat.id == "chat-1")
    assert direct.title == "Direct Chat"
    assert direct.project_id == "workspace-123"
    assert direct.prompt_excerpt == "Hello direct chat prompt"
    assert direct.create_time == pytest.approx(1_700_000_000.0)

    nested = next(chat for chat in chats if chat.id == "chat-2")
    assert nested.title == "Nested Chat"
    assert nested.project_id == "proj-xyz"
    assert nested.prompt_excerpt == "Nested user prompt"
    expected_ct = datetime(2024, 1, 2, tzinfo=timezone.utc).timestamp()
    assert nested.create_time == pytest.approx(expected_ct)


def test_load_chats_from_conversations_json_deduplicates_latest(tmp_path) -> None:
    """NDJSON inputs should be parsed and deduplicated by ``id``."""

    first = {
        "id": "chat-1",
        "title": "Old Title",
        "update_time": 1_700_000_000,
    }
    second = {
        "id": "chat-1",
        "title": "New Title",
        "update_time": 1_700_000_500,
    }

    path = tmp_path / "conversations.ndjson"
    path.write_text("\n".join(json.dumps(item) for item in (first, second)))

    chats = load_chats_from_conversations_json(str(path))

    assert len(chats) == 1
    chat = chats[0]
    assert chat.title == "New Title"
    assert chat.update_time == pytest.approx(1_700_000_500.0)


def test_cluster_text_cohesion_returns_average_similarity() -> None:
    """Cohesion is the average cosine similarity between cluster members."""

    vectors = np.array([[1.0, 1.0], [2.0, 2.0], [1.0, 0.0]])
    labels = np.array([0, 0, 1])

    assert cluster_text_cohesion(vectors, labels, 0) == pytest.approx(1.0)
    assert cluster_text_cohesion(vectors, labels, 1) == 0.0


def test_temporal_cohesion_computes_average_similarity() -> None:
    """Temporal cohesion averages the exponential kernel across pairs."""

    base = 1_700_000_000
    members = [
        Chat(id="a", title="A", create_time=base),
        Chat(id="b", title="B", create_time=base + 86_400),
        Chat(id="c", title="C", create_time=base + 2 * 86_400),
    ]

    expected = (2 * math.exp(-1 / 30) + math.exp(-2 / 30)) / 3
    assert temporal_cohesion(members, time_decay_days=30.0) == pytest.approx(expected)

    missing_times = [Chat(id="x", title="X"), Chat(id="y", title="Y")]
    assert temporal_cohesion(missing_times) == 0.5


def test_cosine_to_euclid_eps_handles_negative_values() -> None:
    """Cosine epsilons are clamped to non-negative values before conversion."""

    assert cosine_to_euclid_eps(-0.5) == 0.0
    assert cosine_to_euclid_eps(0.5) == pytest.approx(math.sqrt(1.0))

