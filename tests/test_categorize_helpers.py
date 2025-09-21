"""Unit tests for helper utilities in :mod:`GptCategorize.categorize`."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import numpy as np
import pytest

import GptCategorize.categorize as categorize
from GptCategorize.categorize import (
    Chat,
    _detect_projectish,
    _conversation_excerpt,
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
    assert to_epoch({"unexpected": "shape"}) is None


def test_detect_projectish_scans_nested_fields() -> None:
    """Project information is detected from direct and nested keys."""

    obj = {
        "metadata": {"folder_id": "folder-123"},
        "conversation": {"project": 7},
    }
    assert _detect_projectish(obj) == "folder-123"

    assert _detect_projectish({"project_id": "direct"}) == "direct"
    assert _detect_projectish({"conversation": {"project": 7}}) == "7"
    assert _detect_projectish({}) is None


def test_conversation_excerpt_orders_and_truncates() -> None:
    """Conversation excerpts should order messages and respect ``max_words``."""

    conversation = {
        "messages": [
            {
                "author": {"role": "assistant"},
                "content": {"parts": ["Assistant reply offering help"]},
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

    result = _conversation_excerpt(conversation, max_words=12)
    # Earliest user message should be first, followed by assistant reply, both truncated
    assert result == "User: Earliest user message with enough words to truncate\n\nAssistant: Assistant reply"


def test_conversation_excerpt_handles_alternate_shapes() -> None:
    """Content stored under ``text`` or as a string should be considered."""

    conversation = {
        "messages": [
            {"author": {"role": "user"}, "content": {"text": "Dict text"}, "create_time": 200},
            {"author": {"role": "user"}, "content": "String message", "create_time": 100},
            {"author": {"role": "user"}, "content": {"text": ""}, "create_time": 50},
        ]
    }

    assert _conversation_excerpt(conversation) == "User: String message\n\nUser: Dict text"


def test_conversation_excerpt_skips_system_and_thinking_messages() -> None:
    """System, search, and thinking messages should be excluded from excerpts."""

    conversation = {
        "messages": [
            {
                "author": {"role": "system"},
                "content": {"parts": ["System greeting"]},
                "create_time": 10,
            },
            {
                "author": {"role": "assistant"},
                "content": {"parts": ["Visible assistant reply"]},
                "create_time": 20,
            },
            {
                "author": {"role": "assistant"},
                "content": {
                    "parts": [
                        {"content_type": "thinking", "text": "Hidden thinking"},
                        {"content_type": "text", "text": "Main answer"},
                    ]
                },
                "create_time": 30,
                "metadata": {"message_type": "assistant_response"},
            },
            {
                "author": {"role": "tool"},
                "content": {"parts": ["Tool output"]},
                "create_time": 40,
            },
            {
                "author": {"role": "assistant"},
                "content": {"parts": [{"content_type": "tether_browsing_result", "text": "Search"}]},
                "create_time": 50,
                "metadata": {"message_type": "web_search_result"},
            },
            {
                "author": {"role": "user"},
                "content": {"parts": ["User follow-up"]},
                "create_time": 60,
            },
        ]
    }

    result = _conversation_excerpt(conversation, max_words=0)
    assert result == "Assistant: Visible assistant reply\n\nAssistant: Main answer\n\nUser: User follow-up"


def test_first_user_prompt_alias_matches_conversation_excerpt() -> None:
    """The legacy helper should return the same value as the new excerpt helper."""

    conversation = {
        "messages": [
            {"author": {"role": "user"}, "content": {"parts": ["Hello there"]}},
            {"author": {"role": "assistant"}, "content": {"parts": ["Hi"]}},
        ]
    }

    assert categorize._first_user_prompt(conversation) == _conversation_excerpt(conversation)


def test_extract_text_parts_handles_various_inputs() -> None:
    """Edge shapes should either return text fragments or an empty list."""

    assert categorize._extract_text_parts(None) == []
    assert categorize._extract_text_parts(123) == []
    assert categorize._extract_text_parts([" first ", {"text": " second "}]) == ["first", "second"]
    assert categorize._extract_text_parts({"data": {"code": "print(1)", "text": "extra"}}) == ["print(1)", "extra"]


def test_message_text_if_allowed_filters_metadata_and_types() -> None:
    """Metadata flags and message types should prevent inclusion."""

    base_message = {
        "id": "msg-1",
        "author": {"role": "assistant"},
        "content": {"parts": [" Visible reply "]},
    }

    assert categorize._message_text_if_allowed(base_message) == "Assistant: Visible reply"

    thinking = dict(base_message)
    thinking["metadata"] = {"is_thinking": True}
    assert categorize._message_text_if_allowed(thinking) is None

    web_search = dict(base_message)
    web_search["metadata"] = {"message_type": "web_search_result"}
    assert categorize._message_text_if_allowed(web_search) is None

    system_type = dict(base_message)
    system_type["message_type"] = "system-broadcast"
    assert categorize._message_text_if_allowed(system_type) is None


def test_message_text_if_allowed_strips_blank_fragments(monkeypatch: pytest.MonkeyPatch) -> None:
    """Whitespace-only fragments should be discarded."""

    message = {
        "id": "blank",
        "author": {"role": "assistant"},
        "content": {"parts": ["ignored"]},
    }

    monkeypatch.setattr(categorize, "_extract_text_parts", lambda _content: ["   "])
    assert categorize._message_text_if_allowed(message) is None


def test_conversation_excerpt_skips_duplicates_and_blank_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Duplicate message IDs and blank results should be ignored."""

    conversation = {
        "messages": [
            {
                "id": "dup",
                "author": {"role": "user"},
                "content": {"parts": ["First message"]},
                "create_time": 10,
            },
            {
                "id": "dup",
                "author": {"role": "assistant"},
                "content": {"parts": ["Duplicate should skip"]},
                "create_time": 20,
            },
            {
                "id": "blank",
                "author": {"role": "assistant"},
                "content": {"parts": ["Will become blank"]},
                "create_time": 30,
            },
            {
                "id": "keep",
                "author": {"role": "assistant"},
                "content": {"parts": ["Second message"]},
                "create_time": 40,
            },
        ]
    }

    original = categorize._message_text_if_allowed

    def fake_message_text(message: dict[str, object]) -> str | None:
        if message.get("id") == "blank":
            return "   "
        return original(message)

    monkeypatch.setattr(categorize, "_message_text_if_allowed", fake_message_text)

    result = _conversation_excerpt(conversation, max_words=5)
    assert result == "User: First message\n\nAssistant: Second"


def test_conversation_excerpt_handles_invalid_limits() -> None:
    """Non-integer limits should fall back without raising."""

    conversation = {
        "messages": [
            {"author": {"role": "user"}, "content": {"parts": ["Hello"]}},
        ]
    }

    assert _conversation_excerpt(conversation, max_words=cast(int, "not-a-number")) == "User: Hello"
    assert _conversation_excerpt(conversation, max_words=-5) == "User: Hello"


def test_conversation_excerpt_ignores_non_dict_messages() -> None:
    """Mapping entries without dict messages are skipped."""

    conversation = {"mapping": {"node": {"message": None}}, "messages": []}
    assert _conversation_excerpt(conversation, max_words=10) is None


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
    assert direct.prompt_excerpt == "User: Hello direct chat prompt"
    assert direct.create_time == pytest.approx(1_700_000_000.0)

    nested = next(chat for chat in chats if chat.id == "chat-2")
    assert nested.title == "Nested Chat"
    assert nested.project_id == "proj-xyz"
    assert nested.prompt_excerpt == "User: Nested user prompt"
    expected_ct = datetime(2024, 1, 2, tzinfo=timezone.utc).timestamp()
    assert nested.create_time == pytest.approx(expected_ct)


def test_extract_chats_from_json_blob_handles_dict_container() -> None:
    """Dictionary containers should also be flattened into chats."""

    data = {
        "conversations": [
            {
                "id": "direct",
                "title": "Direct",
                "messages": [{"author": {"role": "user"}, "content": {"parts": ["Hello"]}}],
            },
            {
                "title": "Missing id",
                "messages": [{"author": {"role": "user"}, "content": {"parts": ["Ignore"]}}],
            },
        ],
        "data": [
            {
                "conversation": {
                    "id": "wrapped",
                    "title": "Wrapped",
                    "messages": [{"author": {"role": "user"}, "content": "Hi"}],
                }
            }
        ],
        "id": "top",
        "title": "Top-level",
        "messages": [{"author": {"role": "user"}, "content": {"parts": ["Top"]}}],
    }

    chats = extract_chats_from_json_blob(data)
    ids = {chat.id for chat in chats}
    assert ids == {"direct", "wrapped", "top"}


def test_extract_chats_skips_non_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    """``convert_one`` should immediately return for non-dict values."""

    monkeypatch.setattr(categorize, "_looks_like_conversation", lambda obj: True)
    assert extract_chats_from_json_blob(["ignore-me"]) == []


def test_extract_chats_ignores_empty_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    """Conversations with blank identifiers are ignored."""

    def forgiving(obj: object) -> bool:
        return isinstance(obj, dict) and "title" in obj and "id" in obj

    monkeypatch.setattr(categorize, "_looks_like_conversation", forgiving)
    data = [
        {
            "id": "",
            "title": "Blank",
            "messages": [{"author": {"role": "user"}, "content": {"parts": ["Hi"]}}],
        }
    ]

    assert extract_chats_from_json_blob(data) == []


def test_load_chats_from_conversations_json_deduplicates_latest(tmp_path: Path) -> None:
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


def test_load_chats_from_conversations_json_missing_file() -> None:
    """A helpful error should be raised when the file does not exist."""

    with pytest.raises(FileNotFoundError):
        load_chats_from_conversations_json("/nonexistent/conversations.json")


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
