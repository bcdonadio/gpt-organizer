"""Tests covering logging configuration and client factories."""

from __future__ import annotations

import builtins
import datetime as dt
import sys
from types import ModuleType
from typing import Iterator

import pytest

import GptCategorize.categorize as categorize


@pytest.fixture(autouse=True)
def restore_logging_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Ensure logging globals are restored after each test."""

    orig_level = categorize.LOG_LEVEL
    orig_verbose = categorize.VERBOSE_LOGGING
    yield
    monkeypatch.setattr(categorize, "LOG_LEVEL", orig_level, raising=False)
    monkeypatch.setattr(categorize, "VERBOSE_LOGGING", orig_verbose, raising=False)


def test_setup_logging_invalid_level_and_verbose(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An invalid log level should fall back to INFO and enable verbose hooks."""

    monkeypatch.setattr(categorize, "LOG_LEVEL", "not-a-level", raising=False)
    monkeypatch.setattr(categorize, "VERBOSE_LOGGING", True, raising=False)

    original_import = builtins.__import__

    def fake_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "urllib3":
            raise ImportError("missing")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    categorize.setup_logging()
    out = capsys.readouterr().out
    assert "Invalid LOG_LEVEL" in out
    assert "Verbose logging enabled" in out


def test_setup_logging_verbose_imports_urllib3(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """When urllib3 is available the verbose branch should configure it."""

    monkeypatch.setattr(categorize, "LOG_LEVEL", "INFO", raising=False)
    monkeypatch.setattr(categorize, "VERBOSE_LOGGING", True, raising=False)
    stub = ModuleType("urllib3")

    def disable_warnings() -> None:
        return None

    setattr(stub, "disable_warnings", disable_warnings)
    monkeypatch.setitem(sys.modules, "urllib3", stub)
    categorize.setup_logging()
    out = capsys.readouterr().out
    assert "Verbose logging enabled" in out


def test_now_iso_produces_timezone_aware_timestamp() -> None:
    """``now_iso`` should return an ISO-8601 timestamp with timezone info."""

    iso = categorize.now_iso()
    parsed = dt.datetime.fromisoformat(iso)
    assert parsed.tzinfo is not None


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({"title": "Chat", "id": "123"}, True),
        ({"conversation": {"title": "Chat", "id": "123"}}, True),
        ({"title": "Missing id"}, False),
    ],
)
def test_looks_like_conversation_detection(payload: dict[str, object], expected: bool) -> None:
    """``_looks_like_conversation`` detects valid shapes."""

    assert bool(categorize._looks_like_conversation(payload)) is expected


def test_get_inference_client_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Factory should raise when the API key placeholder is still set."""

    monkeypatch.setattr(categorize, "INFERENCE_API_KEY", "YOUR_INFERENCE_API_KEY_HERE", raising=False)
    with pytest.raises(RuntimeError):
        categorize.get_inference_client()


def test_get_embedding_client_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Embedding client factory should validate API key presence."""

    monkeypatch.setattr(categorize, "EMBEDDING_API_KEY", "YOUR_EMBEDDING_API_KEY_HERE", raising=False)
    with pytest.raises(RuntimeError):
        categorize.get_embedding_client()


def test_client_factories_return_configured_instances(monkeypatch: pytest.MonkeyPatch) -> None:
    """Factories should instantiate the ``OpenAI`` client with configured options."""

    constructed: dict[str, object] = {}

    class DummyOpenAI:
        def __init__(self, **kwargs: object) -> None:
            constructed.update(kwargs)

    monkeypatch.setattr(categorize, "OpenAI", DummyOpenAI)
    monkeypatch.setattr(categorize, "INFERENCE_API_KEY", "inference-key", raising=False)
    monkeypatch.setattr(categorize, "EMBEDDING_API_KEY", "embedding-key", raising=False)
    monkeypatch.setattr(categorize, "INFERENCE_API_BASE", "https://inference", raising=False)
    monkeypatch.setattr(categorize, "EMBEDDING_API_BASE", "https://embedding", raising=False)
    monkeypatch.setattr(categorize, "OPENAI_TIMEOUT", 123.0, raising=False)

    categorize.get_inference_client()
    assert constructed["api_key"] == "inference-key"
    assert constructed["base_url"] == "https://inference"
    assert constructed["timeout"] == 123.0

    constructed.clear()
    embed_client = categorize.get_embedding_client()
    assert embed_client is not None
    assert constructed["api_key"] == "embedding-key"
    assert constructed["base_url"] == "https://embedding"
