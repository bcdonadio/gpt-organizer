"""Integration-style tests for the categorization pipeline and CLI wrappers."""

from __future__ import annotations

import json
import runpy
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import pytest

import GptCategorize.categorize as categorize
import main as entry_main


def _conversation(user_text: str, **extra: object) -> dict[str, object]:
    words = user_text.split()
    base: dict[str, object] = {
        "id": extra.get("id", "chat"),
        "title": extra.get("title", words[0] if words else ""),
        "messages": [
            {
                "author": {"role": "user"},
                "content": {"parts": [user_text]},
                "create_time": extra.get("create_time", 1_700_000_000),
            }
        ],
    }
    base.update(extra)
    return base


def test_categorize_chats_handles_no_available_items(tmp_path: Path) -> None:
    """If every chat is already assigned to a project we exit early."""

    path = tmp_path / "conversations.json"
    path.write_text(
        json.dumps(
            [
                _conversation("already", id="chat-1", metadata={"project_id": "existing"}),
                _conversation("in project", id="chat-2", conversation={"workspace": "abc"}),
            ]
        )
    )
    out_path = tmp_path / "plan.json"

    code = categorize.categorize_chats(str(path), out=str(out_path), no_qdrant=True)
    assert code == 0

    plan = json.loads(out_path.read_text())
    assert plan["proposed_moves"] == []
    assert sorted(plan["skipped"]["already_in_project"]) == ["chat-1", "chat-2"]


def test_categorize_chats_respects_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The limit flag should cap how many chats enter the processing pipeline."""

    conversations_path = tmp_path / "limit.json"
    conversations_path.write_text("[]")
    out_path = tmp_path / "plan.json"

    chats = [
        categorize.Chat(id="chat-0", title="Alpha", prompt_excerpt="First"),
        categorize.Chat(id="chat-1", title="Beta", prompt_excerpt="Second"),
        categorize.Chat(id="chat-2", title="Gamma", prompt_excerpt="Third"),
    ]

    monkeypatch.setattr(categorize, "load_chats_from_conversations_json", lambda _path: list(chats))

    embed_calls: list[list[str]] = []

    def fake_embed(_: Any, texts: Sequence[str], batch_size: int = 96) -> np.ndarray:
        embed_calls.append(list(texts))
        if not texts:
            return np.empty((0, 2), dtype=np.float32)
        values = np.linspace(1.0, 1.0 + len(texts) - 1, num=len(texts), dtype=float)
        vectors = np.stack([np.array([val, val + 0.1], dtype=float) for val in values], axis=0)
        return vectors.astype(np.float32)

    def fake_cluster(
        vectors: np.ndarray, *, eps_cosine: float, min_samples: int, min_cluster_size: int
    ) -> np.ndarray:
        assert vectors.shape[0] == 2  # limit applied before clustering
        return np.zeros(vectors.shape[0], dtype=int)

    def fake_text(_: np.ndarray, *, labels_mapped: np.ndarray, cid: int) -> float:
        return 0.9

    def fake_temporal(*, members: Sequence[categorize.Chat], time_decay_days: float) -> float:
        return 0.85

    recorded_clusters: list[list[str]] = []

    def fake_label(_: Any, clusters: dict[int, list[categorize.Chat]]) -> dict[int, dict[str, object]]:
        recorded_clusters.append([chat.id for chat in clusters.get(0, [])])
        return {
            0: {
                "label": "Limited Cluster",
                "project_folder_slug": "limited-cluster",
                "project_title": "Limited Cluster",
                "confidence_model": 0.95,
            }
        }

    monkeypatch.setattr(categorize, "get_embedding_client", _simple_client)
    monkeypatch.setattr(categorize, "embed_chats_with_retry", fake_embed)
    monkeypatch.setattr(categorize, "cluster_embeddings", fake_cluster)
    monkeypatch.setattr(categorize, "cluster_text_cohesion", fake_text)
    monkeypatch.setattr(categorize, "temporal_cohesion", fake_temporal)
    monkeypatch.setattr(categorize, "get_inference_client", _simple_client)
    monkeypatch.setattr(categorize, "label_clusters_with_llm", fake_label)

    code = categorize.categorize_chats(str(conversations_path), out=str(out_path), no_qdrant=True, limit=2)
    assert code == 0

    plan = json.loads(out_path.read_text())
    cluster_chat_ids = [chat_info["id"] for chat_info in plan["clusters"][0]["chats"]]
    assert cluster_chat_ids == ["chat-0", "chat-1"]
    assert plan["parameters"]["min_cluster_size"] == 2
    assert embed_calls and len(embed_calls[0]) == 2
    assert recorded_clusters == [["chat-0", "chat-1"]]
    assert "chat-2" not in json.dumps(plan)


def _simple_client() -> SimpleNamespace:
    return SimpleNamespace()


def test_categorize_chats_generates_plan_with_qdrant(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The happy-path flow should orchestrate clustering and persistence."""

    path = tmp_path / "convos.json"
    chats = [
        _conversation("Alpha topic", id="chat-0", title="Alpha", create_time=1000),
        _conversation("Alpha follow up", id="chat-1", title="Alpha Follow", create_time=2000),
        _conversation("Singleton", id="chat-2", title="Solo", create_time=3000),
    ]
    path.write_text(json.dumps(chats))
    out_path = tmp_path / "plan.json"

    def fake_embed(_: Any, __: Sequence[str], batch_size: int = 96) -> np.ndarray:
        return np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=float)

    def fake_clusters(
        _: np.ndarray, *, eps_cosine: float, min_samples: int, min_cluster_size: int
    ) -> np.ndarray:
        return np.array([0, 0, -1])

    def fake_text_cohesion(_: np.ndarray, *, labels_mapped: np.ndarray, cid: int) -> float:
        return 0.8

    def fake_temporal(members: Sequence[categorize.Chat], *, time_decay_days: float) -> float:
        return 0.6

    def fake_label_clusters(_: Any, __: dict[int, list[categorize.Chat]]) -> dict[int, dict[str, object]]:
        return {
            0: {
                "label": "Alpha Project",
                "project_folder_slug": "alpha-project",
                "project_title": "Alpha Project",
                "confidence_model": 0.9,
            }
        }

    qdrant_calls: list[tuple[Any, ...]] = []

    def fake_fetch(_: Any, __: str, ___: Sequence[str]) -> dict[str, np.ndarray]:
        return {}

    def record_ensure(_: Any, name: str, size: int) -> None:
        qdrant_calls.append(("ensure", name, size))

    def record_upsert(_: Any, name: str, chats_subset: Sequence[categorize.Chat], vectors: np.ndarray) -> None:
        qdrant_calls.append(("upsert", len(chats_subset), vectors.shape))

    monkeypatch.setattr(categorize, "get_inference_client", _simple_client)
    monkeypatch.setattr(categorize, "get_embedding_client", _simple_client)
    monkeypatch.setattr(categorize, "embed_chats_with_retry", fake_embed)
    monkeypatch.setattr(categorize, "cluster_embeddings", fake_clusters)
    monkeypatch.setattr(categorize, "cluster_text_cohesion", fake_text_cohesion)
    monkeypatch.setattr(categorize, "temporal_cohesion", fake_temporal)
    monkeypatch.setattr(categorize, "label_clusters_with_llm", fake_label_clusters)
    monkeypatch.setattr(categorize, "get_qdrant_client_with_timeout", _simple_client)
    monkeypatch.setattr(categorize, "fetch_existing_embeddings_from_qdrant", fake_fetch)
    monkeypatch.setattr(categorize, "ensure_qdrant_collection", record_ensure)
    monkeypatch.setattr(categorize, "upsert_to_qdrant", record_upsert)

    code = categorize.categorize_chats(str(path), out=str(out_path), no_qdrant=False, min_cluster_size=1)
    assert code == 0

    plan = json.loads(out_path.read_text())
    assert plan["proposed_moves"] and plan["proposed_moves"][0]["project_folder_slug"] == "alpha-project"
    assert plan["skipped"]["singletons"] == ["chat-2"]
    assert qdrant_calls[0][0] == "ensure" and qdrant_calls[1][0] == "upsert"


def test_embedding_batches_wait_for_qdrant(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Next embedding groups are requested only after prior upserts complete."""

    out_path = tmp_path / "plan.json"

    chats = [
        categorize.Chat(id=f"chat-{idx}", title=f"Title {idx}", prompt_excerpt=f"body {idx}", create_time=idx)
        for idx in range(4)
    ]

    monkeypatch.setattr(categorize, "load_chats_from_conversations_json", lambda _path: list(chats))

    def text_for(chat: categorize.Chat) -> str:
        return f"{chat.title}\n\n{chat.prompt_excerpt}" if chat.prompt_excerpt else chat.title

    text_to_id = {text_for(chat): chat.id for chat in chats}

    call_log: list[tuple[str, list[str]]] = []

    def fake_embed(_: Any, texts: Sequence[str], batch_size: int = 96) -> np.ndarray:
        ids = [text_to_id[text] for text in texts]
        call_log.append(("embed", ids))
        vectors = np.array([[float(idx + 1), 0.0] for idx, _ in enumerate(ids)], dtype=np.float32)
        return vectors

    def fake_cluster(
        vectors: np.ndarray, *, eps_cosine: float, min_samples: int, min_cluster_size: int
    ) -> np.ndarray:
        assert vectors.shape[0] == len(chats)
        return np.zeros(vectors.shape[0], dtype=int)

    def fake_label(_: Any, clusters: dict[int, list[categorize.Chat]]) -> dict[int, dict[str, object]]:
        return {
            0: {
                "label": "Grouped",
                "project_folder_slug": "grouped",
                "project_title": "Grouped",
                "confidence_model": 0.9,
            }
        }

    qdrant_events: list[list[str]] = []

    def fake_upsert(_: Any, __: str, subset: Sequence[categorize.Chat], vectors: np.ndarray) -> None:
        ids = [chat.id for chat in subset]
        call_log.append(("upsert", ids))
        qdrant_events.append(ids)
        assert vectors.shape[0] == len(subset)

    ensure_calls: list[tuple[str, int]] = []

    def fake_ensure(_: Any, name: str, size: int) -> None:
        ensure_calls.append((name, size))

    monkeypatch.setattr(categorize, "get_embedding_client", _simple_client)
    monkeypatch.setattr(categorize, "embed_chats_with_retry", fake_embed)
    monkeypatch.setattr(categorize, "get_inference_client", _simple_client)
    monkeypatch.setattr(categorize, "cluster_embeddings", fake_cluster)
    monkeypatch.setattr(categorize, "cluster_text_cohesion", lambda *_args, **_kwargs: 0.8)
    monkeypatch.setattr(categorize, "temporal_cohesion", lambda *_args, **_kwargs: 0.7)
    monkeypatch.setattr(categorize, "label_clusters_with_llm", fake_label)
    monkeypatch.setattr(categorize, "get_qdrant_client_with_timeout", _simple_client)
    monkeypatch.setattr(categorize, "fetch_existing_embeddings_from_qdrant", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(categorize, "ensure_qdrant_collection", fake_ensure)
    monkeypatch.setattr(categorize, "upsert_to_qdrant", fake_upsert)

    code = categorize.categorize_chats(
        "ignored.json",
        out=str(out_path),
        no_qdrant=False,
        embedding_batch_words=1,
        embedding_batch_parallelism=2,
    )
    assert code == 0

    plan = json.loads(out_path.read_text())
    assert plan["parameters"]["embedding_batch_words"] == 1
    assert plan["parameters"]["embedding_batch_parallelism"] == 2
    assert ensure_calls and ensure_calls[0][1] == 2
    assert qdrant_events == [["chat-0"], ["chat-1"], ["chat-2"], ["chat-3"]]

    def locate(kind: str, target: str) -> int:
        for idx, entry in enumerate(call_log):
            if entry[0] == kind and entry[1] == [target]:
                return idx
        raise AssertionError(f"{kind} call for {target} not found")

    first_upsert_idx = max(locate("upsert", "chat-0"), locate("upsert", "chat-1"))
    assert locate("embed", "chat-2") > first_upsert_idx
    assert locate("embed", "chat-3") > first_upsert_idx


def test_embedding_batch_failure_disables_qdrant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A failed upsert disables further persistence but keeps embedding batches flowing."""

    out_path = tmp_path / "plan.json"
    chats = [
        categorize.Chat(id=f"chat-{idx}", title=f"Title {idx}", prompt_excerpt=f"body {idx}")
        for idx in range(3)
    ]

    monkeypatch.setattr(categorize, "load_chats_from_conversations_json", lambda _path: list(chats))

    def text_for(chat: categorize.Chat) -> str:
        return f"{chat.title}\n\n{chat.prompt_excerpt}" if chat.prompt_excerpt else chat.title

    text_to_id = {text_for(chat): chat.id for chat in chats}

    embed_tally = 0

    def fake_embed(_: Any, texts: Sequence[str], batch_size: int = 96) -> np.ndarray:
        nonlocal embed_tally
        embed_tally += len(texts)
        _ = [text_to_id[text] for text in texts]
        return np.ones((len(texts), 2), dtype=np.float32)

    upsert_attempts = 0

    def failing_upsert(_: Any, __: str, subset: Sequence[categorize.Chat], vectors: np.ndarray) -> None:
        nonlocal upsert_attempts
        upsert_attempts += 1
        assert vectors.shape[0] == len(subset)
        raise RuntimeError("boom")

    monkeypatch.setattr(categorize, "get_embedding_client", _simple_client)
    monkeypatch.setattr(categorize, "embed_chats_with_retry", fake_embed)
    monkeypatch.setattr(categorize, "get_inference_client", _simple_client)
    monkeypatch.setattr(categorize, "cluster_embeddings", lambda *_args, **_kwargs: np.zeros(len(chats), dtype=int))
    monkeypatch.setattr(categorize, "cluster_text_cohesion", lambda *_args, **_kwargs: 0.5)
    monkeypatch.setattr(categorize, "temporal_cohesion", lambda *_args, **_kwargs: 0.5)
    monkeypatch.setattr(categorize, "label_clusters_with_llm", lambda *_args, **_kwargs: {0: {"label": "ok"}})
    monkeypatch.setattr(categorize, "get_qdrant_client_with_timeout", _simple_client)
    monkeypatch.setattr(categorize, "fetch_existing_embeddings_from_qdrant", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(categorize, "ensure_qdrant_collection", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(categorize, "upsert_to_qdrant", failing_upsert)

    code = categorize.categorize_chats(
        "ignored.json",
        out=str(out_path),
        no_qdrant=False,
        embedding_batch_words=1,
        embedding_batch_parallelism=1,
    )
    assert code == 0

    captured = capsys.readouterr().out
    assert "Warning: Qdrant operation failed" in captured
    assert embed_tally == len(chats)
    assert upsert_attempts == 1


def test_embedding_batch_shape_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An embedding batch returning an unexpected shape is treated as fatal."""

    out_path = tmp_path / "plan.json"
    chats = [categorize.Chat(id="bad", title="One", prompt_excerpt="body")]

    monkeypatch.setattr(categorize, "load_chats_from_conversations_json", lambda _path: list(chats))
    monkeypatch.setattr(categorize, "get_embedding_client", _simple_client)
    monkeypatch.setattr(categorize, "embed_chats_with_retry", lambda *_args, **_kwargs: np.zeros((0, 2), dtype=np.float32))

    with pytest.raises(RuntimeError, match="Embedding batch shape mismatch"):
        categorize.categorize_chats(
            "ignored.json",
            out=str(out_path),
            no_qdrant=True,
            embedding_batch_words=1,
            embedding_batch_parallelism=1,
        )


def test_categorize_chats_reuses_cached_embeddings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Existing Qdrant vectors should be reused without requesting new embeddings."""

    path = tmp_path / "cached.json"
    chats = [
        _conversation("Topic A", id="chat-0", title="Topic A"),
        _conversation("Topic B", id="chat-1", title="Topic B"),
    ]
    path.write_text(json.dumps(chats))
    out_path = tmp_path / "plan.json"

    cached_vectors = {
        "chat-0": np.array([1.0, 0.0], dtype=np.float32),
        "chat-1": np.array([0.5, 0.5], dtype=np.float32),
    }

    def fake_fetch(_: Any, __: str, ids: Sequence[str]) -> dict[str, np.ndarray]:
        return {cid: cached_vectors[cid] for cid in ids}

    monkeypatch.setattr(categorize, "get_qdrant_client_with_timeout", _simple_client)
    monkeypatch.setattr(categorize, "fetch_existing_embeddings_from_qdrant", fake_fetch)
    monkeypatch.setattr(categorize, "get_embedding_client", partial(pytest.fail, "should not request embedding client"))
    monkeypatch.setattr(categorize, "embed_chats_with_retry", partial(pytest.fail, "should not embed"))

    def fake_cluster_embeddings(
        _: np.ndarray, *, eps_cosine: float, min_samples: int, min_cluster_size: int
    ) -> np.ndarray:
        return np.array([0, 0])

    monkeypatch.setattr(categorize, "cluster_embeddings", fake_cluster_embeddings)

    def cached_text_cohesion(_: np.ndarray, *, labels_mapped: np.ndarray, cid: int) -> float:
        return 1.0

    def cached_temporal(members: Sequence[categorize.Chat], *, time_decay_days: float) -> float:
        return 1.0

    monkeypatch.setattr(categorize, "cluster_text_cohesion", cached_text_cohesion)
    monkeypatch.setattr(categorize, "temporal_cohesion", cached_temporal)
    monkeypatch.setattr(categorize, "get_inference_client", _simple_client)
    monkeypatch.setattr(categorize, "label_clusters_with_llm", lambda *_: {0: {"label": "Cached"}})
    monkeypatch.setattr(categorize, "ensure_qdrant_collection", partial(pytest.fail, "should not ensure"))
    monkeypatch.setattr(categorize, "upsert_to_qdrant", partial(pytest.fail, "should not upsert"))

    code = categorize.categorize_chats(str(path), out=str(out_path), no_qdrant=False, min_cluster_size=1)
    assert code == 0

    plan = json.loads(out_path.read_text())
    assert plan["clusters"] and plan["clusters"][0]["cohesion_text"] == 1.0


def test_categorize_chats_handles_failures_and_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Qdrant failures and LLM errors trigger graceful fallbacks."""

    path = tmp_path / "input.json"
    chats = [
        _conversation("   ", id="chat-10", title="   "),
        _conversation("backup", id="chat-11", title="Backup"),
    ]
    path.write_text(json.dumps(chats))
    out_path = tmp_path / "plan.json"

    monkeypatch.setattr(categorize, "get_inference_client", _simple_client)
    monkeypatch.setattr(categorize, "get_embedding_client", _simple_client)
    monkeypatch.setattr(categorize, "embed_chats_with_retry", lambda *_: np.ones((2, 2), dtype=float))

    def fallback_clusters(
        _: np.ndarray, *, eps_cosine: float, min_samples: int, min_cluster_size: int
    ) -> np.ndarray:
        return np.array([0, 0])

    monkeypatch.setattr(categorize, "cluster_embeddings", fallback_clusters)

    def failing_text(_: np.ndarray, *, labels_mapped: np.ndarray, cid: int) -> float:
        return 0.0

    def failing_temporal(members: Sequence[categorize.Chat], *, time_decay_days: float) -> float:
        return 0.0

    monkeypatch.setattr(categorize, "cluster_text_cohesion", failing_text)
    monkeypatch.setattr(categorize, "temporal_cohesion", failing_temporal)

    def failing_label(_: Any, clusters: dict[int, list[categorize.Chat]]) -> dict[int, dict[str, object]]:
        clusters[99] = []
        raise RuntimeError("llm")

    def fail_qdrant_client() -> Any:
        raise RuntimeError("qdrant")

    monkeypatch.setattr(categorize, "label_clusters_with_llm", failing_label)
    monkeypatch.setattr(categorize, "get_qdrant_client_with_timeout", fail_qdrant_client)

    code = categorize.categorize_chats(str(path), out=str(out_path), no_qdrant=False, min_cluster_size=1)
    assert code == 0

    plan = json.loads(out_path.read_text())
    assert plan["proposed_moves"] == []
    assert plan["parameters"]["min_cluster_size"] == 2
    assert set(plan["skipped"]["clusters_low_confidence"]) == {"0", "99"}
    captured = capsys.readouterr().out
    assert "Continuing without Qdrant" in captured
    assert "Using fallback cluster labels" in captured


def test_categorize_chats_warns_on_qdrant_cache_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Failures during cache fetch should fall back to embedding all chats."""

    path = tmp_path / "cache-fail.json"
    chats = [
        _conversation("Topic A", id="chat-1", title="Topic A"),
        _conversation("Topic B", id="chat-2", title="Topic B"),
    ]
    path.write_text(json.dumps(chats))
    out_path = tmp_path / "plan.json"

    monkeypatch.setattr(categorize, "get_qdrant_client_with_timeout", _simple_client)

    def raise_fetch(*_: Any, **__: Any) -> dict[str, np.ndarray]:
        raise RuntimeError("fetch boom")

    def fake_embed(_: Any, __: Sequence[str], batch_size: int = 96) -> np.ndarray:
        return np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)

    def fake_label(_: Any, __: dict[int, list[categorize.Chat]]) -> dict[int, dict[str, object]]:
        return {
            0: {
                "label": "Recovered",
                "project_folder_slug": "recovered",
                "project_title": "Recovered",
                "confidence_model": 0.7,
            }
        }

    monkeypatch.setattr(categorize, "fetch_existing_embeddings_from_qdrant", raise_fetch)
    monkeypatch.setattr(categorize, "get_embedding_client", _simple_client)
    monkeypatch.setattr(categorize, "embed_chats_with_retry", fake_embed)

    def warn_clusters(
        _: np.ndarray, *, eps_cosine: float, min_samples: int, min_cluster_size: int
    ) -> np.ndarray:
        return np.array([0, 0])

    monkeypatch.setattr(categorize, "cluster_embeddings", warn_clusters)

    def warn_text(_: np.ndarray, *, labels_mapped: np.ndarray, cid: int) -> float:
        return 0.5

    def warn_temporal(members: Sequence[categorize.Chat], *, time_decay_days: float) -> float:
        return 0.5

    monkeypatch.setattr(categorize, "cluster_text_cohesion", warn_text)
    monkeypatch.setattr(categorize, "temporal_cohesion", warn_temporal)
    monkeypatch.setattr(categorize, "get_inference_client", _simple_client)
    monkeypatch.setattr(categorize, "label_clusters_with_llm", fake_label)

    qdrant_calls: list[tuple[str, int]] = []

    def record_ensure(_: Any, __: str, size: int) -> None:
        qdrant_calls.append(("ensure", size))

    def record_upsert(_: Any, __: str, chats_subset: Sequence[categorize.Chat], vectors: np.ndarray) -> None:
        qdrant_calls.append(("upsert", len(chats_subset)))

    monkeypatch.setattr(categorize, "ensure_qdrant_collection", record_ensure)
    monkeypatch.setattr(categorize, "upsert_to_qdrant", record_upsert)

    code = categorize.categorize_chats(str(path), out=str(out_path), no_qdrant=False)
    assert code == 0
    captured = capsys.readouterr().out
    assert "cache were empty" in captured
    assert qdrant_calls == [("ensure", 2), ("upsert", 2)]


def test_categorize_chats_raises_when_embeddings_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If embeddings remain missing after retries the pipeline aborts."""

    path = tmp_path / "missing.json"
    chats = [
        _conversation("Topic A", id="chat-1", title="Topic A"),
        _conversation("Topic B", id="chat-2", title="Topic B"),
    ]
    path.write_text(json.dumps(chats))

    class NonStoringDict(dict[str, np.ndarray]):
        def __setitem__(self, key: str, value: np.ndarray) -> None:
            return None

    monkeypatch.setattr(categorize, "get_qdrant_client_with_timeout", _simple_client)
    monkeypatch.setattr(categorize, "fetch_existing_embeddings_from_qdrant", lambda *_: NonStoringDict())
    monkeypatch.setattr(categorize, "get_embedding_client", _simple_client)
    monkeypatch.setattr(categorize, "embed_chats_with_retry", lambda *_: np.ones((2, 2), dtype=float))

    with pytest.raises(RuntimeError) as excinfo:
        categorize.categorize_chats(str(path), out=str(path.with_suffix(".out")), no_qdrant=False)

    assert "Missing embeddings" in str(excinfo.value)


def test_categorize_chats_handles_upsert_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Upsert errors should warn but not abort processing."""

    path = tmp_path / "upsert.json"
    chats = [
        _conversation("Topic A", id="chat-1", title="Topic A"),
        _conversation("Topic B", id="chat-2", title="Topic B"),
    ]
    path.write_text(json.dumps(chats))
    out_path = tmp_path / "plan.json"

    monkeypatch.setattr(categorize, "get_qdrant_client_with_timeout", _simple_client)
    monkeypatch.setattr(categorize, "fetch_existing_embeddings_from_qdrant", lambda *_: {})
    monkeypatch.setattr(categorize, "get_embedding_client", _simple_client)
    monkeypatch.setattr(categorize, "embed_chats_with_retry", lambda *_: np.ones((2, 2), dtype=float))

    def resilient_clusters(
        _: np.ndarray, *, eps_cosine: float, min_samples: int, min_cluster_size: int
    ) -> np.ndarray:
        return np.array([0, 0])

    monkeypatch.setattr(categorize, "cluster_embeddings", resilient_clusters)

    def upsert_text(_: np.ndarray, *, labels_mapped: np.ndarray, cid: int) -> float:
        return 0.4

    def upsert_temporal(members: Sequence[categorize.Chat], *, time_decay_days: float) -> float:
        return 0.4

    monkeypatch.setattr(categorize, "cluster_text_cohesion", upsert_text)
    monkeypatch.setattr(categorize, "temporal_cohesion", upsert_temporal)
    monkeypatch.setattr(categorize, "get_inference_client", _simple_client)
    monkeypatch.setattr(
        categorize,
        "label_clusters_with_llm",
        lambda *_: {
            0: {
                "label": "Upsert",
                "project_folder_slug": "upsert",
                "project_title": "Upsert",
                "confidence_model": 0.6,
            }
        },
    )

    def fail_upsert(*_: Any) -> None:
        raise RuntimeError("qdrant upsert fail")

    monkeypatch.setattr(categorize, "ensure_qdrant_collection", lambda *_: None)
    monkeypatch.setattr(categorize, "upsert_to_qdrant", fail_upsert)

    code = categorize.categorize_chats(str(path), out=str(out_path), no_qdrant=False)
    assert code == 0
    captured = capsys.readouterr().out
    assert "Warning: Qdrant operation failed" in captured


def test_categorize_prints_summary_for_many_moves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Large numbers of proposed moves trigger the truncated summary output."""

    path = tmp_path / "bulk.json"
    items: list[dict[str, object]] = []
    labels: list[int] = []
    embeddings: list[list[float]] = []
    for idx in range(51):
        items.append(_conversation(f"topic {idx}a", id=f"chat-{idx*2}", title=f"Topic {idx} A"))
        items.append(_conversation(f"topic {idx}b", id=f"chat-{idx*2+1}", title=f"Topic {idx} B"))
        labels.extend([idx, idx])
        embeddings.extend([[float(idx), 0.1], [float(idx), 0.2]])
    path.write_text(json.dumps(items))
    out_path = tmp_path / "plan.json"

    monkeypatch.setattr(categorize, "get_inference_client", _simple_client)
    monkeypatch.setattr(categorize, "get_embedding_client", _simple_client)
    monkeypatch.setattr(categorize, "embed_chats_with_retry", lambda *_: np.array(embeddings, dtype=float))

    def many_clusters(
        _: np.ndarray, *, eps_cosine: float, min_samples: int, min_cluster_size: int
    ) -> np.ndarray:
        return np.array(labels)

    monkeypatch.setattr(categorize, "cluster_embeddings", many_clusters)

    def summary_text(_: np.ndarray, *, labels_mapped: np.ndarray, cid: int) -> float:
        return 0.9

    def summary_temporal(members: Sequence[categorize.Chat], *, time_decay_days: float) -> float:
        return 0.9

    monkeypatch.setattr(categorize, "cluster_text_cohesion", summary_text)
    monkeypatch.setattr(categorize, "temporal_cohesion", summary_temporal)

    def fake_labels(_: Any, clusters: dict[int, list[categorize.Chat]]) -> dict[int, dict[str, object]]:
        return {
            cid: {
                "label": f"Topic {cid}",
                "project_folder_slug": f"topic-{cid}",
                "project_title": f"Topic {cid}",
                "confidence_model": 0.95,
            }
            for cid in clusters
        }

    monkeypatch.setattr(categorize, "label_clusters_with_llm", fake_labels)

    categorize.categorize_chats(str(path), out=str(out_path), no_qdrant=True)
    output = capsys.readouterr().out
    assert "... and 1 more" in output

    plan = json.loads(out_path.read_text())
    assert len(plan["proposed_moves"]) == 51


def test_entry_main_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Top-level ``main.py`` delegates to the categorization function."""

    path = tmp_path / "src.json"
    path.write_text("[]")
    out_path = tmp_path / "dest.json"

    called: dict[str, object] = {}

    def fake_categorize(**kwargs: object) -> int:
        called.update(kwargs)
        return 7

    monkeypatch.setattr(entry_main, "categorize_chats", fake_categorize)

    argv = [
        "main",
        "--conversations-json",
        str(path),
        "--out",
        str(out_path),
        "--collection",
        "col",
        "--no-qdrant",
        "--eps-cosine",
        "0.4",
        "--min-samples",
        "4",
        "--min-cluster-size",
        "6",
        "--confidence-threshold",
        "0.6",
        "--time-weight",
        "0.2",
        "--limit",
        "5",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    exit_code = entry_main.main()

    assert exit_code == 7
    assert called["no_qdrant"] is True
    assert called["collection"] == "col"
    assert called["min_cluster_size"] == 6


def test_entry_main_module_block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Running ``python -m main`` exits with the CLI return code."""

    path = tmp_path / "input.json"
    path.write_text(json.dumps([_conversation("skip", id="chat-y", metadata={"project_id": "x"})]))

    argv = ["main", "--conversations-json", str(path), "--out", str(tmp_path / "plan.json"), "--no-qdrant"]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc:
        runpy.run_module("main", run_name="__main__")

    assert exc.value.code == 0


def test_entry_main_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The ``main.py`` script should exit with status 1 on Ctrl+C."""

    path = tmp_path / "input.json"
    path.write_text("[]")

    argv = ["main", "--conversations-json", str(path), "--out", str(tmp_path / "plan.json"), "--no-qdrant"]
    monkeypatch.setattr(sys, "argv", argv)

    def raise_interrupt(**_: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(categorize, "categorize_chats", raise_interrupt)

    with pytest.raises(SystemExit) as exc:
        runpy.run_module("main", run_name="__main__")

    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "Interrupted." in out
