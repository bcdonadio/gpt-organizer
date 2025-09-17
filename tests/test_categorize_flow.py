"""Integration-style tests for the categorization pipeline and CLI wrappers."""

from __future__ import annotations

import json
import runpy
import sys
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest

import GptCategorize.categorize as categorize
import main as entry_main


def _conversation(user_text: str, **extra: object) -> dict[str, object]:
    words = user_text.split()
    base = {
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


def test_categorize_chats_handles_no_available_items(tmp_path):
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


def test_categorize_chats_generates_plan_with_qdrant(tmp_path, monkeypatch):
    """The happy-path flow should orchestrate clustering and persistence."""

    path = tmp_path / "convos.json"
    chats = [
        _conversation("Alpha topic", id="chat-0", title="Alpha", create_time=1000),
        _conversation("Alpha follow up", id="chat-1", title="Alpha Follow", create_time=2000),
        _conversation("Singleton", id="chat-2", title="Solo", create_time=3000),
    ]
    path.write_text(json.dumps(chats))
    out_path = tmp_path / "plan.json"

    monkeypatch.setattr(categorize, "get_inference_client", lambda: SimpleNamespace())
    monkeypatch.setattr(categorize, "get_embedding_client", lambda: SimpleNamespace())
    monkeypatch.setattr(
        categorize,
        "embed_chats_with_retry",
        lambda client, texts, batch_size=96: np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=float),
    )
    monkeypatch.setattr(categorize, "cluster_embeddings", lambda vectors, eps_cosine, min_samples: np.array([0, 0, -1]))
    monkeypatch.setattr(categorize, "cluster_text_cohesion", lambda vectors, labels_mapped, cid: 0.8)
    monkeypatch.setattr(categorize, "temporal_cohesion", lambda members, time_decay_days: 0.6)
    monkeypatch.setattr(
        categorize,
        "label_clusters_with_llm",
        lambda client, clusters: {
            0: {
                "label": "Alpha Project",
                "project_folder_slug": "alpha-project",
                "project_title": "Alpha Project",
                "confidence_model": 0.9,
            }
        },
    )

    qdrant_calls: list[tuple[str, object]] = []
    monkeypatch.setattr(categorize, "get_qdrant_client_with_timeout", lambda: SimpleNamespace())
    monkeypatch.setattr(categorize, "fetch_existing_embeddings_from_qdrant", lambda client, name, ids: {})
    monkeypatch.setattr(categorize, "ensure_qdrant_collection", lambda client, name, size: qdrant_calls.append(("ensure", name, size)))
    monkeypatch.setattr(
        categorize,
        "upsert_to_qdrant",
        lambda client, name, chats_subset, vectors: qdrant_calls.append(("upsert", len(chats_subset), vectors.shape)),
    )

    code = categorize.categorize_chats(str(path), out=str(out_path), no_qdrant=False)
    assert code == 0

    plan = json.loads(out_path.read_text())
    assert plan["proposed_moves"] and plan["proposed_moves"][0]["project_folder_slug"] == "alpha-project"
    assert plan["skipped"]["singletons"] == ["chat-2"]
    assert qdrant_calls[0][0] == "ensure" and qdrant_calls[1][0] == "upsert"


def test_categorize_chats_reuses_cached_embeddings(tmp_path, monkeypatch):
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

    monkeypatch.setattr(categorize, "get_qdrant_client_with_timeout", lambda: SimpleNamespace())
    monkeypatch.setattr(categorize, "fetch_existing_embeddings_from_qdrant", lambda client, name, ids: {cid: cached_vectors[cid] for cid in ids})
    monkeypatch.setattr(categorize, "get_embedding_client", partial(pytest.fail, "should not request embedding client"))
    monkeypatch.setattr(categorize, "embed_chats_with_retry", partial(pytest.fail, "should not embed"))
    monkeypatch.setattr(categorize, "cluster_embeddings", lambda vectors, eps_cosine, min_samples: np.array([0, 0]))
    monkeypatch.setattr(categorize, "cluster_text_cohesion", lambda vectors, labels_mapped, cid: 1.0)
    monkeypatch.setattr(categorize, "temporal_cohesion", lambda members, time_decay_days: 1.0)
    monkeypatch.setattr(categorize, "get_inference_client", lambda: SimpleNamespace())
    monkeypatch.setattr(
        categorize,
        "label_clusters_with_llm",
        lambda client, clusters: {
            0: {
                "label": "Cached",
                "project_folder_slug": "cached",
                "project_title": "Cached",
                "confidence_model": 0.8,
            }
        },
    )

    monkeypatch.setattr(categorize, "ensure_qdrant_collection", partial(pytest.fail, "should not ensure"))
    monkeypatch.setattr(categorize, "upsert_to_qdrant", partial(pytest.fail, "should not upsert"))

    code = categorize.categorize_chats(str(path), out=str(out_path), no_qdrant=False)
    assert code == 0

    plan = json.loads(out_path.read_text())
    assert plan["clusters"] and plan["clusters"][0]["cohesion_text"] == 1.0


def test_categorize_chats_handles_failures_and_fallback(tmp_path, monkeypatch, capsys):
    """Qdrant failures and LLM errors trigger graceful fallbacks."""

    path = tmp_path / "input.json"
    chats = [
        _conversation("   ", id="chat-10", title="   "),
        _conversation("backup", id="chat-11", title="Backup"),
    ]
    path.write_text(json.dumps(chats))
    out_path = tmp_path / "plan.json"

    monkeypatch.setattr(categorize, "get_inference_client", lambda: SimpleNamespace())
    monkeypatch.setattr(categorize, "get_embedding_client", lambda: SimpleNamespace())
    monkeypatch.setattr(categorize, "embed_chats_with_retry", lambda client, texts, batch_size=96: np.ones((2, 2), dtype=float))
    monkeypatch.setattr(categorize, "cluster_embeddings", lambda vectors, eps_cosine, min_samples: np.array([0, 0]))
    monkeypatch.setattr(categorize, "cluster_text_cohesion", lambda vectors, labels_mapped, cid: 0.0)
    monkeypatch.setattr(categorize, "temporal_cohesion", lambda members, time_decay_days: 0.0)
    def failing_label(client, clusters):
        clusters[99] = []  # Force an empty cluster to exercise the fallback "else" branch.
        raise RuntimeError("llm")

    monkeypatch.setattr(categorize, "label_clusters_with_llm", failing_label)
    monkeypatch.setattr(categorize, "get_qdrant_client_with_timeout", lambda: (_ for _ in ()).throw(RuntimeError("qdrant")))

    code = categorize.categorize_chats(str(path), out=str(out_path), no_qdrant=False)
    assert code == 0

    plan = json.loads(out_path.read_text())
    assert plan["proposed_moves"] == []
    assert set(plan["skipped"]["clusters_low_confidence"]) == {"0", "99"}
    captured = capsys.readouterr().out
    assert "Continuing without Qdrant" in captured
    assert "Using fallback cluster labels" in captured


def test_categorize_chats_warns_on_qdrant_cache_failure(tmp_path, monkeypatch, capsys):
    """Failures during cache fetch should fall back to embedding all chats."""

    path = tmp_path / "cache-fail.json"
    chats = [
        _conversation("Topic A", id="chat-1", title="Topic A"),
        _conversation("Topic B", id="chat-2", title="Topic B"),
    ]
    path.write_text(json.dumps(chats))
    out_path = tmp_path / "plan.json"

    monkeypatch.setattr(categorize, "get_qdrant_client_with_timeout", lambda: SimpleNamespace())

    def _raise_fetch(*args, **kwargs):
        raise RuntimeError("fetch boom")

    monkeypatch.setattr(categorize, "fetch_existing_embeddings_from_qdrant", _raise_fetch)
    monkeypatch.setattr(categorize, "get_embedding_client", lambda: SimpleNamespace())
    monkeypatch.setattr(
        categorize,
        "embed_chats_with_retry",
        lambda client, texts, batch_size=96: np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float),
    )
    monkeypatch.setattr(categorize, "cluster_embeddings", lambda vectors, eps_cosine, min_samples: np.array([0, 0]))
    monkeypatch.setattr(categorize, "cluster_text_cohesion", lambda vectors, labels_mapped, cid: 0.5)
    monkeypatch.setattr(categorize, "temporal_cohesion", lambda members, time_decay_days: 0.5)
    monkeypatch.setattr(categorize, "get_inference_client", lambda: SimpleNamespace())
    monkeypatch.setattr(
        categorize,
        "label_clusters_with_llm",
        lambda client, clusters: {
            0: {
                "label": "Recovered",
                "project_folder_slug": "recovered",
                "project_title": "Recovered",
                "confidence_model": 0.7,
            }
        },
    )

    qdrant_calls: list[tuple[str, object]] = []

    def _record_ensure(client, name, size):
        qdrant_calls.append(("ensure", size))

    def _record_upsert(client, name, chats_subset, vectors):
        qdrant_calls.append(("upsert", len(chats_subset)))

    monkeypatch.setattr(categorize, "ensure_qdrant_collection", _record_ensure)
    monkeypatch.setattr(categorize, "upsert_to_qdrant", _record_upsert)

    code = categorize.categorize_chats(str(path), out=str(out_path), no_qdrant=False)
    assert code == 0
    captured = capsys.readouterr().out
    assert "cache were empty" in captured
    assert qdrant_calls == [("ensure", 2), ("upsert", 2)]


def test_categorize_chats_raises_when_embeddings_missing(tmp_path, monkeypatch):
    """If embeddings remain missing after retries the pipeline aborts."""

    path = tmp_path / "missing.json"
    chats = [
        _conversation("Topic A", id="chat-1", title="Topic A"),
        _conversation("Topic B", id="chat-2", title="Topic B"),
    ]
    path.write_text(json.dumps(chats))

    class NonStoringDict(dict):
        def __setitem__(self, key, value):
            return None

    monkeypatch.setattr(categorize, "get_qdrant_client_with_timeout", lambda: SimpleNamespace())
    monkeypatch.setattr(categorize, "fetch_existing_embeddings_from_qdrant", lambda *args, **kwargs: NonStoringDict())
    monkeypatch.setattr(categorize, "get_embedding_client", lambda: SimpleNamespace())
    monkeypatch.setattr(
        categorize,
        "embed_chats_with_retry",
        lambda client, texts, batch_size=96: np.ones((2, 2), dtype=float),
    )

    with pytest.raises(RuntimeError) as excinfo:
        categorize.categorize_chats(str(path), out=str(path.with_suffix(".out")), no_qdrant=False)

    assert "Missing embeddings" in str(excinfo.value)


def test_categorize_chats_handles_upsert_failure(tmp_path, monkeypatch, capsys):
    """Upsert errors should warn but not abort processing."""

    path = tmp_path / "upsert.json"
    chats = [
        _conversation("Topic A", id="chat-1", title="Topic A"),
        _conversation("Topic B", id="chat-2", title="Topic B"),
    ]
    path.write_text(json.dumps(chats))
    out_path = tmp_path / "plan.json"

    monkeypatch.setattr(categorize, "get_qdrant_client_with_timeout", lambda: SimpleNamespace())
    monkeypatch.setattr(categorize, "fetch_existing_embeddings_from_qdrant", lambda *args, **kwargs: {})
    monkeypatch.setattr(categorize, "get_embedding_client", lambda: SimpleNamespace())
    monkeypatch.setattr(
        categorize,
        "embed_chats_with_retry",
        lambda client, texts, batch_size=96: np.ones((2, 2), dtype=float),
    )
    monkeypatch.setattr(categorize, "cluster_embeddings", lambda vectors, eps_cosine, min_samples: np.array([0, 0]))
    monkeypatch.setattr(categorize, "cluster_text_cohesion", lambda vectors, labels_mapped, cid: 0.4)
    monkeypatch.setattr(categorize, "temporal_cohesion", lambda members, time_decay_days: 0.4)
    monkeypatch.setattr(categorize, "get_inference_client", lambda: SimpleNamespace())
    monkeypatch.setattr(
        categorize,
        "label_clusters_with_llm",
        lambda client, clusters: {
            0: {
                "label": "Upsert",
                "project_folder_slug": "upsert",
                "project_title": "Upsert",
                "confidence_model": 0.7,
            }
        },
    )

    monkeypatch.setattr(categorize, "ensure_qdrant_collection", lambda *args, **kwargs: None)

    def _fail_upsert(*args, **kwargs):
        raise RuntimeError("upsert boom")

    monkeypatch.setattr(categorize, "upsert_to_qdrant", _fail_upsert)

    code = categorize.categorize_chats(str(path), out=str(out_path), no_qdrant=False)
    assert code == 0
    captured = capsys.readouterr().out
    assert "upsert boom" in captured
    assert "Continuing without Qdrant persistence" in captured
def test_categorize_respects_limit(tmp_path, monkeypatch):
    """Setting ``limit`` should truncate the processed chats."""

    path = tmp_path / "limited.json"
    chats = [
        _conversation("first", id="chat-a"),
        _conversation("second", id="chat-b"),
    ]
    path.write_text(json.dumps(chats))
    out_path = tmp_path / "plan.json"

    recorded: list[int] = []
    monkeypatch.setattr(categorize, "get_inference_client", lambda: SimpleNamespace())
    monkeypatch.setattr(categorize, "get_embedding_client", lambda: SimpleNamespace())

    def fake_embed(client, texts, batch_size=96):
        recorded.append(len(texts))
        return np.zeros((len(texts), 2), dtype=float)

    monkeypatch.setattr(categorize, "embed_chats_with_retry", fake_embed)
    monkeypatch.setattr(categorize, "cluster_embeddings", lambda vectors, eps_cosine, min_samples: np.full(len(vectors), -1))
    monkeypatch.setattr(categorize, "label_clusters_with_llm", lambda client, clusters: {})

    categorize.categorize_chats(str(path), out=str(out_path), no_qdrant=True, limit=1)
    assert recorded == [1]


def test_categorize_prints_summary_for_many_moves(tmp_path, monkeypatch, capsys):
    """Large numbers of proposed moves trigger the truncated summary output."""

    path = tmp_path / "bulk.json"
    items = []
    labels: list[int] = []
    embeddings: list[list[float]] = []
    for idx in range(51):
        items.append(_conversation(f"topic {idx}a", id=f"chat-{idx*2}", title=f"Topic {idx} A"))
        items.append(_conversation(f"topic {idx}b", id=f"chat-{idx*2+1}", title=f"Topic {idx} B"))
        labels.extend([idx, idx])
        embeddings.extend([[float(idx), 0.1], [float(idx), 0.2]])
    path.write_text(json.dumps(items))
    out_path = tmp_path / "plan.json"

    monkeypatch.setattr(categorize, "get_inference_client", lambda: SimpleNamespace())
    monkeypatch.setattr(categorize, "get_embedding_client", lambda: SimpleNamespace())
    monkeypatch.setattr(categorize, "embed_chats_with_retry", lambda client, texts, batch_size=96: np.array(embeddings, dtype=float))
    monkeypatch.setattr(categorize, "cluster_embeddings", lambda vectors, eps_cosine, min_samples: np.array(labels))
    monkeypatch.setattr(categorize, "cluster_text_cohesion", lambda vectors, labels_mapped, cid: 0.9)
    monkeypatch.setattr(categorize, "temporal_cohesion", lambda members, time_decay_days: 0.9)

    def fake_labels(client, clusters):
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


def test_categorize_main_cli(monkeypatch, tmp_path):
    """The module CLI forwards arguments to ``categorize_chats``."""

    path = tmp_path / "source.json"
    path.write_text("[]")
    out_path = tmp_path / "result.json"

    called: dict[str, object] = {}

    def fake_categorize(**kwargs):
        called.update(kwargs)
        return 5

    monkeypatch.setattr(categorize, "categorize_chats", lambda **kwargs: fake_categorize(**kwargs))

    exit_code = categorize.main([
        "--conversations-json",
        str(path),
        "--out",
        str(out_path),
        "--collection",
        "col",
        "--no-qdrant",
        "--eps-cosine",
        "0.5",
        "--min-samples",
        "3",
        "--confidence-threshold",
        "0.4",
        "--time-weight",
        "0.7",
        "--limit",
        "10",
    ])

    assert exit_code == 5
    assert called["no_qdrant"] is True
    assert called["eps_cosine"] == 0.5
    assert called["min_samples"] == 3
    assert called["confidence_threshold"] == 0.4
    assert called["time_weight"] == 0.7
    assert called["limit"] == 10


def test_entry_main_cli(monkeypatch, tmp_path):
    """Top-level ``main.py`` delegates to the categorization function."""

    path = tmp_path / "src.json"
    path.write_text("[]")
    out_path = tmp_path / "dest.json"

    called: dict[str, object] = {}

    def fake_categorize(**kwargs):
        called.update(kwargs)
        return 7

    monkeypatch.setattr(entry_main, "categorize_chats", lambda **kwargs: fake_categorize(**kwargs))

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


def test_categorize_module_main_block(tmp_path, monkeypatch):
    """Executing the module as ``__main__`` should exit with the wrapped status code."""

    path = tmp_path / "input.json"
    path.write_text(json.dumps([_conversation("skip", id="chat-z", metadata={"project_id": "x"})]))

    argv = ["categorize", "--conversations-json", str(path), "--out", str(tmp_path / "plan.json"), "--no-qdrant"]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc:
        runpy.run_module("GptCategorize.categorize", run_name="__main__")

    assert exc.value.code == 0


def test_entry_main_module_block(tmp_path, monkeypatch):
    """Running ``python -m main`` exits with the CLI return code."""

    path = tmp_path / "input.json"
    path.write_text(json.dumps([_conversation("skip", id="chat-y", metadata={"project_id": "x"})]))

    argv = ["main", "--conversations-json", str(path), "--out", str(tmp_path / "plan.json"), "--no-qdrant"]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc:
        runpy.run_module("main", run_name="__main__")

    assert exc.value.code == 0


def test_categorize_module_keyboard_interrupt(tmp_path, monkeypatch, capsys):
    """Keyboard interrupts should surface after printing a message."""

    path = tmp_path / "input.json"
    path.write_text("[]")

    argv = ["categorize", "--conversations-json", str(path), "--out", str(tmp_path / "plan.json"), "--no-qdrant"]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()))

    with pytest.raises(KeyboardInterrupt):
        runpy.run_module("GptCategorize.categorize", run_name="__main__")

    out = capsys.readouterr().out
    assert "Interrupted." in out


def test_entry_main_keyboard_interrupt(monkeypatch, tmp_path, capsys):
    """The ``main.py`` script should exit with status 1 on Ctrl+C."""

    path = tmp_path / "input.json"
    path.write_text("[]")

    argv = ["main", "--conversations-json", str(path), "--out", str(tmp_path / "plan.json"), "--no-qdrant"]
    monkeypatch.setattr(sys, "argv", argv)

    def raise_interrupt(**kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(categorize, "categorize_chats", raise_interrupt)

    with pytest.raises(SystemExit) as exc:
        runpy.run_module("main", run_name="__main__")

    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "Interrupted." in out
