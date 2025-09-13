#!/usr/bin/env python3
from __future__ import annotations
import argparse
import datetime as dt
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

# scikit-learn
from sklearn.cluster import DBSCAN, KMeans  # type: ignore[import-untyped]
from sklearn.preprocessing import normalize  # type: ignore[import-untyped]

# OpenAI + Qdrant
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# ===============================
# Constants (edit to your setup)
# ===============================
# Separate endpoints and keys for inference vs. embeddings
INFERENCE_API_BASE = os.environ.get("INFERENCE_API_BASE", "https://api.openai.com/v1")
INFERENCE_API_KEY = os.environ.get("INFERENCE_API_KEY", "YOUR_INFERENCE_API_KEY_HERE")
EMBEDDING_API_BASE = os.environ.get("EMBEDDING_API_BASE", "https://api.openai.com/v1")
EMBEDDING_API_KEY = os.environ.get("EMBEDDING_API_KEY", "YOUR_EMBEDDING_API_KEY_HERE")

INFERENCE_MODEL = os.environ.get("INFERENCE_MODEL", "gpt-5-latest")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large")

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", None)  # usually not needed for local
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "chatgpt_titles")

# Temporal cohesion timescale (days). This is not exposed as a CLI flag; tweak here if needed.
TIME_DECAY_DAYS = float(os.environ.get("TIME_DECAY_DAYS", 30))

# Timeout configurations
OPENAI_TIMEOUT = float(os.environ.get("OPENAI_TIMEOUT", 300))  # 5 minutes
QDRANT_TIMEOUT = float(os.environ.get("QDRANT_TIMEOUT", 600))  # 10 minutes
QDRANT_BATCH_SIZE = int(os.environ.get("QDRANT_BATCH_SIZE", 100))  # batch size for upserts
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", 3))
RETRY_DELAY = float(os.environ.get("RETRY_DELAY", 2.0))


# ===============================
# Logging Configuration
# ===============================
VERBOSE_LOGGING = os.environ.get("VERBOSE_LOGGING", "false").lower() in ("true", "1", "yes")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "DEBUG" if VERBOSE_LOGGING else "INFO")


def setup_logging() -> None:
    """Configure comprehensive logging including HTTP debugging."""
    # Validate LOG_LEVEL before using it
    valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
    if LOG_LEVEL.upper() not in valid_levels:
        print(f"Warning: Invalid LOG_LEVEL '{LOG_LEVEL}', defaulting to INFO")
        level = logging.INFO
    else:
        level = getattr(logging, LOG_LEVEL.upper())

    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    if VERBOSE_LOGGING:
        # Enable HTTP debugging for requests/httpx
        logging.getLogger("httpx").setLevel(logging.DEBUG)
        logging.getLogger("httpcore").setLevel(logging.DEBUG)
        logging.getLogger("openai").setLevel(logging.DEBUG)
        logging.getLogger("qdrant_client").setLevel(logging.DEBUG)

        # Add urllib3 debug logging for even more detail
        try:
            import urllib3
            urllib3.disable_warnings()
            logging.getLogger("urllib3").setLevel(logging.DEBUG)
        except ImportError:
            pass

        print("🔍 Verbose logging enabled - HTTP requests/responses will be logged")
    else:
        # Reduce noise from third-party libraries
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)


# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)


# Notes
# -----
# • "Already in a project" detection: we look for any of these fields on the conversation record
#   (case-insensitive, nested OK): project_id, workspace_id, folder_id, project, workspace, folder.
# • Embedding dimension for text-embedding-3-large is 3072.
# • DBSCAN runs in Euclidean space over L2-normalized embeddings (≈ cosine distance). We convert the
#   cosine epsilon into an equivalent Euclidean epsilon automatically.
# • Temporal cohesion uses an exponential decay over pairwise creation-time gaps:
#     sim_ij = exp(- |Δt_days| / TIME_DECAY_DAYS )
#   averaged over all pairs. The default TIME_DECAY_DAYS is 30.
# • If DBSCAN finds nothing useful, we fall back to KMeans (k ≈ sqrt(N)).
# • Model prompts are constrained to return strict JSON that we parse.


# Caveats
# -------
# • There is currently no official public API to list personal ChatGPT conversations.
#   Data export is the reliable route; this script reads the JSON directly.
# • This script never attempts the actual move.


# =====================
# Data model / helpers
# =====================
@dataclass
class Chat:
    id: str
    title: str
    create_time: Optional[float] = None
    update_time: Optional[float] = None
    project_id: Optional[str] = None  # any value means: already in a project
    raw: Optional[Dict[str, Any]] = None  # keep the raw object for debugging


def now_iso() -> str:
    return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat()


def to_epoch(ts: Any) -> Optional[float]:
    """Best-effort parse of various timestamp shapes to epoch seconds."""
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        # Heuristic: very large values are probably milliseconds.
        if ts > 10_000_000_000:
            return float(ts) / 1000.0
        return float(ts)
    if isinstance(ts, str):
        try:
            return dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
        except Exception:
            pass
        try:
            return float(ts)
        except Exception:
            return None
    return None


# =====================
# Source: conversations.json
# =====================


def _looks_like_conversation(obj: Dict[str, Any]) -> bool:
    return (
        isinstance(obj, dict)
        and ("title" in obj or obj.get("conversation", {}).get("title"))
        and ("id" in obj or obj.get("conversation_id") or obj.get("conversation", {}).get("id"))
    )


def _detect_projectish(obj: Dict[str, Any]) -> Optional[str]:
    """Return a project-ish identifier if present (workspace/folder/project IDs or names)."""
    keys = [
        "project_id",
        "workspace_id",
        "folder_id",
        "project",
        "workspace",
        "folder",
    ]
    # direct keys
    for k in keys:
        if k in obj and obj[k]:
            return str(obj[k])
    # nested in metadata or conversation
    for container_key in ("metadata", "conversation"):
        sub = obj.get(container_key, {})
        if isinstance(sub, dict):
            for k in keys:
                if k in sub and sub[k]:
                    return str(sub[k])
    return None


def extract_chats_from_json_blob(data: Any) -> List[Chat]:
    out: List[Chat] = []

    def convert_one(o: Dict[str, Any]) -> None:
        if not isinstance(o, dict):
            return
        title = o.get("title") or o.get("conversation", {}).get("title") or "(untitled)"
        cid = o.get("id") or o.get("conversation_id") or o.get("conversation", {}).get("id") or o.get("uuid")
        if not cid:
            return
        ct = to_epoch(o.get("create_time") or o.get("created_at") or o.get("conversation", {}).get("create_time"))
        ut = to_epoch(o.get("update_time") or o.get("updated_at") or o.get("conversation", {}).get("update_time"))
        projectish = _detect_projectish(o)
        out.append(
            Chat(
                id=str(cid),
                title=str(title),
                create_time=ct,
                update_time=ut,
                project_id=projectish,
                raw=o,
            )
        )

    if isinstance(data, list):
        for item in data:
            if _looks_like_conversation(item):
                convert_one(item)
            elif isinstance(item, dict):
                for v in item.values():
                    if _looks_like_conversation(v):
                        convert_one(v)
    elif isinstance(data, dict):
        for key in ("conversations", "items", "data"):
            if isinstance(data.get(key), list):
                for item in data[key]:
                    if _looks_like_conversation(item):
                        convert_one(item)
        if _looks_like_conversation(data):
            convert_one(data)
    return out


def load_chats_from_conversations_json(path: str) -> List[Chat]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(p, "r", encoding="utf-8") as fh:
        blob = fh.read()
    try:
        data = json.loads(blob)
    except Exception:
        # Try NDJSON (one JSON per line)
        lines = [json.loads(line) for line in blob.splitlines() if line.strip()]
        data = lines

    chats = extract_chats_from_json_blob(data)

    # Deduplicate by id (prefer latest update_time)
    dedup: Dict[str, Chat] = {}
    for c in chats:
        if c.id not in dedup or (c.update_time or 0) > (dedup[c.id].update_time or 0):
            dedup[c.id] = c
    return list(dedup.values())


# =====================
# OpenAI Clients & Embeddings
# =====================


def get_inference_client() -> OpenAI:
    if not INFERENCE_API_KEY or INFERENCE_API_KEY == "YOUR_INFERENCE_API_KEY_HERE":
        raise RuntimeError("INFERENCE_API_KEY is not set. Edit the constant or export it in env.")

    logger.info(f"Creating inference client - API Base: {INFERENCE_API_BASE}")
    logger.debug(f"Inference API Key: {INFERENCE_API_KEY[:8]}...{INFERENCE_API_KEY[-4:]}")

    client = OpenAI(api_key=INFERENCE_API_KEY, base_url=INFERENCE_API_BASE, timeout=OPENAI_TIMEOUT)
    logger.info(f"Inference client created successfully with timeout: {OPENAI_TIMEOUT}s")
    return client


def get_embedding_client() -> OpenAI:
    if not EMBEDDING_API_KEY or EMBEDDING_API_KEY == "YOUR_EMBEDDING_API_KEY_HERE":
        raise RuntimeError("EMBEDDING_API_KEY is not set. Edit the constant or export it in env.")

    logger.info(f"Creating embedding client - API Base: {EMBEDDING_API_BASE}")
    logger.debug(f"Embedding API Key: {EMBEDDING_API_KEY[:8]}...{EMBEDDING_API_KEY[-4:]}")

    client = OpenAI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_API_BASE, timeout=OPENAI_TIMEOUT)
    logger.info(f"Embedding client created successfully with timeout: {OPENAI_TIMEOUT}s")
    return client


def embed_titles_with_retry(embed_client: OpenAI, titles: List[str], batch_size: int = 96) -> np.ndarray:
    """Embed titles with retry logic and timeout handling."""
    logger.info(f"Starting embedding process for {len(titles)} titles in batches of {batch_size}")
    vecs: List[List[float]] = []

    for i in tqdm(range(0, len(titles), batch_size), desc="Embedding titles"):
        batch = titles[i : i + batch_size]
        logger.debug(f"Processing batch {i//batch_size + 1}: {len(batch)} titles")

        # Retry logic for embedding API calls
        for attempt in range(MAX_RETRIES):
            try:
                logger.debug(f"Batch {i//batch_size + 1}, attempt {attempt + 1}: Calling embeddings API")
                logger.debug(f"Model: {EMBEDDING_MODEL}, Input size: {len(batch)}")

                resp = embed_client.embeddings.create(model=EMBEDDING_MODEL, input=batch)

                logger.debug(f"Batch {i//batch_size + 1}: API response received, {len(resp.data)} embeddings")
                vecs.extend([d.embedding for d in resp.data])
                break  # Success, exit retry loop

            except Exception as e:
                logger.error(f"Batch {i//batch_size + 1}, attempt {attempt + 1} failed: {type(e).__name__}: {e}")

                # Safely log HTTP details if available
                response = getattr(e, 'response', None)
                if response is not None:
                    logger.error(f"HTTP Response Status: {getattr(response, 'status_code', 'unknown')}")
                    logger.error(f"HTTP Response Headers: {getattr(response, 'headers', {})}")

                request = getattr(e, 'request', None)
                if request is not None:
                    logger.error(f"HTTP Request URL: {getattr(request, 'url', 'unknown')}")
                    logger.error(f"HTTP Request Method: {getattr(request, 'method', 'unknown')}")

                if attempt == MAX_RETRIES - 1:  # Last attempt
                    logger.critical(f"Failed to embed batch {i//batch_size + 1} after {MAX_RETRIES} attempts")
                    print(f"\nFailed to embed batch {i//batch_size + 1} after {MAX_RETRIES} attempts: {e}")
                    raise
                else:
                    wait_time = RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Retrying batch {i//batch_size + 1} in {wait_time}s")
                    print(f"\nEmbedding attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)

    arr = np.array(vecs, dtype=np.float32)
    logger.info(f"Embedding completed: {arr.shape[0]} vectors with {arr.shape[1]} dimensions")

    if arr.ndim != 2 or arr.shape[1] == 0:
        raise RuntimeError("Empty embeddings returned.")
    return arr


# Keep original function for backward compatibility
def embed_titles(embed_client: OpenAI, titles: List[str], batch_size: int = 96) -> np.ndarray:
    return embed_titles_with_retry(embed_client, titles, batch_size)


# =====================
# Qdrant Persistence
# =====================


def get_qdrant_client_with_timeout() -> QdrantClient:
    """Create Qdrant client with proper timeout configuration."""
    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=int(QDRANT_TIMEOUT)
    )


def ensure_qdrant_collection(client: QdrantClient, name: str, vector_size: int) -> None:
    """Ensure Qdrant collection exists with retry logic."""
    for attempt in range(MAX_RETRIES):
        try:
            client.get_collection(name)
            return  # Collection exists, success
        except Exception as e:
            if "Not found" in str(e) or "404" in str(e):
                # Collection doesn't exist, try to create it
                try:
                    client.recreate_collection(
                        collection_name=name,
                        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                    )
                    print(f"Created Qdrant collection: {name}")
                    return
                except Exception as create_e:
                    if attempt == MAX_RETRIES - 1:
                        print(f"Failed to create Qdrant collection after {MAX_RETRIES} attempts: {create_e}")
                        raise
                    else:
                        wait_time = RETRY_DELAY * (2 ** attempt)
                        print(f"Collection creation attempt {attempt + 1} failed, retrying in {wait_time}s: {create_e}")
                        time.sleep(wait_time)
            else:
                # Other error, retry
                if attempt == MAX_RETRIES - 1:
                    print(f"Failed to check Qdrant collection after {MAX_RETRIES} attempts: {e}")
                    raise
                else:
                    wait_time = RETRY_DELAY * (2 ** attempt)
                    print(f"Collection check attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)


def upsert_to_qdrant_batched(client: QdrantClient, name: str, chats: List[Chat], vectors: np.ndarray) -> None:
    """Upsert vectors to Qdrant in batches with retry logic to prevent timeouts."""
    total_points = len(chats)

    if total_points == 0:
        print("No points to upsert to Qdrant")
        return

    print(f"Upserting {total_points} points to Qdrant in batches of {QDRANT_BATCH_SIZE}")

    for start_idx in tqdm(range(0, total_points, QDRANT_BATCH_SIZE), desc="Upserting to Qdrant"):
        end_idx = min(start_idx + QDRANT_BATCH_SIZE, total_points)
        batch_chats = chats[start_idx:end_idx]
        batch_vectors = vectors[start_idx:end_idx]

        # Prepare batch points
        points: List[PointStruct] = []
        for c, v in zip(batch_chats, batch_vectors):
            payload = {
                "chat_id": c.id,
                "title": c.title,
                "create_time": c.create_time,
                "update_time": c.update_time,
            }
            points.append(PointStruct(id=c.id, vector=v.tolist(), payload=payload))

        # Retry logic for batch upsert
        for attempt in range(MAX_RETRIES):
            try:
                client.upsert(collection_name=name, points=points)
                break  # Success, exit retry loop
            except Exception as e:
                if attempt == MAX_RETRIES - 1:  # Last attempt
                    batch_num = start_idx//QDRANT_BATCH_SIZE + 1
                    print(f"\nFailed to upsert batch {batch_num} after {MAX_RETRIES} attempts: {e}")
                    raise
                else:
                    wait_time = RETRY_DELAY * (2 ** attempt)
                    batch_num = start_idx//QDRANT_BATCH_SIZE + 1
                    print(f"\nUpsert batch {batch_num} attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)

    print(f"Successfully upserted {total_points} points to Qdrant collection '{name}'")


# Keep original function for backward compatibility but with improved implementation
def upsert_to_qdrant(client: QdrantClient, name: str, chats: List[Chat], vectors: np.ndarray) -> None:
    return upsert_to_qdrant_batched(client, name, chats, vectors)


# =====================
# Clustering & Cohesion
# =====================


def cosine_to_euclid_eps(cos_eps: float) -> float:
    """For L2-normalized vectors, Euclid^2 ≈ 2 * (1 - cos_sim) = 2 * cos_dist.
    Convert a cosine distance epsilon into an Euclidean epsilon for DBSCAN.
    """
    cos_dist = max(cos_eps, 0.0)
    return math.sqrt(2.0 * cos_dist)


def cluster_embeddings(vectors: np.ndarray, eps_cosine: float, min_samples: int) -> np.ndarray:
    v = normalize(vectors, norm="l2")
    eps_euclid = cosine_to_euclid_eps(eps_cosine)
    db = DBSCAN(eps=eps_euclid, min_samples=min_samples, metric="euclidean")
    labels = db.fit_predict(v)
    if (labels >= 0).sum() == 0:
        # Fallback: naive KMeans with k from sqrt(N)
        n = len(v)
        k = max(2, int(math.sqrt(max(n, 1))))
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(v)
    return np.array(labels, dtype=int)


def cluster_text_cohesion(vectors: np.ndarray, labels_mapped: np.ndarray, cid: int) -> float:
    """Average pairwise cosine similarity within cluster (0..1)."""
    idx = np.where(labels_mapped == cid)[0]
    if len(idx) < 2:
        return 0.0
    sub = normalize(vectors[idx], norm="l2")
    sims = sub @ sub.T  # cosine sim for normalized vectors
    m = sims.shape[0]
    tri = sims[np.triu_indices(m, k=1)]
    return float(np.clip(np.mean(tri), 0.0, 1.0))


def temporal_cohesion(members: List[Chat], time_decay_days: float = TIME_DECAY_DAYS) -> float:
    """Compute time-based cohesion (0..1) from creation-time gaps.

    We map pairwise |Δt_days| into a similarity via an exponential kernel:
        sim_ij = exp(- |Δt_days| / time_decay_days )
    and average across all pairs (diagonal excluded). If fewer than 2 timestamps are
    available, we return 0.5 to avoid penalizing clusters with missing metadata.
    """
    # Prefer create_time, fall back to update_time
    times: List[float] = []
    for c in members:
        t = c.create_time if c.create_time is not None else c.update_time
        if t is not None:
            times.append(float(t))
    if len(times) < 2:
        return 0.5
    times = sorted(times)
    n = len(times)
    # Pairwise average using a simple double loop (clusters are small; fine for speed)
    sim_sum = 0.0
    count = 0
    day = 86400.0
    for i in range(n):
        for j in range(i + 1, n):
            dt_days = abs(times[j] - times[i]) / day
            sim = math.exp(-dt_days / max(1e-6, time_decay_days))
            sim_sum += sim
            count += 1
    return float(np.clip(sim_sum / max(1, count), 0.0, 1.0))


# =====================
# Labeling via LLM
# =====================


def label_clusters_with_llm(infer_client: OpenAI, clusters: Dict[int, List[Chat]]) -> Dict[int, Dict[str, Any]]:
    """Ask the LLM to produce {cluster_id: {label, project_folder_slug, project_title, confidence}}."""
    logger.info(f"Starting LLM labeling for {len(clusters)} clusters")

    serializable = [
        {"cluster_id": int(cid), "titles": [c.title for c in chats[:12]]} for cid, chats in clusters.items()
    ]
    sys_prompt = (
        "You are labeling clusters of ChatGPT chat titles. "
        "Return STRICT JSON: a list of objects with keys: cluster_id (int), label (<=3 words), "
        "project_folder_slug (kebab-case), project_title (short), confidence (0..1). "
        "Labels should be practical (e.g., 'DevOps Incidents', 'Kubernetes', 'Prompt Engineering', 'Golang'). "
        "Do not invent new chats, and do not include any commentary."
    )
    user_text = json.dumps(serializable, ensure_ascii=False)

    logger.debug(f"LLM Request - Model: {INFERENCE_MODEL}")
    logger.debug(f"LLM Request - System prompt length: {len(sys_prompt)} chars")
    logger.debug(f"LLM Request - User text length: {len(user_text)} chars")
    if VERBOSE_LOGGING:
        logger.debug(f"LLM Request - Full user text: {user_text[:500]}...")

    content = None

    # Retry logic for LLM inference
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"LLM inference attempt {attempt + 1}/{MAX_RETRIES}")
            logger.debug(f"Making API call to {INFERENCE_API_BASE} with model {INFERENCE_MODEL}")

            resp = infer_client.chat.completions.create(
                model=INFERENCE_MODEL,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_text},
                ],
                temperature=0.2,
            )
            content = resp.choices[0].message.content
            logger.info(f"LLM inference successful on attempt {attempt + 1}")
            logger.debug(f"LLM response length: {len(content) if content else 0} chars")
            break  # Success, exit retry loop

        except Exception as e:
            logger.error(f"LLM inference attempt {attempt + 1} failed: {type(e).__name__}: {e}")

            # Detailed HTTP debugging
            response = getattr(e, 'response', None)
            if response is not None:
                logger.error(f"HTTP Response Status: {getattr(response, 'status_code', 'unknown')}")
                logger.error(f"HTTP Response Headers: {getattr(response, 'headers', {})}")
                logger.error(f"HTTP Response Text: {getattr(response, 'text', 'unavailable')}")

            request = getattr(e, 'request', None)
            if request is not None:
                logger.error(f"HTTP Request URL: {getattr(request, 'url', 'unknown')}")
                logger.error(f"HTTP Request Method: {getattr(request, 'method', 'unknown')}")
                logger.error(f"HTTP Request Headers: {getattr(request, 'headers', {})}")

            # Log connection details
            if "Connection" in str(e) or "Name or service not known" in str(e):
                logger.error(f"Connection issue detected - Base URL: {INFERENCE_API_BASE}")
                logger.error(f"API Key prefix: {INFERENCE_API_KEY[:10] if INFERENCE_API_KEY else 'None'}...")

            if attempt == MAX_RETRIES - 1:  # Last attempt
                logger.critical(f"Failed to get cluster labels after {MAX_RETRIES} attempts: {e}")
                print(f"\nFailed to get cluster labels after {MAX_RETRIES} attempts: {e}")

                # Try fallback approach
                try:
                    logger.info("Attempting fallback LLM call")
                    cmpl = infer_client.chat.completions.create(
                        model=INFERENCE_MODEL,
                        messages=[
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": user_text},
                        ],
                        temperature=0.2,
                    )
                    content = cmpl.choices[0].message.content
                    logger.info("Fallback LLM call successful")
                    break
                except Exception as fallback_e:
                    logger.error(f"Fallback also failed: {type(fallback_e).__name__}: {fallback_e}")
                    print(f"Fallback also failed: {fallback_e}")
                    raise e  # Raise original exception
            else:
                wait_time = RETRY_DELAY * (2 ** attempt)
                logger.warning(f"Retrying LLM inference in {wait_time}s")
                print(f"\nLLM inference attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                time.sleep(wait_time)

    if not content:
        raise RuntimeError("No content returned from the model for cluster labels.")

    logger.debug("Parsing LLM response JSON")
    try:
        data = json.loads(content)
        logger.debug(f"Successfully parsed JSON with {len(data)} items")
    except Exception as e:
        logger.warning(f"JSON parse failed, attempting regex extraction: {e}")
        m = re.search(r"\[.*\]", content, flags=re.DOTALL)
        if not m:
            logger.error(f"No JSON found in response: {content[:200]}...")
            raise RuntimeError(f"Model did not return JSON: {content}") from e
        data = json.loads(m.group(0))
        logger.debug(f"Regex extraction successful, parsed {len(data)} items")

    results: Dict[int, Dict[str, Any]] = {}
    successful_labels = 0
    for item in data:
        try:
            cid = int(item["cluster_id"])  # may raise
            label = str(item["label"]).strip()
            slug = str(item["project_folder_slug"]).strip()
            ptitle = str(item["project_title"]).strip()
            conf = float(item.get("confidence", 0.6))
            results[cid] = {
                "label": label,
                "project_folder_slug": slug,
                "project_title": ptitle or label,
                "confidence_model": max(0.0, min(1.0, conf)),
            }
            successful_labels += 1
        except Exception as parse_e:
            logger.warning(f"Failed to parse item {item}: {parse_e}")
            continue

    logger.info(f"LLM labeling completed: {successful_labels}/{len(data)} labels parsed successfully")
    return results


# =====================
# Core Function
# =====================


def categorize_chats(
    conversations_json: str,
    out: str = "provisional_move_plan.json",
    collection: str = QDRANT_COLLECTION,
    no_qdrant: bool = False,
    eps_cosine: float = 0.25,
    min_samples: int = 2,
    confidence_threshold: float = 0.60,
    time_weight: float = 0.25,
    limit: int = 0,
) -> int:
    """Core function to categorize ChatGPT chats by title and emit a provisional move plan.

    Args:
        conversations_json: Path to the ChatGPT conversations JSON file
        out: Path to write the provisional plan JSON
        collection: Qdrant collection name for embeddings
        no_qdrant: Disable Qdrant persistence (embeddings kept only in-memory)
        eps_cosine: DBSCAN epsilon in cosine distance space (0..2). Lower = tighter clusters
        min_samples: DBSCAN min_samples (>=2 is sensible)
        confidence_threshold: Minimum combined confidence to propose moves
        time_weight: Weight (0..1) given to temporal cohesion when computing cluster cohesion
        limit: Optional: limit number of chats processed (debug)

    Returns:
        Exit code (0 for success)
    """
    # Clamp time-weight to [0,1]
    time_weight = max(0.0, min(1.0, float(time_weight)))

    # 1) Load chats from JSON
    chats_all = load_chats_from_conversations_json(conversations_json)
    source_desc = {
        "type": "conversations_json",
        "path": str(Path(conversations_json).resolve()),
    }

    # 2) Filter: skip ones already in a project
    skipped_already: List[str] = []
    chats: List[Chat] = []
    for c in chats_all:
        if c.project_id:
            skipped_already.append(c.id)
        else:
            chats.append(c)

    if limit and limit > 0:
        chats = chats[:limit]

    if not chats:
        print("No chats to process after filtering. Exiting.")
        plan = {
            "plan_generated_at": now_iso(),
            "source": source_desc,
            "parameters": {
                "eps_cosine": eps_cosine,
                "min_samples": min_samples,
                "confidence_threshold": confidence_threshold,
                "time_weight": time_weight,
                "time_decay_days": TIME_DECAY_DAYS,
            },
            "collections": {"qdrant_collection": collection},
            "clusters": [],
            "proposed_moves": [],
            "skipped": {
                "clusters_low_confidence": [],
                "singletons": [],
                "already_in_project": skipped_already,
            },
        }
        with open(out, "w", encoding="utf-8") as f:
            json.dump(plan, f, ensure_ascii=False, indent=2)
        return 0

    # 3) OpenAI clients
    infer_client = get_inference_client()
    embed_client = get_embedding_client()

    # 4) Embeddings
    titles = [c.title for c in chats]
    vectors = embed_titles(embed_client, titles)

    # 5) Optional: persist to Qdrant
    if not no_qdrant:
        try:
            qcli = get_qdrant_client_with_timeout()
            ensure_qdrant_collection(qcli, collection, vectors.shape[1])
            upsert_to_qdrant(qcli, collection, chats, vectors)
        except Exception as e:
            print(f"\nWarning: Qdrant operation failed: {e}")
            print("Continuing without Qdrant persistence (embeddings kept in-memory only)...")

    # 6) Cluster
    labels = cluster_embeddings(vectors, eps_cosine=eps_cosine, min_samples=min_samples)

    # Build contiguous cluster IDs (exclude -1 noise) and remap labels
    unique = sorted(set(int(x) for x in labels if int(x) != -1))
    cid_map = {old: i for i, old in enumerate(unique)}
    labels_mapped = np.array([cid_map.get(int(label), -1) for label in labels], dtype=int)

    clusters: Dict[int, List[Chat]] = {}
    singletons: List[str] = []
    for idx, c in enumerate(chats):
        lab = int(labels_mapped[idx])
        if lab == -1:
            singletons.append(c.id)
            continue
        clusters.setdefault(lab, []).append(c)

    # 7) Compute cohesion (text + time) and label with LLM
    text_cohesions: Dict[int, float] = {
        cid: cluster_text_cohesion(vectors, labels_mapped=labels_mapped, cid=cid) for cid in clusters
    }
    time_cohesions: Dict[int, float] = {
        cid: temporal_cohesion(members=members, time_decay_days=TIME_DECAY_DAYS) for cid, members in clusters.items()
    }

    # Try LLM labeling with graceful fallback
    try:
        llm_labels = label_clusters_with_llm(infer_client, clusters)
        print("✅ LLM labeling completed successfully")
    except Exception as e:
        print(f"⚠️ LLM labeling failed: {e}")
        print("Using fallback cluster labels (script will continue)")
        # Generate fallback labels based on most common words in titles
        llm_labels = {}
        for cid, members in clusters.items():
            # Simple fallback: use first few words from most common title
            titles = [m.title for m in members]
            if titles:
                first_title = titles[0]
                # Create a simple label from the first title
                label_words = first_title.split()[:3]
                label = " ".join(label_words) if label_words else f"Cluster {cid}"
                slug = "-".join(word.lower().strip(".,!?") for word in label_words if word.strip(".,!?"))
                slug = slug or f"cluster-{cid}"
            else:
                label = f"Cluster {cid}"
                slug = f"cluster-{cid}"

            llm_labels[cid] = {
                "label": label,
                "project_folder_slug": slug,
                "project_title": label,
                "confidence_model": 0.5,  # Lower confidence for fallback labels
            }

    # 8) Merge data into cluster descriptors
    cluster_descs: List[Dict[str, Any]] = []
    for cid, members in clusters.items():
        info = llm_labels.get(cid, {})
        t_text = float(text_cohesions.get(cid, 0.0))
        t_time = float(time_cohesions.get(cid, 0.5))
        cohesion_weighted = (1.0 - time_weight) * t_text + time_weight * t_time

        cdesc = {
            "cluster_id": cid,
            "label": info.get("label", f"Cluster {cid}"),
            "project_folder_slug": info.get("project_folder_slug", f"cluster-{cid}"),
            "project_title": info.get("project_title", info.get("label", f"Cluster {cid}")),
            "confidence_model": info.get("confidence_model", 0.5),
            "cohesion_text": round(t_text, 4),
            "cohesion_time": round(t_time, 4),
            "confidence_cohesion": round(float(cohesion_weighted), 4),
            "chats": [
                {
                    "id": m.id,
                    "title": m.title,
                    "create_time": m.create_time,
                    "update_time": m.update_time,
                }
                for m in members
            ],
        }
        combined_conf = 0.65 * cdesc["confidence_model"] + 0.35 * cdesc["confidence_cohesion"]
        cdesc["confidence_combined"] = round(float(combined_conf), 4)
        cluster_descs.append(cdesc)

    # 9) Proposed moves (only above threshold and >=2 items)
    proposed_moves: List[Dict[str, Any]] = []
    low_conf_cids: List[int] = []
    for cdesc in cluster_descs:
        if cdesc["confidence_combined"] >= confidence_threshold and len(cdesc["chats"]) >= 2:
            proposed_moves.append(
                {
                    "project_folder_slug": cdesc["project_folder_slug"],
                    "project_title": cdesc["project_title"],
                    "cluster_id": cdesc["cluster_id"],
                    "chats": [c["id"] for c in cdesc["chats"]],
                }
            )
        else:
            low_conf_cids.append(int(cdesc["cluster_id"]))

    plan = {
        "plan_generated_at": now_iso(),
        "source": source_desc,
        "parameters": {
            "eps_cosine": eps_cosine,
            "min_samples": min_samples,
            "confidence_threshold": confidence_threshold,
            "time_weight": time_weight,
            "time_decay_days": TIME_DECAY_DAYS,
        },
        "collections": {"qdrant_collection": collection},
        "clusters": cluster_descs,  # type: ignore[dict-item]
        "proposed_moves": proposed_moves,  # type: ignore[dict-item]
        "skipped": {
            "clusters_low_confidence": [str(cid) for cid in low_conf_cids],
            "singletons": singletons,
            "already_in_project": skipped_already,
        },
    }

    with open(out, "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)

    # Human-readable summary
    print(f"\nProvisional plan written to: {out}")
    print(
        f"Clusters: {len(cluster_descs)} | Proposed moves: {len(proposed_moves)} | "
        f"Singletons: {len(singletons)} | Pre-assigned skipped: {len(skipped_already)}"
    )
    for pm in proposed_moves[:12]:
        print(f"  -> {pm['project_title']} [{pm['project_folder_slug']}] : " f"{len(pm['chats'])} chats")
    if len(proposed_moves) > 12:
        print(f"  ... and {len(proposed_moves)-12} more")

    return 0


# =====================
# Main routine
# =====================


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Categorize ChatGPT chats by title and emit a provisional move plan.")
    p.add_argument(
        "--conversations-json",
        required=True,
        help="Path to the ChatGPT conversations JSON file.",
    )
    p.add_argument(
        "--out",
        default="provisional_move_plan.json",
        help="Path to write the provisional plan JSON.",
    )
    p.add_argument(
        "--collection",
        default=QDRANT_COLLECTION,
        help="Qdrant collection name for embeddings.",
    )
    p.add_argument(
        "--no-qdrant",
        action="store_true",
        help="Disable Qdrant persistence (embeddings kept only in-memory).",
    )
    p.add_argument(
        "--eps-cosine",
        type=float,
        default=0.25,
        help="DBSCAN epsilon in cosine distance space (0..2). Lower = tighter clusters.",
    )
    p.add_argument(
        "--min-samples",
        type=int,
        default=2,
        help="DBSCAN min_samples (>=2 is sensible).",
    )
    p.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.60,
        help="Minimum combined confidence to propose moves.",
    )
    p.add_argument(
        "--time-weight",
        type=float,
        default=0.25,
        help="Weight (0..1) given to temporal cohesion when computing cluster cohesion.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional: limit number of chats processed (debug).",
    )

    args = p.parse_args(argv)

    return categorize_chats(
        conversations_json=args.conversations_json,
        out=args.out,
        collection=args.collection,
        no_qdrant=args.no_qdrant,
        eps_cosine=args.eps_cosine,
        min_samples=args.min_samples,
        confidence_threshold=args.confidence_threshold,
        time_weight=args.time_weight,
        limit=args.limit,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise
