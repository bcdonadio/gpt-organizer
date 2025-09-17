# ChatGPT Organizer

Organize exported ChatGPT conversations into coherent project folders by clustering similar chats and emitting a provisional move plan.

- Input: your exported `conversations.json` (or NDJSON).
- Processing: embeddings, clustering, LLM-based labeling, cohesion scoring.
- Output: JSON move plan with per-cluster labels and confidence.
- Safety: this tool does not move anything; it proposes a plan only.

Primary CLI entry: [main.main()](main.py:119). Core logic: [GptCategorize.categorize_chats()](GptCategorize/categorize.py:704).

## What this program does

- Reads ChatGPT conversations from a JSON file you exported (see Export guide below).
- Skips chats that already belong to a project/workspace/folder (heuristics on common fields).
- Reuses embeddings cached in Qdrant when available, and only embeds remaining chats using `EMBEDDING_MODEL` (default: `text-embedding-3-large`) via a dedicated API base/key.
- Clusters chats via DBSCAN in Euclidean space over L2-normalized embeddings (≈ cosine distance), with KMeans fallback.
- Labels clusters with an inference LLM (default: `gpt-5-latest`) and proposes project folder slugs.
- Computes cluster cohesion from text similarity and temporal proximity (weighted, tunable).
- Emits a provisional move plan JSON (no actual moves performed).

Key components:

- Embedding cache: [GptCategorize.fetch_existing_embeddings_from_qdrant()](GptCategorize/categorize.py:445)
- Embedding: [GptCategorize.embed_chats_with_retry()](GptCategorize/categorize.py:344)
- Clustering: [GptCategorize.cluster_embeddings()](GptCategorize/categorize.py:505)
- Temporal cohesion: [GptCategorize.temporal_cohesion()](GptCategorize/categorize.py:531)
- LLM labeling: [GptCategorize.label_clusters_with_llm()](GptCategorize/categorize.py:567)

## Project layout

- CLI entry and basic usage examples: [main.py](main.py)
- Core categorization module: [GptCategorize/categorize.py](GptCategorize/categorize.py)
  - Public API: [GptCategorize.categorize_chats()](GptCategorize/categorize.py:704)
  - Config constants (API bases/keys, Qdrant, timeouts): see lines 31–52, including [GptCategorize.QDRANT_COLLECTION](GptCategorize/categorize.py:41)
  - Package exports: [GptCategorize/__init__.py](GptCategorize/__init__.py)
- Move module (planned): [GptMove/__init__.py](GptMove/__init__.py)
- Debugging guide: [DEBUG.md](DEBUG.md); helpers in:
  - [debug_config.start_debug_server()](debug_config.py:15)
  - [debug_config.enable_debugging_on_exception()](debug_config.py:45)
  - [debug_config.debug_here()](debug_config.py:68)
  - [debug_config.is_debugger_attached()](debug_config.py:86)
- Editor debugging configs: [.vscode/launch.json](.vscode/launch.json)
- Developer tasks: [Makefile](Makefile)

## Requirements

- Python 3.13+ (see [pyproject.toml](pyproject.toml))
- Dependencies are managed via `uv`.

Install dependencies:

```bash
# runtime only
uv sync

# runtime + dev
uv sync --group dev
```

## Exporting your ChatGPT conversations

1. In ChatGPT: Settings → Data Controls → Export → Create export
2. Download the ZIP and unzip locally
3. Locate the JSON file with conversations (commonly `conversations.json`)
4. Use that path with `--conversations-json /path/to/conversations.json`

## Quick start

Using the main entry:

```bash
uv run python ./main.py \
  --conversations-json ./conversations.json \
  --out ./move_plan.json
```

Tune clustering (lower cosine eps → tighter clusters):

```bash
uv run python ./main.py \
  --conversations-json ./conversations.json \
  --eps-cosine 0.22 --min-samples 2
```

Increase time weight (more emphasis on temporal proximity):

```bash
uv run python ./main.py \
  --conversations-json ./conversations.json \
  --time-weight 0.35
```

Skip Qdrant persistence (compute in-memory only):

```bash
uv run python ./main.py \
  --conversations-json ./conversations.json \
  --no-qdrant
```

## CLI usage

Use the main script [main.main()](main.py:119) via `uv run python ./main.py` or the installed console script `categorize`.

Options:

- --conversations-json PATH (required) — input file path
- --out PATH (default: `provisional_move_plan.json`) — output plan file
- --collection NAME (default: [GptCategorize.QDRANT_COLLECTION](GptCategorize/categorize.py:41)) — Qdrant collection
- --no-qdrant — disable Qdrant persistence
- --eps-cosine FLOAT (default: 0.25) — DBSCAN epsilon in cosine distance (0..2); lower → tighter clusters
- --min-samples INT (default: 2) — DBSCAN min_samples
- --confidence-threshold FLOAT (default: 0.60) — minimum combined confidence to propose moves
- --time-weight FLOAT (default: 0.25) — weight (0..1) applied to temporal cohesion
- --limit INT (default: 0) — process only first N chats (debug)

Return code: 0 on success.

## Environment configuration

Set via environment variables before running. Defaults are shown below and live in:

- API clients: [GptCategorize.get_inference_client()](GptCategorize/categorize.py:320), [GptCategorize.get_embedding_client()](GptCategorize/categorize.py:332)
- Qdrant: creation/upsert helpers at [GptCategorize.ensure_qdrant_collection()](GptCategorize/categorize.py:410), [GptCategorize.upsert_to_qdrant_batched()](GptCategorize/categorize.py:490)

OpenAI (or compatible) endpoints and models:

```bash
# Inference (LLM used for cluster labeling)
export INFERENCE_API_BASE="https://api.openai.com/v1"
export INFERENCE_API_KEY="YOUR_INFERENCE_API_KEY"
export INFERENCE_MODEL="gpt-5-latest"

# Embeddings (used for clustering)
export EMBEDDING_API_BASE="https://api.openai.com/v1"
export EMBEDDING_API_KEY="YOUR_EMBEDDING_API_KEY"
export EMBEDDING_MODEL="text-embedding-3-large"
```

Qdrant (optional, for persistence):

```bash
export QDRANT_URL="http://localhost:6333"
# For local instances, API key is usually not needed:
# export QDRANT_API_KEY="..."

# Collection name used by default:
export QDRANT_COLLECTION="chatgpt_chats"
```

Performance and robustness:

```bash
# time scale for temporal cohesion (days)
export TIME_DECAY_DAYS=30

# timeouts (seconds)
export OPENAI_TIMEOUT=600
export QDRANT_TIMEOUT=600

# Qdrant upsert batching
export QDRANT_BATCH_SIZE=100

# retry logic for API calls
export MAX_RETRIES=3
export RETRY_DELAY=2.0

# logging detail
export VERBOSE_LOGGING=false
export LOG_LEVEL=INFO
```

Debugging (debugpy):

```bash
# enable auto-start debug server
export DEBUGPY_ENABLE=1
export DEBUGPY_HOST=localhost
export DEBUGPY_PORT=5678
export DEBUGPY_WAIT=1
```

See [DEBUG.md](DEBUG.md) and helpers:

- [debug_config.start_debug_server()](debug_config.py:15)
- [debug_config.enable_debugging_on_exception()](debug_config.py:45)
- [debug_config.debug_here()](debug_config.py:68)

## Output

The tool writes a JSON plan (default `./provisional_move_plan.json`) with the schema:

```json
{
  "plan_generated_at": "ISO-8601",
  "source": { "type": "conversations_json", "path": "/abs/path/to/conversations.json" },
  "parameters": {
    "eps_cosine": 0.25,
    "min_samples": 2,
    "confidence_threshold": 0.6,
    "time_weight": 0.25,
    "time_decay_days": 30
  },
  "collections": { "qdrant_collection": "chatgpt_chats" },
  "clusters": [
    {
      "cluster_id": 0,
      "label": "DevOps Incidents",
      "project_folder_slug": "devops-incidents",
      "project_title": "DevOps Incidents",
      "confidence_model": 0.91,
      "cohesion_text": 0.80,
      "cohesion_time": 0.55,
      "confidence_cohesion": 0.74,
      "confidence_combined": 0.86,
      "chats": [
        { "id": "...", "title": "...", "create_time": 1712345678, "update_time": 1712350000 }
      ]
    }
  ],
  "proposed_moves": [
    { "project_folder_slug": "devops-incidents", "project_title": "DevOps Incidents", "cluster_id": 0, "chats": ["chat_id_1", "chat_id_2"] }
  ],
  "skipped": {
    "clusters_low_confidence": ["2", "5"],
    "singletons": ["chat_id_9"],
    "already_in_project": ["chat_id_42"]
  }
}
```

- Combined confidence = 0.65 × model_conf + 0.35 × cohesion_conf
- Cohesion_conf = (1 − time_weight) × text_cohesion + time_weight × time_cohesion

## Notes and caveats

- Already-in-project detection checks common fields (case-insensitive, nested OK): `project_id`, `workspace_id`, `folder_id`, `project`, `workspace`, `folder`. See source comments in [GptCategorize/categorize.py](GptCategorize/categorize.py) around lines 105–124 and implementation in [_detect_projectish()](GptCategorize/categorize.py:178).
- DBSCAN runs in Euclidean space over L2-normalized embeddings (≈ cosine distance); epsilon is auto-converted from cosine to Euclidean. See [GptCategorize.cosine_to_euclid_eps()](GptCategorize/categorize.py:497).
- If DBSCAN yields no useful clusters, we fallback to KMeans with k ≈ √N.
- There is currently no public API to list personal ChatGPT conversations; data export is the reliable route.
- This project does not move chats. A future module will handle execution: see [GptMove/__init__.py](GptMove/__init__.py).

## Debugging

- VSCode: use the provided launch configs in [.vscode/launch.json](.vscode/launch.json).
- Programmatic/manual:
  - Start a server and wait for client: [debug_config.start_debug_server()](debug_config.py:15)
  - Break on an exception automatically: [debug_config.enable_debugging_on_exception()](debug_config.py:45)
  - Break at a specific point: [debug_config.debug_here()](debug_config.py:68)

Full walkthrough: [DEBUG.md](DEBUG.md).

## Development

Common tasks are defined in the [Makefile](Makefile):

- `make install` — create venv and install runtime deps
- `make install-dev` — install runtime + dev deps
- `make check` — run format, lint, and strict static analysis (Black, Flake8, Mypy, Pyright)
- `make update` — update locked dependencies
- `make clean` / `make clean-all` — clean caches and venv
- `make format` — auto-format code

Code style/quality settings:

- Black config: [pyproject.toml](pyproject.toml)
- Flake8 config: [.flake8](.flake8)
- Mypy strict mode: [pyproject.toml](pyproject.toml)
- Pyright strict config: [pyproject.toml](pyproject.toml)

## License

MIT — see [pyproject.toml](pyproject.toml) for metadata.
