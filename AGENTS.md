# Agents Guide

Operational runbook for coding agents integrating with this repository.

## Purpose

- Ingest ChatGPT exports and cluster conversations into projects.
- Emit a provisional move plan JSON. No mutation of external systems is performed.

Primary entry: [`main.main()`](main.py:119). Core API: [`GptCategorize.categorize_chats()`](GptCategorize/categorize.py:704).

## Quickstart

Environment

```bash
# Required (OpenAI-compatible inference + embeddings)
export INFERENCE_API_BASE="https://api.openai.com/v1"
export INFERENCE_API_KEY="REDACTED"
export EMBEDDING_API_BASE="https://api.openai.com/v1"
export EMBEDDING_API_KEY="REDACTED"
# Optional (Qdrant persistence)
export QDRANT_URL="http://localhost:6333"
# export QDRANT_API_KEY=...

# Optional tuning
export LOG_LEVEL=INFO
export VERBOSE_LOGGING=false
```

Install runtime deps

```bash
uv sync
```

Run categorization

```bash
uv run python ./main.py \
  --conversations-json ./conversations.json \
  --out ./provisional_move_plan.json
```

Entrypoints in packaging: see [`pyproject.toml`](pyproject.toml:49)

## Inputs

- Chat export JSON (flat JSON or NDJSON accepted). Detected via:
  - Parser: [`load_chats_from_conversations_json()`](GptCategorize/categorize.py:291)
  - Shape handling: [`extract_chats_from_json_blob()`](GptCategorize/categorize.py:246)
  - Timestamp parsing: [`to_epoch()`](GptCategorize/categorize.py:144)
  - First prompt extraction: [`_first_user_prompt()`](GptCategorize/categorize.py:202)
  - Project detection: [`_detect_projectish()`](GptCategorize/categorize.py:178)

## Outputs

- JSON move plan written to `--out` (default `provisional_move_plan.json`).
- Structure assembled in [`categorize_chats()`](GptCategorize/categorize.py:704).
- Contains `clusters`, `proposed_moves`, and `skipped` sections.

Key confidence formulas come from:

- Text cohesion: [`cluster_text_cohesion()`](GptCategorize/categorize.py:519)
- Time cohesion: [`temporal_cohesion()`](GptCategorize/categorize.py:531)
- Combined confidence computed inside [`categorize_chats()`](GptCategorize/categorize.py:852)

## Tunable CLI Flags

Handled by [`main.main()`](main.py:119):

- `--conversations-json PATH` (required)
- `--out PATH` (default: `provisional_move_plan.json`)
- `--collection NAME` (default: [`QDRANT_COLLECTION`](GptCategorize/categorize.py:41))
- `--no-qdrant` (compute-only; skip persistence)
- `--eps-cosine FLOAT` (default: 0.25) — DBSCAN epsilon in cosine distance; converted by [`cosine_to_euclid_eps()`](GptCategorize/categorize.py:497)
- `--min-samples INT` (default: 2)
- `--confidence-threshold FLOAT` (default: 0.60)
- `--time-weight FLOAT` (default: 0.25)
- `--limit INT` (default: 0) — process first N chats

## Environment Variables and Defaults

Defined near the top of [`categorize.py`](GptCategorize/categorize.py:31):

- Inference (LLM for labels):
  - [`INFERENCE_API_BASE`](GptCategorize/categorize.py:31)
  - [`INFERENCE_API_KEY`](GptCategorize/categorize.py:32)
  - [`INFERENCE_MODEL`](GptCategorize/categorize.py:36)
  - Client factory: [`get_inference_client()`](GptCategorize/categorize.py:320)

- Embeddings (for clustering):
  - [`EMBEDDING_API_BASE`](GptCategorize/categorize.py:33)
  - [`EMBEDDING_API_KEY`](GptCategorize/categorize.py:34)
  - [`EMBEDDING_MODEL`](GptCategorize/categorize.py:37)
  - Client factory: [`get_embedding_client()`](GptCategorize/categorize.py:332)

- Qdrant (optional):
  - [`QDRANT_URL`](GptCategorize/categorize.py:39)
  - [`QDRANT_API_KEY`](GptCategorize/categorize.py:40)
  - [`QDRANT_COLLECTION`](GptCategorize/categorize.py:41)
  - Ensure/create: [`ensure_qdrant_collection()`](GptCategorize/categorize.py:406)
  - Batched upsert: [`upsert_to_qdrant_batched()`](GptCategorize/categorize.py:441)

- Operations and logging:
  - [`TIME_DECAY_DAYS`](GptCategorize/categorize.py:44)
  - [`OPENAI_TIMEOUT`](GptCategorize/categorize.py:47)
  - [`QDRANT_TIMEOUT`](GptCategorize/categorize.py:48)
  - [`QDRANT_BATCH_SIZE`](GptCategorize/categorize.py:49)
  - [`MAX_RETRIES`](GptCategorize/categorize.py:50)
  - [`RETRY_DELAY`](GptCategorize/categorize.py:51)
  - [`VERBOSE_LOGGING`](GptCategorize/categorize.py:57)
  - [`LOG_LEVEL`](GptCategorize/categorize.py:58)

## Processing Pipeline

1) Load and normalize chats
   - [`load_chats_from_conversations_json()`](GptCategorize/categorize.py:291)

2) Filter out chats already in projects
   - project heuristics via [`_detect_projectish()`](GptCategorize/categorize.py:178)

3) Retrieve cached embeddings from Qdrant
   - [`fetch_existing_embeddings_from_qdrant()`](GptCategorize/categorize.py:445)

4) Embed chats missing vectors
   - Robust retry loop: [`embed_chats_with_retry()`](GptCategorize/categorize.py:344)

5) Optional persistence (new vectors only)
   - Qdrant collection management and batched upserts:
     - [`ensure_qdrant_collection()`](GptCategorize/categorize.py:410)
     - [`upsert_to_qdrant_batched()`](GptCategorize/categorize.py:490)

6) Clustering
   - [`cluster_embeddings()`](GptCategorize/categorize.py:505)
   - KMeans fallback with `random_state=42` on degenerate DBSCAN

7) Cohesion metrics
   - [`cluster_text_cohesion()`](GptCategorize/categorize.py:519)
   - [`temporal_cohesion()`](GptCategorize/categorize.py:531)

8) LLM labeling with strict JSON
   - [`label_clusters_with_llm()`](GptCategorize/categorize.py:567)
   - Fallback heuristic if LLM call fails

9) Plan emission
   - Orchestrated by [`categorize_chats()`](GptCategorize/categorize.py:704)

## Failure Modes and Behavior

- Missing API keys:
  - [`get_inference_client()`](GptCategorize/categorize.py:320) and [`get_embedding_client()`](GptCategorize/categorize.py:332) raise descriptive errors.

- Qdrant unavailable:
  - Errors are caught, warning printed, pipeline continues compute-only (see persistence block within [`categorize_chats()`](GptCategorize/categorize.py:786)).

- API/Network flakiness:
  - Exponential backoff and bounded retries for embeddings [`embed_chats_with_retry()`](GptCategorize/categorize.py:344) and LLM calls [`label_clusters_with_llm()`](GptCategorize/categorize.py:567).

- LLM non-determinism:
  - Temperature = 0.2. Fallback labels maintain continuity.

- Determinism:
  - KMeans fallback uses `random_state=42`. DBSCAN is deterministic given inputs.

- Large inputs:
  - Use `--limit` during orchestration to bound runtime and cost.

## Observability and Debugging

- Logging config and levels set in [`setup_logging()`](GptCategorize/categorize.py:61); control via [`VERBOSE_LOGGING`](GptCategorize/categorize.py:57) and [`LOG_LEVEL`](GptCategorize/categorize.py:58).

- VSCode launchers: see [.vscode/launch.json](.vscode/launch.json:1) and guide in [`DEBUG.md`](DEBUG.md:1).

- Programmatic debug hooks:
  - [`debug_config.start_debug_server()`](debug_config.py:15)
  - [`debug_config.enable_debugging_on_exception()`](debug_config.py:45)
  - [`debug_config.debug_here()`](debug_config.py:68)
  - [`debug_config.is_debugger_attached()`](debug_config.py:86)

## Automation Recipes

- Minimal dry run (no persistence, limit N)

```bash
uv run python ./main.py \
  --conversations-json ./conversations.json \
  --no-qdrant \
  --limit 200
```

- Tight clusters, higher confidence gate

```bash
uv run python ./main.py \
  --conversations-json ./conversations.json \
  --eps-cosine 0.2 --min-samples 3 \
  --confidence-threshold 0.7
```

- Emphasize temporal cohesion

```bash
uv run python ./main.py \
  --conversations-json ./conversations.json \
  --time-weight 0.4
```

## Extension Points

- Move executor (future work):
  - Implement in [`GptMove`](GptMove/__init__.py:1).
  - Consume the plan emitted by [`categorize_chats()`](GptCategorize/categorize.py:704).
  - Honor `proposed_moves` only; keep side effects idempotent and resumable.

- Packaging CLI:
  - Installable scripts configured in [`pyproject.toml`](pyproject.toml:49):
    - `categorize` → [`main.main()`](main.py:119)
    - `move` → `GptMove:main` (to be implemented)

## Make Targets for Agents

- Bootstrap: [`install`](Makefile:21), [`install-dev`](Makefile:27)
- Checks: [`check`](Makefile:32)
- Upgrade deps: [`update`](Makefile:41)
- Housekeeping: [`clean`](Makefile:47), [`clean-all`](Makefile:59)
- Style: [`format`](Makefile:67)

## Repository Invariants

- No direct mutation of ChatGPT data; only reads input file and emits a plan.
- External calls:
  - Embeddings and LLM via OpenAI-compatible endpoints.
  - Optional Qdrant persistence; safe to disable with `--no-qdrant`.
- Output file is overwritten if it exists; ensure unique `--out` when needed.

## File Map

- Entrypoint: [`main.py`](main.py)
- Categorization module: [`GptCategorize/categorize.py`](GptCategorize/categorize.py)
- Planned mover: [`GptMove/__init__.py`](GptMove/__init__.py)
- Debug helpers: [`debug_config.py`](debug_config.py)
- IDE configs: [.vscode/launch.json](.vscode/launch.json)
- Dev workflow: [`Makefile`](Makefile)
- Project metadata: [`pyproject.toml`](pyproject.toml)
- Human doc: [`README.md`](README.md), [`DEBUG.md`](DEBUG.md)
