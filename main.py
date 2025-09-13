#!/usr/bin/env python3
"""
Categorize ChatGPT chats by title, propose project folders, and emit a provisional move plan.
An automated functionality to perform the movement will be added later.

What this program does
---------------------
1) Reads a ChatGPT **conversations JSON** file directly (no ZIP handling).
   - Export your data from ChatGPT, unzip it yourself, then pass the path to a
     JSON file that contains the conversation objects (commonly `conversations.json`).
2) Skips any chats already assigned to a project/workspace/folder.
3) Creates embeddings for each chat's title plus the first 250 words of the
   initial user prompt using **text-embedding-3-large** (via a dedicated
   Embeddings API base/key), and optionally persists vectors into local **Qdrant**.
4) Clusters chats (DBSCAN over L2-normalized embeddings ≈ cosine distance),
   falling back to KMeans if DBSCAN yields nothing useful.
5) Uses **gpt-5-mini** (via a separate Inference API base/key) to label clusters
   and propose project folder slugs.
6) Computes **cluster cohesion** as a weighted blend of:
     • textual cohesion (average pairwise cosine similarity of embeddings), and
     • temporal cohesion (how close chats are in creation time, via an exponential decay).
   The temporal contribution weight is CLI-tunable via `--time-weight` (default 0.25).
7) Writes a **provisional move plan** JSON your next script can consume to actually
   move chats into project folders (this script does not perform the move).

Install deps
------------
uv sync

Exporting Chats (how to obtain conversations.json)
--------------------------------------------------
In ChatGPT:
  • Click your name (bottom-left) → Settings → Data Controls → Export → "Create export".
  • You'll receive an email with a download link to a .zip.
  • Download and **unzip** locally.
  • Locate the JSON file that contains conversations (commonly `conversations.json`).
  • Pass that file path with `--conversations-json /path/to/conversations.json`.

Outputs
-------
A JSON file (default `./provisional_move_plan.json`) with schema:
{
  "plan_generated_at": ISO-8601 string,
  "source": { "type": "conversations_json", "path": "..." },
  "parameters": { ... },
  "collections": { "qdrant_collection": "..." },
  "clusters": [
    {
      "cluster_id": 0,
      "label": "DevOps Incidents",
      "project_folder_slug": "devops-incidents",
      "project_title": "DevOps Incidents",
      "confidence_model": 0.91,
      "cohesion_text": 0.80,
      "cohesion_time": 0.55,
      "confidence_cohesion": 0.74,   // (1 - time_weight) * cohesion_text + time_weight * cohesion_time
      "confidence_combined": 0.86,    // 0.65 * confidence_model + 0.35 * confidence_cohesion
      "chats": [{"id": "...", "title": "...", "create_time": 1712345678, "update_time": 1712350000}]
    },
    ...
  ],
  "proposed_moves": [
    {
      "project_folder_slug": "devops-incidents",
      "project_title": "DevOps Incidents",
      "cluster_id": 0,
      "chats": ["chat_id_1", "chat_id_2"]
  ],
  "skipped": {
    "clusters_low_confidence": [2, 5],
    "singletons": ["chat_id_9"],
    "already_in_project": ["chat_id_42"]
  }
}

Usage examples
--------------
python ./main.py \
  --conversations-json conversations.json \
  --out move_plan.json

# Tune clustering (cosine threshold ~ lower = tighter clusters)
python ./main.py \
  --conversations-json conversations.json \
  --eps-cosine 0.22 --min-samples 2

# Add a bit more emphasis on temporal proximity (default is 0.25)
python ./main.py \
  --conversations-json ./conversations.json --time-weight 0.35

# Skip Qdrant persistence (still runs clustering in-memory)
python ./main.py --conversations-json ./conversations.json --no-qdrant

Notes
-----
• "Already in a project" detection: we look for any of these fields on the conversation record
  (case-insensitive, nested OK): project_id, workspace_id, folder_id, project, workspace, folder.
• Embedding dimension for text-embedding-3-large is 3072.
• DBSCAN runs in Euclidean space over L2-normalized embeddings (≈ cosine distance). We convert the
  cosine epsilon into an equivalent Euclidean epsilon automatically.
• Temporal cohesion uses an exponential decay over pairwise creation-time gaps:
    sim_ij = exp(- |Δt_days| / TIME_DECAY_DAYS )
  averaged over all pairs. The default TIME_DECAY_DAYS is 30.
• If DBSCAN finds nothing useful, we fall back to KMeans (k ≈ sqrt(N)).
• Model prompts are constrained to return strict JSON that we parse.

Caveats
-------
• There is currently no official public API to list personal ChatGPT conversations.
  Data export is the reliable route; this script reads the JSON directly.
• This script never attempts the actual move.
"""

import argparse
import sys
from GptCategorize.categorize import categorize_chats, QDRANT_COLLECTION


def main() -> int:
    """Main CLI entry point for categorizing ChatGPT chats."""
    p = argparse.ArgumentParser(description="Categorize ChatGPT chats by title and emit a provisional move plan.")
    p.add_argument(
        "--conversations-json",
        required=True,
        help="Path to the ChatGPT conversations JSON file.",
    )
    p.add_argument(
        "--out",
        default="move_plan.json",
        help="Path to write the move plan JSON.",
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

    args = p.parse_args()

    # Call the categorize_chats function with parsed arguments
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
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
