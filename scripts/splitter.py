from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator

JSONDict = dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a ChatGPT conversations export into individual conversation files."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("conversations.json"),
        help="Path to the ChatGPT conversations export (JSON or NDJSON).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("conversations"),
        help="Directory where individual conversation files will be written.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indent level for emitted JSON files (default: 2).",
    )
    return parser.parse_args()


def load_conversations(path: Path) -> list[JSONDict]:
    """Load conversations from the given path.

    Supports both the default ChatGPT export shape (list or dict with "conversations")
    and newline-delimited JSON (NDJSON).
    """
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError:
        return list(_load_ndjson(path))

    if isinstance(data, list):
        return _ensure_dict_list(data, "root list")

    if isinstance(data, dict):
        conversations = data.get("conversations")
        if isinstance(conversations, list):
            return _ensure_dict_list(conversations, '"conversations" list')
        raise ValueError(
            "JSON object does not contain a \"conversations\" list"
        )

    raise ValueError("Unsupported JSON structure. Expected list or dict with conversations.")


def _ensure_dict_list(items: Iterable[Any], context: str) -> list[JSONDict]:
    conversations: list[JSONDict] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(
                f"Item at index {index} in {context} is not a JSON object: {type(item)!r}"
            )
        conversations.append(item)
    return conversations


def _load_ndjson(path: Path) -> Iterator[JSONDict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}."
                ) from exc
            if not isinstance(data, dict):
                raise ValueError(
                    f"Line {line_number} of {path} is not a JSON object: {type(data)!r}"
                )
            yield data


def ensure_output_dir(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)


def write_conversations(
    conversations: Iterable[JSONDict],
    output_dir: Path,
    indent: int,
) -> list[Path]:
    written_paths: list[Path] = []
    existing_names: set[str] = set()

    for index, conversation in enumerate(conversations):
        conversation_id = _derive_identifier(conversation, index)
        filename = f"{index:04d}-{conversation_id}.json"
        if filename in existing_names:
            raise ValueError(
                f"Duplicate filename generated for conversations at index {index}: {filename}"
            )
        existing_names.add(filename)

        destination = output_dir / filename
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(conversation, handle, ensure_ascii=False, indent=indent)
            handle.write("\n")
        written_paths.append(destination)

    return written_paths


def _derive_identifier(conversation: JSONDict, index: int) -> str:
    for key in ("conversation_id", "id"):
        value = conversation.get(key)
        if isinstance(value, str) and value.strip():
            return _sanitize_identifier(value)

    title = conversation.get("title")
    if isinstance(title, str) and title.strip():
        return _sanitize_identifier(title)

    return f"conversation-{index:04d}"


def _sanitize_identifier(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "-", value)
    token = token.strip("-._")
    return token or "conversation"


def main() -> int:
    args = parse_args()
    try:
        conversations = load_conversations(args.source)
    except FileNotFoundError:
        print(f"Source file not found: {args.source}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Failed to parse conversations: {exc}", file=sys.stderr)
        return 1

    ensure_output_dir(args.output_dir)
    written_paths = write_conversations(conversations, args.output_dir, args.indent)
    print(f"Wrote {len(written_paths)} conversation files to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
