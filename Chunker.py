"""Chunk .txt transcripts, create OpenAI embeddings, and store them in Chroma.

Install: pip install python-dotenv openai chromadb
Set OPENAI_API_KEY in .env, then run: python ingest_transcripts.py ./transcripts
Chunks default to five-minute windows from each transcript's first timestamp.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import chromadb
from dotenv import load_dotenv
from openai import OpenAI


TIMESTAMP_SPEAKER_RE = re.compile(
    r"^\s*\[?(?P<time>\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?)\]?"
    r"\s*(?:[-–]\s*)?(?P<speaker>[^:]{1,60}):\s*(?P<text>.*)$"
)
SPEAKER_TIMESTAMP_RE = re.compile(
    r"^\[(?P<speaker>.+?)\]\s+(?P<time>\d{1,2}:\d{2}:\d{2}(?:[.,]\d+)?)$"
)
CHUNK_MINUTES = 5
BATCH_SIZE = 100


def parse_text_file(text: str) -> list[dict[str, str]]:
    """Read Zoom blocks like `[Speaker] 14:24:57` followed by the spoken text."""
    text = text.lstrip("\ufeff").replace("\r", "")
    turns = []
    for paragraph in re.split(r"\n\s*\n", text):
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        if not lines:
            continue
        header = SPEAKER_TIMESTAMP_RE.match(lines[0])
        body = " ".join(lines[1:]) if header else " ".join(lines)
        inline = TIMESTAMP_SPEAKER_RE.match(body)
        turns.append({
            "start": header["time"] if header else inline["time"] if inline else "",
            "text": inline["text"].strip() if inline else body,
        })
    return turns


def time_to_seconds(value: str) -> float:
    """Convert HH:MM:SS or MM:SS timestamps to seconds."""
    clock, _, fraction = value.replace(",", ".").partition(".")
    parts = [int(part) for part in clock.split(":")]
    seconds = parts[-1] + (parts[-2] * 60 if len(parts) >= 2 else 0)
    if len(parts) == 3:
        seconds += parts[0] * 3600
    return seconds + (float(f"0.{fraction}") if fraction else 0)


def make_chunks(turns: list[dict[str, str]], window_minutes: int = CHUNK_MINUTES
                ) -> list[dict[str, Any]]:
    """Group turns by time; use timestamps for bucketing, but return text only."""
    if not turns:
        return []
    duration = window_minutes * 60
    first_time = next((time_to_seconds(t["start"]) for t in turns if t["start"]), 0)
    groups: dict[int, list[dict[str, str]]] = {}
    current_window = 0

    # This creates time windows that the turns are then placed into
    for turn in turns:
        if turn["start"]:
            current_window = max(0, int((time_to_seconds(turn["start"]) - first_time) // duration))
        groups.setdefault(current_window, []).append(turn)

    #Chunks the text up and then appends all the text in a certain window together
    chunks = []
    for group in groups.values():
        chunks.append({"text": " ".join(turn["text"] for turn in group)})
    return chunks


def ingest_directory(directory: Path, db_path: Path, collection_name: str,
                     chunk_minutes: int, embedding_model: str) -> None:
    files = sorted(p for p in directory.rglob("*.txt") if p.is_file())
    if not files:
        raise SystemExit(f"No .txt files found in {directory}")

    load_dotenv()
    ai = OpenAI()
    db = chromadb.PersistentClient(path=str(db_path))
    collection = db.get_or_create_collection(name=collection_name)
    total = 0

    for path in files:
        try:
            turns = parse_text_file(path.read_text(encoding="utf-8-sig"))
        except UnicodeDecodeError:
            print(f"Skipping non-UTF-8 file: {path}", file=sys.stderr)
            continue
        chunks = make_chunks(turns, chunk_minutes)
        if not chunks:
            continue

        source = path.relative_to(directory).as_posix()
        ids = [f"{source}::chunk-{i:06d}" for i in range(len(chunks))]
        for start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[start:start + BATCH_SIZE]
            response = ai.embeddings.create(model=embedding_model,
                                            input=[chunk["text"] for chunk in batch])
            collection.upsert(
                ids=ids[start:start + len(batch)],
                documents=[chunk["text"] for chunk in batch],
                embeddings=[item.embedding for item in response.data],
                metadatas=[{
                    "source_file": source,
                    "meeting_id": path.stem,
                    "chunk_index": start + i,
                } for i, chunk in enumerate(batch)],
            )

        old_ids = collection.get(where={"source_file": source}, include=[])["ids"]
        stale = [item_id for item_id in old_ids if item_id not in set(ids)]
        for start in range(0, len(stale), BATCH_SIZE):
            collection.delete(ids=stale[start:start + BATCH_SIZE])
        total += len(chunks)
        print(f"Indexed {source}: {len(chunks)} chunks")

    print(f"Done. Indexed {total} chunks in '{collection_name}' at {db_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed .txt transcripts into Chroma.")
    parser.add_argument("directory", type=Path, help="Directory to scan recursively")
    parser.add_argument("--db-path", type=Path, default=Path("./chroma_db"))
    parser.add_argument("--collection", default="class_notes")
    parser.add_argument("--chunk-minutes", type=int, default=CHUNK_MINUTES,
                        help="Minutes per chunk window (default: 5)")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    args = parser.parse_args()
    if not args.directory.is_dir():
        parser.error(f"not a directory: {args.directory}")
    if args.chunk_minutes < 1:
        parser.error("chunk-minutes must be at least 1")
    ingest_directory(args.directory.resolve(), args.db_path, args.collection,
                     args.chunk_minutes, args.embedding_model)


if __name__ == "__main__":
    main()

