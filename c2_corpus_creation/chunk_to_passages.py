"""
Module for chunking Wikipedia documents into passages.
This module provides functions to split Wikipedia articles into fixed-size passages
and convert them to JSONL format.
"""

import io
import os
import re
import sys
import bz2
import json
import hashlib
from pathlib import Path
from typing import Iterator, Tuple, Dict


# Regular expressions for parsing
DOC_OPEN_RE = re.compile(r'<doc\b[^>]*?>', re.IGNORECASE)
DOC_CLOSE_RE = re.compile(r'</doc>', re.IGNORECASE)
ATTR_RE = re.compile(r'(\w+)="([^"]*)"')
SECTION_HEADER_RE = re.compile(r'^\s*==+\s.*?==+\s*$', re.MULTILINE)


def open_maybe_bz2(path: Path) -> io.TextIOBase:
    """
    Opens a file, automatically handling bz2 compression.

    Args:
        path: Path to the file

    Returns:
        Text stream for reading the file
    """
    if str(path).endswith(".bz2"):
        return io.TextIOWrapper(bz2.open(path, mode="rb"), encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")


def normalize_text(text: str, preserve_structure: bool = False) -> str:
    """
    Normalizes Wikipedia text by handling whitespace and section headers.

    Args:
        text: Raw Wikipedia text
        preserve_structure: If True, preserve some text structure (for infoboxes).
                          If False, collapse to single spaces (default behavior).

    Returns:
        Normalized text
    """
    if preserve_structure:
        # Keep section headers for infobox information, but normalize them
        text = SECTION_HEADER_RE.sub(lambda m: m.group(0).strip() + " ", text)
        # Collapse excessive whitespace but preserve some structure
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        return text.strip()
    else:
        # Remove section headers lines like "== References ==", keep body
        text = SECTION_HEADER_RE.sub("\n", text)
        # Collapse whitespace/newlines to single spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text


def chunk_by_words(text: str, words_per_chunk: int) -> Iterator[str]:
    """
    Splits text into chunks of approximately equal word count.

    Args:
        text: Text to chunk
        words_per_chunk: Target number of words per chunk

    Yields:
        Text chunks
    """
    if not text:
        return
    words = text.split()
    for i in range(0, len(words), words_per_chunk):
        chunk_words = words[i:i + words_per_chunk]
        if len(chunk_words) == 0:
            continue
        yield " ".join(chunk_words)


def parse_wikiextractor_xmlish(stream: io.TextIOBase) -> Iterator[Tuple[str, str, str]]:
    """
    Parses WikiExtractor XML-like format.

    Expected format:
        <doc id="123" url="..." title="Some title">
        text...
        </doc>

    Args:
        stream: Text stream to parse

    Yields:
        Tuples of (doc_id, title, text)
    """
    in_doc = False
    buf = []
    title = None
    doc_id = None
    for line in stream:
        if not in_doc:
            if DOC_OPEN_RE.search(line):
                in_doc = True
                buf = []
                attrs = dict(ATTR_RE.findall(line))
                title = attrs.get("title", "").strip()
                doc_id = attrs.get("id", "").strip()
        else:
            if DOC_CLOSE_RE.search(line):
                text = "".join(buf)
                yield (doc_id or "", title or "", text)
                in_doc = False
                buf = []
                title = None
                doc_id = None
            else:
                buf.append(line)
    # If file ended inside a doc (rare), flush
    if in_doc and buf:
        yield (doc_id or "", title or "", "".join(buf))


def parse_jsonl(stream: io.TextIOBase) -> Iterator[Tuple[str, str, str]]:
    """
    Parses JSONL format Wikipedia dumps.

    Args:
        stream: Text stream to parse

    Yields:
        Tuples of (doc_id, title, text)
    """
    for line in stream:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            # Best-effort: strip trailing commas and retry once
            maybe = line.rstrip(",")
            try:
                obj = json.loads(maybe)
            except Exception:
                continue
        title = (obj.get("title") or "").strip()
        text = (obj.get("text") or obj.get("contents") or "").strip()
        doc_id = str(obj.get("id") or obj.get("_id") or "")
        if title and text:
            yield (doc_id, title, text)


def detect_format_first_line(path: Path) -> str:
    """
    Detects whether a file is in JSONL or XML-like format.

    Args:
        path: Path to the file

    Returns:
        Either "jsonl" or "xmlish"
    """
    with open_maybe_bz2(path) as fh:
        for line in fh:
            s = line.lstrip()
            if not s:
                continue
            if s.startswith("{"):
                return "jsonl"
            if "<doc" in s:
                return "xmlish"
            break
    return "xmlish"


def iter_documents_from_file(path: Path) -> Iterator[Tuple[str, str, str]]:
    """
    Iterates over documents in a file, automatically detecting format.

    Args:
        path: Path to the file

    Yields:
        Tuples of (doc_id, title, text)
    """
    fmt = detect_format_first_line(path)
    with open_maybe_bz2(path) as fh:
        if fmt == "jsonl":
            yield from parse_jsonl(fh)
        else:
            yield from parse_wikiextractor_xmlish(fh)


def iter_all_chunk_files(root: Path) -> Iterator[Path]:
    """
    Finds all chunk files in a WikiExtractor output directory.

    Args:
        root: Root directory to search

    Yields:
        Paths to chunk files
    """
    for dirpath, _, filenames in os.walk(root):
        for fn in sorted(filenames):
            if fn.startswith("wiki_") or fn.endswith(".bz2") or fn.endswith(".json") or fn.endswith(".jsonl"):
                yield Path(dirpath) / fn


def chunk_docs_by_words(
    input_root=None,
    output_jsonl=None,
    words_per_passage=100,
    min_words=20,
    skip_empty_titles=False,
    progress_every=1000,
    preserve_structure=False
):
    """
    Split Wikipedia articles into fixed-size word-based passages and write JSONL.

    Args:
        input_root: Path to processed Wikipedia dump root
        output_jsonl: Output JSONL file path
        words_per_passage: Number of words per passage (default: 100)
        min_words: Skip passages with fewer words than this (default: 20)
        skip_empty_titles: Skip docs without a title (default: False)
        progress_every: Log every N passages (default: 1000)
        preserve_structure: Preserve text structure for infoboxes (default: False)

    Returns:
        Tuple of (total_articles, total_passages)
    """
    if input_root is None or output_jsonl is None:
        raise ValueError("input_root and output_jsonl are required arguments")

    input_root_path = Path(input_root)
    out_path = Path(output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_articles = 0
    total_passages = 0

    with open(out_path, "w", encoding="utf-8") as out_f:
        for fpath in iter_all_chunk_files(input_root_path):
            try:
                for doc_id, title, raw_text in iter_documents_from_file(fpath):
                    if skip_empty_titles and not title:
                        continue
                    total_articles += 1

                    text = normalize_text(raw_text, preserve_structure=preserve_structure)
                    if not text:
                        continue

                    # Stable base id: prefer doc_id; if missing, hash title
                    if doc_id:
                        base = doc_id
                    else:
                        base = hashlib.md5(title.encode("utf-8", errors="ignore")).hexdigest()[:12]

                    chunk_idx = 0
                    for chunk in chunk_by_words(text, words_per_passage):
                        if len(chunk.split()) < min_words:
                            continue
                        pid = f"{base}-{chunk_idx:04d}"
                        rec: Dict[str, str] = {
                            "id": pid,
                            "title": title,
                            "contents": chunk
                        }
                        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        total_passages += 1
                        chunk_idx += 1

                        if progress_every and (total_passages % progress_every == 0):
                            print(f"[info] {total_passages} passages written ...", file=sys.stderr)
            except Exception as e:
                print(f"[warn] Skipping {fpath} due to error: {e}", file=sys.stderr)

    print(f"[done] Articles processed: {total_articles}", file=sys.stderr)
    print(f"[done] Passages written: {total_passages}", file=sys.stderr)

    return total_articles, total_passages


def read_jsonl_line(filepath: str, index: int) -> str:
    """
    Reads a specific line (by index) from a large JSONL file without loading the entire file.

    Args:
        filepath: Path to the .jsonl file
        index: Zero-based index of the line to read

    Returns:
        The line at the given index (stripped of newline)

    Raises:
        IndexError: If index is out of range
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == index:
                return line.strip()
    raise IndexError(f"Index {index} out of range for file {filepath}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert Wikipedia dump to passage-based JSONL corpus"
    )
    parser.add_argument("--input-root", type=str, required=True,
                        help="Path to processed Wikipedia dump root")
    parser.add_argument("--output-jsonl", type=str, required=True,
                        help="Output JSONL file path")
    parser.add_argument("--words-per-passage", type=int, default=100,
                        help="Number of words per passage (default: 100)")
    parser.add_argument("--min-words", type=int, default=20,
                        help="Skip passages with fewer words (default: 20)")
    parser.add_argument("--skip-empty-titles", action="store_true",
                        help="Skip docs without a title")
    parser.add_argument("--progress-every", type=int, default=10000,
                        help="Log every N passages (default: 10000)")
    parser.add_argument("--preserve-structure", action="store_true",
                        help="Preserve text structure (useful for infoboxes)")

    args = parser.parse_args()

    chunk_docs_by_words(
        input_root=args.input_root,
        output_jsonl=args.output_jsonl,
        words_per_passage=args.words_per_passage,
        min_words=args.min_words,
        skip_empty_titles=args.skip_empty_titles,
        progress_every=args.progress_every,
        preserve_structure=args.preserve_structure
    )
