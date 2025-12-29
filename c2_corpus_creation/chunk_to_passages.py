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
MARKDOWN_HEADER_RE = re.compile(r'^(#{1,6})\s+(.+?)(?:\s*#*)?$', re.MULTILINE)


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


# === V2 ================
def parse_sections_with_hierarchy(text: str) -> list:
    """
    Parse text to extract sections with their hierarchical context.

    Returns a list of tuples: (start_pos, end_pos, section_hierarchy)
    where section_hierarchy is a list of section titles from ## onwards (excluding #).

    Args:
        text: Raw text with markdown-style headers

    Returns:
        List of (start_pos, end_pos, section_hierarchy) tuples
    """
    # Find all headers in the text
    headers = []
    for match in MARKDOWN_HEADER_RE.finditer(text):
        level = len(match.group(1))  # Number of # symbols
        title = match.group(2).strip()
        # Remove trailing period if it exists
        if title.endswith('.'):
            title = title[:-1]
        pos = match.start()
        headers.append((pos, level, title))

    if not headers:
        # No headers found, return entire text as one section with empty hierarchy
        return [(0, len(text), [])]

    # Build sections with their hierarchies
    sections = []
    for i, (pos, level, title) in enumerate(headers):
        # Skip level 1 headers (page title)
        if level == 1:
            continue

        # Find the end position (start of next header at same or higher level, or end of text)
        end_pos = len(text)
        for j in range(i + 1, len(headers)):
            if headers[j][1] <= level:
                end_pos = headers[j][0]
                break

        # Build hierarchy: all ancestor headers from level 2 onwards
        hierarchy = []
        for k in range(i + 1):
            h_pos, h_level, h_title = headers[k]
            if h_pos > pos:
                break
            if 2 <= h_level <= level:
                # Maintain proper hierarchy depth
                while len(hierarchy) >= h_level - 1:
                    hierarchy.pop()
                hierarchy.append(h_title)

        sections.append((pos, end_pos, hierarchy))

    # Handle text before first section (if any)
    if sections:
        first_section_start = min(s[0] for s in sections)
        if first_section_start > 0:
            sections.insert(0, (0, first_section_start, []))
    else:
        # No sections at all (only level 1 header or no headers)
        sections.append((0, len(text), []))

    return sections


def chunk_by_words_v2(text: str, words_per_chunk: int) -> Iterator[str]:
    """
    Splits text into chunks of approximately equal word count without cutting sentences.

    Each chunk will be close to words_per_chunk but may be slightly more or less
    to avoid cutting sentences mid-way.

    Args:
        text: Text to chunk
        words_per_chunk: Target number of words per chunk (approximate)

    Yields:
        Text chunks that end at sentence boundaries
    """
    if not text:
        return

    # Split text into sentences using common sentence terminators
    # This regex splits on . ! ? followed by space/newline or end of string
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$'
    sentences = re.split(sentence_pattern, text)

    # Filter empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return

    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        sentence_words = sentence.split()
        sentence_word_count = len(sentence_words)

        # If adding this sentence would significantly exceed target, yield current chunk
        if current_chunk and (current_word_count + sentence_word_count) > words_per_chunk:
            # Decide whether to include this sentence or start a new chunk
            # Include if we're closer to target with it than without it
            distance_with = abs((current_word_count + sentence_word_count) - words_per_chunk)
            distance_without = abs(current_word_count - words_per_chunk)

            if distance_without <= distance_with:
                # Better to yield current chunk and start new one
                yield " ".join(current_chunk)
                current_chunk = [sentence]
                current_word_count = sentence_word_count
            else:
                # Better to include this sentence in current chunk
                current_chunk.append(sentence)
                current_word_count += sentence_word_count
                yield " ".join(current_chunk)
                current_chunk = []
                current_word_count = 0
        else:
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_word_count += sentence_word_count

    # Yield any remaining content
    if current_chunk:
        yield " ".join(current_chunk)


def parse_and_normalize_section(section_text: str) -> list:
    """
    Parse a section into text segments, list items, and infoboxes.

    Returns a list of tuples: (segment_type, content, word_count)
    where segment_type is either 'text', 'list_items', or 'infobox'
    """
    list_start_pattern = r'^--- List:Start ---\s*$'
    list_end_pattern = r'^--- List:End ---\s*$'
    list_item_pattern = r'^\s*-\s+(.+)$'
    infobox_start_pattern = r'^--- Infobox:Start ---\s*$'
    infobox_end_pattern = r'^--- Infobox:End ---\s*$'

    segments = []
    lines = section_text.split('\n')
    i = 0

    while i < len(lines):
        line = lines[i]

        # Check if we're at the start of an infobox
        if re.match(infobox_start_pattern, line.strip()):
            # Find the end of the infobox
            infobox_lines = []
            i += 1
            while i < len(lines):
                if re.match(infobox_end_pattern, lines[i].strip()):
                    i += 1
                    break
                # Collect key-value pairs
                infobox_lines.append(lines[i])
                i += 1

            # Add infobox as a segment if we have any content
            if infobox_lines:
                # Format infobox as key: value pairs
                infobox_text = " ".join(
                    f"{line.split('=')[0].strip()}: {line.split('=', 1)[1].strip()}"
                    if '=' in line else line.strip()
                    for line in infobox_lines if line.strip()
                )
                word_count = len(infobox_text.split())
                segments.append(('infobox', infobox_text, word_count))

        # Check if we're at the start of an explicit list
        elif re.match(list_start_pattern, line.strip()):
            # Find the end of the list
            list_items = []
            i += 1
            while i < len(lines):
                if re.match(list_end_pattern, lines[i].strip()):
                    i += 1
                    break
                # Extract list item
                item_match = re.match(list_item_pattern, lines[i])
                if item_match:
                    list_items.append(item_match.group(1).strip())
                i += 1

            # Add list items as a segment if we have any
            if list_items:
                list_text = " ".join(list_items)
                word_count = len(list_text.split())
                segments.append(('list_items', list_items, word_count))

        # Check if we're at a List:End without a matching List:Start (implicit list)
        elif re.match(list_end_pattern, line.strip()):
            # This is a list that started implicitly - skip the end tag
            i += 1

        else:
            # Regular text - accumulate until we hit a list/infobox marker or end
            text_lines = []
            list_items = []
            in_implicit_list = False

            should_skip_end_tag = False
            while i < len(lines) and not re.match(list_start_pattern, lines[i].strip()) and not re.match(infobox_start_pattern, lines[i].strip()):
                # Check for end of implicit list
                if re.match(list_end_pattern, lines[i].strip()):
                    should_skip_end_tag = True
                    break
                # Check for infobox end (in case of malformed input)
                if re.match(infobox_end_pattern, lines[i].strip()):
                    should_skip_end_tag = True
                    break

                # Check if this is a list item
                item_match = re.match(list_item_pattern, lines[i])
                if item_match:
                    # If we have accumulated text, save it first
                    if text_lines and not in_implicit_list:
                        text = '\n'.join(text_lines)
                        text = MARKDOWN_HEADER_RE.sub("\n", text)
                        text = re.sub(r'\s+', ' ', text).strip()
                        if text:
                            word_count = len(text.split())
                            segments.append(('text', text, word_count))
                        text_lines = []

                    in_implicit_list = True
                    list_items.append(item_match.group(1).strip())
                else:
                    # Regular text line
                    if in_implicit_list:
                        # We've transitioned out of implicit list, save list items
                        if list_items:
                            list_text = " ".join(list_items)
                            word_count = len(list_text.split())
                            segments.append(('list_items', list_items, word_count))
                            list_items = []
                        in_implicit_list = False
                    text_lines.append(lines[i])

                i += 1

            # Save any remaining list items
            if list_items:
                list_text = " ".join(list_items)
                word_count = len(list_text.split())
                segments.append(('list_items', list_items, word_count))

            # Save any remaining text
            if text_lines:
                text = '\n'.join(text_lines)
                text = MARKDOWN_HEADER_RE.sub("\n", text)
                text = re.sub(r'\s+', ' ', text).strip()
                if text:
                    word_count = len(text.split())
                    segments.append(('text', text, word_count))

            # If we broke due to hitting an end tag, skip it
            if should_skip_end_tag:
                i += 1

    return segments


def chunk_text_with_sections(raw_text: str, words_per_chunk: int) -> Iterator[Tuple[str, list]]:
    """
    Splits text into chunks with section hierarchy information.

    Handles lists and infoboxes specially:
    - Small lists (< words_per_chunk/2) are merged with previous text
    - Medium lists (< words_per_chunk) are kept as one passage
    - Large lists are split by list items
    - Infoboxes are kept together if possible, or split by key-value pairs if too large

    Args:
        raw_text: Raw text with markdown-style section headers
        words_per_chunk: Target number of words per chunk (approximate)

    Yields:
        Tuples of (chunk_text, section_hierarchy) where section_hierarchy is a list
        of section titles (from ## onwards, excluding # page title)
    """
    if not raw_text:
        return

    # Parse sections from the raw text (before normalization)
    sections = parse_sections_with_hierarchy(raw_text)

    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$'

    # Process each section
    for section_start, section_end, section_hierarchy in sections:
        section_text = raw_text[section_start:section_end]

        # Parse section into text and list segments
        segments = parse_and_normalize_section(section_text)

        if not segments:
            continue

        # Chunk the segments
        current_chunk = []
        current_word_count = 0

        for seg_type, content, word_count in segments:
            if seg_type == 'text':
                # Split text into sentences
                sentences = re.split(sentence_pattern, content)
                sentences = [s.strip() for s in sentences if s.strip()]

                for sentence in sentences:
                    sentence_words = sentence.split()
                    sentence_word_count = len(sentence_words)

                    if current_chunk and (current_word_count + sentence_word_count) > words_per_chunk:
                        distance_with = abs((current_word_count + sentence_word_count) - words_per_chunk)
                        distance_without = abs(current_word_count - words_per_chunk)

                        if distance_without <= distance_with:
                            yield (" ".join(current_chunk), section_hierarchy.copy())
                            current_chunk = [sentence]
                            current_word_count = sentence_word_count
                        else:
                            current_chunk.append(sentence)
                            current_word_count += sentence_word_count
                            yield (" ".join(current_chunk), section_hierarchy.copy())
                            current_chunk = []
                            current_word_count = 0
                    else:
                        current_chunk.append(sentence)
                        current_word_count += sentence_word_count

            elif seg_type == 'list_items':
                list_items = content

                # If list is small enough, try to merge with previous chunk
                if word_count < words_per_chunk // 2:
                    list_text = " ".join(list_items)
                    if current_chunk and (current_word_count + word_count) <= words_per_chunk * 1.2:
                        # Merge with previous chunk
                        current_chunk.append(list_text)
                        current_word_count += word_count
                    else:
                        # Yield previous chunk and start new one with list
                        if current_chunk:
                            yield (" ".join(current_chunk), section_hierarchy.copy())
                        current_chunk = [list_text]
                        current_word_count = word_count

                # If list fits in one passage
                elif word_count <= words_per_chunk:
                    # Yield previous chunk if exists
                    if current_chunk:
                        yield (" ".join(current_chunk), section_hierarchy.copy())
                        current_chunk = []
                        current_word_count = 0

                    # Yield list as one passage
                    list_text = " ".join(list_items)
                    yield (list_text, section_hierarchy.copy())

                # Large list - split by items
                else:
                    # Yield previous chunk if exists
                    if current_chunk:
                        yield (" ".join(current_chunk), section_hierarchy.copy())
                        current_chunk = []
                        current_word_count = 0

                    # Split list by items
                    list_chunk = []
                    list_chunk_words = 0

                    for item in list_items:
                        item_words = len(item.split())

                        if list_chunk and (list_chunk_words + item_words) > words_per_chunk:
                            # Yield current list chunk
                            yield (" ".join(list_chunk), section_hierarchy.copy())
                            list_chunk = [item]
                            list_chunk_words = item_words
                        else:
                            list_chunk.append(item)
                            list_chunk_words += item_words

                    # Yield remaining list items
                    if list_chunk:
                        current_chunk = list_chunk
                        current_word_count = list_chunk_words

            elif seg_type == 'infobox':
                infobox_text = content

                # If infobox is small enough, try to merge with previous or next text
                if word_count < words_per_chunk // 2:
                    if current_chunk and (current_word_count + word_count) <= words_per_chunk * 1.2:
                        # Merge with previous chunk
                        current_chunk.append(f"Infobox: {infobox_text}")
                        current_word_count += word_count
                    else:
                        # Yield previous chunk and start new one with infobox
                        if current_chunk:
                            yield (" ".join(current_chunk), section_hierarchy.copy())
                        current_chunk = [f"Infobox: {infobox_text}"]
                        current_word_count = word_count

                # If infobox fits in one passage
                elif word_count <= words_per_chunk:
                    # Try to merge with previous chunk if it doesn't exceed limit too much
                    if current_chunk and (current_word_count + word_count) <= words_per_chunk * 1.2:
                        current_chunk.append(f"Infobox: {infobox_text}")
                        current_word_count += word_count
                    else:
                        # Yield previous chunk if exists
                        if current_chunk:
                            yield (" ".join(current_chunk), section_hierarchy.copy())
                        # Yield infobox as one passage
                        yield (f"Infobox: {infobox_text}", section_hierarchy.copy())
                        current_chunk = []
                        current_word_count = 0

                # Large infobox - split by key-value pairs
                else:
                    # Yield previous chunk if exists
                    if current_chunk:
                        yield (" ".join(current_chunk), section_hierarchy.copy())
                        current_chunk = []
                        current_word_count = 0

                    # Split infobox into key-value pairs
                    # Parse key-value pairs by finding pattern "key: value" where next key starts with a capital letter followed by colon
                    # or reaches end of string
                    key_value_pairs = []
                    current_pair = []
                    words = infobox_text.split()

                    for i, word in enumerate(words):
                        # Check if this word looks like a key (ends with colon)
                        if word.endswith(':') and i > 0 and current_pair:
                            # Save the previous pair
                            key_value_pairs.append(' '.join(current_pair))
                            current_pair = [word]
                        else:
                            current_pair.append(word)

                    # Add the last pair
                    if current_pair:
                        key_value_pairs.append(' '.join(current_pair))

                    # Now chunk by key-value pairs
                    infobox_chunk = ["Infobox:"]
                    infobox_chunk_words = 1

                    i_pair = 0
                    while i_pair < len(key_value_pairs):
                        pair = key_value_pairs[i_pair]
                        pair_words = len(pair.split())

                        if infobox_chunk_words + pair_words > words_per_chunk and len(infobox_chunk) > 1:
                            # Yield current infobox chunk
                            yield (" ".join(infobox_chunk), section_hierarchy.copy())
                            infobox_chunk = ["Infobox:", pair]
                            infobox_chunk_words = 1 + pair_words
                        else:
                            infobox_chunk.append(pair)
                            infobox_chunk_words += pair_words

                        i_pair += 1

                    # Add remaining infobox content to current chunk for potential merging with next text
                    if infobox_chunk and len(infobox_chunk) > 1:
                        current_chunk = infobox_chunk
                        current_word_count = infobox_chunk_words

        # Yield any remaining content from this section
        if current_chunk:
            yield (" ".join(current_chunk), section_hierarchy.copy())


def chunk_docs_context_aware(
    input_root=None,
    output_jsonl=None,
    words_per_passage=100,
    min_words=20,
    skip_empty_titles=False,
    progress_every=1000,
    preserve_structure=False
):
    """
    Split Wikipedia articles into context-aware passages.

    This function creates passages that respect semantic boundaries (sentences, lists,
    infoboxes) and preserves section hierarchy information for each passage.
    Chunks end at sentence boundaries, making each passage approximately
    words_per_passage words (may be slightly more or less).

    Args:
        input_root: Path to processed Wikipedia dump root
        output_jsonl: Output JSONL file path
        words_per_passage: Target number of words per passage (default: 100)
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

                    # Stable base id: prefer doc_id; if missing, hash title
                    if doc_id:
                        base = doc_id
                    else:
                        base = hashlib.md5(title.encode("utf-8", errors="ignore")).hexdigest()[:12]

                    chunk_idx = 0
                    for chunk_text, section_hierarchy in chunk_text_with_sections(raw_text, words_per_passage):
                        if len(chunk_text.split()) < min_words:
                            continue
                        pid = f"{base}-{chunk_idx:04d}"
                        rec = {
                            "id": pid,
                            "title": title,
                            "section": section_hierarchy,
                            "contents": chunk_text

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert Wikipedia dump to passage-based JSONL corpus"
    )
    parser.add_argument("--input-root", type=str, 
                        default="corpus_datasets/sample_wikiextractorV2",
                        help="Path to processed Wikipedia dump root")
    parser.add_argument("--output-jsonl", type=str, 
                        default="corpus_datasets/output.jsonl",
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

    # chunk_docs_by_words(
    #     input_root=args.input_root,
    #     output_jsonl=args.output_jsonl,
    #     words_per_passage=args.words_per_passage,
    #     min_words=args.min_words,
    #     skip_empty_titles=args.skip_empty_titles,
    #     progress_every=args.progress_every,
    #     preserve_structure=args.preserve_structure
    # )

    chunk_docs_context_aware(
        input_root=args.input_root,
        output_jsonl=args.output_jsonl,
        words_per_passage=args.words_per_passage,
        min_words=args.min_words,
        skip_empty_titles=args.skip_empty_titles,
        progress_every=args.progress_every,
        preserve_structure=args.preserve_structure
    )
    
    
    # python c2_corpus_creation/chunk_to_passages.py
