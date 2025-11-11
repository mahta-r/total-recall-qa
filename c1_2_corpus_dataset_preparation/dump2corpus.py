### Src: https://github.com/GermanT5/wikipedia2corpus
###      https://github.com/attardi/wikiextractor


import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import io
import os
import re
import sys
import bz2
import json
import requests
import argparse
from pathlib import Path
from typing import Iterator, Tuple, Dict
from SPARQLWrapper import SPARQLWrapper, JSON



def download_raw_dump(input_url: str, output_path: str, chunk_size: int = 8192):
    """
    Downloads a file from the given URL and saves it to the specified output path.
    
    Args:
        input_url (str): The URL to download the file from.
        output_path (str): The path (including filename) to save the downloaded file.
        chunk_size (int): Size of chunks to stream (default: 8192 bytes).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(input_url, stream=True) as response:
        response.raise_for_status()  # raise error for bad responses
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        print(f"Downloading {input_url}")
        with open(output_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:  # filter out keep-alive chunks
                    file.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        percent = downloaded / total_size * 100
                        print(f"\rProgress: {percent:.2f}%", end="")
        print(f"\nDownload completed: {output_path}")


DOC_OPEN_RE = re.compile(r'<doc\b[^>]*?>', re.IGNORECASE)
DOC_CLOSE_RE = re.compile(r'</doc>', re.IGNORECASE)
ATTR_RE = re.compile(r'(\w+)="([^"]*)"')

# Simple cleaner: collapse whitespace, drop very short boilerplate sections
SECTION_HEADER_RE = re.compile(r'^\s*==+\s.*?==+\s*$', re.MULTILINE)

def open_maybe_bz2(path: Path) -> io.TextIOBase:
    if str(path).endswith(".bz2"):
        return io.TextIOWrapper(bz2.open(path, mode="rb"), encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")

def normalize_text(text: str) -> str:
    # Remove section headers lines like "== References ==", keep body
    text = SECTION_HEADER_RE.sub("\n", text)
    # Collapse whitespace/newlines to single spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def chunk_by_words(text: str, words_per_chunk: int) -> Iterator[str]:
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
    Yields (doc_id, title, text)
    For files formatted like:
      <doc id="123" url="..." title="Some title">
      text...
      </doc>
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
                # Parse attributes on the same line
                attrs = dict(ATTR_RE.findall(line))
                title = attrs.get("title", "").strip()
                doc_id = attrs.get("id", "").strip()
            # else keep scanning
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
    Yields (doc_id, title, text) for JSONL lines with at least a title and text/content.
    """
    for line in stream:
        line = line.strip()
        if not line:
            continue
        # Some WikiExtractor JSONL files may have trailing commas or stray characters;
        # attempt a tolerant load.
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
    with open_maybe_bz2(path) as fh:
        for line in fh:
            s = line.lstrip()
            if not s:
                continue
            if s.startswith("{"):
                return "jsonl"
            if "<doc" in s:
                return "xmlish"
            # fallback keep scanning a few lines
            break
    # default guess
    return "xmlish"

def iter_documents_from_file(path: Path) -> Iterator[Tuple[str, str, str]]:
    fmt = detect_format_first_line(path)
    with open_maybe_bz2(path) as fh:
        if fmt == "jsonl":
            yield from parse_jsonl(fh)
        else:
            yield from parse_wikiextractor_xmlish(fh)

def iter_all_chunk_files(root: Path) -> Iterator[Path]:
    # Common WikiExtractor layout is nested subdirs with files named wiki_**
    for dirpath, _, filenames in os.walk(root):
        for fn in sorted(filenames):
            if fn.startswith("wiki_") or fn.endswith(".bz2") or fn.endswith(".json") or fn.endswith(".jsonl"):
                yield Path(dirpath) / fn

def get_wikidata_qid(pageid=None, title=None):
    if not pageid and not title:
        raise ValueError("You must provide either a pageid or a title.")

    base_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "pageprops",
        "format": "json",
    }

    if pageid:
        params["pageids"] = pageid
    else:
        params["titles"] = title

    headers = {
        "User-Agent": "HeydarSoudaniBot/1.0 (heydar@example.com)"
        # replace with your email or website if you have one
    }

    response = requests.get(base_url, params=params, headers=headers)
    response.raise_for_status()
    data = response.json()

    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return None

    page_info = next(iter(pages.values()))
    if "missing" in page_info:
        return None

    return page_info.get("pageprops", {}).get("wikibase_item")


def to_passages():
    ap = argparse.ArgumentParser(description="Split Wikipedia articles into fixed-size passages and write JSONL.")
    ap.add_argument("--input_root", type=str, default="corpus_datasets/enwiki-20251001", help="Path to processed Wikipedia dump root (e.g., enwiki-20251001)")
    ap.add_argument("--output_jsonl", type=str, default="corpus_datasets/enwiki_20251001.jsonl", help="Output JSONL file path")
    ap.add_argument("--words-per-passage", type=int, default=100, help="Number of words per passage (default: 100)")
    ap.add_argument("--min-words", type=int, default=20, help="Skip passages with fewer words than this (default: 20)")
    ap.add_argument("--skip-empty-titles", action="store_true", help="Skip docs without a title")
    ap.add_argument("--progress-every", type=int, default=1000, help="Log every N passages (default: 1000)")
    args = ap.parse_args()

    input_root = Path(args.input_root)
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Create a unique passage id scheme: <doc_id or hash>-<chunk_idx>
    import hashlib

    total_articles = 0
    total_passages = 0

    with open(out_path, "w", encoding="utf-8") as out_f:
        for fpath in iter_all_chunk_files(input_root):
            try:
                for doc_id, title, raw_text in iter_documents_from_file(fpath):
                    if args.skip_empty_titles and not title:
                        continue
                    total_articles += 1

                    text = normalize_text(raw_text)
                    if not text:
                        continue

                    # Stable base id: prefer doc_id; if missing, hash title
                    if doc_id:
                        base = doc_id
                    else:
                        base = hashlib.md5(title.encode("utf-8", errors="ignore")).hexdigest()[:12]
                    # wikidata_qid = get_wikidata_qid(doc_id, title)

                    chunk_idx = 0
                    for chunk in chunk_by_words(text, args.words_per_passage):
                        # enforce min words after splitting
                        if len(chunk.split()) < args.min_words:
                            continue
                        pid = f"{base}-{chunk_idx:04d}"
                        rec: Dict[str, str] = {
                            "id": pid,
                            "title": title,
                            # "wikidata_qid": wikidata_qid,
                            "contents": chunk
                        }
                        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        total_passages += 1
                        chunk_idx += 1

                        if args.progress_every and (total_passages % args.progress_every == 0):
                            print(f"[info] {total_passages} passages written ...", file=sys.stderr)
            except Exception as e:
                # Keep going even if a file is malformed
                print(f"[warn] Skipping {fpath} due to error: {e}", file=sys.stderr)

    print(f"[done] Articles processed: {total_articles}", file=sys.stderr)
    print(f"[done] Passages written: {total_passages}", file=sys.stderr)

def read_jsonl_line(filepath: str, index: int) -> str:
    """
    Reads a specific line (by index) from a large JSONL file without loading the entire file.

    Args:
        filepath (str): Path to the .jsonl file.
        index (int): Zero-based index of the line to read.

    Returns:
        str: The line at the given index (including the newline at the end).
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == index:
                return line.strip()
    raise IndexError(f"Index {index} out of range for file {filepath}")


if __name__ == "__main__":
    
    dump_date = "20251001" # (25M pages, 18.5M articles)
    
    ### --- Step 1: Download the Wikipedia XML dump
    # raw_dump_path = f"https://dumps.wikimedia.org/enwiki/{dump_date}/enwiki-{dump_date}-pages-articles.xml.bz2"
    # output_path = f"data/enwiki-{dump_date}-pages-articles.xml.bz2"
    # download_raw_dump(raw_dump_path, output_path)
    
    
    ### --- Step 2: Extract clean text from a Wikipedia XML dump
    # conda create -n wiki_env python=3.8  -> The version is important. The wikiextractor has problem with more recent version of python
    # conda activate wiki_env
    # pip install wikiextractor SoMaJo
    
    # - Without --no-templates: a preprocessed step is added to get templates, it almost takes ~2h
    # - The main process takes ~2h
    # python -m wikiextractor.WikiExtractor data/enwiki-20251001-pages-articles.xml.bz2 -o data/enwiki-20251001 --processes 16
    # python -m wikiextractor.WikiExtractor data/enwiki-20251001-pages-articles.xml.bz2 -o data/enwiki-20251001 --processes 16 --no-templates
    
    # conda deactivate
    
    ### --- Step 3: Convert wiki dump to jsonl corpus
    # Src: Dense Passage Retrieval for Open-Domain Question Answering, EMNLP 2020
    to_passages()
    
    # index = 1
    # line = read_jsonl_line('downloads/enwiki_20251001.jsonl', index)
    # print(line)

    
# python c1_corpus_dataset_preparation/dump2corpus.py
    
