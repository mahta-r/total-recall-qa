### Src: https://github.com/GermanT5/wikipedia2corpus
###      https://github.com/attardi/wikiextractor
### Enhanced version with infobox handling using mwparserfromhell + WikiExtractor

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import io
import bz2
import json
import requests
import argparse
import tempfile
import subprocess
import mwxml
import mwparserfromhell
from pathlib import Path
from typing import Iterator, Tuple, Dict
from tqdm import tqdm


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
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        print(f"Downloading {input_url}")
        with open(output_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    file.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        percent = downloaded / total_size * 100
                        print(f"\rProgress: {percent:.2f}%", end="")
        print(f"\nDownload completed: {output_path}")

def create_subsample(input_dump_path: str, output_dump_path: str, num_pages: int = 1000):
    """
    Creates a subsample of the Wikipedia dump for testing.
    Extracts the first N pages from the full dump.

    Args:
        input_dump_path (str): Path to full Wikipedia dump (bz2 format)
        output_dump_path (str): Path to output subsample dump (bz2 format)
        num_pages (int): Number of pages to include in subsample (default: 1000)
    """
    print(f"[info] Creating subsample from: {input_dump_path}")
    print(f"[info] Output will be saved to: {output_dump_path}")
    print(f"[info] Number of pages: {num_pages}")

    output_path = Path(output_dump_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pages_extracted = 0
    in_mediawiki = False
    in_page = False
    page_buffer = []
    header_buffer = []

    with bz2.open(input_dump_path, "rt", encoding="utf-8") as input_file:
        with bz2.open(output_dump_path, "wt", encoding="utf-8") as output_file:
            for line in tqdm(input_file, desc="Extracting pages", total=num_pages):
                # Capture XML header/namespace definitions
                if "<mediawiki" in line:
                    in_mediawiki = True
                    header_buffer.append(line)
                    continue

                # Continue capturing header until first page
                if in_mediawiki and not in_page and "<page>" not in line:
                    header_buffer.append(line)
                    continue

                # Write header on first page encounter
                if header_buffer and "<page>" in line:
                    output_file.writelines(header_buffer)
                    header_buffer = []

                # Detect page start
                if "<page>" in line:
                    in_page = True
                    page_buffer = [line]
                    continue

                # Collect page content
                if in_page:
                    page_buffer.append(line)

                    # Detect page end
                    if "</page>" in line:
                        in_page = False

                        # Write the page
                        output_file.writelines(page_buffer)
                        pages_extracted += 1

                        if pages_extracted >= num_pages:
                            # Write closing tag
                            output_file.write("</mediawiki>\n")
                            break

                        page_buffer = []

    print(f"\n[info] Subsample created: {pages_extracted} pages")
    print(f"[info] Output file: {output_dump_path}")
    return output_dump_path

def preprocess_dump_with_infoboxes(input_dump_path: str, output_dump_path: str, max_pages: int = None):
    """
    Preprocesses Wikipedia dump to convert infoboxes to readable text format.
    Uses mwparserfromhell to parse and transform infobox templates.

    Args:
        input_dump_path (str): Path to input Wikipedia dump (bz2 format)
        output_dump_path (str): Path to output preprocessed dump (bz2 format)
        max_pages (int): Maximum number of pages to process (None for all)
    """
    print(f"[info] Preprocessing dump with infobox handling: {input_dump_path}")
    print(f"[info] Output will be saved to: {output_dump_path}")

    output_path = Path(output_dump_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pages_processed = 0
    pages_with_infoboxes = 0

    with bz2.open(input_dump_path, "rt", encoding="utf-8") as input_file:
        with bz2.open(output_dump_path, "wt", encoding="utf-8") as output_file:
            # Read and write XML header
            in_page = False
            page_buffer = []

            for line in tqdm(input_file, desc="Preprocessing pages"):
                # Detect page start
                if "<page>" in line:
                    in_page = True
                    page_buffer = [line]
                    continue

                # Collect page content
                if in_page:
                    page_buffer.append(line)

                    # Detect page end
                    if "</page>" in line:
                        in_page = False

                        # Process the page
                        page_xml = "".join(page_buffer)
                        processed_page = process_page_infoboxes(page_xml)
                        output_file.write(processed_page)

                        pages_processed += 1
                        if processed_page != page_xml:
                            pages_with_infoboxes += 1

                        if max_pages and pages_processed >= max_pages:
                            break

                        page_buffer = []
                else:
                    # Write non-page content (headers, etc.) directly
                    output_file.write(line)

    print(f"\n[info] Preprocessing completed!")
    print(f"[info] Total pages processed: {pages_processed}")
    print(f"[info] Pages with infoboxes: {pages_with_infoboxes}")

def process_page_infoboxes(page_xml: str) -> str:
    """
    Process a single Wikipedia page to convert infoboxes to readable text.

    Args:
        page_xml (str): XML content of a single Wikipedia page

    Returns:
        str: Modified XML with infoboxes converted to text
    """
    try:
        # Extract the text content between <text> tags
        text_start = page_xml.find("<text")
        if text_start == -1:
            return page_xml

        # Find the end of the opening tag
        text_tag_end = page_xml.find(">", text_start)
        text_end = page_xml.find("</text>", text_tag_end)

        if text_tag_end == -1 or text_end == -1:
            return page_xml

        # Extract the actual wiki text content
        wiki_text = page_xml[text_tag_end + 1:text_end]

        # Parse with mwparserfromhell
        wikicode = mwparserfromhell.parse(wiki_text)
        templates = wikicode.filter_templates()

        # Track if any infoboxes were found
        modified = False

        # Process templates in reverse order to handle nested templates correctly
        for template in reversed(templates):
            name = template.name.strip().lower()

            if name.startswith("infobox"):
                modified = True
                # Convert infobox to readable text format
                infobox_text_lines = [f"== {template.name.strip()} =="]

                for param in template.params:
                    param_name = param.name.strip()
                    param_value = param.value.strip()

                    # Skip empty values
                    if not param_value:
                        continue

                    # Format as "name: value"
                    infobox_text_lines.append(f"{param_name}: {param_value}")

                infobox_text = "\n".join(infobox_text_lines) + "\n"

                try:
                    # Try to replace the template node
                    wikicode.replace(template, infobox_text)
                except ValueError:
                    try:
                        # Fallback: replace as string
                        wikicode.replace(str(template), infobox_text)
                    except:
                        # If replacement fails, keep original
                        pass

        if modified:
            # Reconstruct the page XML with modified text
            modified_wiki_text = str(wikicode)
            modified_page = (
                page_xml[:text_tag_end + 1] +
                modified_wiki_text +
                page_xml[text_end:]
            )
            return modified_page

        return page_xml

    except Exception as e:
        # If any error occurs, return original page
        print(f"[warn] Error processing page: {e}", file=sys.stderr)
        return page_xml

def extract_with_wikiextractor(input_dump_path: str, output_dir: str, processes: int = 16, no_templates: bool = False):
    """
    Runs WikiExtractor on the preprocessed dump.

    Args:
        input_dump_path (str): Path to preprocessed Wikipedia dump
        output_dir (str): Output directory for extracted text
        processes (int): Number of parallel processes
        no_templates (bool): Skip template preprocessing (faster)
    """
    print(f"[info] Extracting Wikipedia dump with WikiExtractor")
    print(f"[info] Input: {input_dump_path}")
    print(f"[info] Output: {output_dir}")

    cmd = [
        "python", "-m", "wikiextractor.WikiExtractor",
        input_dump_path,
        "-o", output_dir,
        "--processes", str(processes)
    ]

    if no_templates:
        cmd.append("--no-templates")

    print(f"[info] Running command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        print(f"[info] WikiExtractor completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"[error] WikiExtractor failed: {e}", file=sys.stderr)
        print(f"[error] stdout: {e.stdout}", file=sys.stderr)
        print(f"[error] stderr: {e.stderr}", file=sys.stderr)
        raise


# Import corpus conversion functions from dump2corpus.py
import re

DOC_OPEN_RE = re.compile(r'<doc\b[^>]*?>', re.IGNORECASE)
DOC_CLOSE_RE = re.compile(r'</doc>', re.IGNORECASE)
ATTR_RE = re.compile(r'(\w+)="([^"]*)"')
SECTION_HEADER_RE = re.compile(r'^\s*==+\s.*?==+\s*$', re.MULTILINE)


def open_maybe_bz2(path: Path) -> io.TextIOBase:
    if str(path).endswith(".bz2"):
        return io.TextIOWrapper(bz2.open(path, mode="rb"), encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")


def normalize_text(text: str) -> str:
    # Keep section headers for infobox information, but normalize them
    text = SECTION_HEADER_RE.sub(lambda m: m.group(0).strip() + " ", text)
    # Collapse excessive whitespace but preserve some structure
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()


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
    if in_doc and buf:
        yield (doc_id or "", title or "", "".join(buf))


def parse_jsonl(stream: io.TextIOBase) -> Iterator[Tuple[str, str, str]]:
    """
    Yields (doc_id, title, text) for JSONL lines.
    """
    for line in stream:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
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
            break
    return "xmlish"


def iter_documents_from_file(path: Path) -> Iterator[Tuple[str, str, str]]:
    fmt = detect_format_first_line(path)
    with open_maybe_bz2(path) as fh:
        if fmt == "jsonl":
            yield from parse_jsonl(fh)
        else:
            yield from parse_wikiextractor_xmlish(fh)


def iter_all_chunk_files(root: Path) -> Iterator[Path]:
    for dirpath, _, filenames in os.walk(root):
        for fn in sorted(filenames):
            if fn.startswith("wiki_") or fn.endswith(".bz2") or fn.endswith(".json") or fn.endswith(".jsonl"):
                yield Path(dirpath) / fn


def to_passages(input_root=None, output_jsonl=None, words_per_passage=100, min_words=20,
                skip_empty_titles=False, progress_every=1000):
    """
    Split Wikipedia articles into fixed-size passages and write JSONL.

    Args:
        input_root: Path to processed Wikipedia dump root
        output_jsonl: Output JSONL file path
        words_per_passage: Number of words per passage (default: 100)
        min_words: Skip passages with fewer words than this (default: 20)
        skip_empty_titles: Skip docs without a title (default: False)
        progress_every: Log every N passages (default: 1000)
    """
    if input_root is None or output_jsonl is None:
        raise ValueError("input_root and output_jsonl are required arguments")

    input_root_path = Path(input_root)
    out_path = Path(output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import hashlib

    total_articles = 0
    total_passages = 0

    with open(out_path, "w", encoding="utf-8") as out_f:
        for fpath in iter_all_chunk_files(input_root_path):
            try:
                for doc_id, title, raw_text in iter_documents_from_file(fpath):
                    if skip_empty_titles and not title:
                        continue
                    total_articles += 1

                    text = normalize_text(raw_text)
                    if not text:
                        continue

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wikipedia dump processing with infobox handling")
    parser.add_argument("--step", type=str, required=True,
                        choices=["download", "subsample", "preprocess", "extract", "convert", "full", "test"],
                        help="Which step to execute: download, subsample, preprocess, extract, convert, full (all steps), or test (test pipeline on subsample)")
    parser.add_argument("--dump-date", type=str, default="20251001",
                        help="Wikipedia dump date (default: 20251001)")
    parser.add_argument("--output-dir", type=str, default="/projects/0/prjs0834/heydars/INDICES",
                        help="Output directory for dumps and corpus")

    # For subsample step
    parser.add_argument("--num-pages", type=int, default=1000,
                        help="Number of pages for subsample (default: 1000)")
    parser.add_argument("--subsample-dump", type=str,
                        help="Path to subsample dump output (default: based on dump-date)")

    # For preprocess step
    parser.add_argument("--input-dump", type=str,
                        help="Path to input Wikipedia dump (default: based on dump-date)")
    parser.add_argument("--preprocessed-dump", type=str,
                        help="Path to preprocessed dump (default: based on dump-date)")
    parser.add_argument("--max-pages", type=int,
                        help="Maximum number of pages to preprocess (for testing)")

    # For extract step
    parser.add_argument("--extracted-dir", type=str,
                        help="Directory for extracted text (default: based on dump-date)")
    parser.add_argument("--processes", type=int, default=16,
                        help="Number of parallel processes for WikiExtractor (default: 16)")
    parser.add_argument("--no-templates", action="store_true",
                        help="Skip template preprocessing in WikiExtractor (faster)")

    # For convert step
    parser.add_argument("--output-jsonl", type=str,
                        help="Output JSONL corpus file (default: based on dump-date)")
    parser.add_argument("--words-per-passage", type=int, default=100,
                        help="Number of words per passage (default: 100)")
    parser.add_argument("--min-words", type=int, default=20,
                        help="Skip passages with fewer words (default: 20)")
    parser.add_argument("--skip-empty-titles", action="store_true",
                        help="Skip docs without a title")
    parser.add_argument("--progress-every", type=int, default=10000,
                        help="Log every N passages (default: 10000)")

    args = parser.parse_args()

    dump_date = args.dump_date

    # Set up default paths
    raw_dump_path = args.input_dump or f"{args.output_dir}/enwiki-{dump_date}-pages-articles.xml.bz2"
    subsample_dump_path = args.subsample_dump or f"{args.output_dir}/enwiki-{dump_date}-subsample.xml.bz2"

    # For test mode, use subsample paths
    if args.step == "test":
        preprocessed_dump_path = args.preprocessed_dump or f"{args.output_dir}/enwiki-{dump_date}-subsample-preprocessed.xml.bz2"
        extracted_dir = args.extracted_dir or f"{args.output_dir}/enwiki-{dump_date}-subsample-infoboxes"
        output_jsonl = args.output_jsonl or f"{args.output_dir}/enwiki_{dump_date}_subsample_with_infoboxes.jsonl"
    else:
        preprocessed_dump_path = args.preprocessed_dump or f"{args.output_dir}/enwiki-{dump_date}-preprocessed.xml.bz2"
        extracted_dir = args.extracted_dir or f"{args.output_dir}/enwiki-{dump_date}-infoboxes"
        output_jsonl = args.output_jsonl or f"{args.output_dir}/enwiki_{dump_date}_with_infoboxes.jsonl"

    if args.step == "download" or args.step == "full":
        print("\n" + "="*80)
        print("STEP 1: Download Wikipedia dump")
        print("="*80)
        dump_url = f"https://dumps.wikimedia.org/enwiki/{dump_date}/enwiki-{dump_date}-pages-articles.xml.bz2"
        download_raw_dump(dump_url, raw_dump_path)
        print(f"[info] Download completed: {raw_dump_path}")

        if args.step != "full":
            print(f"\n[info] Next step: python {__file__} --step subsample --dump-date {dump_date} --num-pages 1000")
            print(f"[info] Or go directly to: python {__file__} --step preprocess --dump-date {dump_date}")

    if args.step == "subsample":
        print("\n" + "="*80)
        print("STEP 2: Create subsample for testing")
        print("="*80)
        create_subsample(raw_dump_path, subsample_dump_path, args.num_pages)
        print(f"[info] Subsample created: {subsample_dump_path}")
        print(f"\n[info] To test the full pipeline on this subsample, run:")
        print(f"[info] python {__file__} --step test --dump-date {dump_date}")

    if args.step == "test":
        print("\n" + "="*80)
        print("TEST MODE: Running full pipeline on subsample")
        print("="*80)
        print(f"[info] Using subsample: {subsample_dump_path}")

        # Run all steps on subsample
        print("\n" + "-"*80)
        print("Step 1/3: Preprocessing subsample with infobox handling")
        print("-"*80)
        preprocess_dump_with_infoboxes(subsample_dump_path, preprocessed_dump_path, args.max_pages)

        print("\n" + "-"*80)
        print("Step 2/3: Extracting text with WikiExtractor")
        print("-"*80)
        extract_with_wikiextractor(preprocessed_dump_path, extracted_dir, args.processes, args.no_templates)

        print("\n" + "-"*80)
        print("Step 3/3: Converting to JSONL corpus")
        print("-"*80)
        to_passages(
            input_root=extracted_dir,
            output_jsonl=output_jsonl,
            words_per_passage=args.words_per_passage,
            min_words=args.min_words,
            skip_empty_titles=args.skip_empty_titles,
            progress_every=args.progress_every
        )

        print("\n" + "="*80)
        print("TEST COMPLETED!")
        print("="*80)
        print(f"[info] Test corpus created: {output_jsonl}")
        print(f"[info] Review the output to verify infoboxes are properly included")
        print(f"\n[info] If the output looks good, run the full pipeline:")
        print(f"[info] python {__file__} --step full --dump-date {dump_date}")

    if args.step == "preprocess" or args.step == "full":
        print("\n" + "="*80)
        print("STEP 2: Preprocess dump with infobox handling")
        print("="*80)
        preprocess_dump_with_infoboxes(raw_dump_path, preprocessed_dump_path, args.max_pages)
        print(f"[info] Preprocessing completed: {preprocessed_dump_path}")

        if args.step != "full":
            print(f"\n[info] Next step: python {__file__} --step extract --dump-date {dump_date}")

    if args.step == "extract" or args.step == "full":
        print("\n" + "="*80)
        print("STEP 3: Extract text with WikiExtractor")
        print("="*80)
        print("[info] Requirements: pip install wikiextractor")
        extract_with_wikiextractor(preprocessed_dump_path, extracted_dir, args.processes, args.no_templates)
        print(f"[info] Extraction completed: {extracted_dir}")

        if args.step != "full":
            print(f"\n[info] Next step: python {__file__} --step convert --dump-date {dump_date}")

    if args.step == "convert" or args.step == "full":
        print("\n" + "="*80)
        print("STEP 4: Convert to passage-based JSONL corpus")
        print("="*80)
        to_passages(
            input_root=extracted_dir,
            output_jsonl=output_jsonl,
            words_per_passage=args.words_per_passage,
            min_words=args.min_words,
            skip_empty_titles=args.skip_empty_titles,
            progress_every=args.progress_every
        )
        print(f"[info] Corpus created: {output_jsonl}")

    if args.step == "full":
        print("\n" + "="*80)
        print("ALL STEPS COMPLETED!")
        print("="*80)
        print(f"Final corpus: {output_jsonl}")


# Usage examples:
#
# RECOMMENDED WORKFLOW (test on subsample first):
# 1. Download the full dump
# python c2_corpus_annotation/dump2corpus_infoboxes.py --step download --dump-date 20251001
#
# 2. Create a subsample for testing (e.g., 1000 pages)
# python c2_corpus_annotation/dump2corpus_infoboxes.py --step subsample --dump-date 20251001 --num-pages 1000
#
# 3. Test the entire pipeline on the subsample
# python c2_corpus_annotation/dump2corpus_infoboxes.py --step test --dump-date 20251001
#
# 4. Review the output at: /projects/0/prjs0834/heydars/INDICES/enwiki_20251001_subsample_with_infoboxes.jsonl
#
# 5. If output looks good, run on full dump
# python c2_corpus_annotation/dump2corpus_infoboxes.py --step full --dump-date 20251001
#
# ALTERNATIVE: Step-by-step execution on full dump:
# python c2_corpus_annotation/dump2corpus_infoboxes.py --step download --dump-date 20251001
# python c2_corpus_annotation/dump2corpus_infoboxes.py --step preprocess --dump-date 20251001
# python c2_corpus_annotation/dump2corpus_infoboxes.py --step extract --dump-date 20251001
# python c2_corpus_annotation/dump2corpus_infoboxes.py --step convert --dump-date 20251001
