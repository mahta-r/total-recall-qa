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
                if chunk:  # filter out keep-alive chunks
                    file.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        percent = downloaded / total_size * 100
                        print(f"\rProgress: {percent:.2f}%", end="")
        print(f"\nDownload completed: {output_path}")


# Import chunking and passage conversion functions
from chunk_to_passages import chunk_docs_by_words, read_jsonl_line


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wikipedia dump processing pipeline")
    parser.add_argument("--step", type=str, required=True,
                        choices=["download", "extract", "convert"],
                        help="Which step to execute: download, extract, or convert")
    parser.add_argument("--dump-date", type=str, default="20251001",
                        help="Wikipedia dump date (default: 20251001)")
    parser.add_argument("--output-dir", type=str, default="/projects/0/prjs0834/heydars/INDICES",
                        help="Output directory for downloaded dump")
    
    # For convert step
    parser.add_argument("--input-root", type=str,
                        help="Path to processed Wikipedia dump root (default: based on dump-date)")
    parser.add_argument("--output-jsonl", type=str,
                        help="Output JSONL file path (default: based on dump-date)")
    parser.add_argument("--words-per-passage", type=int, default=100,
                        help="Number of words per passage (default: 100)")
    parser.add_argument("--min-words", type=int, default=20,
                        help="Skip passages with fewer words than this (default: 20)")
    parser.add_argument("--skip-empty-titles", action="store_true",
                        help="Skip docs without a title")
    parser.add_argument("--progress-every", type=int, default=10000,
                        help="Log every N passages (default: 10000)")
    
    args = parser.parse_args()

    dump_date = args.dump_date


    if args.step == "download":
        ### --- Step 1: Download the Wikipedia XML dump
        raw_dump_path = f"https://dumps.wikimedia.org/enwiki/{dump_date}/enwiki-{dump_date}-pages-articles.xml.bz2"
        output_path = f"{args.output_dir}/enwiki-{dump_date}-pages-articles.xml.bz2"
        download_raw_dump(raw_dump_path, output_path)
        print(f"\n[info] Download completed. Next step: extract")
        print(f"[info] Run: python c2_corpus_annotation/src/dump2corpus.py --step extract")

    elif args.step == "extract":
        ### --- Step 2: Extract clean text from a Wikipedia XML dump
        print("[info] Step 2: Extract clean text from Wikipedia XML dump")
        print("[info] Requirements:")
        print("  - conda create -n wiki_env python=3.8")
        print("  - conda activate wiki_env")
        print("  - pip install wikiextractor SoMaJo")
        print("\n[info] Run one of the following commands:")
        print(f"  python -m wikiextractor.WikiExtractor {args.output_dir}/enwiki-{dump_date}-pages-articles.xml.bz2 -o {args.output_dir}/enwiki-{dump_date} --processes 16")
        print(f"  python -m wikiextractor.WikiExtractor {args.output_dir}/enwiki-{dump_date}-pages-articles.xml.bz2 -o {args.output_dir}/enwiki-{dump_date} --processes 16 --no-templates")
        print("\n[info] Note: --no-templates is faster (~2h) but skips template preprocessing")
        print("[info] After extraction completes, run: python c2_corpus_annotation/src/dump2corpus.py --step convert")

    elif args.step == "convert":
        ### --- Step 3: Convert wiki dump to jsonl corpus
        print("[info] Step 3: Convert wiki dump to jsonl corpus")
        print("[info] This will run the chunk_docs_by_words() function")

        # Set defaults based on dump_date if not provided
        input_root = args.input_root or f"{args.output_dir}/enwiki-{dump_date}"
        output_jsonl = args.output_jsonl or f"{args.output_dir}/enwiki_{dump_date}.jsonl"

        chunk_docs_by_words(
            input_root=input_root,
            output_jsonl=output_jsonl,
            words_per_passage=args.words_per_passage,
            min_words=args.min_words,
            skip_empty_titles=args.skip_empty_titles,
            progress_every=args.progress_every
        )


# python c2_corpus_annotation/src/dump2corpus.py --step download
# python c2_corpus_annotation/src/dump2corpus.py --step extract
# python c2_corpus_annotation/src/dump2corpus.py --step convert
    
