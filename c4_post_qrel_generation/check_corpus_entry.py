"""
Show original and rewritten passage for a given passage ID.

Scans both corpus files until the passage ID is found.
Corpus paths (relative to repo root):
  - corpus_datasets/corpus/enwiki_20251001_infoboxconv.jsonl (original)
  - corpus_datasets/corpus/enwiki_20251001_infoboxconv_rewritten.jsonl (rewritten)
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple


def get_passage_by_id(
    passage_id: str,
    corpus_original: str = "corpus_datasets/corpus/enwiki_20251001_infoboxconv.jsonl",
    corpus_rewritten: str = "corpus_datasets/corpus/enwiki_20251001_infoboxconv_rewritten.jsonl",
) -> Tuple[Optional[dict], Optional[dict]]:
    """
    Given a passage ID, return (original_passage, rewritten_passage) from the two corpora.

    Scans both files in parallel until a line with matching id is found in the original corpus,
    then returns that line and the corresponding line from the rewritten corpus.

    Returns:
        (original_passage_dict, rewritten_passage_dict). Either can be None if not found.
    """
    path_orig = Path(corpus_original)
    path_rew = Path(corpus_rewritten)
    return _find_passage_by_scan(passage_id, path_orig, path_rew)


def _find_passage_by_scan(
    passage_id: str, path_orig: Path, path_rew: Path
) -> Tuple[Optional[dict], Optional[dict]]:
    """Scan both files until the passage with the given id is found."""
    orig_passage = None
    rew_passage = None

    with open(path_orig, "r", encoding="utf-8") as fo, open(
        path_rew, "r", encoding="utf-8"
    ) as fr:
        for line_orig, line_rew in zip(fo, fr):
            line_orig = line_orig.strip()
            line_rew = line_rew.strip()
            if not line_orig:
                continue
            obj_orig = json.loads(line_orig)
            if obj_orig.get("id") == passage_id:
                orig_passage = obj_orig
                if line_rew:
                    rew_passage = json.loads(line_rew)
                break
    return (orig_passage, rew_passage)


def format_passage(p: dict, label: str) -> str:
    """Format a passage dict for printing."""
    if p is None:
        return f"--- {label} ---\n(not found)\n"
    lines = [
        f"--- {label} ---",
        f"id: {p.get('id', '')}",
        f"title: {p.get('title', '')}",
        "",
        "contents:",
        (p.get("contents") or p.get("text") or "").strip(),
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Show original and rewritten passage for a given passage ID"
    )
    parser.add_argument(
        "--corpus-original",
        type=str,
        default="corpus_datasets/corpus/enwiki_20251001_infoboxconv.jsonl",
        help="Path to original corpus JSONL",
    )
    parser.add_argument(
        "--corpus-rewritten",
        type=str,
        default="corpus_datasets/corpus/enwiki_20251001_infoboxconv_rewritten.jsonl",
        help="Path to rewritten corpus JSONL",
    )
    args = parser.parse_args()
    
    
    passage_id = "932171-0000"
    
    orig, rew = get_passage_by_id(
        passage_id,
        corpus_original=args.corpus_original,
        corpus_rewritten=args.corpus_rewritten,
    )

    print(format_passage(orig, "Original"))
    print(format_passage(rew, "Rewritten"))


if __name__ == "__main__":
    main()
    

# python c4_post_qrel_generation/check_corpus_entry.py