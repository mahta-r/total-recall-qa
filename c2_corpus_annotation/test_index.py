"""
Test script to verify the page index optimization works correctly.

This script tests the build_page_index.py and the optimized get_passages_for_page function.
"""

import json
import tempfile
import os
from pathlib import Path

# Test corpus data
TEST_CORPUS = [
    {"id": "12345-0000", "title": "Test Page 1", "contents": "First passage of page 12345"},
    {"id": "12345-0001", "title": "Test Page 1", "contents": "Second passage of page 12345"},
    {"id": "67890-0000", "title": "Test Page 2", "contents": "First passage of page 67890"},
    {"id": "12345-0002", "title": "Test Page 1", "contents": "Third passage of page 12345"},
    {"id": "99999-0000", "title": "Test Page 3", "contents": "Only passage of page 99999"},
]


def create_test_corpus(corpus_path: str):
    """Create a test corpus JSONL file."""
    with open(corpus_path, 'w', encoding='utf-8') as f:
        for entry in TEST_CORPUS:
            f.write(json.dumps(entry) + '\n')
    print(f"Created test corpus at {corpus_path}")


def test_build_index():
    """Test the index building functionality."""
    print("\n" + "="*80)
    print("TEST 1: Building Index")
    print("="*80)

    from build_page_index import build_page_to_line_index

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        corpus_path = f.name

    try:
        create_test_corpus(corpus_path)

        # Build index
        index = build_page_to_line_index(corpus_path, show_progress=False)

        # Verify index
        assert "12345" in index, "Page 12345 should be in index"
        assert "67890" in index, "Page 67890 should be in index"
        assert "99999" in index, "Page 99999 should be in index"

        assert len(index["12345"]) == 3, f"Page 12345 should have 3 passages, got {len(index['12345'])}"
        assert len(index["67890"]) == 1, f"Page 67890 should have 1 passage, got {len(index['67890'])}"
        assert len(index["99999"]) == 1, f"Page 99999 should have 1 passage, got {len(index['99999'])}"

        # Verify line numbers
        assert set(index["12345"]) == {0, 1, 3}, f"Page 12345 should be at lines 0,1,3, got {index['12345']}"
        assert index["67890"] == [2], f"Page 67890 should be at line 2, got {index['67890']}"
        assert index["99999"] == [4], f"Page 99999 should be at line 4, got {index['99999']}"

        print("✓ Index built correctly")
        print(f"  - Found {len(index)} unique pages")
        print(f"  - Page 12345 has passages at lines: {index['12345']}")
        print(f"  - Page 67890 has passages at lines: {index['67890']}")
        print(f"  - Page 99999 has passages at lines: {index['99999']}")

        return corpus_path, index

    except Exception as e:
        os.unlink(corpus_path)
        raise e


def test_get_passages(corpus_path: str, index: dict):
    """Test the optimized get_passages_for_page function."""
    print("\n" + "="*80)
    print("TEST 2: Getting Passages with Index")
    print("="*80)

    import sys
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    from qrel_generation import get_passages_for_page

    # Test with index
    passages_12345 = get_passages_for_page("12345", corpus_path, index)
    assert len(passages_12345) == 3, f"Should get 3 passages for page 12345, got {len(passages_12345)}"
    assert passages_12345[0]["id"] == "12345-0000", f"First passage should be 12345-0000"
    assert passages_12345[1]["id"] == "12345-0001", f"Second passage should be 12345-0001"
    assert passages_12345[2]["id"] == "12345-0002", f"Third passage should be 12345-0002"
    print(f"✓ Retrieved 3 passages for page 12345")

    passages_67890 = get_passages_for_page("67890", corpus_path, index)
    assert len(passages_67890) == 1, f"Should get 1 passage for page 67890, got {len(passages_67890)}"
    assert passages_67890[0]["id"] == "67890-0000", f"Passage should be 67890-0000"
    print(f"✓ Retrieved 1 passage for page 67890")

    passages_99999 = get_passages_for_page("99999", corpus_path, index)
    assert len(passages_99999) == 1, f"Should get 1 passage for page 99999, got {len(passages_99999)}"
    assert passages_99999[0]["id"] == "99999-0000", f"Passage should be 99999-0000"
    print(f"✓ Retrieved 1 passage for page 99999")

    # Test non-existent page
    passages_none = get_passages_for_page("00000", corpus_path, index)
    assert len(passages_none) == 0, f"Should get 0 passages for non-existent page, got {len(passages_none)}"
    print(f"✓ Retrieved 0 passages for non-existent page")


def test_without_index(corpus_path: str):
    """Test fallback without index (should still work but with warning)."""
    print("\n" + "="*80)
    print("TEST 3: Getting Passages without Index (Fallback)")
    print("="*80)

    import sys
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    from qrel_generation import get_passages_for_page

    # Test without index
    passages_12345 = get_passages_for_page("12345", corpus_path, None)
    assert len(passages_12345) == 3, f"Should get 3 passages for page 12345, got {len(passages_12345)}"
    print(f"✓ Fallback mode works: Retrieved 3 passages for page 12345")


def main():
    """Run all tests."""
    print("="*80)
    print("Page Index Optimization Tests")
    print("="*80)

    corpus_path = None
    try:
        # Test 1: Build index
        corpus_path, index = test_build_index()

        # Test 2: Get passages with index
        test_get_passages(corpus_path, index)

        # Test 3: Get passages without index
        test_without_index(corpus_path)

        print("\n" + "="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80)

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        # Cleanup
        if corpus_path and os.path.exists(corpus_path):
            os.unlink(corpus_path)
            print(f"\nCleaned up test corpus")

    return 0


if __name__ == "__main__":
    exit(main())
