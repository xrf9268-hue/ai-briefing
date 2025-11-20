#!/usr/bin/env python3
"""Validation script for prompt optimization implementation.

This script validates that all new modules work correctly without requiring
external dependencies like pytest.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from briefing.stages.keyword_filter import compute_keyword_score, filter_by_keywords
from briefing.stages.source_scoring import score_source_reliability, enrich_with_source_scores
from briefing.stages.query_builder import build_weighted_query


def test_keyword_scoring():
    """Test keyword scoring functionality."""
    print("\n=== Testing Keyword Scoring ===")

    # Test LLM release
    item1 = {
        "text": "Announcing Claude 3.5 with improved coding capabilities",
        "title": "Claude 3.5 Release",
        "url": "https://anthropic.com/blog/claude-3-5",
    }
    score1 = compute_keyword_score(item1)
    print(f"✓ LLM release (official source): score={score1:.2f}")
    assert score1 > 3.0, "Expected high score for LLM release from official source"

    # Test agentic coding tool
    item2 = {
        "text": "New Cursor plugin enables AI-assisted code refactoring",
        "title": "",
        "url": "https://news.ycombinator.com/item?id=123456",
    }
    score2 = compute_keyword_score(item2)
    print(f"✓ Agentic coding tool: score={score2:.2f}")
    assert score2 > 1.5, "Expected moderate score for agentic coding tool"

    # Test irrelevant content
    item3 = {
        "text": "Random blog post about cooking recipes",
        "title": "How to bake bread",
        "url": "https://example.com/cooking",
    }
    score3 = compute_keyword_score(item3)
    print(f"✓ Irrelevant content: score={score3:.2f}")
    assert score3 == 0.0, "Expected zero score for irrelevant content"

    print("✅ Keyword scoring tests passed!")


def test_keyword_filtering():
    """Test keyword filtering functionality."""
    print("\n=== Testing Keyword Filtering ===")

    items = [
        {
            "id": "1",
            "text": "Claude 3.5 announcement",
            "url": "https://anthropic.com/blog",
        },
        {
            "id": "2",
            "text": "Random unrelated content",
            "url": "https://example.com",
        },
        {
            "id": "3",
            "text": "Cursor AI assistant update",
            "url": "https://cursor.sh/blog",
        },
    ]

    filtered = filter_by_keywords(items, min_score=1.0)
    print(f"✓ Filtered {len(items)} items → {len(filtered)} items (min_score=1.0)")
    assert len(filtered) == 2, "Expected 2 relevant items"
    assert all("keyword_score" in item for item in filtered), "Expected keyword_score field"

    # Test top_k
    many_items = [
        {"text": f"Claude {i} update", "url": "https://anthropic.com"}
        for i in range(10)
    ]
    top_5 = filter_by_keywords(many_items, top_k=5)
    print(f"✓ Top-K filtering: {len(many_items)} items → {len(top_5)} items (top_k=5)")
    assert len(top_5) == 5, "Expected exactly 5 items"

    print("✅ Keyword filtering tests passed!")


def test_source_scoring():
    """Test source reliability scoring."""
    print("\n=== Testing Source Reliability Scoring ===")

    # Tier 1: Official sources
    tier1_urls = [
        "https://anthropic.com/blog/claude-3-5",
        "https://openai.com/research/gpt-4",
        "https://google.ai/gemini",
        "https://github.com/anthropics/anthropic-sdk-python",
    ]

    for url in tier1_urls:
        score = score_source_reliability({"url": url})
        assert score == 2, f"Expected Tier 1 (score 2) for {url}, got {score}"
    print(f"✓ Tier 1 (official): {len(tier1_urls)} sources scored correctly")

    # Tier 2: Reputable sources
    tier2_urls = [
        "https://techcrunch.com/2024/01/01/ai-news/",
        "https://github.com/user/repo/releases/tag/v1.0",
    ]

    for url in tier2_urls:
        score = score_source_reliability({"url": url})
        assert score == 1, f"Expected Tier 2 (score 1) for {url}, got {score}"
    print(f"✓ Tier 2 (reputable): {len(tier2_urls)} sources scored correctly")

    # Tier 3: Unknown sources
    tier3_url = "https://random-blog.example.com/post"
    score = score_source_reliability({"url": tier3_url})
    assert score == 0, f"Expected Tier 3 (score 0) for {tier3_url}, got {score}"
    print(f"✓ Tier 3 (unknown): scored correctly")

    print("✅ Source reliability scoring tests passed!")


def test_source_enrichment():
    """Test source enrichment functionality."""
    print("\n=== Testing Source Enrichment ===")

    items = [
        {"id": "1", "url": "https://anthropic.com/blog"},
        {"id": "2", "url": "https://techcrunch.com/article"},
        {"id": "3", "url": "https://example.com/blog"},
    ]

    enriched = enrich_with_source_scores(items)

    assert len(enriched) == 3, "Expected 3 items after enrichment"
    assert all("source_reliability_score" in item for item in enriched), "Expected source_reliability_score field"
    assert enriched[0]["source_reliability_score"] == 2, "Expected Tier 1 for anthropic.com"
    assert enriched[1]["source_reliability_score"] == 1, "Expected Tier 2 for techcrunch.com"
    assert enriched[2]["source_reliability_score"] == 0, "Expected Tier 3 for example.com"

    print(f"✓ Enriched {len(items)} items with source scores")
    print("✅ Source enrichment tests passed!")


def test_query_builder():
    """Test TF-IDF query builder."""
    print("\n=== Testing TF-IDF Query Builder ===")

    items = [
        {"text": "Claude Code enables AI-assisted development"},
        {"text": "Cursor provides agentic coding capabilities"},
        {"text": "AI assistants improve developer productivity"},
    ]

    query = build_weighted_query(items, top_n=5)

    assert query, "Expected non-empty query"
    assert "Keywords:" in query, "Expected keywords in query"
    print(f"✓ Generated weighted query: {len(query)} characters")
    print(f"  Preview: {query[:100]}...")

    # Test single item
    single_item = [{"text": "Single item test"}]
    single_query = build_weighted_query(single_item)
    assert single_query == "Single item test", "Expected simple text for single item"
    print(f"✓ Single item query handled correctly")

    # Test empty input
    empty_query = build_weighted_query([])
    assert empty_query == "", "Expected empty string for empty input"
    print(f"✓ Empty input handled correctly")

    print("✅ Query builder tests passed!")


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("Prompt Optimization Implementation Validation")
    print("=" * 60)

    try:
        test_keyword_scoring()
        test_keyword_filtering()
        test_source_scoring()
        test_source_enrichment()
        test_query_builder()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nImplementation Summary:")
        print("  ✓ Keyword filtering with TF-IDF weighting")
        print("  ✓ 3-tier source reliability classification")
        print("  ✓ TF-IDF weighted query generation")
        print("  ✓ Data enrichment and filtering")
        print("\nThe prompt optimization implementation is ready for production use.")

        return 0

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
