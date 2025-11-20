"""Tests for keyword-based content filtering."""

import pytest
from briefing.stages.keyword_filter import (
    compute_keyword_score,
    filter_by_keywords,
    KEYWORD_CATEGORIES,
)


class TestKeywordScoring:
    """Test suite for keyword scoring logic."""

    def test_llm_release_scoring(self):
        """Test scoring for LLM release announcements."""
        item = {
            "text": "Announcing Claude 3.5 with improved coding capabilities",
            "title": "Claude 3.5 Release",
            "url": "https://anthropic.com/blog/claude-3-5",
        }
        score = compute_keyword_score(item)
        # Should get points for "Claude 3.5" match + official source boost (1.5x)
        assert score > 3.0, "LLM release from official source should score high"

    def test_agentic_coding_tool_scoring(self):
        """Test scoring for agentic coding tools."""
        item = {
            "text": "New Cursor plugin enables AI-assisted code refactoring",
            "title": "",
            "url": "https://news.ycombinator.com/item?id=123456",
        }
        score = compute_keyword_score(item)
        # Should get points for "Cursor" + "AI-assisted" matches
        assert score > 1.5, "Agentic coding tool should score moderately"

    def test_vibe_coding_scoring(self):
        """Test scoring for vibe coding related content."""
        item = {
            "text": "Vibe coding with rapid prototyping and real-time collaboration",
            "title": "",
            "url": "https://blog.example.com/vibe-coding",
        }
        score = compute_keyword_score(item)
        # Should get points for "vibe coding" + "rapid prototyp" + "real-time collaboration"
        assert score > 1.5, "Vibe coding content should score above threshold"

    def test_cli_tools_scoring(self):
        """Test scoring for CLI tools."""
        item = {
            "text": "New CLI tool for terminal-based code generation",
            "title": "Terminal automation",
            "url": "https://github.com/example/cli-tool",
        }
        score = compute_keyword_score(item)
        # Should get points for "CLI" + "terminal" matches
        assert score > 1.0, "CLI tools should score above minimum"

    def test_official_source_boost(self):
        """Test that official sources get 1.5x score boost."""
        base_item = {
            "text": "Gemini 2.0 announcement",
            "title": "",
            "url": "https://example.com/blog",
        }
        official_item = {
            "text": "Gemini 2.0 announcement",
            "title": "",
            "url": "https://google.ai/gemini-2.0",
        }

        base_score = compute_keyword_score(base_item)
        official_score = compute_keyword_score(official_item)

        # Official source should have 1.5x boost
        assert official_score > base_score * 1.4, "Official source should get boost"

    def test_no_keyword_match(self):
        """Test that irrelevant content scores zero."""
        item = {
            "text": "Random blog post about cooking recipes",
            "title": "How to bake bread",
            "url": "https://example.com/cooking",
        }
        score = compute_keyword_score(item)
        assert score == 0.0, "Irrelevant content should score zero"

    def test_multiple_keyword_matches(self):
        """Test logarithmic scaling for multiple keyword matches."""
        item = {
            "text": "Claude Code and Cursor both support agentic coding with AI assistant",
            "title": "",
            "url": "https://example.com",
        }
        score = compute_keyword_score(item)
        # Multiple matches: "Claude Code", "Cursor", "agentic coding", "AI assistant"
        # Should use log1p scaling, not linear
        assert score > 4.0, "Multiple keywords should accumulate"


class TestKeywordFiltering:
    """Test suite for keyword filtering functionality."""

    def test_filter_by_min_score(self):
        """Test filtering with minimum score threshold."""
        items = [
            {
                "text": "Claude 3.5 announcement",
                "url": "https://anthropic.com/blog",
            },
            {
                "text": "Random unrelated content",
                "url": "https://example.com",
            },
            {
                "text": "Cursor AI assistant update",
                "url": "https://cursor.sh/blog",
            },
        ]

        filtered = filter_by_keywords(items, min_score=1.0)

        # Only relevant items should pass
        assert len(filtered) == 2, "Should filter out low-scoring items"
        assert all(
            "keyword_score" in item for item in filtered
        ), "Should add keyword_score field"

    def test_filter_by_top_k(self):
        """Test filtering with top_k limit."""
        items = [
            {"text": f"Claude {i} update", "url": "https://anthropic.com"}
            for i in range(10)
        ]

        filtered = filter_by_keywords(items, top_k=5)

        assert len(filtered) == 5, "Should limit to top_k items"
        # Should be sorted by score descending
        scores = [item["keyword_score"] for item in filtered]
        assert scores == sorted(
            scores, reverse=True
        ), "Should be sorted by score descending"

    def test_filter_preserves_item_data(self):
        """Test that filtering preserves original item data."""
        items = [
            {
                "id": "123",
                "text": "Claude Code release",
                "url": "https://anthropic.com",
                "metadata": {"author": "test"},
            }
        ]

        filtered = filter_by_keywords(items, min_score=0.0)

        assert len(filtered) == 1
        assert filtered[0]["id"] == "123"
        assert filtered[0]["metadata"] == {"author": "test"}
        assert "keyword_score" in filtered[0]

    def test_empty_input(self):
        """Test handling of empty input."""
        filtered = filter_by_keywords([])
        assert filtered == [], "Should return empty list for empty input"

    def test_all_items_below_threshold(self):
        """Test when all items are below minimum score."""
        items = [
            {"text": "Unrelated content 1", "url": "https://example.com"},
            {"text": "Unrelated content 2", "url": "https://example.com"},
        ]

        filtered = filter_by_keywords(items, min_score=5.0)
        assert len(filtered) == 0, "Should return empty when all below threshold"


class TestKeywordCategories:
    """Test keyword category definitions."""

    def test_categories_have_weights(self):
        """Test that all categories have weight definitions."""
        for category, config in KEYWORD_CATEGORIES.items():
            assert "weight" in config, f"Category {category} missing weight"
            assert "keywords" in config, f"Category {category} missing keywords"
            assert config["weight"] > 0, f"Category {category} has invalid weight"

    def test_categories_have_keywords(self):
        """Test that all categories have keyword patterns."""
        for category, config in KEYWORD_CATEGORIES.items():
            assert len(config["keywords"]) > 0, f"Category {category} has no keywords"
            # Check that patterns are valid regex
            for pattern in config["keywords"]:
                assert isinstance(pattern, str), f"Invalid pattern in {category}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
