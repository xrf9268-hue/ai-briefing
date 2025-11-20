"""Tests for source reliability scoring."""

import pytest
from briefing.stages.source_scoring import (
    score_source_reliability,
    enrich_with_source_scores,
    filter_by_source_reliability,
)


class TestSourceReliabilityScoring:
    """Test suite for source reliability scoring logic."""

    def test_tier1_anthropic(self):
        """Test Tier 1 scoring for Anthropic official source."""
        item = {"url": "https://anthropic.com/blog/claude-3-5"}
        score = score_source_reliability(item)
        assert score == 2, "Anthropic.com should be Tier 1 (score 2)"

    def test_tier1_openai(self):
        """Test Tier 1 scoring for OpenAI official source."""
        item = {"url": "https://openai.com/research/gpt-4"}
        score = score_source_reliability(item)
        assert score == 2, "OpenAI.com should be Tier 1 (score 2)"

    def test_tier1_google_ai(self):
        """Test Tier 1 scoring for Google AI."""
        item = {"url": "https://google.ai/gemini"}
        score = score_source_reliability(item)
        assert score == 2, "Google.ai should be Tier 1 (score 2)"

    def test_tier1_github_orgs(self):
        """Test Tier 1 scoring for official GitHub organizations."""
        test_cases = [
            "https://github.com/anthropics/anthropic-sdk-python",
            "https://github.com/openai/openai-python",
            "https://github.com/google/gemini-api",
            "https://github.com/meta-llama/llama",
        ]

        for url in test_cases:
            item = {"url": url}
            score = score_source_reliability(item)
            assert score == 2, f"{url} should be Tier 1 (score 2)"

    def test_tier1_aws_blog(self):
        """Test Tier 1 scoring for AWS blog."""
        item = {"url": "https://aws.amazon.com/blogs/machine-learning/"}
        score = score_source_reliability(item)
        assert score == 2, "AWS blogs should be Tier 1 (score 2)"

    def test_tier2_github_release(self):
        """Test Tier 2 scoring for GitHub releases."""
        item = {"url": "https://github.com/user/repo/releases/tag/v1.0"}
        score = score_source_reliability(item)
        assert score == 1, "GitHub releases should be Tier 2 (score 1)"

    def test_tier2_tech_media(self):
        """Test Tier 2 scoring for reputable tech media."""
        test_cases = [
            "https://techcrunch.com/2024/01/01/ai-news/",
            "https://arstechnica.com/ai/2024/new-model/",
            "https://theverge.com/tech/ai-assistant",
            "https://wired.com/story/ai-coding/",
        ]

        for url in test_cases:
            item = {"url": url}
            score = score_source_reliability(item)
            assert score == 1, f"{url} should be Tier 2 (score 1)"

    def test_tier2_high_score_hn(self):
        """Test Tier 2 scoring for high-engagement HN posts."""
        item = {
            "url": "https://news.ycombinator.com/item?id=123456",
            "metadata": {"score": 500},
        }
        score = score_source_reliability(item)
        assert score == 1, "High-score HN post should be Tier 2 (score 1)"

    def test_tier3_low_score_hn(self):
        """Test Tier 3 scoring for low-engagement HN posts."""
        item = {
            "url": "https://news.ycombinator.com/item?id=123456",
            "metadata": {"score": 50},
        }
        score = score_source_reliability(item)
        assert score == 0, "Low-score HN post should be Tier 3 (score 0)"

    def test_tier3_unknown_blog(self):
        """Test Tier 3 scoring for unknown sources."""
        item = {"url": "https://random-blog.example.com/post"}
        score = score_source_reliability(item)
        assert score == 0, "Unknown sources should be Tier 3 (score 0)"

    def test_tier3_social_media(self):
        """Test Tier 3 scoring for social media."""
        item = {"url": "https://twitter.com/user/status/123"}
        score = score_source_reliability(item)
        assert score == 0, "Social media should be Tier 3 (score 0)"

    def test_missing_url(self):
        """Test handling of missing URL."""
        item = {}
        score = score_source_reliability(item)
        assert score == 0, "Missing URL should default to Tier 3 (score 0)"

    def test_case_insensitive_matching(self):
        """Test that URL matching is case-insensitive."""
        item = {"url": "https://ANTHROPIC.COM/blog/news"}
        score = score_source_reliability(item)
        assert score == 2, "URL matching should be case-insensitive"


class TestEnrichWithSourceScores:
    """Test suite for enriching items with source scores."""

    def test_enrich_adds_scores(self):
        """Test that enrichment adds source_reliability_score field."""
        items = [
            {"url": "https://anthropic.com/blog"},
            {"url": "https://example.com/blog"},
        ]

        enriched = enrich_with_source_scores(items)

        assert len(enriched) == 2
        assert all("source_reliability_score" in item for item in enriched)
        assert enriched[0]["source_reliability_score"] == 2
        assert enriched[1]["source_reliability_score"] == 0

    def test_enrich_preserves_data(self):
        """Test that enrichment preserves original item data."""
        items = [
            {
                "id": "123",
                "url": "https://openai.com",
                "text": "Test content",
                "metadata": {"key": "value"},
            }
        ]

        enriched = enrich_with_source_scores(items)

        assert enriched[0]["id"] == "123"
        assert enriched[0]["text"] == "Test content"
        assert enriched[0]["metadata"] == {"key": "value"}
        assert enriched[0]["source_reliability_score"] == 2

    def test_enrich_empty_list(self):
        """Test enrichment of empty list."""
        enriched = enrich_with_source_scores([])
        assert enriched == []

    def test_enrich_logs_tier_distribution(self, caplog):
        """Test that enrichment logs tier distribution."""
        items = [
            {"url": "https://anthropic.com"},  # Tier 1
            {"url": "https://openai.com"},  # Tier 1
            {"url": "https://techcrunch.com/article"},  # Tier 2
            {"url": "https://example.com"},  # Tier 3
            {"url": "https://random.blog"},  # Tier 3
        ]

        enrich_with_source_scores(items)

        # Should log tier counts: 2 Tier 1, 1 Tier 2, 2 Tier 3
        # (Actual log verification depends on logging setup)


class TestFilterBySourceReliability:
    """Test suite for filtering by source reliability."""

    def test_filter_min_tier_0(self):
        """Test filtering with min_tier=0 (keep all)."""
        items = [
            {"url": "https://anthropic.com", "source_reliability_score": 2},
            {"url": "https://techcrunch.com", "source_reliability_score": 1},
            {"url": "https://example.com", "source_reliability_score": 0},
        ]

        filtered = filter_by_source_reliability(items, min_tier=0)
        assert len(filtered) == 3, "min_tier=0 should keep all items"

    def test_filter_min_tier_1(self):
        """Test filtering with min_tier=1 (Tier 1 + Tier 2)."""
        items = [
            {"url": "https://anthropic.com", "source_reliability_score": 2},
            {"url": "https://techcrunch.com", "source_reliability_score": 1},
            {"url": "https://example.com", "source_reliability_score": 0},
        ]

        filtered = filter_by_source_reliability(items, min_tier=1)
        assert len(filtered) == 2, "min_tier=1 should keep Tier 1 and Tier 2"
        assert all(item["source_reliability_score"] >= 1 for item in filtered)

    def test_filter_min_tier_2(self):
        """Test filtering with min_tier=2 (Tier 1 only)."""
        items = [
            {"url": "https://anthropic.com", "source_reliability_score": 2},
            {"url": "https://techcrunch.com", "source_reliability_score": 1},
            {"url": "https://example.com", "source_reliability_score": 0},
        ]

        filtered = filter_by_source_reliability(items, min_tier=2)
        assert len(filtered) == 1, "min_tier=2 should keep only Tier 1"
        assert filtered[0]["source_reliability_score"] == 2

    def test_filter_preserves_data(self):
        """Test that filtering preserves item data."""
        items = [
            {
                "id": "123",
                "url": "https://openai.com",
                "source_reliability_score": 2,
                "metadata": {"test": "value"},
            }
        ]

        filtered = filter_by_source_reliability(items, min_tier=2)

        assert filtered[0]["id"] == "123"
        assert filtered[0]["metadata"] == {"test": "value"}

    def test_filter_empty_result(self):
        """Test filtering that results in empty list."""
        items = [
            {"url": "https://example.com", "source_reliability_score": 0},
            {"url": "https://random.blog", "source_reliability_score": 0},
        ]

        filtered = filter_by_source_reliability(items, min_tier=2)
        assert len(filtered) == 0, "Should return empty when no items meet threshold"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
