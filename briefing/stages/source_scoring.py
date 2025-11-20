"""Source reliability scoring based on URL patterns.

This module implements a tier-based source classification system:
- Tier 1 (score 2): Official sources (anthropic.com, openai.com, GitHub orgs)
- Tier 2 (score 1): Reputable tech media, high-engagement HN posts
- Tier 3 (score 0): Unknown or low-quality sources

Based on the prompt optimization proposal (docs/PROMPT_OPTIMIZATION_PROPOSAL.md).
"""

import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Tier 1: Official sources (score = 2)
TIER1_PATTERNS = [
    r"anthropic\.com",
    r"openai\.com",
    r"google\.ai",
    r"deepmind\.google",
    r"github\.com/(anthropics|openai|google|meta-llama)",
    r"blog\.cloudflare\.com",
    r"aws\.amazon\.com/blogs",
    r"microsoft\.com/.*research",
]

# Tier 2: Reputable tech media and releases (score = 1)
TIER2_PATTERNS = [
    r"github\.com/[\w-]+/[\w-]+/releases",  # GitHub releases
    r"news\.ycombinator\.com/item",  # HN (will check score separately)
    r"(techcrunch|arstechnica|theverge|wired)\.com",
    r"blog\.(.*)\.(com|io|org)",  # Technical blogs (general pattern)
]


def score_source_reliability(item: Dict[str, Any]) -> int:
    """Score source reliability based on URL patterns.

    Args:
        item: Content item with 'url' field and optional 'metadata'

    Returns:
        Source reliability score:
        - 2: Official sources (Tier 1)
        - 1: Reputable tech media or high-engagement HN (Tier 2)
        - 0: Unknown/low-quality sources (Tier 3)
    """
    url = item.get("url", "")

    # Tier 1: Official sources
    for pattern in TIER1_PATTERNS:
        if re.search(pattern, url, re.IGNORECASE):
            return 2

    # Tier 2: Reputable tech media or high-engagement HN
    for pattern in TIER2_PATTERNS:
        if re.search(pattern, url, re.IGNORECASE):
            # For HN, check score/engagement if available
            if "news.ycombinator.com" in url:
                score = item.get("metadata", {}).get("score", 0)
                if score >= 100:
                    return 1
                return 0
            return 1

    return 0


def enrich_with_source_scores(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add source_reliability_score to each item.

    Args:
        items: List of content items

    Returns:
        Same list with 'source_reliability_score' field added to each item
    """
    tier_counts = {0: 0, 1: 0, 2: 0}

    for item in items:
        score = score_source_reliability(item)
        item["source_reliability_score"] = score
        tier_counts[score] += 1

    logger.info(
        "source_scoring: %d items scored - Tier 1 (official): %d, Tier 2 (reputable): %d, Tier 3 (unknown): %d",
        len(items),
        tier_counts[2],
        tier_counts[1],
        tier_counts[0],
    )

    return items


def filter_by_source_reliability(
    items: List[Dict[str, Any]],
    *,
    min_tier: int = 0,
) -> List[Dict[str, Any]]:
    """Filter items by minimum source reliability tier.

    Args:
        items: List of content items (must have 'source_reliability_score' field)
        min_tier: Minimum reliability score (0, 1, or 2)

    Returns:
        Filtered list of items meeting the minimum tier requirement
    """
    filtered = [
        item
        for item in items
        if item.get("source_reliability_score", 0) >= min_tier
    ]

    logger.info(
        "source_filter: %d items filtered to %d (min_tier=%d)",
        len(items),
        len(filtered),
        min_tier,
    )

    return filtered
