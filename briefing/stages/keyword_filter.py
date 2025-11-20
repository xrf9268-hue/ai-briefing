"""Keyword-based content filtering with TF-IDF weighting.

This module implements upstream filtering to prioritize content about:
- LLM releases (Claude, GPT, Gemini, Llama)
- Agentic coding tools (Claude Code, Cursor, Devin, Gemini CLI)
- Vibe coding and rapid prototyping
- CLI tools and automation

Based on the prompt optimization proposal (docs/PROMPT_OPTIMIZATION_PROPOSAL.md).
"""

import logging
import math
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Priority keyword categories with weights
KEYWORD_CATEGORIES = {
    "llm_releases": {
        "weight": 3.0,
        "keywords": [
            r"\bClaude\s+\d+\.?\d*",  # Claude 3, Claude 3.5
            r"\bGPT-?\d+",  # GPT-4, GPT-4o
            r"\bGemini\s+\d+\.?\d*",  # Gemini 1.5, Gemini 2.0
            r"\bLlama\s*\d+",  # Llama 3, Llama3
            r"\bmodel\s+release\b",
            r"\bAI\s+model\b",
            r"\bnew\s+version\b",
            r"\bAPI\s+update\b",
        ],
    },
    "agentic_coding": {
        "weight": 2.5,
        "keywords": [
            r"\bClaude\s+Code\b",
            r"\bCursor\b",
            r"\bGemini\s+CLI\b",
            r"\bDevin\b",
            r"\bCopilot\b",
            r"\bagentic\s+coding\b",
            r"\bcode\s+agent\b",
            r"\bAI\s+assistant\b",
            r"\bautonomous\s+agent\b",
            r"\bpair\s+programming\b",
        ],
    },
    "vibe_coding": {
        "weight": 2.0,
        "keywords": [
            r"\bvibe\s+coding\b",
            r"\brapid\s+prototyp",
            r"\bconversational\s+programming\b",
            r"\breal-?time\s+collaboration\b",
            r"\biterative\s+development\b",
        ],
    },
    "cli_tools": {
        "weight": 2.0,
        "keywords": [
            r"\bCLI\b",
            r"\bcommand[- ]line\b",
            r"\bterminal\b",
            r"\bshell\s+script\b",
            r"\bbash\s+automation\b",
            r"\bREPL\b",
        ],
    },
}

OFFICIAL_SOURCE_DOMAINS = [
    "anthropic.com",
    "openai.com",
    "google.ai",
    "deepmind.google",
    "github.com/anthropics",
    "github.com/openai",
    "github.com/google",
    "github.com/meta-llama",
    "blog.cloudflare.com",
    "aws.amazon.com/blogs",
]


def compute_keyword_score(item: Dict[str, Any]) -> float:
    """Compute keyword relevance score using TF-IDF inspired approach.

    Args:
        item: Content item with 'text', 'title', and 'url' fields

    Returns:
        Keyword relevance score (0.0+). Higher scores indicate better topic match.
        Typical ranges:
        - 0.0: No keyword matches
        - 0.5-2.0: Weak relevance (1-2 keyword hits)
        - 2.0-5.0: Moderate relevance (multiple keywords)
        - 5.0+: High relevance (many keywords + official source boost)
    """
    text = f"{item.get('text', '')} {item.get('title', '')}".lower()
    url = item.get("url", "").lower()

    # Base score from keyword matching
    score = 0.0
    matches_by_category = {}

    for category, config in KEYWORD_CATEGORIES.items():
        weight = config["weight"]
        matches = 0
        for pattern in config["keywords"]:
            if re.search(pattern, text, re.IGNORECASE):
                matches += 1
        if matches > 0:
            # Logarithmic scaling to prevent over-weighting
            matches_by_category[category] = matches
            score += weight * math.log1p(matches)

    # Boost for official sources (1.5x multiplier)
    for domain in OFFICIAL_SOURCE_DOMAINS:
        if domain in url:
            score *= 1.5
            break

    return score


def filter_by_keywords(
    items: List[Dict[str, Any]],
    *,
    min_score: float = 0.0,
    top_k: Optional[int] = None,
    boost_official_sources: bool = True,
) -> List[Dict[str, Any]]:
    """Filter items by keyword relevance.

    Args:
        items: List of content items to filter
        min_score: Minimum keyword score threshold (default: 0.0, no filtering)
        top_k: Keep only top K items by score (default: None, keep all above threshold)
        boost_official_sources: Apply 1.5x multiplier to official sources (default: True)

    Returns:
        Filtered list of items, sorted by keyword score descending.
        Each item has an added 'keyword_score' field.
    """
    scored_items = []
    for item in items:
        score = compute_keyword_score(item)
        if score >= min_score:
            item_copy = dict(item)
            item_copy["keyword_score"] = score
            scored_items.append(item_copy)

    # Sort by score descending
    scored_items.sort(key=lambda x: x["keyword_score"], reverse=True)

    if top_k:
        scored_items = scored_items[:top_k]

    logger.info(
        "keyword_filter: %d items filtered to %d (min_score=%.2f, top_k=%s)",
        len(items),
        len(scored_items),
        min_score,
        top_k,
    )

    return scored_items
