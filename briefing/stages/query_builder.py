"""Build weighted queries for reranking using TF-IDF.

This module enhances cluster queries by extracting the most important keywords
using TF-IDF (Term Frequency-Inverse Document Frequency) weighting. This helps
the reranker focus on the most distinctive and relevant terms in each cluster.

Based on the prompt optimization proposal (docs/PROMPT_OPTIMIZATION_PROPOSAL.md).
"""

import logging
from typing import Any, Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


def build_weighted_query(
    items: List[Dict[str, Any]],
    *,
    top_n: int = 10,
    max_features: int = 100,
) -> str:
    """Build a TF-IDF weighted query from cluster items.

    Args:
        items: List of content items in the cluster
        top_n: Number of top keywords to extract (default: 10)
        max_features: Maximum features for TF-IDF vectorizer (default: 100)

    Returns:
        Weighted query string combining representative text and top keywords.
        Format: "<representative_text> Keywords: <keyword1> <keyword2> ..."
    """
    texts = [item.get("text", "") for item in items]

    if not texts:
        return ""

    if len(texts) == 1:
        # Single item: just return the text
        return texts[0]

    try:
        # Compute TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            max_df=0.8,  # Ignore terms appearing in >80% of docs
            min_df=1,  # Must appear at least once
        )
        tfidf_matrix = vectorizer.fit_transform(texts)

        # Get global importance scores (sum across documents)
        feature_names = vectorizer.get_feature_names_out()
        global_scores = np.asarray(tfidf_matrix.sum(axis=0)).flatten()

        # Get top keywords
        top_indices = np.argsort(-global_scores)[: min(top_n, len(feature_names))]
        top_keywords = [feature_names[i] for i in top_indices]

        # Use middle item as representative (centroid approximation)
        centroid_text = texts[len(texts) // 2]

        # Combine representative text with top keywords
        query = f"{centroid_text} Keywords: {' '.join(top_keywords)}"

        logger.debug(
            "query_builder: Built weighted query with %d keywords from %d items",
            len(top_keywords),
            len(texts),
        )

        return query

    except Exception as e:
        # Fallback to simple representative text on error
        logger.warning(
            "query_builder: TF-IDF failed (%s), falling back to centroid text", e
        )
        return texts[len(texts) // 2] if texts else ""


def build_weighted_queries_batch(
    clusters: Dict[int, List[Dict[str, Any]]],
    *,
    top_n: int = 10,
    max_features: int = 100,
) -> Dict[int, str]:
    """Build weighted queries for multiple clusters.

    Args:
        clusters: Dictionary mapping cluster_id -> list of items
        top_n: Number of top keywords per cluster
        max_features: Maximum features for TF-IDF vectorizer

    Returns:
        Dictionary mapping cluster_id -> weighted query string
    """
    queries = {}
    for cluster_id, items in clusters.items():
        queries[cluster_id] = build_weighted_query(
            items, top_n=top_n, max_features=max_features
        )

    logger.info("query_builder: Built %d weighted queries", len(queries))
    return queries
