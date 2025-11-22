# Configuration Examples for Prompt Optimization Features

This guide provides practical examples for configuring the new prompt optimization features introduced in Phase 4.

**Related Documentation:**
- [Implementation Summary](IMPLEMENTATION_SUMMARY.md) - Overview of features
- [Validation Report](VALIDATION_REPORT.md) - Testing and validation results
- [Prompt Optimization Proposal](PROMPT_OPTIMIZATION_PROPOSAL.md) - Design rationale

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [TF-IDF Weighted Queries](#tf-idf-weighted-queries)
3. [Keyword Filtering](#keyword-filtering)
4. [Custom Keyword Patterns](#custom-keyword-patterns)
5. [Complete Examples](#complete-examples)
6. [A/B Testing](#ab-testing)

---

## Quick Start

### Enable All Optimizations (Recommended)

The simplest way to enable all prompt optimization features:

```yaml
processing:
  # Enable multi-stage chain prompts
  multi_stage: true

  # Enable keyword filtering with defaults
  keyword_filter:
    enabled: true
    min_score: 0.5
    top_k: 500
    boost_official_sources: true

  # Enable TF-IDF weighted queries
  rerank:
    strategy: ce+mmr
    lambda: 0.4
    use_tfidf_query: true
    tfidf_top_n: 10

  # Enhanced scoring weights
  scoring_weights:
    agentic_bonus: 2.5
    source_reliability: 2.0

summarization:
  # Use multi-stage prompts
  prompt_file: prompts/stage1_extract_facts.yaml
```

**What this does:**
- ✅ Filters content for LLM releases and agentic coding topics
- ✅ Boosts official sources (anthropic.com, openai.com, etc.)
- ✅ Uses TF-IDF weighted queries for better reranking
- ✅ Employs multi-stage chain prompts for clarity
- ✅ Prioritizes agentic coding content in scoring

---

## TF-IDF Weighted Queries

### Basic Usage

Enable TF-IDF weighted query generation for reranking:

```yaml
processing:
  rerank:
    strategy: ce+mmr
    lambda: 0.4
    use_tfidf_query: true    # Enable TF-IDF queries
    tfidf_top_n: 10          # Number of keywords to extract
```

**How it works:**
1. For each cluster, extracts top N keywords using TF-IDF
2. Combines keywords with representative text to build query
3. Uses this enhanced query for cross-encoder reranking
4. Falls back to centroid text if TF-IDF fails

### Conservative Settings (Lower Overhead)

```yaml
processing:
  rerank:
    strategy: ce+mmr
    use_tfidf_query: true
    tfidf_top_n: 5           # Fewer keywords = faster
```

### Aggressive Settings (Better Quality)

```yaml
processing:
  rerank:
    strategy: ce+mmr
    use_tfidf_query: true
    tfidf_top_n: 20          # More keywords = better semantic coverage
```

### Disable TF-IDF (Baseline)

```yaml
processing:
  rerank:
    strategy: ce+mmr
    use_tfidf_query: false   # Use simple centroid text
```

---

## Keyword Filtering

### Default Keyword Filtering

Use built-in keyword patterns for LLM/agentic content:

```yaml
processing:
  keyword_filter:
    enabled: true
    min_score: 0.5           # Minimum keyword relevance score
    top_k: 500               # Keep top 500 items after filtering
    boost_official_sources: true
```

**Built-in Categories:**
- **llm_releases** (weight 3.0): Claude, GPT, Gemini, Llama model releases
- **agentic_coding** (weight 2.5): Claude Code, Cursor, Devin, Copilot
- **vibe_coding** (weight 2.0): Rapid prototyping, conversational programming
- **cli_tools** (weight 2.0): Command-line tools, terminal automation

### Aggressive Filtering

Filter more content to focus on high-relevance items:

```yaml
processing:
  keyword_filter:
    enabled: true
    min_score: 1.0           # Higher threshold
    top_k: 200               # Fewer items
```

### Permissive Filtering

Allow more content through:

```yaml
processing:
  keyword_filter:
    enabled: true
    min_score: 0.3           # Lower threshold
    top_k: 1000              # More items
```

### Disable Official Source Boost

```yaml
processing:
  keyword_filter:
    enabled: true
    boost_official_sources: false  # No 1.5x multiplier
```

---

## Custom Keyword Patterns

### Override Default Categories

Replace or extend built-in keyword patterns:

```yaml
processing:
  keyword_filter:
    enabled: true
    min_score: 0.5

    # Custom keyword categories
    keyword_categories:
      # Override default llm_releases category
      llm_releases:
        weight: 3.5
        keywords:
          - "\\bClaude\\s+\\d+\\.?\\d*"
          - "\\bGPT-?\\d+"
          - "\\bGemini\\s+\\d+\\.?\\d*"
          - "\\bLlama\\s*\\d+"
          - "\\bMistral\\s+\\d+"        # Add Mistral
          - "\\bQwen\\s+\\d+"           # Add Qwen

      # Add custom category for your domain
      game_engines:
        weight: 2.0
        keywords:
          - "\\bUnity\\s+\\d+"
          - "\\bUnreal\\s+Engine"
          - "\\bGodot\\b"
          - "\\bBevy\\b"
```

### Custom Official Domains

Add your own trusted sources:

```yaml
processing:
  keyword_filter:
    enabled: true

    # Custom official domains
    official_domains:
      - "anthropic.com"
      - "openai.com"
      - "google.ai"
      - "your-company.com"      # Add your domain
      - "your-blog.io"           # Add your blog
```

### Domain-Specific Configuration

Example for game development briefing:

```yaml
briefing_id: game-dev-briefing
briefing_title: Game Development News

processing:
  keyword_filter:
    enabled: true
    min_score: 0.5

    keyword_categories:
      game_engines:
        weight: 3.0
        keywords:
          - "\\bUnity\\b"
          - "\\bUnreal\\s+Engine\\b"
          - "\\bGodot\\b"
          - "\\bBevy\\b"

      game_ai:
        weight: 2.5
        keywords:
          - "\\bprocedural\\s+generation\\b"
          - "\\bNPC\\s+AI\\b"
          - "\\bpathfinding\\b"
          - "\\bmachine\\s+learning\\b"

      graphics:
        weight: 2.0
        keywords:
          - "\\bray\\s+tracing\\b"
          - "\\bshader\\b"
          - "\\brendering\\b"

    official_domains:
      - "unity.com"
      - "unrealengine.com"
      - "godotengine.org"
```

---

## Complete Examples

### Example 1: Maximum Optimization (Production)

Full-featured configuration for production use:

```yaml
briefing_id: ai-briefing-optimized
briefing_title: AI Briefing (Optimized)

source:
  type: hackernews
  hn_story_type: top
  hn_limit: 60

processing:
  time_window_hours: 24
  min_cluster_size: 2

  # Multi-stage chain prompts
  multi_stage: true
  agentic_section: true

  # Keyword filtering
  keyword_filter:
    enabled: true
    min_score: 0.5
    top_k: 500
    boost_official_sources: true

  # Enhanced reranking
  rerank:
    strategy: ce+mmr
    lambda: 0.4
    use_tfidf_query: true
    tfidf_top_n: 10

  # Enhanced scoring
  scoring_weights:
    actionability: 3.0
    novelty: 2.0
    impact: 2.0
    reusability: 2.0
    reliability: 1.0
    agentic_bonus: 2.5
    source_reliability: 2.0

summarization:
  prompt_file: prompts/stage1_extract_facts.yaml
  llm_provider: gemini
  gemini_model: gemini-2.5-flash
  temperature: 0.2

output:
  dir: out/ai-briefing-optimized
  formats: [md, json]
```

### Example 2: Baseline (For A/B Testing)

Configuration without optimizations for comparison:

```yaml
briefing_id: ai-briefing-baseline
briefing_title: AI Briefing (Baseline)

source:
  type: hackernews
  hn_story_type: top
  hn_limit: 60

processing:
  time_window_hours: 24
  min_cluster_size: 2

  # Single-stage prompts
  multi_stage: false

  # No keyword filtering
  keyword_filter:
    enabled: false

  # Basic reranking (no TF-IDF)
  rerank:
    strategy: ce+mmr
    lambda: 0.4
    use_tfidf_query: false

  # Original scoring weights
  scoring_weights:
    actionability: 3.0
    novelty: 2.0
    impact: 2.0
    reusability: 2.0
    reliability: 1.0
    agentic_bonus: 1.0
    source_reliability: 2.0

summarization:
  prompt_file: prompts/daily_briefing_multisource.yaml
  llm_provider: gemini
  gemini_model: gemini-2.5-flash
  temperature: 0.2

output:
  dir: out/ai-briefing-baseline
  formats: [md, json]
```

### Example 3: Gradual Rollout (Stage 1)

Start with keyword filtering only:

```yaml
processing:
  # Stage 1: Enable keyword filtering only
  multi_stage: false

  keyword_filter:
    enabled: true
    min_score: 0.3           # Lower threshold for testing

  rerank:
    use_tfidf_query: false   # Keep disabled

  scoring_weights:
    agentic_bonus: 1.0       # Keep original weight

summarization:
  prompt_file: prompts/daily_briefing_multisource.yaml
```

### Example 4: Gradual Rollout (Stage 2)

Add multi-stage prompts:

```yaml
processing:
  # Stage 2: Enable multi-stage prompts
  multi_stage: true

  keyword_filter:
    enabled: true
    min_score: 0.5           # Increase threshold

  rerank:
    use_tfidf_query: false   # Still disabled

  scoring_weights:
    agentic_bonus: 2.5       # Increase weight

summarization:
  prompt_file: prompts/stage1_extract_facts.yaml
```

### Example 5: Gradual Rollout (Stage 3 - Full)

Enable all features:

```yaml
processing:
  # Stage 3: Enable all features
  multi_stage: true

  keyword_filter:
    enabled: true
    min_score: 0.5

  rerank:
    use_tfidf_query: true    # Now enabled
    tfidf_top_n: 10

  scoring_weights:
    agentic_bonus: 2.5
    source_reliability: 2.0

summarization:
  prompt_file: prompts/stage1_extract_facts.yaml
```

---

## A/B Testing

### Running A/B Comparison

Use the comparison script to evaluate optimizations:

```bash
python tools/compare_ab.py \
  --baseline-config configs/ai-briefing-hackernews-baseline.yaml \
  --optimized-config configs/ai-briefing-hackernews-optimized.yaml \
  --output reports/ab_comparison.json
```

### Interpreting Results

The script generates a report with:

```json
{
  "baseline": {
    "agentic_ratio": 0.200,
    "official_ratio": 0.350,
    "processing_time_sec": 120.5
  },
  "optimized": {
    "agentic_ratio": 0.500,
    "official_ratio": 0.650,
    "processing_time_sec": 135.2
  },
  "comparison": {
    "agentic_ratio_improvement_pct": 30.0,
    "official_ratio_improvement_pct": 30.0,
    "time_overhead_pct": 12.2
  },
  "targets_met": {
    "agentic_content_40pct": true,
    "official_source_60pct": true,
    "time_under_150sec": true
  }
}
```

**Target Metrics:**
- ✅ Agentic content ratio ≥ 40%
- ✅ Official source ratio ≥ 60%
- ✅ Processing time < 150 seconds

### A/B Testing Best Practices

1. **Run multiple iterations:**
   ```bash
   for i in {1..5}; do
     python tools/compare_ab.py \
       --baseline-config configs/baseline.yaml \
       --optimized-config configs/optimized.yaml \
       --output reports/ab_run_$i.json
   done
   ```

2. **Compare on same time window:**
   - Use same `hn_limit`, `time_window_hours`, etc.
   - Run both configs on same day for fair comparison

3. **Monitor processing time:**
   - TF-IDF adds ~5-10% overhead
   - Acceptable if quality improvement justifies it

---

## Configuration Reference

### All New Parameters

```yaml
processing:
  # Keyword filtering
  keyword_filter:
    enabled: boolean             # Enable/disable filtering
    min_score: number            # Minimum relevance score (0.0+)
    top_k: integer               # Max items to keep (1+)
    boost_official_sources: boolean
    keyword_categories:          # Optional custom patterns
      category_name:
        weight: number           # Category weight (0.0+)
        keywords: [string]       # Regex patterns
    official_domains: [string]   # Optional custom domains

  # Reranking with TF-IDF
  rerank:
    strategy: string             # "ce", "mmr", "ce+mmr"
    lambda: number               # MMR diversity param (0-1)
    use_tfidf_query: boolean     # Enable TF-IDF queries
    tfidf_top_n: integer         # Keywords to extract (1-50)

  # Enhanced scoring
  scoring_weights:
    agentic_bonus: number        # 0-3 scale weight
    source_reliability: number   # 0-2 scale weight
```

### Default Values

```yaml
processing:
  keyword_filter:
    enabled: false
    min_score: 0.5
    top_k: 500
    boost_official_sources: true

  rerank:
    use_tfidf_query: false
    tfidf_top_n: 10

  scoring_weights:
    agentic_bonus: 1.0
    source_reliability: 2.0
```

---

## Troubleshooting

### Issue: Too much content filtered out

**Symptom:** `keyword_filter: 100 -> 5 items`

**Solution:** Lower thresholds
```yaml
keyword_filter:
  min_score: 0.3    # Was 0.5
  top_k: 1000       # Was 500
```

### Issue: TF-IDF queries failing

**Symptom:** Logs show "TF-IDF query builder failed, falling back to centroid text"

**Solution:** Check cluster sizes
```yaml
# Ensure clusters are large enough for TF-IDF
clustering:
  min_cluster_size: 3    # At least 3 items per cluster
```

### Issue: No performance improvement

**Symptom:** A/B test shows no improvement in agentic content ratio

**Solution:** Check keyword patterns
```yaml
# Add logging to debug
LOG_LEVEL=DEBUG python cli.py --config your-config.yaml

# Check if keywords match your content
# Consider customizing keyword_categories
```

---

## Further Reading

- **[VALIDATION_REPORT.md](VALIDATION_REPORT.md)** - Testing results and metrics
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Feature overview
- **[PROMPT_OPTIMIZATION_PROPOSAL.md](PROMPT_OPTIMIZATION_PROPOSAL.md)** - Design rationale
- **[Main README](../README.md)** - General usage guide

---

**Document Version:** 1.0
**Last Updated:** 2025-11-22
**Author:** AI-Briefing Development Team
