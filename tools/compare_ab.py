#!/usr/bin/env python3
"""A/B comparison tool for evaluating prompt optimization improvements.

This script compares outputs from the old system (single-stage, no keyword filtering)
versus the new optimized system (multi-stage, keyword filtering, TF-IDF queries).

Usage:
    python tools/compare_ab.py --old-config configs/ai-briefing-hackernews-baseline.yaml \
                                --new-config configs/ai-briefing-hackernews.yaml \
                                --output reports/ab_comparison.json

Metrics tracked:
    - Agentic content ratio (topics with agentic_bonus > 0)
    - Official source ratio (items with source_reliability = 2)
    - Topic relevance (manual review scores)
    - Processing time
    - Total topics generated
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from briefing.orchestrator import main as run_briefing
from briefing.utils import get_logger

logger = get_logger(__name__)


def count_agentic_topics(briefing_data: Dict[str, Any]) -> Tuple[int, int]:
    """Count topics with agentic content.

    Returns:
        (agentic_count, total_count)
    """
    if not briefing_data or "topics" not in briefing_data:
        return 0, 0

    topics = briefing_data["topics"]
    agentic_count = 0

    for topic in topics:
        # Check if any bullet mentions agentic coding keywords
        bullets = topic.get("bullets", [])
        for bullet in bullets:
            text = bullet.get("text", "").lower() if isinstance(bullet, dict) else str(bullet).lower()
            if any(
                keyword in text
                for keyword in [
                    "claude code",
                    "cursor",
                    "devin",
                    "copilot",
                    "agentic",
                    "code agent",
                    "ai assistant",
                ]
            ):
                agentic_count += 1
                break

    return agentic_count, len(topics)


def count_official_sources(briefing_data: Dict[str, Any]) -> Tuple[int, int]:
    """Count bullets from official sources.

    Returns:
        (official_count, total_count)
    """
    if not briefing_data or "topics" not in briefing_data:
        return 0, 0

    topics = briefing_data["topics"]
    official_count = 0
    total_count = 0

    official_domains = [
        "anthropic.com",
        "openai.com",
        "google.ai",
        "deepmind.google",
        "github.com/anthropics",
        "github.com/openai",
        "github.com/google",
        "github.com/meta-llama",
    ]

    for topic in topics:
        bullets = topic.get("bullets", [])
        for bullet in bullets:
            total_count += 1
            if isinstance(bullet, dict):
                url = bullet.get("url", "").lower()
                if any(domain in url for domain in official_domains):
                    official_count += 1

    return official_count, total_count


def analyze_briefing(config_path: str, label: str) -> Dict[str, Any]:
    """Run a briefing and collect metrics.

    Args:
        config_path: Path to configuration YAML
        label: Label for this run (e.g., "baseline", "optimized")

    Returns:
        Dictionary of metrics
    """
    logger.info(f"Running {label} configuration: {config_path}")

    start_time = time.time()
    try:
        # Run the briefing
        result = run_briefing(config_path)

        # Load the generated JSON output
        output_dir = Path(result.get("output_dir", "out"))
        json_files = sorted(output_dir.glob("*.json"))

        if not json_files:
            logger.warning(f"No JSON output found for {label}")
            return {
                "label": label,
                "config": config_path,
                "success": False,
                "error": "No JSON output generated",
            }

        latest_json = json_files[-1]
        with open(latest_json, "r", encoding="utf-8") as f:
            briefing_data = json.load(f)

        processing_time = time.time() - start_time

        # Calculate metrics
        agentic_count, total_topics = count_agentic_topics(briefing_data)
        official_count, total_bullets = count_official_sources(briefing_data)

        agentic_ratio = agentic_count / total_topics if total_topics > 0 else 0.0
        official_ratio = official_count / total_bullets if total_bullets > 0 else 0.0

        return {
            "label": label,
            "config": config_path,
            "success": True,
            "processing_time_sec": round(processing_time, 2),
            "total_topics": total_topics,
            "agentic_topics": agentic_count,
            "agentic_ratio": round(agentic_ratio, 3),
            "total_bullets": total_bullets,
            "official_bullets": official_count,
            "official_ratio": round(official_ratio, 3),
            "output_file": str(latest_json),
        }

    except Exception as e:
        logger.error(f"Failed to run {label} configuration: {e}")
        return {
            "label": label,
            "config": config_path,
            "success": False,
            "error": str(e),
        }


def generate_report(baseline: Dict[str, Any], optimized: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comparison report.

    Args:
        baseline: Metrics from baseline configuration
        optimized: Metrics from optimized configuration

    Returns:
        Comparison report with deltas
    """
    if not baseline["success"] or not optimized["success"]:
        return {
            "baseline": baseline,
            "optimized": optimized,
            "comparison": "One or both runs failed",
        }

    # Calculate improvements
    agentic_delta = optimized["agentic_ratio"] - baseline["agentic_ratio"]
    official_delta = optimized["official_ratio"] - baseline["official_ratio"]
    time_delta = optimized["processing_time_sec"] - baseline["processing_time_sec"]
    time_delta_pct = (time_delta / baseline["processing_time_sec"]) * 100 if baseline["processing_time_sec"] > 0 else 0

    comparison = {
        "agentic_ratio_improvement": round(agentic_delta, 3),
        "agentic_ratio_improvement_pct": round(agentic_delta * 100, 1),
        "official_ratio_improvement": round(official_delta, 3),
        "official_ratio_improvement_pct": round(official_delta * 100, 1),
        "time_overhead_sec": round(time_delta, 2),
        "time_overhead_pct": round(time_delta_pct, 1),
    }

    # Determine if targets are met (from implementation summary)
    targets_met = {
        "agentic_content_40pct": optimized["agentic_ratio"] >= 0.40,
        "official_source_60pct": optimized["official_ratio"] >= 0.60,
        "time_under_150sec": optimized["processing_time_sec"] < 150,
    }

    return {
        "baseline": baseline,
        "optimized": optimized,
        "comparison": comparison,
        "targets_met": targets_met,
        "summary": f"Agentic content: {optimized['agentic_ratio']:.1%} (target: 40%), "
        f"Official sources: {optimized['official_ratio']:.1%} (target: 60%), "
        f"Time: {optimized['processing_time_sec']:.1f}s (target: <150s)",
    }


def main():
    """Run A/B comparison."""
    parser = argparse.ArgumentParser(description="A/B comparison for prompt optimization")
    parser.add_argument(
        "--baseline-config",
        required=True,
        help="Path to baseline configuration (old system)",
    )
    parser.add_argument(
        "--optimized-config",
        required=True,
        help="Path to optimized configuration (new system)",
    )
    parser.add_argument(
        "--output",
        default="reports/ab_comparison.json",
        help="Output path for comparison report",
    )

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Run both configurations
    baseline_metrics = analyze_briefing(args.baseline_config, "baseline")
    optimized_metrics = analyze_briefing(args.optimized_config, "optimized")

    # Generate comparison report
    report = generate_report(baseline_metrics, optimized_metrics)

    # Save report
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"Comparison report saved to {output_path}")

    # Print summary to console
    print("\n" + "=" * 60)
    print("A/B Comparison Report")
    print("=" * 60)
    print(f"\nBaseline: {args.baseline_config}")
    print(f"  Topics: {baseline_metrics.get('total_topics', 'N/A')}")
    print(f"  Agentic ratio: {baseline_metrics.get('agentic_ratio', 0):.1%}")
    print(f"  Official ratio: {baseline_metrics.get('official_ratio', 0):.1%}")
    print(f"  Time: {baseline_metrics.get('processing_time_sec', 0):.1f}s")

    print(f"\nOptimized: {args.optimized_config}")
    print(f"  Topics: {optimized_metrics.get('total_topics', 'N/A')}")
    print(f"  Agentic ratio: {optimized_metrics.get('agentic_ratio', 0):.1%}")
    print(f"  Official ratio: {optimized_metrics.get('official_ratio', 0):.1%}")
    print(f"  Time: {optimized_metrics.get('processing_time_sec', 0):.1f}s")

    if "comparison" in report and isinstance(report["comparison"], dict):
        comp = report["comparison"]
        print("\nImprovements:")
        print(f"  Agentic content: {comp['agentic_ratio_improvement_pct']:+.1f}%")
        print(f"  Official sources: {comp['official_ratio_improvement_pct']:+.1f}%")
        print(f"  Time overhead: {comp['time_overhead_pct']:+.1f}%")

        print("\nTargets met:")
        targets = report.get("targets_met", {})
        print(f"  ✓ Agentic content ≥40%: {targets.get('agentic_content_40pct', False)}")
        print(f"  ✓ Official sources ≥60%: {targets.get('official_source_60pct', False)}")
        print(f"  ✓ Time <150s: {targets.get('time_under_150sec', False)}")

    print("\n" + "=" * 60)

    return 0 if baseline_metrics["success"] and optimized_metrics["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
