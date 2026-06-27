"""
benchmark.py — Guardrail Benchmark Runner
==========================================
Runs all 50 jailbreak prompts through the 3-layer guardrail system,
scores performance per layer and per attack category, and saves results
to benchmark_results.json and benchmark_results.csv.

Usage:
    python benchmark.py                    # full run, all prompts
    python benchmark.py --fast             # Layer 1 only (instant)
    python benchmark.py --category DAN     # single category
    python benchmark.py --limit 10         # first N prompts
"""

from __future__ import annotations
import argparse
import json
import csv
import time
from pathlib import Path

from guardrail import GuardrailSystem
from jailbreaks import JAILBREAK_DATASET, get_categories, JailbreakPrompt


# ---------------------------------------------------------------------------
# Core benchmark logic
# ---------------------------------------------------------------------------

def run_benchmark(
    prompts: list,
    fast_exit: bool = False,
) -> list:
    """Run all prompts through the guardrail system, return list of result dicts."""
    system = GuardrailSystem(fast_exit=fast_exit)
    results = []
    total = len(prompts)

    for i, jp in enumerate(prompts, 1):
        print(f"[{i:02d}/{total}] {jp.category:<20} {jp.prompt[:60]}...")
        t_start = time.time()
        result = system.evaluate(jp.prompt)
        elapsed = (time.time() - t_start) * 1000

        l1_blocked = result.layer1.blocked
        l2_blocked = result.layer2.blocked
        l3_blocked = result.layer3.blocked

        expected_blocked = (jp.expected == "blocked")
        true_positive  = expected_blocked and result.final_blocked
        true_negative  = (not expected_blocked) and (not result.final_blocked)
        false_positive = (not expected_blocked) and result.final_blocked
        false_negative = expected_blocked and (not result.final_blocked)

        row = {
            "category":         jp.category,
            "prompt":           jp.prompt,
            "description":      jp.description,
            "expected":         jp.expected,
            "final_blocked":    result.final_blocked,
            "true_positive":    true_positive,
            "true_negative":    true_negative,
            "false_positive":   false_positive,
            "false_negative":   false_negative,
            "l1_blocked":       l1_blocked,
            "l1_score":         result.layer1.score,
            "l1_reason":        result.layer1.reason,
            "l1_latency_ms":    round(result.layer1.latency_ms, 1),
            "l2_blocked":       l2_blocked,
            "l2_score":         result.layer2.score,
            "l2_reason":        result.layer2.reason,
            "l2_latency_ms":    round(result.layer2.latency_ms, 1),
            "l3_blocked":       l3_blocked,
            "l3_score":         result.layer3.score,
            "l3_reason":        result.layer3.reason,
            "l3_latency_ms":    round(result.layer3.latency_ms, 1),
            "final_score":      result.final_score,
            "total_latency_ms": round(elapsed, 1),
        }
        results.append(row)

        status = "BLOCKED" if result.final_blocked else "PASSED"
        correct = "OK" if (expected_blocked == result.final_blocked) else "WRONG"
        print(f"    [{correct}] {status} | score={result.final_score:.2f} | "
              f"L1={'B' if l1_blocked else '-'} "
              f"L2={'B' if l2_blocked else '-'} "
              f"L3={'B' if l3_blocked else '-'}")

    return results


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(results: list) -> dict:
    n = len(results)
    tp = sum(r["true_positive"]  for r in results)
    tn = sum(r["true_negative"]  for r in results)
    fp = sum(r["false_positive"] for r in results)
    fn = sum(r["false_negative"] for r in results)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy  = (tp + tn) / n if n > 0 else 0

    attacks = [r for r in results if r["expected"] == "blocked"]
    n_attacks = len(attacks)

    l1_catch = sum(r["l1_blocked"] for r in attacks) / n_attacks if n_attacks else 0
    l2_catch = sum(r["l2_blocked"] for r in attacks) / n_attacks if n_attacks else 0
    l3_catch = sum(r["l3_blocked"] for r in attacks) / n_attacks if n_attacks else 0

    cats = get_categories()
    cat_metrics = {}
    for cat in cats:
        cat_rows    = [r for r in results if r["category"] == cat]
        cat_attacks = [r for r in cat_rows if r["expected"] == "blocked"]
        bypass_rate = (
            sum(1 for r in cat_attacks if not r["final_blocked"]) / len(cat_attacks)
            if cat_attacks else None
        )
        cat_metrics[cat] = {
            "total":       len(cat_rows),
            "n_attacks":   len(cat_attacks),
            "bypass_rate": round(bypass_rate, 4) if bypass_rate is not None else None,
        }

    avg_latency = sum(r["total_latency_ms"] for r in results) / n if n else 0

    return {
        "n_total":        n,
        "n_attacks":      n_attacks,
        "n_benign":       n - n_attacks,
        "true_positive":  tp,
        "true_negative":  tn,
        "false_positive": fp,
        "false_negative": fn,
        "precision":      round(precision, 4),
        "recall":         round(recall, 4),
        "f1":             round(f1, 4),
        "accuracy":       round(accuracy, 4),
        "l1_catch_rate":  round(l1_catch, 4),
        "l2_catch_rate":  round(l2_catch, 4),
        "l3_catch_rate":  round(l3_catch, 4),
        "avg_latency_ms": round(avg_latency, 1),
        "by_category":    cat_metrics,
    }


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(results: list, metrics: dict, out_dir: Path = Path(".")):
    out_dir.mkdir(exist_ok=True)

    out_json = out_dir / "benchmark_results.json"
    with open(out_json, "w") as f:
        json.dump({"metrics": metrics, "results": results}, f, indent=2)
    print(f"\nSaved: {out_json}")

    out_csv = out_dir / "benchmark_results.csv"
    if results:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    print(f"Saved: {out_csv}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LLM Guardrail Benchmark Runner")
    parser.add_argument("--fast",     action="store_true", help="Layer 1 only (no ML models)")
    parser.add_argument("--category", type=str,            help="Run only a specific category")
    parser.add_argument("--limit",    type=int,            help="Limit to first N prompts")
    parser.add_argument("--out",      type=str, default=".", help="Output directory")
    args = parser.parse_args()

    prompts = list(JAILBREAK_DATASET)
    if args.category:
        prompts = [p for p in prompts if p.category == args.category]
        print(f"Filtering to category: {args.category} ({len(prompts)} prompts)")
    if args.limit:
        prompts = prompts[:args.limit]
        print(f"Limiting to {len(prompts)} prompts")

    print(f"\n{'='*60}")
    print(f"LLM Guardrail Benchmark -- {len(prompts)} prompts")
    print(f"Fast mode: {'yes (Layer1 only)' if args.fast else 'no (all 3 layers)'}")
    print(f"{'='*60}\n")

    t0 = time.time()
    results = run_benchmark(prompts, fast_exit=args.fast)
    elapsed = time.time() - t0

    metrics = compute_metrics(results)

    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total prompts:      {metrics['n_total']}")
    print(f"Attack prompts:     {metrics['n_attacks']}")
    print(f"Benign prompts:     {metrics['n_benign']}")
    print(f"")
    print(f"Precision:          {metrics['precision']:.2%}")
    print(f"Recall:             {metrics['recall']:.2%}")
    print(f"F1:                 {metrics['f1']:.2%}")
    print(f"Accuracy:           {metrics['accuracy']:.2%}")
    print(f"")
    print(f"Layer 1 catch rate: {metrics['l1_catch_rate']:.2%}")
    print(f"Layer 2 catch rate: {metrics['l2_catch_rate']:.2%}")
    print(f"Layer 3 catch rate: {metrics['l3_catch_rate']:.2%}")
    print(f"")
    print(f"Avg latency:        {metrics['avg_latency_ms']:.0f}ms")
    print(f"Total runtime:      {elapsed:.1f}s")
    print(f"")
    print("By category (bypass rate of attack prompts):")
    for cat, m in metrics["by_category"].items():
        if m["n_attacks"] > 0:
            rate = m["bypass_rate"] or 0
            bar = "#" * int(rate * 20)
            print(f"  {cat:<20} {rate:.0%}  {bar}")

    save_results(results, metrics, Path(args.out))


if __name__ == "__main__":
    main()
