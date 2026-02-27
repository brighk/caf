#!/usr/bin/env python3
"""
Compare RAG vs CAF Results
===========================

Analyzes and compares results from:
- Pure RAG baseline (retrieval + single-shot generation)
- CAF (neuro-symbolic with SPARQL verification)
"""

import json
from pathlib import Path
from typing import Dict, Any


def load_results(results_dir: Path) -> Dict[str, Any]:
    """Load results from directory."""
    # Try different result file names
    for filename in ['results.json', 'summary.txt', 'report.txt']:
        filepath = results_dir / filename
        if filepath.exists():
            if filename.endswith('.json'):
                with open(filepath) as f:
                    return json.load(f)
            else:
                with open(filepath) as f:
                    return {'text': f.read()}
    return None


def parse_caf_report(report_path: Path) -> Dict[str, Any]:
    """Parse CAF report.txt file."""
    with open(report_path) as f:
        lines = f.readlines()

    metrics = {}
    for line in lines:
        if 'Total examples:' in line:
            metrics['total'] = int(line.split(':')[1].strip())
        elif 'Correct:' in line:
            metrics['correct'] = int(line.split(':')[1].strip())
        elif 'Accuracy:' in line:
            metrics['accuracy'] = float(line.split(':')[1].strip().rstrip('%')) / 100
        elif 'Avg iterations:' in line:
            metrics['avg_iterations'] = float(line.split(':')[1].strip())
        elif 'Avg score:' in line:
            metrics['avg_score'] = float(line.split(':')[1].strip())

    return metrics


def parse_rag_summary(summary_path: Path) -> Dict[str, Any]:
    """Parse RAG summary.txt file."""
    with open(summary_path) as f:
        lines = f.readlines()

    metrics = {}
    for line in lines:
        if 'Total examples:' in line:
            metrics['total'] = int(line.split(':')[1].strip())
        elif 'Correct:' in line:
            metrics['correct'] = int(line.split(':')[1].strip())
        elif 'Accuracy:' in line:
            metrics['accuracy'] = float(line.split(':')[1].strip().rstrip('%')) / 100
        elif 'Avg iterations:' in line:
            metrics['avg_iterations'] = float(line.split(':')[1].strip().split()[0])

    return metrics


def compare_systems(rag_dir: Path, caf_dir: Path):
    """Compare RAG and CAF results."""

    # Load results
    rag_metrics = parse_rag_summary(rag_dir / 'summary.txt')
    caf_metrics = parse_caf_report(caf_dir / 'report.txt')

    print("="*70)
    print("RAG vs CAF: Comparative Analysis")
    print("="*70)
    print()

    print("System Performance Comparison")
    print("-"*70)
    print(f"{'Metric':<25} | {'RAG (Pure Neural)':<20} | {'CAF (Neuro-Symbolic)':<20}")
    print("-"*70)
    print(f"{'Accuracy':<25} | {rag_metrics['accuracy']*100:>18.2f}% | {caf_metrics['accuracy']*100:>18.2f}%")
    print(f"{'Correct / Total':<25} | {rag_metrics['correct']:>4d} / {rag_metrics['total']:<13d} | {caf_metrics['correct']:>4d} / {caf_metrics['total']:<13d}")
    print(f"{'Iterations':<25} | {rag_metrics.get('avg_iterations', 1.0):>18.1f} | {caf_metrics.get('avg_iterations', 0):>18.1f}")

    if 'avg_score' in caf_metrics:
        print(f"{'Verification Score':<25} | {'N/A':>18s} | {caf_metrics['avg_score']:>18.2f}")

    print()

    # Determine winner
    rag_acc = rag_metrics['accuracy']
    caf_acc = caf_metrics['accuracy']

    print("Analysis")
    print("-"*70)

    if abs(rag_acc - caf_acc) < 0.05:
        print("✓ Both systems perform similarly (within 5%)")
    elif rag_acc > caf_acc:
        diff = (rag_acc - caf_acc) * 100
        print(f"✓ RAG outperforms CAF by {diff:.1f} percentage points")
        print(f"  Reason: Single-shot generation works well for this task,")
        print(f"  while SPARQL verification may be too strict")
    else:
        diff = (caf_acc - rag_acc) * 100
        print(f"✓ CAF outperforms RAG by {diff:.1f} percentage points")
        print(f"  Reason: Iterative refinement and symbolic verification")
        print(f"  help correct LLM errors")

    print()

    print("Key Differences")
    print("-"*70)
    print("RAG Approach:")
    print("  - Single-shot LLM generation (fast)")
    print("  - Keyword-based retrieval")
    print("  - No verification or iteration")
    print("  - Pure neural approach")
    print()
    print("CAF Approach:")
    print("  - Multi-iteration refinement (slower)")
    print("  - SPARQL-based symbolic verification")
    print("  - Decision engine for quality control")
    print("  - Neuro-symbolic hybrid")
    print()

    # Speed comparison
    rag_iters = rag_metrics.get('avg_iterations', 1.0)
    caf_iters = caf_metrics.get('avg_iterations', 1.0)

    if caf_iters > rag_iters:
        speedup = caf_iters / rag_iters
        print(f"Speed: RAG is ~{speedup:.1f}x faster (fewer iterations)")

    print()
    print("="*70)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Compare RAG vs CAF results')
    parser.add_argument('--rag-dir', type=Path, required=True, help='RAG results directory')
    parser.add_argument('--caf-dir', type=Path, required=True, help='CAF results directory')

    args = parser.parse_args()

    compare_systems(args.rag_dir, args.caf_dir)


if __name__ == '__main__':
    main()
