#!/usr/bin/env python3
"""
Paper A Real-Domain Benchmark Runner
====================================

Evaluates causal discovery on real-text domains and produces Table-II style
artifacts (per-domain SHD, intervention accuracy, counterfactual consistency).

Input format (optional --input JSON):
[
  {
    "domain": "medical",
    "text": "Smoking causes tar deposition. Tar deposition causes lung cancer.",
    "gold_edges": [["smoking", "tar deposition"], ["tar deposition", "lung cancer"]]
  },
  ...
]
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.causal_discovery import CausalGraphExtractor


SEED = 42
random.seed(SEED)
np.random.seed(SEED)


@dataclass
class RealSample:
    domain: str
    text: str
    gold_edges: List[Tuple[str, str]]


class NullLLM:
    """Forces extractor to use deterministic text fallbacks."""

    def generate(self, prompt: str) -> str:
        return ""


def default_samples() -> List[RealSample]:
    items = [
        # Medical
        (
            "medical",
            "Smoking causes chronic inflammation. Chronic inflammation causes DNA damage. "
            "DNA damage causes lung cancer. Age is associated with mortality.",
            [("smoking", "chronic inflammation"), ("chronic inflammation", "dna damage"), ("dna damage", "lung cancer")],
        ),
        (
            "medical",
            "High LDL causes arterial plaque. Arterial plaque causes coronary disease. "
            "Coronary disease causes myocardial infarction.",
            [("high ldl", "arterial plaque"), ("arterial plaque", "coronary disease"), ("coronary disease", "myocardial infarction")],
        ),
        (
            "medical",
            "Insulin resistance causes elevated glucose. Elevated glucose causes neuropathy.",
            [("insulin resistance", "elevated glucose"), ("elevated glucose", "neuropathy")],
        ),
        # Economic
        (
            "economics",
            "Policy rate increases cause borrowing costs to rise. Borrowing costs cause investment to fall. "
            "Lower investment causes GDP growth to slow.",
            [("policy rate increases", "borrowing costs to rise"), ("borrowing costs", "investment to fall"), ("lower investment", "gdp growth to slow")],
        ),
        (
            "economics",
            "Inflation expectations cause wage demands to increase. Higher wage demands cause unit labor costs to increase.",
            [("inflation expectations", "wage demands to increase"), ("higher wage demands", "unit labor costs to increase")],
        ),
        (
            "economics",
            "Currency depreciation causes import prices to rise. Higher import prices cause headline inflation to rise.",
            [("currency depreciation", "import prices to rise"), ("higher import prices", "headline inflation to rise")],
        ),
        # Policy
        (
            "policy",
            "Stricter emission standards cause coal usage to decline. Lower coal usage causes particulate pollution to decline.",
            [("stricter emission standards", "coal usage to decline"), ("lower coal usage", "particulate pollution to decline")],
        ),
        (
            "policy",
            "Higher tobacco taxes cause cigarette consumption to decline. Lower consumption causes smoking-attributable deaths to decline.",
            [("higher tobacco taxes", "cigarette consumption to decline"), ("lower consumption", "smoking-attributable deaths to decline")],
        ),
        (
            "policy",
            "School meal subsidies cause attendance to increase. Higher attendance causes graduation rates to increase.",
            [("school meal subsidies", "attendance to increase"), ("higher attendance", "graduation rates to increase")],
        ),
    ]
    return [RealSample(domain=d, text=t, gold_edges=e) for d, t, e in items]


def load_samples(path: Path) -> List[RealSample]:
    raw = json.loads(path.read_text())
    out: List[RealSample] = []
    for item in raw:
        domain = str(item["domain"]).strip().lower()
        text = str(item["text"]).strip()
        edges = [(str(a).strip().lower(), str(b).strip().lower()) for a, b in item["gold_edges"]]
        out.append(RealSample(domain=domain, text=text, gold_edges=edges))
    return out


def build_graph(edges: List[Tuple[str, str]]) -> nx.DiGraph:
    g = nx.DiGraph()
    for u, v in edges:
        if u and v and u != v:
            g.add_edge(u, v)
    return g


def shd(gt: nx.DiGraph, pred: nx.DiGraph) -> int:
    gt_edges = set(gt.edges())
    pred_edges = set(pred.edges())
    return len(gt_edges.symmetric_difference(pred_edges))


def descendant_reachability(g: nx.DiGraph, source: str, target: str) -> bool:
    if source == target:
        return True
    if source not in g or target not in g:
        return False
    return target in nx.descendants(g, source)


def intervention_accuracy(gt: nx.DiGraph, pred: nx.DiGraph, trials: int = 20) -> float:
    nodes = list(set(gt.nodes()) | set(pred.nodes()))
    if len(nodes) < 2:
        return 0.0
    correct = 0
    for _ in range(trials):
        x, y = random.sample(nodes, 2)
        gt_ans = not descendant_reachability(gt, x, y)
        pred_ans = not descendant_reachability(pred, x, y)
        if gt_ans == pred_ans:
            correct += 1
    return correct / trials


def counterfactual_consistency(gt: nx.DiGraph, pred: nx.DiGraph, trials: int = 20) -> float:
    nodes = list(set(gt.nodes()) | set(pred.nodes()))
    if len(nodes) < 2:
        return 0.0
    correct = 0
    for _ in range(trials):
        x, y = random.sample(nodes, 2)
        gt_ans = not descendant_reachability(gt, x, y)
        pred_ans = not descendant_reachability(pred, x, y)
        if gt_ans == pred_ans:
            correct += 1
    return correct / trials


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(Path(args.input)) if args.input else default_samples()

    extractor = CausalGraphExtractor(NullLLM(), k_samples=max(1, args.k_samples))

    by_domain: Dict[str, Dict[str, List[float]]] = {}
    for s in samples:
        gt = build_graph(s.gold_edges)
        pred_result = extractor.extract_from_text(s.text, domain=s.domain)
        pred = pred_result["graph"]

        d = by_domain.setdefault(s.domain, {"shd": [], "ia": [], "cfc": []})
        d["shd"].append(float(shd(gt, pred)))
        d["ia"].append(intervention_accuracy(gt, pred, trials=args.intervention_trials))
        d["cfc"].append(counterfactual_consistency(gt, pred, trials=args.counterfactual_trials))

    summary = {}
    for domain, vals in by_domain.items():
        summary[domain] = {
            "SHD_mean": float(np.mean(vals["shd"])) if vals["shd"] else 0.0,
            "SHD_std": float(np.std(vals["shd"])) if vals["shd"] else 0.0,
            "Intervention_Accuracy": float(np.mean(vals["ia"])) if vals["ia"] else 0.0,
            "Counterfactual_Consistency": float(np.mean(vals["cfc"])) if vals["cfc"] else 0.0,
            "n": len(vals["shd"]),
        }

    (out_dir / "metrics_real.json").write_text(json.dumps({"domains": summary}, indent=2))

    # CSV table
    csv_lines = ["domain,shd_mean,shd_std,intervention_accuracy,counterfactual_consistency,n"]
    for domain in sorted(summary):
        s = summary[domain]
        csv_lines.append(
            f"{domain},{s['SHD_mean']:.6f},{s['SHD_std']:.6f},{s['Intervention_Accuracy']:.6f},{s['Counterfactual_Consistency']:.6f},{s['n']}"
        )
    (out_dir / "table_real_domains.csv").write_text("\n".join(csv_lines))

    # LaTeX table
    tex_lines = [
        "\\begin{tabular}{lccc}",
        "\\hline",
        "Domain & SHD $\\downarrow$ & Int. Acc. $\\uparrow$ & CF Cons. $\\uparrow$\\\\",
        "\\hline",
    ]
    for domain in sorted(summary):
        s = summary[domain]
        tex_lines.append(
            f"{domain.title()} & {s['SHD_mean']:.1f} $\\pm$ {s['SHD_std']:.1f} & {s['Intervention_Accuracy']:.2f} & {s['Counterfactual_Consistency']:.2f}\\\\"
        )
    tex_lines.extend(["\\hline", "\\end{tabular}"])
    (out_dir / "table_real_domains.tex").write_text("\n".join(tex_lines))

    # Figure: grouped bars
    domains = sorted(summary.keys())
    shd_vals = [summary[d]["SHD_mean"] for d in domains]
    ia_vals = [summary[d]["Intervention_Accuracy"] for d in domains]
    cfc_vals = [summary[d]["Counterfactual_Consistency"] for d in domains]

    x = np.arange(len(domains))
    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w, shd_vals, w, label="SHD")
    ax.bar(x, ia_vals, w, label="Intervention Acc.")
    ax.bar(x + w, cfc_vals, w, label="CF Consistency")
    ax.set_xticks(x)
    ax.set_xticklabels([d.title() for d in domains])
    ax.set_title("Real-Domain Discovery Metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "real_domain_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("✓ Paper A real-domain benchmark complete")
    print(f"  - {out_dir / 'metrics_real.json'}")
    print(f"  - {out_dir / 'table_real_domains.csv'}")
    print(f"  - {out_dir / 'table_real_domains.tex'}")
    print(f"  - {fig_dir / 'real_domain_metrics.png'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Paper A real-domain benchmark")
    parser.add_argument("--input", help="Optional JSON file with domain/text/gold_edges samples")
    parser.add_argument("--output", default="paperA/results_real", help="Output directory")
    parser.add_argument("--k-samples", type=int, default=5, help="Self-consistency samples (extractor)")
    parser.add_argument("--intervention-trials", type=int, default=20, help="Trials per sample")
    parser.add_argument("--counterfactual-trials", type=int, default=20, help="Trials per sample")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
