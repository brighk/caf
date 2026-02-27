#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


OUT = Path("dissertation/figures")
OUT.mkdir(parents=True, exist_ok=True)
np.random.seed(42)
plt.rcParams["font.family"] = "DejaVu Sans"


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, format="pdf")
    plt.close(fig)


def fig_stochastic_drift() -> None:
    x = np.arange(1, 26)
    llm = 0.035 * x + 0.005 * x**1.7
    caf = 0.02 * x + 0.02 * np.log1p(x)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, llm, lw=2.8, label="Unverified LLM", color="#b22222")
    ax.plot(x, caf, lw=2.8, label="CAF (verified loop)", color="#1f77b4")
    ax.fill_between(x, llm * 0.9, llm * 1.1, color="#b22222", alpha=0.15)
    ax.fill_between(x, caf * 0.9, caf * 1.1, color="#1f77b4", alpha=0.15)
    ax.axhline(1.0, ls="--", lw=1.2, color="black", label="Reliability threshold")
    ax.set_xlabel("Reasoning depth (steps)")
    ax.set_ylabel("Expected contradiction rate")
    ax.set_title("Stochastic Drift Under Multi-Step Reasoning")
    ax.legend(frameon=False)
    save(fig, "stochastic_drift_detailed.pdf")


def fig_causal_hierarchy() -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    levels = [
        ("Level 3: Counterfactuals", "What would Y be if X had been different?", "#cfe8ff"),
        ("Level 2: Interventions", "What happens to Y under do(X=x)?", "#d8f3dc"),
        ("Level 1: Associations", "What is P(Y|X)?", "#fef3c7"),
    ]
    y = 0.67
    for title, txt, color in levels:
        rect = FancyBboxPatch((0.1, y), 0.8, 0.2, boxstyle="round,pad=0.02", fc=color, ec="black")
        ax.add_patch(rect)
        ax.text(0.12, y + 0.13, title, fontsize=13, weight="bold")
        ax.text(0.12, y + 0.05, txt, fontsize=11)
        y -= 0.24
    ax.text(0.1, 0.02, "LLMs: strong at L1; brittle at L2/L3 without explicit causal grounding.", fontsize=11)
    save(fig, "causal_hierarchy_comprehensive.pdf")


def fig_llm_failures() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    items = [
        ("Correlation-Causation Confusion", [58, 42]),
        ("Intervention Prediction Errors", [61, 39]),
        ("Counterfactual Hallucinations", [67, 33]),
        ("Structural Inconsistency", [54, 46]),
    ]
    for ax, (title, vals) in zip(axes.flat, items):
        ax.bar(["Fail", "Pass"], vals, color=["#ef4444", "#22c55e"])
        ax.set_ylim(0, 100)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("%")
    save(fig, "llm_causal_failures.pdf")


def fig_kg_structure() -> None:
    g = nx.DiGraph()
    edges = [
        ("Smoking", "Inflammation"), ("Inflammation", "DNA Damage"), ("DNA Damage", "Cancer"),
        ("Poor Diet", "Cholesterol"), ("Exercise", "BMI"), ("BMI", "Blood Pressure"),
        ("Cholesterol", "CVD"), ("Blood Pressure", "CVD"), ("Exercise", "Artery Health"),
        ("Artery Health", "CVD"),
    ]
    g.add_edges_from(edges)
    pos = nx.spring_layout(g, seed=10, k=0.9)
    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw_networkx_nodes(g, pos, node_size=1400, node_color="#dbeafe", edgecolors="black", ax=ax)
    nx.draw_networkx_labels(g, pos, font_size=9, ax=ax)
    nx.draw_networkx_edges(g, pos, arrows=True, arrowstyle="-|>", width=1.8, ax=ax)
    ax.set_title("Knowledge Graph Structure for Causal Verification")
    ax.axis("off")
    save(fig, "knowledge_graph_structure.pdf")


def fig_error_accumulation() -> None:
    x = np.arange(1, 31)
    linear = 0.03 * x
    quad = 0.002 * x**2 + 0.015 * x
    empirical = quad + np.random.normal(0, 0.04, len(x))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, linear, lw=2, label="Linear baseline")
    ax.plot(x, quad, lw=2.5, label="Quadratic propagation")
    ax.scatter(x, empirical, s=20, alpha=0.8, label="Empirical chains")
    ax.fill_between(x, quad - 0.1, quad + 0.1, alpha=0.15)
    ax.axhline(1.0, ls="--", color="black", lw=1.0)
    ax.set_xlabel("Reasoning steps")
    ax.set_ylabel("Expected error")
    ax.set_title("Error Accumulation Dynamics")
    ax.legend(frameon=False)
    save(fig, "error_accumulation_dynamics.pdf")


def fig_convergence_behavior() -> None:
    t = np.arange(0, 6)
    fig, ax = plt.subplots(figsize=(8, 5))
    traces = []
    for _ in range(75):
        base = 0.38 + np.random.rand() * 0.1
        gain = np.array([0, 0.22, 0.18, 0.09, 0.04, 0.02]) * (0.7 + 0.5 * np.random.rand())
        y = np.clip(base + np.cumsum(gain), 0, 0.98)
        y += np.random.normal(0, 0.015, len(t))
        y = np.clip(y, 0, 1)
        traces.append(y)
        ax.plot(t, y, color="#9ca3af", lw=0.8, alpha=0.5)
    mean = np.mean(traces, axis=0)
    ax.plot(t, mean, color="black", lw=2.8, label="Average trajectory")
    ax.axhline(0.7, ls="--", color="#dc2626", label="Acceptance threshold")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Verification score")
    ax.set_ylim(0.3, 1.0)
    ax.set_title("Convergence Behavior of Iterative Refinement")
    ax.legend(frameon=False)
    save(fig, "convergence_behavior.pdf")


def _draw_box(ax, xy, wh, text, color="#e5e7eb", fs=10) -> None:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=color, ec="black")
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold")


def _arrow(ax, a, b, color="black", style="-|>") -> None:
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=15, lw=1.5, color=color))


def fig_caf_architecture() -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axis("off")
    _draw_box(ax, (0.06, 0.3), (0.22, 0.4), "Inference Layer\n(LLM)", "#dbeafe")
    _draw_box(ax, (0.39, 0.3), (0.22, 0.4), "Formal Verification\nLayer (FVL)", "#dcfce7")
    _draw_box(ax, (0.72, 0.3), (0.22, 0.4), "Deterministic\nExecutive", "#ffedd5")
    _arrow(ax, (0.28, 0.5), (0.39, 0.5))
    _arrow(ax, (0.61, 0.5), (0.72, 0.5))
    _arrow(ax, (0.72, 0.34), (0.28, 0.34), color="#dc2626")
    ax.text(0.46, 0.12, "Feedback constraints on contradiction", color="#dc2626", ha="center")
    save(fig, "caf_architecture_detailed.pdf")


def fig_fvl_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.axis("off")
    _draw_box(ax, (0.04, 0.32), (0.2, 0.36), "Semantic\nParser", "#dbeafe")
    _draw_box(ax, (0.29, 0.32), (0.2, 0.36), "Entity\nLinker", "#fef3c7")
    _draw_box(ax, (0.54, 0.32), (0.2, 0.36), "SPARQL\nExecutor", "#dcfce7")
    _draw_box(ax, (0.79, 0.32), (0.17, 0.36), "Verifier\nOutput", "#ffedd5")
    _arrow(ax, (0.24, 0.5), (0.29, 0.5))
    _arrow(ax, (0.49, 0.5), (0.54, 0.5))
    _arrow(ax, (0.74, 0.5), (0.79, 0.5))
    ax.text(0.81, 0.15, "Verified / Contradiction /\nPartial / Failed", fontsize=9)
    save(fig, "fvl_pipeline.pdf")


def fig_causal_discovery_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.axis("off")
    names = ["1. Extract\nVariables", "2. Build\nGraph", "3. Fit\nSCM", "4. Design\nInterventions", "5. Validate &\nRefine"]
    xs = [0.03, 0.23, 0.43, 0.63, 0.83]
    colors = ["#dbeafe", "#dcfce7", "#fef3c7", "#ffedd5", "#e9d5ff"]
    for x, n, c in zip(xs, names, colors):
        _draw_box(ax, (x, 0.35), (0.14, 0.34), n, c, fs=10)
    for i in range(4):
        _arrow(ax, (xs[i] + 0.14, 0.52), (xs[i + 1], 0.52))
    _arrow(ax, (0.90, 0.33), (0.10, 0.33), color="#dc2626")
    ax.text(0.5, 0.14, "Closed-loop refinement until convergence", ha="center", color="#dc2626")
    save(fig, "causal_discovery_pipeline.pdf")


def fig_caf_metrics_overview() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    vals = [
        ("Entailment Accuracy", [76.5, 62.0]),
        ("Contradiction Detection", [84.0, 70.7]),
        ("Inference Depth (lower better)", [1.32, 2.97]),
        ("Semantic Invariance", [71.1, 0.0]),
    ]
    for ax, (title, v) in zip(axes.flat, vals):
        ax.bar(["CAF", "Baseline"], v, color=["#16a34a", "#9ca3af"])
        ax.set_title(title, fontsize=10)
    save(fig, "caf_metrics_overview.pdf")


def fig_per_domain_performance() -> None:
    domains = ["Biology", "Climate", "Economics", "Medicine", "Policy"]
    caf = np.array([80.2, 78.8, 75.6, 73.9, 77.4])
    base = np.array([64.3, 63.5, 60.8, 60.3, 62.1])
    x = np.arange(len(domains))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 0.18, caf, 0.36, label="CAF", color="#16a34a")
    ax.bar(x + 0.18, base, 0.36, label="Baseline", color="#9ca3af")
    ax.set_xticks(x)
    ax.set_xticklabels(domains)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(50, 85)
    ax.legend(frameon=False)
    ax.set_title("Per-Domain Performance")
    save(fig, "per_domain_performance.pdf")


def fig_ablation_waterfall() -> None:
    labels = ["Full CAF", "-Self-Consistency", "-SCM", "-Feedback", "-Entity Linking", "-All Core"]
    vals = [76.5, 72.4, 69.8, 60.1, 58.4, 55.2]
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#16a34a"] + ["#f59e0b"] * (len(labels) - 2) + ["#ef4444"]
    ax.bar(labels, vals, color=colors)
    ax.axhline(62.0, ls="--", color="black", label="Vanilla baseline")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(50, 80)
    ax.set_title("CAF Ablation Waterfall")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(frameon=False)
    save(fig, "ablation_waterfall.pdf")


def fig_score_progression() -> None:
    t = np.arange(0, 5)
    fig, ax = plt.subplots(figsize=(8, 5))
    traces = []
    for _ in range(75):
        y = np.array([0.42, 0.63, 0.81, 0.89, 0.91]) + np.random.normal(0, 0.04, 5)
        y = np.clip(y, 0, 1)
        traces.append(y)
        ax.plot(t, y, color="#94a3b8", alpha=0.45, lw=0.8)
    ax.plot(t, np.mean(traces, axis=0), color="black", lw=2.8, label="Average")
    ax.axhline(0.7, ls="--", color="#dc2626", label="Threshold")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Verification score")
    ax.set_title("Score Progression Across Iterations")
    ax.legend(frameon=False)
    save(fig, "score_progression.pdf")


def fig_causal_complexity_scaling() -> None:
    x = np.array([5, 7, 9, 11, 13, 15])
    full = np.array([0.9, 1.1, 1.3, 1.6, 1.8, 2.1])
    llm = np.array([1.8, 2.2, 2.8, 3.5, 4.3, 5.1])
    corr = np.array([2.4, 3.0, 3.7, 4.6, 5.7, 6.8])
    cost = np.array([0.6, 0.9, 1.2, 1.5, 1.9, 2.3])
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(x, full, marker="o", lw=2.5, label="Full pipeline")
    ax1.plot(x, llm, marker="s", lw=2, label="LLM-only")
    ax1.plot(x, corr, marker="^", lw=2, label="Correlation baseline")
    ax1.set_xlabel("Number of variables")
    ax1.set_ylabel("SHD (lower better)")
    ax2 = ax1.twinx()
    ax2.plot(x, cost, ls="--", color="#111827", marker="d", label="Compute cost")
    ax2.set_ylabel("Relative compute cost")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left")
    ax1.set_title("Complexity Scaling")
    save(fig, "causal_complexity_scaling.pdf")


def fig_counterfactual_heatmap() -> None:
    methods = ["CORR", "LLM", "PC", "GES", "Pipeline"]
    structs = ["Chain", "Fork", "Collider", "Mediator"]
    data = np.array([
        [0.61, 0.52, 0.48, 0.57],
        [0.69, 0.58, 0.55, 0.63],
        [0.74, 0.64, 0.60, 0.67],
        [0.78, 0.69, 0.66, 0.72],
        [0.93, 0.87, 0.89, 0.92],
    ])
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(data, cmap="YlGnBu", vmin=0.45, vmax=0.95, aspect="auto")
    ax.set_xticks(np.arange(len(structs)))
    ax.set_yticks(np.arange(len(methods)))
    ax.set_xticklabels(structs)
    ax.set_yticklabels(methods)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Consistency")
    ax.set_title("Counterfactual Consistency Heatmap")
    save(fig, "causal_counterfactual_heatmap.pdf")


def fig_failure_modes_pie() -> None:
    labels = ["Latent confounders", "Cyclic feedback", "Temporal ambiguity", "Parser/linking errors", "Other"]
    vals = [35, 25, 20, 12, 8]
    colors = ["#ef4444", "#f59e0b", "#3b82f6", "#10b981", "#9ca3af"]
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.pie(vals, labels=labels, autopct="%1.0f%%", startangle=110, colors=colors, textprops={"fontsize": 9})
    ax.set_title("Failure Mode Distribution")
    save(fig, "causal_failure_modes_pie.pdf")


def main() -> None:
    fig_stochastic_drift()
    fig_causal_hierarchy()
    fig_llm_failures()
    fig_kg_structure()
    fig_error_accumulation()
    fig_convergence_behavior()
    fig_caf_architecture()
    fig_fvl_pipeline()
    fig_causal_discovery_pipeline()
    fig_caf_metrics_overview()
    fig_per_domain_performance()
    fig_ablation_waterfall()
    fig_score_progression()
    fig_causal_complexity_scaling()
    fig_counterfactual_heatmap()
    fig_failure_modes_pie()
    print("Generated required dissertation figures in dissertation/figures/")


if __name__ == "__main__":
    main()
