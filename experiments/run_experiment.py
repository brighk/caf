"""
CAF Full Experiment Runner
==========================
Executes the complete experiment pipeline:

1. Generate synthetic causal chain dataset (75 chains, 3 perturbations each)
2. Run CAF verification loop on all chains
3. Compute all metrics (inference depth, contradiction rate, entailment accuracy)
4. Generate paper-ready artifacts (tables, figures, JSON results)

Usage:
    python -m experiments.run_experiment [--output-dir OUTPUT_DIR] [--num-chains N]
"""

import argparse
import json
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
from dataclasses import asdict
import random

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.synthetic_dataset import (
    SyntheticDatasetGenerator,
    CausalChain,
    PerturbationType,
)
from experiments.caf_algorithm import (
    CAFLoop,
    CAFConfig,
    CAFOutput,
    get_algorithm_pseudocode,
    get_algorithm_latex,
)
from experiments.metrics import (
    MetricsCalculator,
    ExperimentMetrics,
    compute_baseline_comparison,
    generate_latex_results_table,
)


class ExperimentRunner:
    """
    Complete experiment runner for CAF evaluation.

    Orchestrates dataset generation, CAF execution, and metrics computation
    to produce paper-ready results.
    """

    def __init__(
        self,
        output_dir: str = "experiments/results",
        seed: int = 42,
        verbose: bool = True
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.verbose = verbose
        random.seed(seed)

        # Initialize components
        self.generator = SyntheticDatasetGenerator(seed=seed)
        self.caf = CAFLoop(config=CAFConfig(
            max_iterations=5,
            verification_threshold=0.8,
        ))
        self.metrics_calc = MetricsCalculator()

        # Results storage
        self.dataset: List[CausalChain] = []
        self.caf_outputs: List[CAFOutput] = []
        self.baseline_outputs: List[CAFOutput] = []
        self.perturbation_outputs: List[Dict[PerturbationType, CAFOutput]] = []

    def log(self, message: str):
        """Log message if verbose mode enabled."""
        if self.verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

    def generate_dataset(
        self,
        num_chains: int = 75,
        perturbations_per_chain: int = 3
    ) -> List[CausalChain]:
        """
        Generate synthetic causal chain dataset.

        Args:
            num_chains: Number of chains (50-100 for paper)
            perturbations_per_chain: Perturbations per chain (2-3 for paper)

        Returns:
            List of generated CausalChain objects
        """
        self.log(f"Generating dataset: {num_chains} chains, {perturbations_per_chain} perturbations each...")

        self.dataset = self.generator.generate_dataset(
            num_chains=num_chains,
            min_depth=2,
            max_depth=10,
            perturbations_per_chain=perturbations_per_chain,
            contradiction_rate=0.2,
            domains=["physics", "biology", "economics", "logic", "causality"]
        )

        self.log(f"Generated {len(self.dataset)} chains with "
                 f"{sum(len(c.perturbations) for c in self.dataset)} total perturbations")

        return self.dataset

    def run_caf_evaluation(self) -> Tuple[List[CAFOutput], List[CAFOutput]]:
        """
        Run CAF evaluation on all chains.

        Returns:
            Tuple of (caf_outputs, baseline_outputs)
        """
        self.log("Running CAF evaluation...")
        self.caf_outputs = []
        self.baseline_outputs = []
        self.perturbation_outputs = []

        for i, chain in enumerate(self.dataset):
            if self.verbose and (i + 1) % 10 == 0:
                self.log(f"  Processing chain {i + 1}/{len(self.dataset)}...")

            # Run CAF on original prompt
            caf_output, baseline_output = self.caf.execute_with_baseline(
                chain.to_prompt()
            )
            self.caf_outputs.append(caf_output)
            self.baseline_outputs.append(baseline_output)

            # Run on perturbations
            perturbation_results = {}
            for perturbation in chain.perturbations:
                pert_output = self.caf.execute(perturbation.perturbed_prompt)
                perturbation_results[perturbation.perturbation_type] = pert_output

            self.perturbation_outputs.append(perturbation_results)

        self.log(f"Completed CAF evaluation on {len(self.dataset)} chains")
        return self.caf_outputs, self.baseline_outputs

    def compute_metrics(self) -> Tuple[ExperimentMetrics, ExperimentMetrics]:
        """
        Compute all evaluation metrics.

        Returns:
            Tuple of (caf_metrics, baseline_metrics)
        """
        self.log("Computing metrics...")

        caf_metrics = self.metrics_calc.compute_all_metrics(
            self.dataset,
            self.caf_outputs,
            self.perturbation_outputs
        )

        baseline_metrics = self.metrics_calc.compute_all_metrics(
            self.dataset,
            self.baseline_outputs,
            None  # No perturbations for baseline
        )

        self.log("Metrics computed successfully")
        return caf_metrics, baseline_metrics

    def export_results(
        self,
        caf_metrics: ExperimentMetrics,
        baseline_metrics: ExperimentMetrics
    ) -> Dict[str, str]:
        """
        Export all results to files.

        Returns:
            Dictionary mapping result type to file path
        """
        self.log("Exporting results...")
        exported_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Export dataset
        dataset_path = self.output_dir / f"synthetic_dataset_{timestamp}.json"
        self.generator.export_dataset(self.dataset, str(dataset_path))
        exported_files["dataset"] = str(dataset_path)

        # 2. Export metrics JSON
        metrics_data = {
            "experiment_info": {
                "timestamp": timestamp,
                "seed": self.seed,
                "num_chains": len(self.dataset),
                "total_perturbations": sum(len(c.perturbations) for c in self.dataset),
            },
            "caf_metrics": caf_metrics.to_dict(),
            "baseline_metrics": baseline_metrics.to_dict(),
            "comparison": compute_baseline_comparison(caf_metrics, baseline_metrics),
        }

        metrics_path = self.output_dir / f"experiment_metrics_{timestamp}.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics_data, f, indent=2)
        exported_files["metrics"] = str(metrics_path)

        # 3. Export LaTeX results table
        latex_table = generate_latex_results_table(caf_metrics, baseline_metrics)
        latex_path = self.output_dir / f"results_table_{timestamp}.tex"
        with open(latex_path, "w") as f:
            f.write(latex_table)
        exported_files["latex_table"] = str(latex_path)

        # 4. Export algorithm LaTeX
        algorithm_path = self.output_dir / f"algorithm_{timestamp}.tex"
        with open(algorithm_path, "w") as f:
            f.write(get_algorithm_latex())
        exported_files["algorithm_latex"] = str(algorithm_path)

        # 5. Export summary report
        report = self._generate_report(caf_metrics, baseline_metrics)
        report_path = self.output_dir / f"experiment_report_{timestamp}.txt"
        with open(report_path, "w") as f:
            f.write(report)
        exported_files["report"] = str(report_path)

        self.log(f"Exported {len(exported_files)} files to {self.output_dir}")
        return exported_files

    def _generate_report(
        self,
        caf_metrics: ExperimentMetrics,
        baseline_metrics: ExperimentMetrics
    ) -> str:
        """Generate human-readable experiment report."""
        comparison = compute_baseline_comparison(caf_metrics, baseline_metrics)

        report = f"""
{'='*70}
CAF EXPERIMENT REPORT
{'='*70}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Seed: {self.seed}

DATASET SUMMARY
---------------
Total Chains: {len(self.dataset)}
Total Perturbations: {sum(len(c.perturbations) for c in self.dataset)}
Domains: {', '.join(set(c.domain for c in self.dataset))}
Depth Range: {min(c.depth for c in self.dataset)} - {max(c.depth for c in self.dataset)}
Chains with Contradictions: {sum(1 for c in self.dataset if c.injected_contradictions)}

{'='*70}
PRIMARY METRICS (Table 1)
{'='*70}

Metric                    | Baseline | CAF      | Improvement
--------------------------|----------|----------|------------
Inference Depth (d)       | {baseline_metrics.mean_inference_depth:>8.2f} | {caf_metrics.mean_inference_depth:>8.2f} | +{comparison['inference_depth']['improvement_pct']:.1f}%
Contradiction Rate (%)    | {baseline_metrics.contradiction_rate_percent:>8.1f} | {caf_metrics.contradiction_rate_percent:>8.1f} | -{comparison['contradiction_rate']['delta']:.1f}pp
Entailment Accuracy       | {baseline_metrics.entailment_accuracy:>8.4f} | {caf_metrics.entailment_accuracy:>8.4f} | +{comparison['entailment_accuracy']['improvement_pct']:.1f}%
Semantic Invariance       | {baseline_metrics.semantic_invariance_mean:>8.4f} | {caf_metrics.semantic_invariance_mean:>8.4f} | +{comparison['semantic_invariance']['improvement_pct']:.1f}%

{'='*70}
STATISTICAL DETAILS
{'='*70}

CAF Results:
  - Inference Depth: {caf_metrics.mean_inference_depth:.2f} ± {caf_metrics.std_inference_depth:.2f}
  - 95% CI for Entailment: [{caf_metrics.confidence_interval_95[0]:.4f}, {caf_metrics.confidence_interval_95[1]:.4f}]

{'='*70}
PER-DOMAIN BREAKDOWN
{'='*70}
"""

        for domain, metrics in caf_metrics.metrics_by_domain.items():
            report += f"\n{domain.upper()}:\n"
            report += f"  Mean Depth: {metrics['mean_depth']:.2f}\n"
            report += f"  Contradiction Rate: {metrics['contradiction_rate']:.1f}%\n"
            report += f"  Entailment Accuracy: {metrics['entailment_accuracy']:.4f}\n"

        report += f"""
{'='*70}
ALGORITHM PSEUDOCODE
{'='*70}
{get_algorithm_pseudocode()}

{'='*70}
END OF REPORT
{'='*70}
"""
        return report

    def run_full_experiment(
        self,
        num_chains: int = 75,
        perturbations_per_chain: int = 3
    ) -> Dict[str, Any]:
        """
        Run the complete experiment pipeline.

        Args:
            num_chains: Number of chains (50-100)
            perturbations_per_chain: Perturbations per chain (2-3)

        Returns:
            Complete experiment results dictionary
        """
        start_time = time.time()
        self.log("=" * 60)
        self.log("STARTING CAF FULL EXPERIMENT")
        self.log("=" * 60)

        # Step 1: Generate dataset
        self.generate_dataset(num_chains, perturbations_per_chain)

        # Step 2: Run CAF evaluation
        self.run_caf_evaluation()

        # Step 3: Compute metrics
        caf_metrics, baseline_metrics = self.compute_metrics()

        # Step 4: Export results
        exported_files = self.export_results(caf_metrics, baseline_metrics)

        total_time = time.time() - start_time
        self.log("=" * 60)
        self.log(f"EXPERIMENT COMPLETE in {total_time:.1f}s")
        self.log("=" * 60)

        # Print summary
        comparison = compute_baseline_comparison(caf_metrics, baseline_metrics)
        print("\n" + "=" * 60)
        print("FINAL RESULTS SUMMARY")
        print("=" * 60)
        print(f"\nMean Inference Depth: {caf_metrics.mean_inference_depth:.2f} "
              f"(+{comparison['inference_depth']['improvement_pct']:.1f}% vs baseline)")
        print(f"Contradiction Rate: {caf_metrics.contradiction_rate_percent:.1f}% "
              f"(-{comparison['contradiction_rate']['delta']:.1f}pp vs baseline)")
        print(f"Entailment Accuracy: {caf_metrics.entailment_accuracy:.4f} "
              f"(+{comparison['entailment_accuracy']['improvement_pct']:.1f}% vs baseline)")
        print(f"Semantic Invariance: {caf_metrics.semantic_invariance_mean:.4f}")
        print(f"\nResults exported to: {self.output_dir}")
        print("=" * 60)

        return {
            "caf_metrics": caf_metrics.to_dict(),
            "baseline_metrics": baseline_metrics.to_dict(),
            "comparison": comparison,
            "exported_files": exported_files,
            "execution_time_seconds": total_time,
        }


def main():
    """Main entry point for experiment execution."""
    parser = argparse.ArgumentParser(
        description="Run CAF full experiment for paper"
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/results",
        help="Directory for output files"
    )
    parser.add_argument(
        "--num-chains",
        type=int,
        default=75,
        help="Number of causal chains (50-100)"
    )
    parser.add_argument(
        "--perturbations",
        type=int,
        default=3,
        help="Perturbations per chain (2-3)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages"
    )

    args = parser.parse_args()

    # Validate parameters
    if not 50 <= args.num_chains <= 100:
        print(f"Warning: num_chains={args.num_chains} is outside recommended range [50, 100]")

    if not 2 <= args.perturbations <= 3:
        print(f"Warning: perturbations={args.perturbations} is outside recommended range [2, 3]")

    # Run experiment
    runner = ExperimentRunner(
        output_dir=args.output_dir,
        seed=args.seed,
        verbose=not args.quiet
    )

    results = runner.run_full_experiment(
        num_chains=args.num_chains,
        perturbations_per_chain=args.perturbations
    )

    return results


if __name__ == "__main__":
    main()
