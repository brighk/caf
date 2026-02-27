#!/usr/bin/env python3
"""
CounterBench Dataset Loader for CAF
====================================

Load and process CounterBench dataset from HuggingFace for CAF experiments.

CounterBench tests causal reasoning with counterfactual queries on deterministic
structural causal models (SCMs). This loader adapts the dataset format to CAF's
input requirements.

Dataset: https://huggingface.co/datasets/CounterBench/CounterBench
Paper: "CounterBench: A Benchmark for Counterfactual Reasoning" (2025)

Usage:
    python scripts/load_counterbench.py --output data/counterbench_caf.json
    python scripts/load_counterbench.py --subset basic --limit 100
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datasets import load_dataset


@dataclass
class CounterBenchExample:
    """Single CounterBench example adapted for CAF."""

    question_id: str
    given_info: str  # Causal structure description
    question: str  # Counterfactual query
    answer: str  # Yes/No ground truth
    reasoning_type: str  # Basic, Conditional, Joint, Nested

    # Metadata
    graph_id: Optional[str] = None
    model_id: Optional[str] = None
    query_type: Optional[str] = None
    rung: Optional[int] = None  # Pearl's ladder of causation
    story_id: Optional[str] = None

    def to_caf_format(self) -> Dict[str, Any]:
        """
        Convert to CAF input format.

        CAF expects:
        - query: The question to reason about
        - context: Background information
        - expected_answer: Ground truth (for evaluation)
        """
        return {
            "id": self.question_id,
            "query": self.question,
            "context": self.given_info,
            "expected_answer": self.answer,
            "metadata": {
                "reasoning_type": self.reasoning_type,
                "graph_id": self.graph_id,
                "model_id": self.model_id,
                "query_type": self.query_type,
                "rung": self.rung,
                "story_id": self.story_id
            }
        }


class CounterBenchLoader:
    """Load and process CounterBench dataset for CAF."""

    def __init__(
        self,
        dataset_name: str = "CounterBench/CounterBench",
        cache_dir: Optional[str] = None
    ):
        """
        Initialize loader.

        Args:
            dataset_name: HuggingFace dataset identifier
            cache_dir: Cache directory for downloaded data
        """
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.dataset = None

    def load(self, split: str = "train") -> None:
        """
        Load dataset from HuggingFace.

        Args:
            split: Dataset split (train, validation, test)
        """
        print(f"Loading {self.dataset_name}...")

        # CounterBench has mixed file formats - load only the compatible ones
        try:
            # Use only data_balanced_alpha_V1.json which has consistent schema
            full_dataset = load_dataset(
                self.dataset_name,
                data_files='data_balanced_alpha_V1.json',
                cache_dir=self.cache_dir
            )

            # Extract the train split
            if hasattr(full_dataset, 'keys'):
                if split in full_dataset:
                    self.dataset = full_dataset[split]
                else:
                    # Use first available split
                    first_split = list(full_dataset.keys())[0]
                    self.dataset = full_dataset[first_split]
            else:
                self.dataset = full_dataset

            print(f"✓ Loaded {len(self.dataset)} examples from data_balanced_alpha_V1.json")
        except Exception as e:
            print(f"✗ Failed to load: {e}")
            raise RuntimeError(f"Failed to load dataset: {e}")

    def parse_example(self, example: Dict[str, Any]) -> CounterBenchExample:
        """
        Parse raw example into CounterBenchExample.

        Args:
            example: Raw example from dataset

        Returns:
            Parsed CounterBenchExample
        """
        # Extract metadata (if nested in 'meta' field)
        meta = example.get('meta', {})
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except json.JSONDecodeError:
                meta = {}

        return CounterBenchExample(
            question_id=str(example.get('question_id', example.get('id', 'unknown'))),
            given_info=example.get('given_info', ''),
            question=example.get('question', ''),
            answer=example.get('answer', ''),
            reasoning_type=example.get('type', 'Unknown'),
            graph_id=meta.get('graph_id'),
            model_id=meta.get('model_id'),
            query_type=meta.get('query_type'),
            rung=meta.get('rung'),
            story_id=meta.get('story_id')
        )

    def filter_by_type(
        self,
        reasoning_types: Optional[List[str]] = None
    ) -> List[CounterBenchExample]:
        """
        Filter examples by reasoning type.

        Args:
            reasoning_types: List of types to include (Basic, Conditional, Joint, Nested)
                           If None, include all types

        Returns:
            Filtered list of examples
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load() first.")

        examples = []
        for raw_example in self.dataset:
            example = self.parse_example(raw_example)

            if reasoning_types is None or example.reasoning_type in reasoning_types:
                examples.append(example)

        return examples

    def to_caf_format(
        self,
        examples: List[CounterBenchExample]
    ) -> List[Dict[str, Any]]:
        """
        Convert examples to CAF input format.

        Args:
            examples: List of CounterBenchExample

        Returns:
            List of CAF-formatted dictionaries
        """
        return [ex.to_caf_format() for ex in examples]

    def save(
        self,
        examples: List[CounterBenchExample],
        output_path: str,
        format: str = 'caf'
    ) -> None:
        """
        Save examples to file.

        Args:
            examples: List of examples to save
            output_path: Output file path
            format: Output format ('caf' or 'raw')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'caf':
            data = self.to_caf_format(examples)
        else:
            data = [asdict(ex) for ex in examples]

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✓ Saved {len(examples)} examples to {output_path}")

    def get_statistics(self, examples: List[CounterBenchExample]) -> Dict[str, Any]:
        """
        Compute dataset statistics.

        Args:
            examples: List of examples

        Returns:
            Statistics dictionary
        """
        stats = {
            'total': len(examples),
            'by_type': {},
            'by_answer': {'Yes': 0, 'No': 0},
            'by_rung': {}
        }

        for ex in examples:
            # Count by reasoning type
            stats['by_type'][ex.reasoning_type] = \
                stats['by_type'].get(ex.reasoning_type, 0) + 1

            # Count by answer
            if ex.answer in stats['by_answer']:
                stats['by_answer'][ex.answer] += 1

            # Count by rung (if available)
            if ex.rung is not None:
                stats['by_rung'][ex.rung] = \
                    stats['by_rung'].get(ex.rung, 0) + 1

        return stats


def main():
    parser = argparse.ArgumentParser(
        description="Load CounterBench dataset for CAF experiments"
    )

    parser.add_argument(
        '--output',
        default='data/counterbench_caf.json',
        help='Output file path'
    )

    parser.add_argument(
        '--split',
        default='train',
        help='Dataset split (train, validation, test)'
    )

    parser.add_argument(
        '--subset',
        nargs='+',
        choices=['Basic', 'Conditional', 'Joint', 'Nested'],
        help='Filter by reasoning type'
    )

    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of examples'
    )

    parser.add_argument(
        '--cache-dir',
        help='Cache directory for downloaded data'
    )

    parser.add_argument(
        '--stats',
        action='store_true',
        help='Print dataset statistics'
    )

    args = parser.parse_args()

    # Load dataset
    loader = CounterBenchLoader(cache_dir=args.cache_dir)

    try:
        loader.load(split=args.split)
    except Exception as e:
        print(f"\n✗ Error loading dataset: {e}")
        print("\nMake sure you have internet connection and datasets library installed:")
        print("  pip install datasets")
        return 1

    # Filter examples
    examples = loader.filter_by_type(reasoning_types=args.subset)

    # Apply limit
    if args.limit:
        examples = examples[:args.limit]

    # Print statistics
    if args.stats or True:  # Always show stats
        print("\n" + "=" * 60)
        print("DATASET STATISTICS")
        print("=" * 60)

        stats = loader.get_statistics(examples)

        print(f"\nTotal examples: {stats['total']}")

        print("\nBy reasoning type:")
        for rtype, count in sorted(stats['by_type'].items()):
            print(f"  {rtype}: {count}")

        print("\nBy answer:")
        for answer, count in stats['by_answer'].items():
            print(f"  {answer}: {count}")

        if stats['by_rung']:
            print("\nBy rung (Pearl's ladder):")
            for rung, count in sorted(stats['by_rung'].items()):
                print(f"  Rung {rung}: {count}")

    # Save
    loader.save(examples, args.output, format='caf')

    # Print example
    if examples:
        print("\n" + "=" * 60)
        print("EXAMPLE (CAF FORMAT)")
        print("=" * 60)
        example_caf = examples[0].to_caf_format()
        print(json.dumps(example_caf, indent=2))

    print(f"\n✓ Dataset ready for CAF experiments!")
    print(f"\nNext steps:")
    print(f"  python -m experiments.run_counterbench_experiment --input {args.output}")

    return 0


if __name__ == '__main__':
    exit(main())
