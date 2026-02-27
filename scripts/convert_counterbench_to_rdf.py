#!/usr/bin/env python3
"""
Convert CounterBench Causal Relations to RDF
=============================================

Extract causal knowledge from CounterBench dataset and convert to RDF format
for SPARQL verification. This enables CAF's Formal Verification Layer to verify
LLM reasoning against the ground-truth causal structure.

CounterBench provides causal relations in natural language (e.g., "Blaf causes Ziklo").
This script:
1. Parses the context to extract causal relations
2. Converts them to RDF triples (subject-predicate-object)
3. Outputs N-Triples format for loading into SPARQL endpoint

Usage:
    # Convert full dataset
    python scripts/convert_counterbench_to_rdf.py \
        data/counterbench_caf.json \
        --output data/counterbench_kb.nt

    # Convert with statistics
    python scripts/convert_counterbench_to_rdf.py \
        data/counterbench_caf.json \
        --output data/counterbench_kb.nt \
        --stats

    # Preview only (don't save)
    python scripts/convert_counterbench_to_rdf.py \
        data/counterbench_caf.json \
        --preview
"""

import json
import argparse
import re
from pathlib import Path
from typing import List, Dict, Set, Tuple, Any, Optional
from collections import defaultdict
from urllib.parse import quote


class CausalRelation:
    """Represents a single causal relation."""

    def __init__(
        self,
        cause: str,
        effect: str,
        relation_type: str = "causes",
        negated: bool = False,
        conjunctive: bool = False
    ):
        """
        Initialize causal relation.

        Args:
            cause: Cause variable(s)
            effect: Effect variable
            relation_type: Type of relation (causes, prevents, etc.)
            negated: Whether effect is negated (causes NOT X)
            conjunctive: Whether multiple causes are required (AND)
        """
        self.cause = cause
        self.effect = effect
        self.relation_type = relation_type
        self.negated = negated
        self.conjunctive = conjunctive

    def to_rdf(self, namespace: str = "http://counterbench.org/") -> List[str]:
        """
        Convert to RDF triples (N-Triples format).

        Args:
            namespace: Base namespace URI

        Returns:
            List of N-Triple strings
        """
        triples = []

        # Clean and encode variable names
        cause_uri = self._to_uri(self.cause, namespace)
        effect_uri = self._to_uri(self.effect, namespace)

        # Basic causal relation
        predicate = f"<{namespace}causes>"
        if self.negated:
            predicate = f"<{namespace}prevents>"

        triple = f"{cause_uri} {predicate} {effect_uri} ."
        triples.append(triple)

        # Add relation metadata
        relation_uri = f"<{namespace}relation/{quote(self.cause)}_{quote(self.effect)}>"

        triples.append(
            f"{relation_uri} <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> "
            f"<{namespace}CausalRelation> ."
        )

        triples.append(
            f"{relation_uri} <{namespace}hasCause> {cause_uri} ."
        )

        triples.append(
            f"{relation_uri} <{namespace}hasEffect> {effect_uri} ."
        )

        if self.conjunctive:
            triples.append(
                f"{relation_uri} <{namespace}isConjunctive> \"true\"^^<http://www.w3.org/2001/XMLSchema#boolean> ."
            )

        return triples

    def _to_uri(self, name: str, namespace: str) -> str:
        """Convert variable name to URI."""
        # Handle conjunctive causes (e.g., "X and Y")
        if " and " in name.lower():
            # For now, create a compound URI
            # In full implementation, could create intermediate node
            clean_name = name.replace(" and ", "_AND_").replace(" ", "_")
        else:
            clean_name = name.strip()

        return f"<{namespace}variable/{quote(clean_name)}>"


class CounterBenchKBExtractor:
    """Extract causal knowledge base from CounterBench dataset."""

    def __init__(self, namespace: str = "http://counterbench.org/"):
        """
        Initialize extractor.

        Args:
            namespace: RDF namespace URI
        """
        self.namespace = namespace
        self.relations: List[CausalRelation] = []
        self.variables: Set[str] = set()

    def parse_context(self, context: str) -> List[CausalRelation]:
        """
        Parse causal relations from context string.

        Handles patterns like:
        - "X causes Y"
        - "X and Y cause Z"
        - "X or Y causes Z"
        - "X causes not Y"
        - "X prevents Y"

        Args:
            context: Natural language causal description

        Returns:
            List of extracted CausalRelation objects
        """
        relations = []

        # Split into sentences
        sentences = re.split(r'[.;]\s*', context)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Pattern 1: "X causes Y"
            match = re.search(
                r'(\w+(?:\s+and\s+\w+)?)\s+causes?\s+(?:not\s+)?(\w+)',
                sentence,
                re.IGNORECASE
            )
            if match:
                cause = match.group(1).strip()
                effect = match.group(2).strip()
                negated = 'not' in sentence.lower()
                conjunctive = 'and' in cause.lower()

                relation = CausalRelation(
                    cause=cause,
                    effect=effect,
                    relation_type="causes",
                    negated=negated,
                    conjunctive=conjunctive
                )
                relations.append(relation)

                # Track variables
                self.variables.add(effect)
                if conjunctive:
                    for var in cause.split(' and '):
                        self.variables.add(var.strip())
                else:
                    self.variables.add(cause)
                continue

            # Pattern 2: "X prevents Y" or "X inhibits Y"
            match = re.search(
                r'(\w+)\s+(?:prevents?|inhibits?)\s+(\w+)',
                sentence,
                re.IGNORECASE
            )
            if match:
                cause = match.group(1).strip()
                effect = match.group(2).strip()

                relation = CausalRelation(
                    cause=cause,
                    effect=effect,
                    relation_type="prevents",
                    negated=True
                )
                relations.append(relation)

                self.variables.add(cause)
                self.variables.add(effect)
                continue

            # Pattern 3: "If X then Y" (implies causation)
            match = re.search(
                r'if\s+(\w+)\s+then\s+(\w+)',
                sentence,
                re.IGNORECASE
            )
            if match:
                cause = match.group(1).strip()
                effect = match.group(2).strip()

                relation = CausalRelation(
                    cause=cause,
                    effect=effect,
                    relation_type="causes"
                )
                relations.append(relation)

                self.variables.add(cause)
                self.variables.add(effect)
                continue

        return relations

    def extract_from_dataset(
        self,
        examples: List[Dict[str, Any]]
    ) -> None:
        """
        Extract all causal relations from dataset.

        Args:
            examples: List of CounterBench examples (CAF format)
        """
        print(f"Extracting causal relations from {len(examples)} examples...")

        for example in examples:
            context = example.get('context', '')
            if not context:
                continue

            relations = self.parse_context(context)
            self.relations.extend(relations)

        # Remove duplicates (same cause-effect pairs)
        seen = set()
        unique_relations = []
        for rel in self.relations:
            key = (rel.cause, rel.effect, rel.negated)
            if key not in seen:
                seen.add(key)
                unique_relations.append(rel)

        self.relations = unique_relations
        print(f"✓ Extracted {len(self.relations)} unique causal relations")
        print(f"✓ Found {len(self.variables)} unique variables")

    def to_rdf(self) -> List[str]:
        """
        Convert all relations to RDF N-Triples.

        Returns:
            List of N-Triple strings
        """
        triples = []

        # Add namespace declarations (as comments, N-Triples doesn't support @prefix)
        triples.append(f"# CounterBench Causal Knowledge Base")
        triples.append(f"# Namespace: {self.namespace}")
        triples.append(f"# Generated: {len(self.relations)} relations")
        triples.append("")

        # Convert each relation
        for relation in self.relations:
            relation_triples = relation.to_rdf(self.namespace)
            triples.extend(relation_triples)
            triples.append("")  # Blank line between relations

        return triples

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about extracted knowledge.

        Returns:
            Statistics dictionary
        """
        stats = {
            'total_relations': len(self.relations),
            'unique_variables': len(self.variables),
            'relation_types': defaultdict(int),
            'negated_count': 0,
            'conjunctive_count': 0
        }

        for rel in self.relations:
            stats['relation_types'][rel.relation_type] += 1
            if rel.negated:
                stats['negated_count'] += 1
            if rel.conjunctive:
                stats['conjunctive_count'] += 1

        return stats

    def save(self, output_path: str) -> None:
        """
        Save RDF triples to file.

        Args:
            output_path: Output file path (N-Triples format)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        triples = self.to_rdf()

        with open(output_path, 'w') as f:
            f.write('\n'.join(triples))

        print(f"✓ Saved {len(self.relations)} relations to {output_path}")

        # Get file size
        size_kb = output_path.stat().st_size / 1024
        print(f"  File size: {size_kb:.1f} KB")


def main():
    parser = argparse.ArgumentParser(
        description="Extract causal relations from CounterBench and convert to RDF"
    )

    parser.add_argument(
        'input',
        help='Input JSON file (CounterBench in CAF format)'
    )

    parser.add_argument(
        '--output',
        default='data/counterbench_kb.nt',
        help='Output N-Triples file'
    )

    parser.add_argument(
        '--namespace',
        default='http://counterbench.org/',
        help='RDF namespace URI'
    )

    parser.add_argument(
        '--stats',
        action='store_true',
        help='Print detailed statistics'
    )

    parser.add_argument(
        '--preview',
        action='store_true',
        help='Preview extraction without saving'
    )

    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset from {args.input}...")
    with open(args.input) as f:
        examples = json.load(f)
    print(f"✓ Loaded {len(examples)} examples")

    # Extract relations
    extractor = CounterBenchKBExtractor(namespace=args.namespace)
    extractor.extract_from_dataset(examples)

    # Print statistics
    if args.stats or args.preview:
        print("\n" + "=" * 70)
        print("KNOWLEDGE BASE STATISTICS")
        print("=" * 70)

        stats = extractor.get_statistics()

        print(f"\nTotal causal relations: {stats['total_relations']}")
        print(f"Unique variables: {stats['unique_variables']}")

        print("\nRelation types:")
        for rel_type, count in stats['relation_types'].items():
            print(f"  {rel_type}: {count}")

        print(f"\nNegated relations (X prevents Y): {stats['negated_count']}")
        print(f"Conjunctive relations (X and Y cause Z): {stats['conjunctive_count']}")

        print("\nSample variables:")
        for var in list(extractor.variables)[:10]:
            print(f"  - {var}")

    # Preview sample triples
    if args.preview:
        print("\n" + "=" * 70)
        print("SAMPLE RDF TRIPLES")
        print("=" * 70)

        sample_triples = extractor.to_rdf()[:20]
        for triple in sample_triples:
            if triple.strip() and not triple.startswith('#'):
                print(triple)

        print("\n... (showing first 20 triples)")
        print(f"\nTotal triples: {len([t for t in extractor.to_rdf() if t.strip() and not t.startswith('#')])}")

        if not args.output:
            return 0

    # Save
    if not args.preview or args.output:
        extractor.save(args.output)

        print("\n" + "=" * 70)
        print("NEXT STEPS")
        print("=" * 70)
        print("\n1. Start Fuseki server:")
        print("   cd ~/apache-jena-fuseki-4.10.0")
        print("   ./fuseki-server --mem /counterbench &")
        print("\n2. Load knowledge base:")
        print(f"   curl -X POST -H 'Content-Type: application/n-triples' \\")
        print(f"        --data-binary @{args.output} \\")
        print(f"        http://localhost:3030/counterbench/data")
        print("\n3. Run CAF with SPARQL verification:")
        print("   python -m experiments.run_counterbench_experiment \\")
        print(f"        --input {args.input} \\")
        print("        --use-llm --llm-4bit \\")
        print("        --use-real-sparql \\")
        print("        --sparql-endpoint http://localhost:3030/counterbench/query \\")
        print("        --output results/counterbench_full")

    return 0


if __name__ == '__main__':
    exit(main())
