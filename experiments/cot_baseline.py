#!/usr/bin/env python3
"""
Chain-of-Thought Baseline for CounterBench
==========================================

Single-shot CoT baseline WITHOUT verification or iteration:
1. Provide context + question
2. Prompt with step-by-step reasoning instruction
3. Generate one answer (strict yes/no mode by default)

This is a reasoning-prompt baseline to compare against RAG and CAF.
"""

import sys
import re
import json
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.llm_integration import InferenceLayer, create_llama_layer


@dataclass
class CoTResult:
    """Result from CoT baseline."""
    query: str
    context: str
    prompt: str
    llm_response: str
    extracted_answer: str
    expected_answer: str
    correct: bool


class CoTBaseline:
    """Single-shot Chain-of-Thought baseline."""

    def __init__(self, llm: InferenceLayer, strict_binary: bool = True):
        self.llm = llm
        self.strict_binary = strict_binary
        self.results: List[CoTResult] = []

    def extract_answer(self, response: str) -> str:
        """Extract yes/no/unknown answer."""
        if not response:
            return "unknown"

        if self.strict_binary:
            first_line = ""
            for line in response.splitlines():
                line = line.strip()
                if line:
                    first_line = line.lower()
                    break
            if first_line:
                match = re.search(r"\b(yes|no)\b", first_line)
                if match:
                    return match.group(1)

        text = response.lower()
        if 'cannot determine' in text or 'uncertain' in text:
            return 'unknown'
        if 'would not occur' in text or 'would not happen' in text:
            return 'no'
        if 'would occur' in text or 'would happen' in text:
            return 'yes'

        yes_idx = text.find('yes')
        no_idx = text.find('no')
        if yes_idx != -1 and no_idx == -1:
            return 'yes'
        if no_idx != -1 and yes_idx == -1:
            return 'no'
        if yes_idx != -1 and no_idx != -1:
            return 'yes' if yes_idx < no_idx else 'no'
        return 'unknown'

    def build_prompt(self, query: str, context: str) -> str:
        """Build CoT prompt with explicit binary output requirement."""
        return f"""Context: {context}

Question: {query}

Think step by step about the causal chain and intervention effect.
Then answer with exactly one word: yes or no.
"""

    def process_query(self, query: str, context: str, expected_answer: str) -> CoTResult:
        """Run single-shot CoT generation."""
        prompt = self.build_prompt(query, context)
        llm_response = self.llm.generate(prompt)
        extracted = self.extract_answer(llm_response)

        result = CoTResult(
            query=query,
            context=context,
            prompt=prompt,
            llm_response=llm_response,
            extracted_answer=extracted,
            expected_answer=expected_answer,
            correct=(extracted == expected_answer)
        )
        self.results.append(result)
        return result

    def evaluate(self, examples: List[Dict[str, Any]], limit: int = None) -> Dict[str, Any]:
        """Evaluate CoT on multiple examples."""
        if limit:
            examples = examples[:limit]

        total = len(examples)
        correct = 0

        print(f"\nEvaluating CoT on {total} examples...")
        print("=" * 70)

        for i, example in enumerate(examples, 1):
            result = self.process_query(
                query=example['query'],
                context=example['context'],
                expected_answer=example['expected_answer']
            )
            if result.correct:
                correct += 1
            print(f"Progress: {i}/{total} ({100*i/total:.0f}%)", end='\r')

        print()
        accuracy = correct / total if total > 0 else 0.0

        print(f"\n{'='*70}")
        print("COT BASELINE RESULTS")
        print(f"{'='*70}")
        print(f"Total examples: {total}")
        print(f"Correct: {correct}")
        print(f"Accuracy: {accuracy*100:.2f}%")
        print()

        answer_dist: Dict[str, int] = {}
        for r in self.results:
            ans = r.extracted_answer
            answer_dist[ans] = answer_dist.get(ans, 0) + 1

        print("Answer Distribution:")
        for ans, count in sorted(answer_dist.items()):
            print(f"  {ans:10s} | {count:3d} ({100*count/total:.1f}%)")

        return {
            'accuracy': accuracy,
            'total': total,
            'correct': correct,
            'answer_distribution': answer_dist
        }

    def save_results(self, output_dir: Path) -> None:
        """Save detailed and summary outputs."""
        output_dir.mkdir(parents=True, exist_ok=True)

        results_file = output_dir / 'results.json'
        with open(results_file, 'w') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)

        total = len(self.results)
        correct = sum(1 for r in self.results if r.correct)
        accuracy = correct / total if total > 0 else 0.0

        summary_file = output_dir / 'summary.txt'
        with open(summary_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("CoT Baseline - CounterBench Evaluation\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("Configuration: Chain-of-Thought (single-shot, NO verification)\n\n")
            f.write("OVERALL PERFORMANCE\n")
            f.write("-" * 70 + "\n")
            f.write(f"Total examples: {total}\n")
            f.write(f"Correct: {correct}\n")
            f.write(f"Accuracy: {accuracy*100:.2f}%\n")
            f.write("Avg iterations: 1 (CoT is single-shot)\n\n")

            answer_dist: Dict[str, int] = {}
            for r in self.results:
                ans = r.extracted_answer
                answer_dist[ans] = answer_dist.get(ans, 0) + 1

            f.write("ANSWER DISTRIBUTION\n")
            f.write("-" * 70 + "\n")
            for ans, count in sorted(answer_dist.items()):
                f.write(f"{ans:10s} | {count:3d} ({100*count/total:.1f}%)\n")

        print(f"\n✓ Results saved to {output_dir}/")
        print(f"  - {results_file}")
        print(f"  - {summary_file}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description='CoT Baseline for CounterBench')
    parser.add_argument('--input', required=True, help='Input JSON file')
    parser.add_argument('--limit', type=int, default=None, help='Limit examples')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--llm-model', default='tiny', choices=['tiny', 'phi2', '7b'])
    parser.add_argument('--llm-4bit', action='store_true', help='Use 4-bit quantization')
    parser.add_argument('--non-strict', action='store_true', help='Disable strict binary mode')
    args = parser.parse_args()

    print(f"Loading {args.llm_model} model...")
    llm = create_llama_layer(args.llm_model, use_4bit=args.llm_4bit)
    if hasattr(llm, "config") and not args.non_strict:
        llm.config.max_new_tokens = 8
        llm.config.do_sample = False
        llm.config.temperature = 0.0
        llm.config.top_p = 1.0
    print("✓ LLM loaded")

    with open(args.input) as f:
        data = json.load(f)
    print(f"✓ Loaded {len(data)} examples")

    cot = CoTBaseline(llm, strict_binary=not args.non_strict)
    metrics = cot.evaluate(data, limit=args.limit)

    output_dir = Path(args.output)
    cot.save_results(output_dir)

    print(f"\n{'='*70}")
    print("CoT Baseline Evaluation Complete!")
    print(f"{'='*70}")
    print(f"Accuracy: {metrics['accuracy']*100:.2f}%")


if __name__ == '__main__':
    main()
