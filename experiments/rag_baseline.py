#!/usr/bin/env python3
"""
Pure RAG Baseline for CounterBench
===================================

True RAG approach WITHOUT any verification or iteration:
1. Extract causal facts from context
2. Retrieve relevant facts using keyword similarity
3. Single-shot LLM generation with retrieved context
4. No iteration, no verification, no symbolic reasoning

This is the standard neural RAG approach for comparison with CAF.
"""

import sys
import re
import json
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.llm_integration import InferenceLayer


@dataclass
class CausalFact:
    """A causal fact extracted from text."""
    cause: str
    effect: str

    def __str__(self):
        return f"{self.cause} causes {self.effect}"


class RAGRetriever:
    """Simple keyword-based retrieval for causal facts."""

    def __init__(self):
        self.facts: List[CausalFact] = []

    def extract_facts_from_context(self, context: str) -> List[CausalFact]:
        """Extract causal facts using regex patterns."""
        facts = []

        # Pattern: "X causes Y"
        pattern = r'(\w+)\s+causes?\s+(\w+)'
        for match in re.finditer(pattern, context, re.IGNORECASE):
            cause, effect = match.groups()
            facts.append(CausalFact(cause, effect))

        return facts

    def build_index(self, context: str):
        """Build retrieval index from context."""
        self.facts = self.extract_facts_from_context(context)

    def retrieve(self, query: str, top_k: int = 5) -> List[CausalFact]:
        """
        Retrieve relevant facts using keyword overlap.

        In production RAG, this would use:
        - Sentence embeddings (BERT, etc.)
        - Vector similarity (cosine distance)
        - Vector DB (FAISS, Pinecone, etc.)

        For simplicity, we use keyword matching.
        """
        query_terms = set(re.findall(r'\w+', query.lower()))

        # Score facts by keyword overlap
        scored = []
        for fact in self.facts:
            fact_terms = set(re.findall(r'\w+', str(fact).lower()))
            overlap = len(query_terms & fact_terms)
            if overlap > 0:
                scored.append((overlap, fact))

        # Sort by relevance and return top_k
        scored.sort(reverse=True, key=lambda x: x[0])
        return [fact for _, fact in scored[:top_k]]


@dataclass
class RAGResult:
    """Result from RAG system."""
    query: str
    context: str
    retrieved_facts: List[str]
    llm_response: str
    extracted_answer: str
    expected_answer: str
    correct: bool


class RAGBaseline:
    """Pure RAG system for causal reasoning."""

    def __init__(self, llm: InferenceLayer, strict_binary: bool = True):
        self.llm = llm
        self.strict_binary = strict_binary
        self.retriever = RAGRetriever()
        self.results: List[RAGResult] = []

    def extract_answer(self, response: str) -> str:
        """Extract yes/no answer (same logic as CAF for fairness)."""
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

        response_lower = response.lower()

        # Check uncertainty
        if 'cannot determine' in response_lower or 'uncertain' in response_lower:
            return 'unknown'

        # Check counterfactual patterns
        if 'would not occur' in response_lower or 'would not happen' in response_lower:
            return 'no'
        elif 'would occur' in response_lower or 'would happen' in response_lower:
            return 'yes'

        # Check explicit yes/no
        yes_idx = response_lower.find('yes')
        no_idx = response_lower.find('no')

        if yes_idx != -1 and no_idx == -1:
            return 'yes'
        elif no_idx != -1 and yes_idx == -1:
            return 'no'
        elif yes_idx != -1 and no_idx != -1:
            return 'yes' if yes_idx < no_idx else 'no'
        else:
            return 'unknown'

    def process_query(self, query: str, context: str, expected_answer: str) -> RAGResult:
        """
        Pure RAG pipeline: Retrieve → Generate.

        NO iteration, NO verification, NO symbolic reasoning.
        """
        # 1. Build retrieval index from context
        self.retriever.build_index(context)

        # 2. Retrieve relevant facts
        retrieved = self.retriever.retrieve(query, top_k=5)

        # 3. Construct RAG prompt with retrieved context
        if retrieved:
            facts_str = "\n".join([f"- {fact}" for fact in retrieved])
            prompt = f"""Context: {context}

Relevant causal relationships:
{facts_str}

Question: {query}

Based on the causal relationships above, answer with exactly one word: yes or no."""
        else:
            # Fallback if no facts retrieved
            prompt = f"""Context: {context}

Question: {query}

Answer with exactly one word: yes or no."""

        # 4. Single-shot LLM generation (NO ITERATION)
        llm_response = self.llm.generate(prompt)

        # 5. Extract answer
        extracted = self.extract_answer(llm_response)

        # 6. Create result
        result = RAGResult(
            query=query,
            context=context,
            retrieved_facts=[str(f) for f in retrieved],
            llm_response=llm_response,
            extracted_answer=extracted,
            expected_answer=expected_answer,
            correct=(extracted == expected_answer)
        )

        self.results.append(result)
        return result

    def evaluate(self, examples: List[Dict[str, Any]], limit: int = None):
        """Evaluate RAG on multiple examples."""
        if limit:
            examples = examples[:limit]

        total = len(examples)
        correct = 0

        print(f"\nEvaluating RAG on {total} examples...")
        print("="*70)

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
        print(f"RAG BASELINE RESULTS")
        print(f"{'='*70}")
        print(f"Total examples: {total}")
        print(f"Correct: {correct}")
        print(f"Accuracy: {accuracy*100:.2f}%")
        print()

        # Answer distribution
        answer_dist = {}
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

    def save_results(self, output_dir: Path):
        """Save results to JSON."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed results
        results_file = output_dir / 'results.json'
        with open(results_file, 'w') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)

        # Save summary
        total = len(self.results)
        correct = sum(1 for r in self.results if r.correct)
        accuracy = correct / total if total > 0 else 0.0

        summary_file = output_dir / 'summary.txt'
        with open(summary_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("RAG Baseline - CounterBench Evaluation\n")
            f.write("="*70 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Configuration: Pure RAG (Retrieve + Generate, NO verification)\n\n")
            f.write(f"OVERALL PERFORMANCE\n")
            f.write("-"*70 + "\n")
            f.write(f"Total examples: {total}\n")
            f.write(f"Correct: {correct}\n")
            f.write(f"Accuracy: {accuracy*100:.2f}%\n")
            f.write(f"Avg iterations: 1 (RAG is single-shot)\n\n")

            # Answer distribution
            answer_dist = {}
            for r in self.results:
                ans = r.extracted_answer
                answer_dist[ans] = answer_dist.get(ans, 0) + 1

            f.write("ANSWER DISTRIBUTION\n")
            f.write("-"*70 + "\n")
            for ans, count in sorted(answer_dist.items()):
                f.write(f"{ans:10s} | {count:3d} ({100*count/total:.1f}%)\n")

        print(f"\n✓ Results saved to {output_dir}/")
        print(f"  - {results_file}")
        print(f"  - {summary_file}")


def main():
    """Run RAG baseline evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description='RAG Baseline for CounterBench')
    parser.add_argument('--input', required=True, help='Input JSON file')
    parser.add_argument('--limit', type=int, default=None, help='Limit examples')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--llm-model', default='tiny', choices=['tiny', 'phi2', '7b'])
    parser.add_argument('--llm-4bit', action='store_true', help='Use 4-bit quantization')
    parser.add_argument('--non-strict', action='store_true', help='Disable strict binary mode')

    args = parser.parse_args()

    # Load LLM
    from experiments.llm_integration import create_llama_layer

    print(f"Loading {args.llm_model} model...")
    llm = create_llama_layer(args.llm_model, use_4bit=args.llm_4bit)
    # Match CAF strict settings for fair comparison.
    if hasattr(llm, "config") and not args.non_strict:
        llm.config.max_new_tokens = 8
        llm.config.do_sample = False
        llm.config.temperature = 0.0
        llm.config.top_p = 1.0
    print("✓ LLM loaded")

    # Load data
    with open(args.input) as f:
        data = json.load(f)

    print(f"✓ Loaded {len(data)} examples")

    # Create RAG system
    rag = RAGBaseline(llm, strict_binary=not args.non_strict)

    # Evaluate
    metrics = rag.evaluate(data, limit=args.limit)

    # Save results
    output_dir = Path(args.output)
    rag.save_results(output_dir)

    print(f"\n{'='*70}")
    print(f"RAG Baseline Evaluation Complete!")
    print(f"{'='*70}")
    print(f"Accuracy: {metrics['accuracy']*100:.2f}%")


if __name__ == '__main__':
    main()
