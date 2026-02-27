#!/usr/bin/env python3
"""
Quick test to see what TinyLlama actually generates for CounterBench queries.
"""
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent))

from experiments.llm_integration import create_llama_layer

# Load TinyLlama
print("Loading TinyLlama...")
llm = create_llama_layer("tiny", use_4bit=True)
print("✓ Model loaded\n")

# Load a CounterBench example
with open("data/counterbench.json") as f:
    data = json.load(f)

example = data[0]
print("=" * 70)
print("EXAMPLE QUERY:")
print("=" * 70)
print(f"Query: {example['query']}")
print(f"Expected Answer: {example['expected_answer']}")
print(f"Context: {example['context']}")
print()

# Generate response
print("=" * 70)
print("LLM RESPONSE:")
print("=" * 70)
# Combine context and query into single prompt
full_prompt = f"{example['context']}\n\n{example['query']}"
response = llm.generate(full_prompt)
print(response)
print()

# Test answer extraction
response_lower = response.lower()
print("=" * 70)
print("ANSWER EXTRACTION DEBUG:")
print("=" * 70)
print(f"Contains 'yes': {'yes' in response_lower}")
print(f"Contains 'no': {'no' in response_lower}")
print(f"Contains 'would occur': {'would occur' in response_lower}")
print(f"Contains 'would not occur': {'would not occur' in response_lower}")
print(f"Contains 'cannot determine': {'cannot determine' in response_lower}")

# Apply extraction logic
if 'yes' in response_lower and 'no' not in response_lower:
    extracted = 'Yes'
elif 'no' in response_lower and 'yes' not in response_lower:
    extracted = 'No'
elif 'cannot determine' in response_lower or 'uncertain' in response_lower:
    extracted = 'Unknown'
elif 'would occur' in response_lower or 'would happen' in response_lower:
    extracted = 'Yes'
elif 'would not occur' in response_lower or 'would not happen' in response_lower:
    extracted = 'No'
else:
    extracted = 'Unknown'

print(f"\nExtracted Answer: {extracted}")
print(f"Expected Answer: {example['expected_answer']}")
print(f"Correct: {extracted == example['expected_answer']}")
