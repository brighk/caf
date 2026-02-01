"""
Step C: Test Dataset for Benchmarking

Ground truth dataset for comparing CAF vs RAG.
"""
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class TestCase:
    """Single test case with prompt and ground truth"""
    id: str
    prompt: str
    ground_truth_facts: List[Dict[str, str]]
    expected_answer_contains: List[str]
    description: str


# Test dataset based on the KB we loaded
TEST_DATASET = [
    TestCase(
        id="causal_01",
        prompt="What does rain cause on roads?",
        ground_truth_facts=[
            {
                'subject': 'rain',
                'predicate': 'causes',
                'object': 'wet roads',
                'uri_subject': 'http://example.org/Rain',
                'uri_predicate': 'http://example.org/causes',
                'uri_object': 'http://example.org/WetRoads'
            }
        ],
        expected_answer_contains=['wet', 'roads'],
        description="Basic causal relationship - rain to wet roads"
    ),

    TestCase(
        id="causal_02",
        prompt="What effect does the sun have on roads?",
        ground_truth_facts=[
            {
                'subject': 'sun',
                'predicate': 'causes',
                'object': 'dry roads',
                'uri_subject': 'http://example.org/Sun',
                'uri_predicate': 'http://example.org/causes',
                'uri_object': 'http://example.org/DryRoads'
            }
        ],
        expected_answer_contains=['dry', 'roads'],
        description="Causal relationship - sun to dry roads"
    ),

    TestCase(
        id="composition_01",
        prompt="What does rain contain?",
        ground_truth_facts=[
            {
                'subject': 'rain',
                'predicate': 'contains',
                'object': 'water',
                'uri_subject': 'http://example.org/Rain',
                'uri_predicate': 'http://example.org/contains',
                'uri_object': 'http://example.org/Water'
            }
        ],
        expected_answer_contains=['water'],
        description="Compositional relationship - rain contains water"
    ),

    TestCase(
        id="type_01",
        prompt="What type of phenomenon is rain?",
        ground_truth_facts=[
            {
                'subject': 'rain',
                'predicate': 'type',
                'object': 'weather phenomenon',
                'uri_subject': 'http://example.org/Rain',
                'uri_predicate': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type',
                'uri_object': 'http://example.org/WeatherPhenomenon'
            }
        ],
        expected_answer_contains=['weather', 'phenomenon'],
        description="Type classification"
    ),

    TestCase(
        id="contradiction_test_01",
        prompt="Does rain cause dry roads?",
        ground_truth_facts=[
            {
                'subject': 'rain',
                'predicate': 'causes',
                'object': 'wet roads',
                'uri_subject': 'http://example.org/Rain',
                'uri_predicate': 'http://example.org/causes',
                'uri_object': 'http://example.org/WetRoads'
            }
        ],
        expected_answer_contains=['no', 'wet', 'not dry'],
        description="Test contradiction detection - should say NO"
    ),

    TestCase(
        id="multi_hop_01",
        prompt="If it rains, what substance makes the roads wet?",
        ground_truth_facts=[
            {
                'subject': 'rain',
                'predicate': 'contains',
                'object': 'water',
                'uri_subject': 'http://example.org/Rain',
                'uri_predicate': 'http://example.org/contains',
                'uri_object': 'http://example.org/Water'
            },
            {
                'subject': 'rain',
                'predicate': 'causes',
                'object': 'wet roads',
                'uri_subject': 'http://example.org/Rain',
                'uri_predicate': 'http://example.org/causes',
                'uri_object': 'http://example.org/WetRoads'
            }
        ],
        expected_answer_contains=['water'],
        description="Multi-hop reasoning - rain contains water, rain causes wet roads"
    ),
]


def get_test_dataset() -> List[TestCase]:
    """Get the full test dataset"""
    return TEST_DATASET


def get_ground_truth_all() -> List[Dict[str, str]]:
    """Get all ground truth facts (KB knowledge)"""
    all_facts = []
    for test_case in TEST_DATASET:
        all_facts.extend(test_case.ground_truth_facts)

    # Deduplicate
    seen = set()
    unique_facts = []
    for fact in all_facts:
        key = (fact['subject'], fact['predicate'], fact['object'])
        if key not in seen:
            seen.add(key)
            unique_facts.append(fact)

    return unique_facts
