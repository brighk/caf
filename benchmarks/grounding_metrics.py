"""
Step C: Benchmarking Metrics

Defines "Grounding Success" and other metrics for comparing
CAF constraint satisfaction vs standard RAG.
"""
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import re


class FactAlignment(Enum):
    """Classification of generated facts vs KB"""
    CORRECT = "correct"          # Fact exists in KB and is accurate
    HALLUCINATED = "hallucinated" # Fact not in KB
    CONTRADICTS = "contradicts"   # Fact contradicts KB
    PARTIAL = "partial"           # Partially correct


@dataclass
class GroundingResult:
    """Result of grounding evaluation"""
    total_facts: int
    correct_facts: int
    hallucinated_facts: int
    contradicted_facts: int
    partial_facts: int

    @property
    def grounding_success(self) -> float:
        """
        Grounding Success Rate = (Correct Facts) / (Total Facts)

        Key metric for Step C comparison.
        """
        if self.total_facts == 0:
            return 0.0
        return self.correct_facts / self.total_facts

    @property
    def hallucination_rate(self) -> float:
        """Rate of facts not grounded in KB"""
        if self.total_facts == 0:
            return 0.0
        return self.hallucinated_facts / self.total_facts

    @property
    def contradiction_rate(self) -> float:
        """Rate of facts that contradict KB"""
        if self.total_facts == 0:
            return 0.0
        return self.contradicted_facts / self.total_facts

    @property
    def accuracy(self) -> float:
        """
        Overall accuracy = 1 - (hallucinations + contradictions)
        """
        return 1.0 - (self.hallucination_rate + self.contradiction_rate)


class GroundingEvaluator:
    """
    Evaluates how well generated text is grounded in KB.

    Compares generated facts against SPARQL KB to measure:
    - Grounding Success (% facts verified in KB)
    - Hallucination Rate (% facts not in KB)
    - Contradiction Rate (% facts contradicting KB)
    """

    def __init__(self, truth_anchor):
        """
        Args:
            truth_anchor: TruthAnchor instance for KB verification
        """
        self.truth_anchor = truth_anchor

    async def evaluate(
        self,
        generated_facts: List[str],
        ground_truth_facts: List[Dict[str, str]]
    ) -> GroundingResult:
        """
        Evaluate grounding success of generated facts.

        Args:
            generated_facts: List of fact statements from model
            ground_truth_facts: List of KB facts as dicts
                               {'subject': 'rain', 'predicate': 'causes', 'object': 'wet roads'}

        Returns:
            GroundingResult with metrics
        """
        if not generated_facts:
            return GroundingResult(0, 0, 0, 0, 0)

        correct = 0
        hallucinated = 0
        contradicted = 0
        partial = 0

        for fact_text in generated_facts:
            alignment = await self._classify_fact(fact_text, ground_truth_facts)

            if alignment == FactAlignment.CORRECT:
                correct += 1
            elif alignment == FactAlignment.HALLUCINATED:
                hallucinated += 1
            elif alignment == FactAlignment.CONTRADICTS:
                contradicted += 1
            elif alignment == FactAlignment.PARTIAL:
                partial += 1

        return GroundingResult(
            total_facts=len(generated_facts),
            correct_facts=correct,
            hallucinated_facts=hallucinated,
            contradicted_facts=contradicted,
            partial_facts=partial
        )

    async def _classify_fact(
        self,
        fact_text: str,
        ground_truth: List[Dict[str, str]]
    ) -> FactAlignment:
        """
        Classify a single fact against ground truth KB.

        Uses both exact matching and semantic matching.
        """
        fact_text_lower = fact_text.lower()

        # Extract subject-predicate-object from text
        spo = self._extract_spo(fact_text_lower)

        if not spo:
            return FactAlignment.HALLUCINATED

        # Check against ground truth
        for gt_fact in ground_truth:
            if self._matches_fact(spo, gt_fact):
                return FactAlignment.CORRECT
            elif self._contradicts_fact(spo, gt_fact):
                return FactAlignment.CONTRADICTS

        # Check if partially correct (subject and predicate match)
        for gt_fact in ground_truth:
            if (spo.get('subject') in gt_fact.get('subject', '').lower() and
                spo.get('predicate') in gt_fact.get('predicate', '').lower()):
                return FactAlignment.PARTIAL

        # Not found in KB
        return FactAlignment.HALLUCINATED

    def _extract_spo(self, fact_text: str) -> Dict[str, str]:
        """
        Extract subject-predicate-object from natural language fact.

        Examples:
        - "rain causes wet roads" → {subject: rain, predicate: causes, object: wet roads}
        - "sun makes roads dry" → {subject: sun, predicate: makes, object: dry roads}
        """
        # Common causal patterns
        patterns = [
            r'(\w+)\s+(causes?|leads? to|results? in|makes?)\s+(.+)',
            r'(\w+)\s+is\s+(.+)',
            r'(\w+)\s+has\s+(.+)',
            r'(\w+)\s+contains?\s+(.+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, fact_text)
            if match:
                if len(match.groups()) == 3:
                    return {
                        'subject': match.group(1).strip(),
                        'predicate': match.group(2).strip(),
                        'object': match.group(3).strip()
                    }
                elif len(match.groups()) == 2:
                    return {
                        'subject': match.group(1).strip(),
                        'predicate': 'is',
                        'object': match.group(2).strip()
                    }

        return {}

    def _matches_fact(
        self,
        spo: Dict[str, str],
        gt_fact: Dict[str, str]
    ) -> bool:
        """Check if extracted SPO matches ground truth fact"""
        # Normalize for comparison
        subject_match = spo.get('subject', '') in gt_fact.get('subject', '').lower()

        # Normalize predicates
        pred = spo.get('predicate', '')
        gt_pred = gt_fact.get('predicate', '').lower()
        predicate_match = (
            pred in gt_pred or
            (pred in ['causes', 'cause'] and 'cause' in gt_pred) or
            (pred in ['makes', 'make'] and 'cause' in gt_pred)
        )

        # Check object
        obj = spo.get('object', '')
        gt_obj = gt_fact.get('object', '').lower()
        object_match = (
            obj in gt_obj or
            gt_obj in obj or
            self._semantic_match(obj, gt_obj)
        )

        return subject_match and predicate_match and object_match

    def _contradicts_fact(
        self,
        spo: Dict[str, str],
        gt_fact: Dict[str, str]
    ) -> bool:
        """Check if SPO contradicts ground truth"""
        subject_match = spo.get('subject', '') in gt_fact.get('subject', '').lower()
        pred = spo.get('predicate', '')
        gt_pred = gt_fact.get('predicate', '').lower()
        predicate_match = pred in gt_pred or 'cause' in gt_pred

        if not (subject_match and predicate_match):
            return False

        # Same subject and predicate but different object = contradiction
        obj = spo.get('object', '')
        gt_obj = gt_fact.get('object', '').lower()

        # Check for opposite terms
        contradictions = [
            ('wet', 'dry'),
            ('hot', 'cold'),
            ('light', 'dark'),
            ('day', 'night')
        ]

        for term1, term2 in contradictions:
            if ((term1 in obj and term2 in gt_obj) or
                (term2 in obj and term1 in gt_obj)):
                return True

        return False

    def _semantic_match(self, text1: str, text2: str) -> bool:
        """Simple semantic matching"""
        # Check for synonyms
        synonyms = {
            'wet': ['moist', 'damp', 'soaked'],
            'dry': ['arid', 'parched'],
            'road': ['roads', 'street', 'streets']
        }

        for key, values in synonyms.items():
            if ((key in text1 and any(v in text2 for v in values)) or
                (key in text2 and any(v in text1 for v in values))):
                return True

        return False
