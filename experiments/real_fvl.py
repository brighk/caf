"""
Real FVL Implementation with SPARQL/RDF
========================================
Production-oriented Formal Verification Layer for CAF experiments.
"""

from typing import List, Optional, Any
from difflib import SequenceMatcher

from SPARQLWrapper import SPARQLWrapper, JSON
from SPARQLWrapper.SPARQLExceptions import QueryBadFormed
import spacy

from experiments.caf_algorithm import (
    FormalVerificationLayer,
    RDFTriplet,
    VerificationResult,
    VerificationStatus,
)


class RealFVL(FormalVerificationLayer):
    """
    FVL that executes SPARQL queries against a real triplestore.
    """

    def __init__(
        self,
        sparql_endpoint: str = "http://localhost:3030/dataset/query",
        entity_threshold: float = 0.7,
        enable_fuzzy_match: bool = True,
        spacy_model: str = "en_core_web_sm",
    ):
        self.sparql = SPARQLWrapper(sparql_endpoint)
        self.sparql.setReturnFormat(JSON)

        self.entity_threshold = entity_threshold
        self.enable_fuzzy_match = enable_fuzzy_match
        self.nlp = spacy.load(spacy_model)
        self.entity_cache = {}

    def parse(self, response: str) -> List[RDFTriplet]:
        """
        Extract RDF triplets from response text via basic SVO patterns.
        """
        doc = self.nlp(response)
        triplets = []

        for sent in doc.sents:
            subjects = [token for token in sent if token.dep_ == "nsubj"]
            for subj in subjects:
                verb = subj.head
                objects = [child for child in verb.children if child.dep_ in ("dobj", "pobj", "attr")]
                for obj in objects:
                    triplets.append(
                        RDFTriplet(
                            subject=self._extract_entity_text(subj),
                            predicate=verb.lemma_,
                            obj=self._extract_entity_text(obj),
                            confidence=0.8,
                        )
                    )

        return triplets

    def _extract_entity_text(self, token) -> str:
        for chunk in token.doc.noun_chunks:
            if token in chunk:
                return chunk.text
        return token.text

    def verify(
        self,
        triplets: List[RDFTriplet],
        knowledge_base: Any = None,
        accuracy_hint: Optional[float] = None,
    ) -> List[VerificationResult]:
        """
        Verify triplets through SPARQL ASK queries.
        """
        results = []

        for triplet in triplets:
            subj_uri = self._link_entity(triplet.subject)
            obj_uri = self._link_entity(triplet.obj)

            if not subj_uri or not obj_uri:
                results.append(
                    VerificationResult(
                        triplet=triplet,
                        status=VerificationStatus.FAILED,
                        kb_support=False,
                        contradiction_found=False,
                        supporting_facts=[],
                        contradicting_facts=[],
                        confidence_score=0.0,
                    )
                )
                continue

            query = self._build_ask_query(subj_uri, triplet.predicate, obj_uri)
            status = self._execute_verification(query)
            is_verified = status == VerificationStatus.VERIFIED
            is_contradiction = status == VerificationStatus.CONTRADICTION

            results.append(
                VerificationResult(
                    triplet=triplet,
                    status=status,
                    kb_support=is_verified,
                    contradiction_found=is_contradiction,
                    supporting_facts=[query] if is_verified else [],
                    contradicting_facts=[query] if is_contradiction else [],
                    confidence_score=1.0 if is_verified else 0.0,
                )
            )

        return results

    def _link_entity(self, entity_text: str) -> Optional[str]:
        if entity_text in self.entity_cache:
            return self.entity_cache[entity_text]

        safe_entity_text = self._escape_sparql_string(entity_text)
        exact_query = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?uri WHERE {{
            ?uri rdfs:label "{safe_entity_text}"@en .
        }} LIMIT 1
        """
        try:
            self.sparql.setQuery(exact_query)
            results = self.sparql.query().convert()
        except QueryBadFormed:
            return None

        if results["results"]["bindings"]:
            uri = results["results"]["bindings"][0]["uri"]["value"]
            self.entity_cache[entity_text] = uri
            return uri

        if self.enable_fuzzy_match:
            uri = self._fuzzy_entity_search(entity_text)
            if uri:
                self.entity_cache[entity_text] = uri
                return uri

        return None

    def _fuzzy_entity_search(self, entity_text: str) -> Optional[str]:
        first_token = entity_text.split()[0] if entity_text.split() else entity_text
        safe_first_token = self._escape_sparql_string(first_token)
        fuzzy_query = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?uri ?label WHERE {{
            ?uri rdfs:label ?label .
            FILTER(LANG(?label) = "en")
            FILTER(CONTAINS(LCASE(?label), LCASE("{safe_first_token}")))
        }} LIMIT 10
        """

        try:
            self.sparql.setQuery(fuzzy_query)
            results = self.sparql.query().convert()
        except QueryBadFormed:
            return None

        best_match = None
        best_score = 0.0
        for binding in results["results"]["bindings"]:
            label = binding["label"]["value"]
            score = SequenceMatcher(None, entity_text.lower(), label.lower()).ratio()
            if score > best_score and score >= self.entity_threshold:
                best_score = score
                best_match = binding["uri"]["value"]

        return best_match

    def _build_ask_query(self, subject_uri: str, predicate: str, object_uri: str) -> str:
        safe_predicate = self._escape_sparql_string(predicate)
        return f"""
        ASK {{
            <{subject_uri}> ?p <{object_uri}> .
            FILTER(CONTAINS(LCASE(STR(?p)), LCASE("{safe_predicate}")))
        }}
        """

    def _escape_sparql_string(self, value: str) -> str:
        """Escape string literal content for SPARQL queries."""
        return value.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", " ").replace("\r", " ")

    def _execute_verification(self, query: str) -> VerificationStatus:
        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()
            if results.get("boolean", False):
                return VerificationStatus.VERIFIED
            return VerificationStatus.FAILED
        except Exception:
            return VerificationStatus.FAILED
