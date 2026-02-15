"""
Integration Tests for Real SPARQL FVL
======================================
Tests the complete Real FVL pipeline with a running triplestore.

Requirements:
    - Apache Jena Fuseki running on localhost:3030
    - Dataset loaded at /conceptnet endpoint
    - Test data loaded (see setup instructions below)

Setup:
    # Start test Fuseki instance
    ./fuseki-server --mem /test &

    # Load minimal test data
    python scripts/convert_conceptnet_to_rdf.py \
        conceptnet-assertions-5.7.0.csv \
        --limit 1000 \
        --output test_data.nt

    curl -X POST -H "Content-Type: application/n-triples" \
        --data-binary @test_data.nt \
        http://localhost:3030/test/data

    # Run tests
    pytest tests/test_real_fvl_integration.py -v

Usage:
    # Run all tests
    pytest tests/test_real_fvl_integration.py -v

    # Run specific test
    pytest tests/test_real_fvl_integration.py::test_parsing -v

    # Skip if triplestore not available
    pytest tests/test_real_fvl_integration.py --skip-integration
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Check if dependencies are installed
try:
    from experiments.real_fvl import RealFVL
    from experiments.caf_algorithm import RDFTriplet, VerificationStatus
except ImportError as e:
    pytest.skip(f"Real FVL dependencies not installed: {e}", allow_module_level=True)


# Fixture to check if triplestore is available
@pytest.fixture(scope="module")
def sparql_available():
    """Check if SPARQL endpoint is reachable."""
    import requests
    try:
        response = requests.get("http://localhost:3030/$/ping", timeout=2)
        return response.status_code == 200
    except:
        return False


@pytest.fixture(scope="module")
def fvl(sparql_available):
    """Create RealFVL instance if triplestore is available."""
    if not sparql_available:
        pytest.skip("SPARQL endpoint not available at http://localhost:3030")

    return RealFVL(
        sparql_endpoint="http://localhost:3030/test/query",
        entity_threshold=0.7,
        enable_fuzzy_match=True
    )


class TestRealFVLParsing:
    """Test text parsing and triplet extraction."""

    def test_basic_parsing(self, fvl):
        """Test parsing simple sentences."""
        text = "Dogs are animals. Cats are mammals."
        triplets = fvl.parse(text)

        assert len(triplets) > 0, "Should extract at least one triplet"
        assert all(isinstance(t, RDFTriplet) for t in triplets), "All results should be RDFTriplets"

    def test_causal_parsing(self, fvl):
        """Test parsing causal statements."""
        text = "Water causes erosion. Erosion leads to soil degradation."
        triplets = fvl.parse(text)

        # Check for causal predicates
        predicates = [t.predicate for t in triplets]
        assert any("cause" in p.lower() or "lead" in p.lower() for p in predicates), \
            "Should extract causal predicates"

    def test_complex_sentence_parsing(self, fvl):
        """Test parsing complex sentences with multiple clauses."""
        text = """
        Climate change results in rising temperatures, which causes
        ice caps to melt, leading to sea level rise.
        """
        triplets = fvl.parse(text)

        assert len(triplets) >= 2, "Should extract multiple triplets from complex sentence"

    def test_empty_input(self, fvl):
        """Test handling of empty input."""
        triplets = fvl.parse("")
        assert triplets == [], "Empty input should return empty list"

    def test_non_factual_text(self, fvl):
        """Test parsing questions and imperatives."""
        text = "What causes rain? Please explain the process."
        triplets = fvl.parse(text)

        # May extract zero or few triplets from non-declarative sentences
        assert isinstance(triplets, list), "Should return list even for non-factual text"


class TestRealFVLEntityLinking:
    """Test entity linking to KB URIs."""

    def test_entity_cache(self, fvl):
        """Test that entity cache works."""
        # Clear cache
        fvl.entity_cache = {}
        fvl.reset_stats()

        # First lookup
        uri1 = fvl._link_entity("water")
        assert fvl.stats["cache_misses"] == 1

        # Second lookup (should hit cache)
        uri2 = fvl._link_entity("water")
        assert fvl.stats["cache_hits"] == 1
        assert uri1 == uri2, "Cached result should match"

    def test_entity_normalization(self, fvl):
        """Test that entities are normalized."""
        uri1 = fvl._link_entity("Water")
        uri2 = fvl._link_entity("water")

        # Should be normalized to same entity
        assert uri1 == uri2, "Capitalization should be normalized"


class TestRealFVLVerification:
    """Test SPARQL verification against KB."""

    def test_verification_structure(self, fvl):
        """Test that verification returns correct structure."""
        triplets = [
            RDFTriplet(subject="dog", predicate="is", obj="animal"),
            RDFTriplet(subject="water", predicate="causes", obj="erosion")
        ]

        results = fvl.verify(triplets)

        assert len(results) == len(triplets), "Should return one result per triplet"
        assert all(hasattr(r, 'status') for r in results), "Results should have status attribute"
        assert all(hasattr(r, 'confidence_score') for r in results), "Results should have confidence"

    def test_verification_statuses(self, fvl):
        """Test that verification produces valid statuses."""
        triplets = [
            RDFTriplet(subject="test_entity_1", predicate="test_rel", obj="test_entity_2")
        ]

        results = fvl.verify(triplets)

        valid_statuses = {VerificationStatus.VERIFIED, VerificationStatus.PARTIAL,
                         VerificationStatus.FAILED, VerificationStatus.CONTRADICTION}

        for result in results:
            assert result.status in valid_statuses, f"Invalid status: {result.status}"

    def test_confidence_scores(self, fvl):
        """Test that confidence scores are in valid range."""
        triplets = [
            RDFTriplet(subject="water", predicate="causes", obj="erosion"),
            RDFTriplet(subject="unknown_x", predicate="unknown_rel", obj="unknown_y")
        ]

        results = fvl.verify(triplets)

        for result in results:
            assert 0.0 <= result.confidence_score <= 1.0, \
                f"Confidence score out of range: {result.confidence_score}"


class TestRealFVLMetrics:
    """Test metrics tracking and statistics."""

    def test_stats_tracking(self, fvl):
        """Test that statistics are tracked correctly."""
        fvl.reset_stats()

        # Parse and verify some text
        text = "Water causes erosion. Dogs are animals."
        triplets = fvl.parse(text)
        results = fvl.verify(triplets)

        stats = fvl.get_stats()

        assert stats["queries_executed"] > 0, "Should track query count"
        assert stats["avg_query_time_ms"] >= 0, "Should track average query time"
        assert 0.0 <= stats["cache_hit_rate"] <= 1.0, "Cache hit rate should be in [0, 1]"

    def test_cache_metrics(self, fvl):
        """Test cache-related metrics."""
        fvl.reset_stats()
        fvl.entity_cache = {}

        # Link entities multiple times
        entities = ["water", "air", "fire", "water", "air"]
        for entity in entities:
            fvl._link_entity(entity)

        stats = fvl.get_stats()

        total_lookups = stats["cache_hits"] + stats["cache_misses"]
        assert total_lookups == len(entities), "Should track all lookups"
        assert stats["cache_hits"] >= 2, "Should have cache hits from repeated entities"


class TestRealFVLErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_endpoint(self):
        """Test handling of invalid SPARQL endpoint."""
        fvl_invalid = RealFVL(sparql_endpoint="http://invalid:9999/sparql")

        # Should handle connection error gracefully
        triplets = [RDFTriplet(subject="test", predicate="test", obj="test")]
        results = fvl_invalid.verify(triplets)

        assert len(results) == 1
        assert results[0].status == VerificationStatus.FAILED

    def test_malformed_triplets(self, fvl):
        """Test handling of malformed triplets."""
        triplets = [
            RDFTriplet(subject="", predicate="causes", obj="result"),  # Empty subject
            RDFTriplet(subject="cause", predicate="", obj="result"),   # Empty predicate
            RDFTriplet(subject="cause", predicate="causes", obj=""),   # Empty object
        ]

        # Should handle gracefully without crashing
        results = fvl.verify(triplets)
        assert len(results) == len(triplets)


class TestRealFVLIntegration:
    """Integration tests with CAF pipeline."""

    def test_caf_loop_integration(self, fvl, sparql_available):
        """Test RealFVL integration with CAF loop."""
        if not sparql_available:
            pytest.skip("SPARQL endpoint not available")

        from experiments.caf_algorithm import CAFLoop, CAFConfig, SimulatedInferenceLayer

        # Create CAF loop with Real FVL
        caf = CAFLoop(
            config=CAFConfig(max_iterations=2, verification_threshold=0.8),
            inference_layer=SimulatedInferenceLayer(),
            verification_layer=fvl
        )

        # Execute
        result = caf.execute("What causes rain?")

        assert result is not None
        assert hasattr(result, 'final_response')
        assert hasattr(result, 'final_score')
        assert result.iterations_used >= 1

    def test_end_to_end_pipeline(self, fvl):
        """Test complete parse -> verify pipeline."""
        # Real-world example
        text = """
        Deforestation reduces tree coverage. Reduced tree coverage
        leads to increased carbon dioxide levels. Increased CO2
        contributes to climate change.
        """

        # Parse
        triplets = fvl.parse(text)
        assert len(triplets) > 0, "Should extract triplets from causal chain"

        # Verify
        results = fvl.verify(triplets)
        assert len(results) == len(triplets), "Should verify all triplets"

        # Check statistics
        stats = fvl.get_stats()
        assert stats["queries_executed"] >= len(triplets), \
            "Should execute at least one query per triplet"


@pytest.mark.benchmark
class TestRealFVLPerformance:
    """Performance benchmarks (optional)."""

    def test_parsing_performance(self, fvl, benchmark):
        """Benchmark parsing performance."""
        text = "Water causes erosion. " * 10

        result = benchmark(fvl.parse, text)
        assert len(result) > 0

    def test_verification_performance(self, fvl, benchmark):
        """Benchmark verification performance."""
        triplets = [
            RDFTriplet(subject=f"entity_{i}", predicate="causes", obj=f"entity_{i+1}")
            for i in range(5)
        ]

        result = benchmark(fvl.verify, triplets)
        assert len(result) == len(triplets)


def test_sparql_connection():
    """Standalone test to check SPARQL connectivity."""
    import requests

    try:
        response = requests.get("http://localhost:3030/$/ping", timeout=2)
        assert response.status_code == 200, "SPARQL endpoint should respond to ping"
        print("✓ SPARQL endpoint is reachable")
    except Exception as e:
        pytest.skip(f"SPARQL endpoint not reachable: {e}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
