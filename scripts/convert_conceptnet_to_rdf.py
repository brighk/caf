#!/usr/bin/env python3
"""
ConceptNet to RDF Converter
============================
Converts ConceptNet CSV assertions to RDF N-Triples format for loading
into Apache Jena Fuseki or GraphDB.

Usage:
    python convert_conceptnet_to_rdf.py conceptnet-assertions-5.7.0.csv > conceptnet.nt
    python convert_conceptnet_to_rdf.py conceptnet-assertions-5.7.0.csv --filter-english --limit 100000 > conceptnet_en_100k.nt

ConceptNet Format:
    Tab-separated: uri, relation, start, end, metadata
    Example: /a/[...], /r/RelatedTo, /c/en/dog, /c/en/animal, {...}

RDF Output:
    <http://conceptnet.io/c/en/dog> <http://conceptnet.io/r/RelatedTo> <http://conceptnet.io/c/en/animal> .
"""

import csv
import sys
import json
import argparse
from typing import Optional, Dict, Set
from pathlib import Path
import gzip


# ConceptNet relation mappings to standardized predicates
RELATION_MAPPING = {
    "/r/RelatedTo": "RelatedTo",
    "/r/IsA": "IsA",
    "/r/PartOf": "PartOf",
    "/r/HasA": "HasA",
    "/r/UsedFor": "UsedFor",
    "/r/CapableOf": "CapableOf",
    "/r/AtLocation": "AtLocation",
    "/r/Causes": "Causes",
    "/r/HasSubevent": "HasSubevent",
    "/r/HasFirstSubevent": "HasFirstSubevent",
    "/r/HasLastSubevent": "HasLastSubevent",
    "/r/HasPrerequisite": "HasPrerequisite",
    "/r/HasProperty": "HasProperty",
    "/r/MotivatedByGoal": "MotivatedByGoal",
    "/r/ObstructedBy": "ObstructedBy",
    "/r/Desires": "Desires",
    "/r/CreatedBy": "CreatedBy",
    "/r/Synonym": "Synonym",
    "/r/Antonym": "Antonym",
    "/r/DistinctFrom": "DistinctFrom",
    "/r/DerivedFrom": "DerivedFrom",
    "/r/SymbolOf": "SymbolOf",
    "/r/DefinedAs": "DefinedAs",
    "/r/MannerOf": "MannerOf",
    "/r/LocatedNear": "LocatedNear",
    "/r/HasContext": "HasContext",
    "/r/SimilarTo": "SimilarTo",
    "/r/EtymologicallyRelatedTo": "EtymologicallyRelatedTo",
    "/r/EtymologicallyDerivedFrom": "EtymologicallyDerivedFrom",
    "/r/CausesDesire": "CausesDesire",
    "/r/MadeOf": "MadeOf",
    "/r/ReceivesAction": "ReceivesAction",
    "/r/InstanceOf": "InstanceOf",
}


def extract_concept(uri: str) -> Optional[str]:
    """
    Extract concept name from ConceptNet URI.

    Args:
        uri: ConceptNet URI like /c/en/dog or /c/en/hot_dog/n/wn/food

    Returns:
        Concept name or None if invalid
    """
    if not uri.startswith("/c/"):
        return None

    parts = uri.split("/")
    if len(parts) < 4:
        return None

    # Extract language and concept
    lang = parts[2]
    concept = parts[3]

    # Optionally include POS and domain for disambiguation
    # For now, just use the base concept
    return concept


def extract_language(uri: str) -> Optional[str]:
    """Extract language code from ConceptNet URI."""
    if not uri.startswith("/c/"):
        return None

    parts = uri.split("/")
    if len(parts) < 3:
        return None

    return parts[2]


def extract_relation(rel_uri: str) -> str:
    """
    Extract and normalize relation from ConceptNet URI.

    Args:
        rel_uri: Relation URI like /r/IsA

    Returns:
        Standardized relation name
    """
    return RELATION_MAPPING.get(rel_uri, rel_uri.split("/")[-1])


def create_uri(concept: str, namespace: str = "http://conceptnet.io/c/en/") -> str:
    """
    Create RDF URI from concept name.

    Args:
        concept: Concept name
        namespace: Base namespace URI

    Returns:
        Full URI
    """
    # Normalize concept (replace spaces with underscores)
    normalized = concept.replace(" ", "_").replace("/", "_")
    return f"{namespace}{normalized}"


def create_relation_uri(relation: str) -> str:
    """Create RDF URI for relation."""
    return f"http://conceptnet.io/r/{relation}"


def parse_weight(metadata_json: str) -> float:
    """
    Extract edge weight from ConceptNet metadata JSON.

    Args:
        metadata_json: JSON string with edge metadata

    Returns:
        Weight value (default 1.0)
    """
    try:
        metadata = json.loads(metadata_json)
        return metadata.get("weight", 1.0)
    except (json.JSONDecodeError, KeyError):
        return 1.0


def convert_to_ntriple(
    subject_uri: str,
    predicate_uri: str,
    object_uri: str,
    weight: Optional[float] = None
) -> str:
    """
    Create N-Triple RDF statement.

    Args:
        subject_uri: Subject URI
        predicate_uri: Predicate URI
        object_uri: Object URI
        weight: Optional weight (added as comment)

    Returns:
        N-Triple string
    """
    triple = f"<{subject_uri}> <{predicate_uri}> <{object_uri}> ."

    if weight is not None:
        triple += f" # weight: {weight:.3f}"

    return triple


def convert_conceptnet_to_rdf(
    input_file: str,
    output_file: Optional[str] = None,
    filter_language: str = "en",
    include_weights: bool = False,
    limit: Optional[int] = None,
    relations_filter: Optional[Set[str]] = None,
    min_weight: float = 0.0,
    verbose: bool = False
) -> Dict[str, int]:
    """
    Convert ConceptNet CSV to RDF N-Triples.

    Args:
        input_file: Path to ConceptNet CSV file
        output_file: Output file (stdout if None)
        filter_language: Only include this language (None for all)
        include_weights: Include weights as comments
        limit: Maximum number of triples to output
        relations_filter: Set of relations to include (None for all)
        min_weight: Minimum edge weight to include
        verbose: Print progress messages

    Returns:
        Statistics dict
    """
    stats = {
        "total_rows": 0,
        "filtered_language": 0,
        "filtered_relation": 0,
        "filtered_weight": 0,
        "output_triples": 0,
        "invalid_rows": 0
    }

    # Determine if input is gzipped
    is_gzipped = input_file.endswith('.gz')

    # Open output file
    if output_file:
        out_f = open(output_file, 'w', encoding='utf-8')
    else:
        out_f = sys.stdout

    try:
        # Open input file
        if is_gzipped:
            f = gzip.open(input_file, 'rt', encoding='utf-8')
        else:
            f = open(input_file, 'r', encoding='utf-8')

        reader = csv.reader(f, delimiter='\t')

        for row_num, row in enumerate(reader, 1):
            stats["total_rows"] += 1

            # Progress reporting
            if verbose and stats["total_rows"] % 100000 == 0:
                print(f"Processed {stats['total_rows']:,} rows, output {stats['output_triples']:,} triples",
                      file=sys.stderr)

            # Check limit
            if limit and stats["output_triples"] >= limit:
                if verbose:
                    print(f"Reached limit of {limit:,} triples", file=sys.stderr)
                break

            # Parse row
            if len(row) < 4:
                stats["invalid_rows"] += 1
                continue

            rel_uri = row[1]
            start_uri = row[2]
            end_uri = row[3]
            metadata = row[4] if len(row) > 4 else "{}"

            # Filter by language
            if filter_language:
                start_lang = extract_language(start_uri)
                end_lang = extract_language(end_uri)

                if start_lang != filter_language or end_lang != filter_language:
                    stats["filtered_language"] += 1
                    continue

            # Filter by relation
            if relations_filter:
                relation = extract_relation(rel_uri)
                if relation not in relations_filter:
                    stats["filtered_relation"] += 1
                    continue

            # Filter by weight
            if min_weight > 0:
                weight = parse_weight(metadata)
                if weight < min_weight:
                    stats["filtered_weight"] += 1
                    continue
            else:
                weight = None

            # Extract concepts
            start_concept = extract_concept(start_uri)
            end_concept = extract_concept(end_uri)

            if not start_concept or not end_concept:
                stats["invalid_rows"] += 1
                continue

            # Create RDF URIs
            subject_uri = create_uri(start_concept)
            object_uri = create_uri(end_concept)
            predicate_uri = create_relation_uri(extract_relation(rel_uri))

            # Generate N-Triple
            triple = convert_to_ntriple(
                subject_uri,
                predicate_uri,
                object_uri,
                weight if include_weights else None
            )

            # Output
            print(triple, file=out_f)
            stats["output_triples"] += 1

        f.close()

    finally:
        if output_file:
            out_f.close()

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert ConceptNet CSV to RDF N-Triples format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "input_file",
        help="Input ConceptNet CSV file (can be .gz)"
    )

    parser.add_argument(
        "-o", "--output",
        help="Output file (default: stdout)"
    )

    parser.add_argument(
        "-l", "--language",
        default="en",
        help="Filter by language code (e.g., 'en'). Use 'all' for no filter."
    )

    parser.add_argument(
        "-n", "--limit",
        type=int,
        help="Maximum number of triples to output"
    )

    parser.add_argument(
        "-w", "--include-weights",
        action="store_true",
        help="Include edge weights as comments"
    )

    parser.add_argument(
        "-m", "--min-weight",
        type=float,
        default=0.0,
        help="Minimum edge weight to include"
    )

    parser.add_argument(
        "-r", "--relations",
        help="Comma-separated list of relations to include (e.g., 'IsA,PartOf,Causes')"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print progress messages to stderr"
    )

    args = parser.parse_args()

    # Validate input file
    if not Path(args.input_file).exists():
        print(f"Error: Input file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)

    # Parse relations filter
    relations_filter = None
    if args.relations:
        relations_filter = set(args.relations.split(","))

    # Parse language filter
    language_filter = args.language if args.language != "all" else None

    if args.verbose:
        print(f"Converting ConceptNet to RDF...", file=sys.stderr)
        print(f"  Input: {args.input_file}", file=sys.stderr)
        print(f"  Output: {args.output or 'stdout'}", file=sys.stderr)
        print(f"  Language filter: {language_filter or 'none'}", file=sys.stderr)
        print(f"  Limit: {args.limit or 'none'}", file=sys.stderr)
        print(f"  Min weight: {args.min_weight}", file=sys.stderr)
        if relations_filter:
            print(f"  Relations: {', '.join(relations_filter)}", file=sys.stderr)
        print("", file=sys.stderr)

    # Convert
    stats = convert_conceptnet_to_rdf(
        input_file=args.input_file,
        output_file=args.output,
        filter_language=language_filter,
        include_weights=args.include_weights,
        limit=args.limit,
        relations_filter=relations_filter,
        min_weight=args.min_weight,
        verbose=args.verbose
    )

    # Print statistics
    if args.verbose:
        print("\nConversion complete!", file=sys.stderr)
        print(f"  Total rows processed: {stats['total_rows']:,}", file=sys.stderr)
        print(f"  Output triples: {stats['output_triples']:,}", file=sys.stderr)
        print(f"  Filtered (language): {stats['filtered_language']:,}", file=sys.stderr)
        print(f"  Filtered (relation): {stats['filtered_relation']:,}", file=sys.stderr)
        print(f"  Filtered (weight): {stats['filtered_weight']:,}", file=sys.stderr)
        print(f"  Invalid rows: {stats['invalid_rows']:,}", file=sys.stderr)


if __name__ == "__main__":
    main()
