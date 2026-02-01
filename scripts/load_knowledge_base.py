#!/usr/bin/env python3
"""
Script to load ConceptNet and Wikidata into the CAF knowledge base.

Usage:
    python scripts/load_knowledge_base.py --conceptnet data/conceptnet.csv --wikidata data/wikidata.nt
"""
import argparse
import asyncio
import sys
from pathlib import Path
from loguru import logger
from rdflib import Graph
import requests

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.semantic_parser import EntityLinker
from utils.config import get_settings


async def load_conceptnet(file_path: str, entity_linker: EntityLinker):
    """
    Load ConceptNet 5.7 CSV into the vector database.

    ConceptNet format: /c/en/entity, relation, /c/en/entity2, metadata...
    """
    logger.info(f"Loading ConceptNet from {file_path}")

    import csv
    entity_count = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')

        for row in reader:
            if len(row) < 3:
                continue

            # Extract subject, predicate, object
            subject = row[0]
            relation = row[1]
            obj = row[2]

            # Add entities to vector database
            if subject.startswith('/c/en/'):
                label = subject.split('/')[-1].replace('_', ' ')
                entity_linker.add_entity(
                    uri=subject,
                    label=label,
                    source="conceptnet"
                )
                entity_count += 1

            if obj.startswith('/c/en/'):
                label = obj.split('/')[-1].replace('_', ' ')
                entity_linker.add_entity(
                    uri=obj,
                    label=label,
                    source="conceptnet"
                )
                entity_count += 1

            if entity_count % 1000 == 0:
                logger.info(f"Loaded {entity_count} entities from ConceptNet")

    logger.info(f"✓ ConceptNet loaded: {entity_count} entities")


def load_wikidata_to_fuseki(file_path: str, fuseki_endpoint: str):
    """
    Load Wikidata N-Triples into Apache Jena Fuseki.

    Wikidata format: <subject> <predicate> <object> .
    """
    logger.info(f"Loading Wikidata from {file_path}")

    # Load RDF graph
    graph = Graph()
    logger.info("Parsing RDF file...")
    graph.parse(file_path, format='nt')

    logger.info(f"Loaded {len(graph)} triples")

    # Upload to Fuseki
    logger.info("Uploading to Fuseki...")

    # Get upload endpoint (replace /query with /data)
    upload_endpoint = fuseki_endpoint.replace('/query', '/data')

    # Serialize to turtle format for upload
    turtle_data = graph.serialize(format='turtle')

    # POST to Fuseki
    response = requests.post(
        upload_endpoint,
        data=turtle_data,
        headers={'Content-Type': 'text/turtle'},
        params={'default': ''}
    )

    if response.status_code in [200, 201, 204]:
        logger.info(f"✓ Wikidata loaded: {len(graph)} triples")
    else:
        logger.error(f"Failed to upload to Fuseki: {response.status_code} {response.text}")
        raise Exception("Fuseki upload failed")


async def main():
    parser = argparse.ArgumentParser(
        description="Load ConceptNet and Wikidata into CAF knowledge base"
    )
    parser.add_argument(
        '--conceptnet',
        type=str,
        help='Path to ConceptNet CSV file'
    )
    parser.add_argument(
        '--wikidata',
        type=str,
        help='Path to Wikidata N-Triples file'
    )
    parser.add_argument(
        '--sample',
        action='store_true',
        help='Use sample data for testing'
    )

    args = parser.parse_args()

    settings = get_settings()

    logger.info("Starting knowledge base loading...")

    # Initialize entity linker
    entity_linker = EntityLinker(
        chromadb_host=settings.chromadb_host,
        chromadb_port=settings.chromadb_port
    )

    # Load ConceptNet
    if args.conceptnet:
        await load_conceptnet(args.conceptnet, entity_linker)
    elif args.sample:
        logger.info("Creating sample ConceptNet data...")
        # Add sample entities
        sample_entities = [
            ('/c/en/dog', 'dog', 'conceptnet'),
            ('/c/en/cat', 'cat', 'conceptnet'),
            ('/c/en/animal', 'animal', 'conceptnet'),
            ('/c/en/water', 'water', 'conceptnet'),
            ('/c/en/fire', 'fire', 'conceptnet'),
        ]
        for uri, label, source in sample_entities:
            entity_linker.add_entity(uri, label, source)
        logger.info("✓ Sample ConceptNet data loaded")

    # Load Wikidata
    if args.wikidata:
        load_wikidata_to_fuseki(args.wikidata, settings.fuseki_endpoint)
    elif args.sample:
        logger.info("Creating sample Wikidata...")
        # Create sample RDF graph
        from rdflib import Namespace, Literal, RDF

        graph = Graph()
        EX = Namespace("http://example.org/")

        graph.add((EX.Dog, RDF.type, EX.Animal))
        graph.add((EX.Cat, RDF.type, EX.Animal))
        graph.add((EX.Water, EX.hasState, Literal("liquid")))

        # Upload to Fuseki
        upload_endpoint = settings.fuseki_endpoint.replace('/query', '/data')
        turtle_data = graph.serialize(format='turtle')

        response = requests.post(
            upload_endpoint,
            data=turtle_data,
            headers={'Content-Type': 'text/turtle'},
            params={'default': ''}
        )

        if response.status_code in [200, 201, 204]:
            logger.info("✓ Sample Wikidata loaded")
        else:
            logger.warning(f"Fuseki upload status: {response.status_code}")

    logger.info("========================================")
    logger.info("Knowledge Base Loading Complete!")
    logger.info("========================================")


if __name__ == "__main__":
    asyncio.run(main())
