#!/usr/bin/env python3
"""
Populate KB from Text Corpus
=============================
End-to-end pipeline:
1. Extract causal graphs from text
2. Convert to RDF
3. Load into Fuseki

Usage:
    python scripts/populate_kb_from_text.py \
        --input medical_abstracts.txt \
        --domain medical \
        --fuseki http://localhost:3030/caf/data
"""

import argparse
from pathlib import Path
from experiments.causal_discovery import CausalGraphExtractor
from experiments.llm_integration import create_llama_layer
from scripts.causal_graph_to_rdf import CausalGraphToRDF
import requests

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input text file')
    parser.add_argument('--domain', required=True, help='Domain name')
    parser.add_argument(
        '--fuseki',
        default='http://localhost:3030/caf/data',
        help='Fuseki upload endpoint'
    )
    parser.add_argument('--llm-model', default='7b', help='LLM size')
    parser.add_argument('--confidence-threshold', type=float, default=0.6)
    parser.add_argument('--output-rdf', help='Optional: save RDF to file')
    parser.add_argument(
        '--skip-fuseki',
        action='store_true',
        help='Skip Fuseki upload (dry-run extraction + RDF export)'
    )
    
    args = parser.parse_args()
    
    print(f"Loading text from {args.input}...")
    text = Path(args.input).read_text()
    
    print(f"Initializing LLM ({args.llm_model})...")
    llm = create_llama_layer(model_size=args.llm_model, use_4bit=True)
    
    print("Extracting causal graph from text...")
    extractor = CausalGraphExtractor(llm, k_samples=10)
    result = extractor.extract_from_text(text, domain=args.domain)
    
    print(f"Extracted {len(result['variables'])} variables")
    print(f"Extracted {len(result['edges'])} causal edges")
    print(f"Overall confidence: {result['confidence']:.2f}")
    
    print("Converting to RDF...")
    converter = CausalGraphToRDF()
    rdf_graph = converter.convert_causal_graph(
        result['graph'],
        domain=args.domain,
        confidence_threshold=args.confidence_threshold
    )
    
    print(f"Generated {len(rdf_graph)} RDF triples")
    
    if args.output_rdf:
        print(f"Saving RDF to {args.output_rdf}...")
        converter.export_to_ntriples(args.output_rdf)
    
    if args.skip_fuseki:
        print("Skipping Fuseki upload (--skip-fuseki enabled).")
    else:
        print(f"Loading into Fuseki at {args.fuseki}...")
        # Serialize to N-Triples
        rdf_data = rdf_graph.serialize(format='nt')

        # POST to Fuseki
        response = requests.post(
            args.fuseki,
            data=rdf_data,
            headers={'Content-Type': 'application/n-triples'}
        )

        if response.status_code == 200:
            print("✓ Successfully loaded into KB!")
        else:
            print(f"✗ Failed: {response.status_code} - {response.text}")
            return 1
    
    print("\nKB Population Summary:")
    print(f"  Domain: {args.domain}")
    print(f"  Variables: {len(result['variables'])}")
    print(f"  Edges: {len(result['edges'])}")
    print(f"  RDF Triples: {len(rdf_graph)}")
    print(f"  Confidence: {result['confidence']:.2%}")
    
    return 0

if __name__ == "__main__":
    exit(main())
