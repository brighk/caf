"""
Convert Causal Graphs to RDF for KB Population
===============================================
Takes extracted causal graphs and converts them to RDF triples
for loading into Fuseki.
"""

from rdflib import Graph, URIRef, Literal, Namespace, RDF, RDFS
from typing import List, Dict
import networkx as nx

# Define CAF ontology namespace
CAF = Namespace("http://caf.ai/ontology/")
CAUSES = Namespace("http://caf.ai/relations/")

class CausalGraphToRDF:
    """Convert extracted causal graphs to RDF."""
    
    def __init__(self):
        self.graph = Graph()
        self.graph.bind("caf", CAF)
        self.graph.bind("causes", CAUSES)
        self.graph.bind("rdf", RDF)
        self.graph.bind("rdfs", RDFS)
    
    def convert_causal_graph(
        self, 
        causal_graph: nx.DiGraph,
        domain: str,
        confidence_threshold: float = 0.6
    ) -> Graph:
        """
        Convert NetworkX causal graph to RDF.
        
        Args:
            causal_graph: Extracted causal graph
            domain: Domain name (medical, economic, etc.)
            confidence_threshold: Min confidence for including edges
            
        Returns:
            RDF Graph ready for Fuseki
        """
        
        # Add domain
        domain_uri = CAF[domain]
        self.graph.add((domain_uri, RDF.type, CAF.Domain))
        self.graph.add((domain_uri, RDFS.label, Literal(domain)))
        
        # Add variables as entities
        for node in causal_graph.nodes():
            node_data = causal_graph.nodes[node]
            node_uri = CAF[self._normalize_uri(node)]
            
            self.graph.add((node_uri, RDF.type, CAF.CausalVariable))
            self.graph.add((node_uri, RDFS.label, Literal(node)))
            self.graph.add((node_uri, CAF.domain, domain_uri))
            
            if 'confidence' in node_data:
                self.graph.add((
                    node_uri, 
                    CAF.confidence, 
                    Literal(node_data['confidence'])
                ))
        
        # Add causal edges
        for source, target, edge_data in causal_graph.edges(data=True):
            confidence = edge_data.get('confidence', 1.0)
            
            if confidence < confidence_threshold:
                continue
            
            source_uri = CAF[self._normalize_uri(source)]
            target_uri = CAF[self._normalize_uri(target)]
            relation_type = edge_data.get('relation', 'causes')
            
            # Create causal relation
            relation_uri = CAUSES[relation_type]
            self.graph.add((source_uri, relation_uri, target_uri))
            
            # Add metadata
            edge_uri = CAF[f"edge_{source}_{target}"]
            self.graph.add((edge_uri, RDF.type, CAF.CausalEdge))
            self.graph.add((edge_uri, CAF.source, source_uri))
            self.graph.add((edge_uri, CAF.target, target_uri))
            self.graph.add((edge_uri, CAF.confidence, Literal(confidence)))
            
            if 'evidence' in edge_data:
                self.graph.add((
                    edge_uri,
                    CAF.evidence,
                    Literal(edge_data['evidence'])
                ))
        
        return self.graph
    
    def _normalize_uri(self, name: str) -> str:
        """Normalize variable name for URI."""
        return name.lower().replace(" ", "_").replace("-", "_")
    
    def export_to_ntriples(self, output_path: str):
        """Export to N-Triples format for Fuseki."""
        self.graph.serialize(destination=output_path, format='nt')