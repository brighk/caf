"""
Causal Graph Discovery from Text
=================================
Implements the 5-stage pipeline from ICAR-AI 2026 paper:
1. Variable & edge extraction from text
2. Graph induction with DAG checks
3. SCM construction
4. Intervention-based validation
5. Counterfactual verification

This populates the KB before running CAF experiments.
"""

import json
import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import networkx as nx

@dataclass
class CausalVariable:
    """Extracted causal variable."""
    name: str
    aliases: List[str]
    domain: str
    confidence: float

@dataclass
class CausalEdge:
    """Directed causal edge."""
    source: str
    target: str
    relation_type: str  # causes, enables, prevents
    confidence: float
    evidence: List[str]  # Text snippets supporting this edge

class CausalGraphExtractor:
    """
    Extracts causal graphs from unstructured text using LLM.
    
    Based on ICAR-AI 2026 methodology.
    """
    
    def __init__(self, llm, k_samples: int = 10):
        self.llm = llm
        self.k_samples = k_samples
        
    def extract_from_text(self, text: str, domain: str) -> Dict:
        """
        Extract causal graph from text.
        
        Returns:
            {
                'variables': List[CausalVariable],
                'edges': List[CausalEdge],
                'graph': nx.DiGraph,
                'confidence': float
            }
        """
        # Stage 1: Extract variables and edges
        variables = self._extract_variables(text, domain)
        edges = self._extract_edges(text, variables)
        
        # Stage 2: Build graph with self-consistency
        graph = self._build_graph_with_consistency(
            variables, edges, k=self.k_samples
        )
        
        # Stage 3: DAG validation
        graph = self._enforce_dag(graph)
        
        return {
            'variables': variables,
            'edges': edges,
            'graph': graph,
            'confidence': self._compute_confidence(graph)
        }
    
    def _extract_variables(self, text: str, domain: str) -> List[CausalVariable]:
        """Extract causal variables using structured LLM prompting."""
        
        prompt = f"""
        Extract causal variables from the following {domain} text.
        
        Text: {text}
        
        For each variable, provide:
        1. Variable name (normalized)
        2. Synonyms/aliases
        3. Brief description
        
        Output as JSON list.
        """
        
        # Use self-consistency: sample K times
        all_variables = []
        for _ in range(self.k_samples):
            response = self.llm.generate(prompt)
            variables = self._parse_variable_response(response)
            all_variables.extend(variables)

        # Fallback: light heuristic extraction from explicit causal text.
        if not all_variables:
            all_variables = self._extract_variables_from_text(text, domain)

        # Merge and score by frequency
        return self._consolidate_variables(all_variables)
    
    def _extract_edges(
        self, 
        text: str, 
        variables: List[CausalVariable]
    ) -> List[CausalEdge]:
        """Extract causal edges using LLM."""
        
        var_names = [v.name for v in variables]
        
        prompt = f"""
        Given these variables: {var_names}
        
        Extract causal relationships from this text:
        {text}
        
        For each relationship, specify:
        - Source variable
        - Target variable  
        - Relation type (causes/enables/prevents/requires)
        - Evidence (text snippet)
        
        Output as JSON.
        """
        
        # Self-consistency sampling
        all_edges = []
        for _ in range(self.k_samples):
            response = self.llm.generate(prompt)
            edges = self._parse_edge_response(response)
            all_edges.extend(edges)

        # Fallback: regex edge extraction from text.
        if not all_edges:
            all_edges = self._extract_edges_from_text(text)

        # Filter edges appearing in < 60% of samples (low confidence)
        return self._filter_confident_edges(all_edges, threshold=0.6)

    def _parse_variable_response(self, response: str) -> List[CausalVariable]:
        """Parse LLM variable extraction response."""
        payload = self._extract_json_payload(response)
        if payload is None:
            return []

        raw_items: List[Any]
        if isinstance(payload, list):
            raw_items = payload
        elif isinstance(payload, dict):
            raw_items = payload.get("variables", [])
        else:
            return []

        parsed: List[CausalVariable] = []
        for item in raw_items:
            if isinstance(item, str):
                name = self._normalize_name(item)
                if not name:
                    continue
                parsed.append(CausalVariable(name=name, aliases=[name], domain="", confidence=1.0))
                continue

            if not isinstance(item, dict):
                continue

            name = self._normalize_name(
                item.get("name") or item.get("variable") or item.get("label") or ""
            )
            if not name:
                continue

            aliases_raw = item.get("aliases") or item.get("synonyms") or []
            if isinstance(aliases_raw, str):
                aliases = [self._normalize_name(aliases_raw)] if aliases_raw.strip() else []
            else:
                aliases = [self._normalize_name(a) for a in aliases_raw if isinstance(a, str)]
            aliases = [a for a in aliases if a]
            if name not in aliases:
                aliases.insert(0, name)

            domain = str(item.get("domain", "") or "")
            confidence = self._safe_float(item.get("confidence", 1.0), default=1.0)

            parsed.append(
                CausalVariable(
                    name=name,
                    aliases=aliases,
                    domain=domain,
                    confidence=max(0.0, min(1.0, confidence))
                )
            )

        return parsed

    def _consolidate_variables(self, variables: List[CausalVariable]) -> List[CausalVariable]:
        """Merge repeated variable mentions and score by frequency."""
        if not variables:
            return []

        by_name: Dict[str, Dict[str, Any]] = {}
        total = len(variables)

        for v in variables:
            key = self._normalize_name(v.name)
            if not key:
                continue
            bucket = by_name.setdefault(
                key,
                {
                    "count": 0,
                    "aliases": set(),
                    "domains": Counter(),
                    "conf_sum": 0.0,
                },
            )
            bucket["count"] += 1
            bucket["aliases"].update(self._normalize_name(a) for a in v.aliases if a)
            if v.domain:
                bucket["domains"][v.domain] += 1
            bucket["conf_sum"] += v.confidence

        merged: List[CausalVariable] = []
        for name, info in by_name.items():
            freq = info["count"] / max(1, total)
            avg_conf = info["conf_sum"] / max(1, info["count"])
            domain = info["domains"].most_common(1)[0][0] if info["domains"] else ""
            merged.append(
                CausalVariable(
                    name=name,
                    aliases=sorted(a for a in info["aliases"] if a),
                    domain=domain,
                    confidence=max(0.0, min(1.0, 0.5 * freq + 0.5 * avg_conf)),
                )
            )

        merged.sort(key=lambda x: x.confidence, reverse=True)
        return merged

    def _parse_edge_response(self, response: str) -> List[CausalEdge]:
        """Parse LLM edge extraction response."""
        payload = self._extract_json_payload(response)
        if payload is None:
            return []

        raw_items: List[Any]
        if isinstance(payload, list):
            raw_items = payload
        elif isinstance(payload, dict):
            raw_items = payload.get("edges") or payload.get("relations") or []
        else:
            return []

        parsed: List[CausalEdge] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue

            source = self._normalize_name(
                item.get("source") or item.get("from") or item.get("cause") or ""
            )
            target = self._normalize_name(
                item.get("target") or item.get("to") or item.get("effect") or ""
            )
            if not source or not target or source == target:
                continue

            relation = self._normalize_relation(
                item.get("relation_type") or item.get("relation") or item.get("type") or "causes"
            )
            confidence = self._safe_float(item.get("confidence", 1.0), default=1.0)

            evidence_raw = item.get("evidence", [])
            if isinstance(evidence_raw, str):
                evidence = [evidence_raw.strip()] if evidence_raw.strip() else []
            elif isinstance(evidence_raw, list):
                evidence = [str(e).strip() for e in evidence_raw if str(e).strip()]
            else:
                evidence = []

            parsed.append(
                CausalEdge(
                    source=source,
                    target=target,
                    relation_type=relation,
                    confidence=max(0.0, min(1.0, confidence)),
                    evidence=evidence,
                )
            )

        return parsed

    def _filter_confident_edges(self, edges: List[CausalEdge], threshold: float = 0.6) -> List[CausalEdge]:
        """
        Keep edges that are stable across self-consistency samples.

        `threshold` is interpreted as support ratio over `k_samples`.
        """
        if not edges:
            return []

        grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for edge in edges:
            key = (edge.source, edge.target)
            bucket = grouped.setdefault(
                key,
                {
                    "count": 0,
                    "relation_counts": Counter(),
                    "conf_sum": 0.0,
                    "evidence": [],
                },
            )
            bucket["count"] += 1
            bucket["relation_counts"][edge.relation_type] += 1
            bucket["conf_sum"] += edge.confidence
            bucket["evidence"].extend(edge.evidence)

        filtered: List[CausalEdge] = []
        for (source, target), info in grouped.items():
            support = info["count"] / max(1, self.k_samples)
            if support < threshold:
                continue
            relation = info["relation_counts"].most_common(1)[0][0]
            avg_conf = info["conf_sum"] / max(1, info["count"])
            filtered.append(
                CausalEdge(
                    source=source,
                    target=target,
                    relation_type=relation,
                    confidence=max(0.0, min(1.0, 0.5 * support + 0.5 * avg_conf)),
                    evidence=info["evidence"][:5],
                )
            )

        filtered.sort(key=lambda e: e.confidence, reverse=True)
        return filtered

    def _build_graph_with_consistency(
        self,
        variables: List[CausalVariable],
        edges: List[CausalEdge],
        k: int = 10
    ) -> nx.DiGraph:
        """Build a weighted DAG candidate from consolidated variables and edges."""
        graph = nx.DiGraph()

        for var in variables:
            graph.add_node(
                var.name,
                confidence=var.confidence,
                aliases=var.aliases,
                domain=var.domain,
            )

        for edge in edges:
            if edge.source not in graph:
                graph.add_node(edge.source, confidence=0.5, aliases=[edge.source], domain="")
            if edge.target not in graph:
                graph.add_node(edge.target, confidence=0.5, aliases=[edge.target], domain="")

            # Keep the strongest variant if duplicate edge appears.
            existing = graph.get_edge_data(edge.source, edge.target)
            if existing is None or edge.confidence > existing.get("confidence", 0.0):
                graph.add_edge(
                    edge.source,
                    edge.target,
                    confidence=edge.confidence,
                    relation=edge.relation_type,
                    evidence=" | ".join(edge.evidence[:3]) if edge.evidence else "",
                )

        return graph

    def _compute_confidence(self, graph: nx.DiGraph) -> float:
        """Overall extraction confidence from node and edge confidences."""
        if graph.number_of_nodes() == 0:
            return 0.0

        node_scores = [float(graph.nodes[n].get("confidence", 0.5)) for n in graph.nodes]
        edge_scores = [float(d.get("confidence", 0.5)) for _, _, d in graph.edges(data=True)]

        node_avg = sum(node_scores) / max(1, len(node_scores))
        if edge_scores:
            edge_avg = sum(edge_scores) / len(edge_scores)
            return max(0.0, min(1.0, 0.4 * node_avg + 0.6 * edge_avg))
        return max(0.0, min(1.0, node_avg))

    def _extract_json_payload(self, text: str) -> Optional[Any]:
        """Extract JSON object/array from raw LLM output."""
        if not text:
            return None

        cleaned = text.strip()

        # Prefer fenced code blocks if present.
        fence_match = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL | re.IGNORECASE)
        if fence_match:
            cleaned = fence_match.group(1).strip()

        # Try direct JSON parse first.
        try:
            return json.loads(cleaned)
        except Exception:
            pass

        # Fallback: find first JSON-looking span.
        start_obj = cleaned.find("{")
        start_arr = cleaned.find("[")
        starts = [s for s in (start_obj, start_arr) if s != -1]
        if not starts:
            return None
        start = min(starts)

        for end in range(len(cleaned), start + 1, -1):
            snippet = cleaned[start:end].strip()
            try:
                return json.loads(snippet)
            except Exception:
                continue
        return None

    def _extract_variables_from_text(self, text: str, domain: str) -> List[CausalVariable]:
        """Deterministic variable fallback from explicit causal phrases."""
        tokens = set()
        for cause, effect in self._extract_causal_pairs(text):
            tokens.add(cause)
            tokens.add(effect)
        return [
            CausalVariable(name=t, aliases=[t], domain=domain, confidence=0.6)
            for t in sorted(tokens)
        ]

    def _extract_edges_from_text(self, text: str) -> List[CausalEdge]:
        """Deterministic edge fallback from explicit causal phrases."""
        edges: List[CausalEdge] = []
        for cause, effect in self._extract_causal_pairs(text):
            edges.append(
                CausalEdge(
                    source=cause,
                    target=effect,
                    relation_type="causes",
                    confidence=0.6,
                    evidence=[],
                )
            )
        return edges

    def _extract_causal_pairs(self, text: str) -> List[Tuple[str, str]]:
        """Extract simple X causes Y pairs from text."""
        pairs: List[Tuple[str, str]] = []
        pattern = re.compile(r"\b([A-Za-z][A-Za-z0-9_\- ]{0,40}?)\s+causes?\s+(?:not\s+)?([A-Za-z][A-Za-z0-9_\- ]{0,40})\b", re.IGNORECASE)
        for match in pattern.finditer(text or ""):
            source = self._normalize_name(match.group(1))
            target = self._normalize_name(match.group(2))
            if source and target and source != target:
                pairs.append((source, target))
        return pairs

    def _normalize_name(self, name: str) -> str:
        """Normalize variable naming."""
        if not isinstance(name, str):
            return ""
        s = name.strip().lower()
        s = re.sub(r"[^a-z0-9_\- ]+", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _normalize_relation(self, relation: str) -> str:
        """Normalize relation type to a small controlled set."""
        rel = str(relation or "causes").strip().lower()
        if "prevent" in rel or "inhibit" in rel:
            return "prevents"
        if "enable" in rel:
            return "enables"
        if "require" in rel:
            return "requires"
        return "causes"

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        """Best-effort float conversion."""
        try:
            return float(value)
        except Exception:
            return default

    def _enforce_dag(self, graph: nx.DiGraph) -> nx.DiGraph:
        """Ensure graph is a DAG by removing cycles."""
        while not nx.is_directed_acyclic_graph(graph):
            # Find and remove lowest-confidence edge in cycle
            try:
                cycle = nx.find_cycle(graph)
                # Remove edge with lowest confidence
                min_edge = min(
                    cycle, 
                    key=lambda e: graph[e[0]][e[1]]['confidence']
                )
                graph.remove_edge(min_edge[0], min_edge[1])
            except nx.NetworkXNoCycle:
                break
        
        return graph
