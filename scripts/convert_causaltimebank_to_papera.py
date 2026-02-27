#!/usr/bin/env python3
"""
Convert Causal-TimeBank CAT XML files into Paper A benchmark JSON format.

Output format:
[
  {
    "domain": "news",
    "text": "...",
    "gold_edges": [["event phrase a", "event phrase b"], ...],
    "source_file": "wsj_0006.xml"
  }
]
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple


def normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def collect_tokens(root: ET.Element) -> Dict[str, str]:
    tokens: Dict[str, str] = {}
    for tok in root.findall("token"):
        tid = tok.attrib.get("id")
        if not tid:
            continue
        tokens[tid] = (tok.text or "").strip()
    return tokens


def event_spans(markables: ET.Element | None, tokens: Dict[str, str]) -> Dict[str, str]:
    spans: Dict[str, str] = {}
    if markables is None:
        return spans

    for item in markables.findall("EVENT"):
        eid = item.attrib.get("id")
        if not eid:
            continue
        tok_ids = [a.attrib.get("id") for a in item.findall("token_anchor")]
        words = [tokens[t] for t in tok_ids if t in tokens and tokens[t]]
        phrase = normalize(" ".join(words))
        if phrase:
            spans[eid] = phrase
    return spans


def clink_edges(relations: ET.Element | None, eid_to_text: Dict[str, str]) -> List[Tuple[str, str]]:
    edges: List[Tuple[str, str]] = []
    if relations is None:
        return edges

    for rel in relations.findall("CLINK"):
        src = rel.find("source")
        tgt = rel.find("target")
        if src is None or tgt is None:
            continue
        sid = src.attrib.get("id", "")
        tid = tgt.attrib.get("id", "")
        if sid not in eid_to_text or tid not in eid_to_text:
            continue
        s = eid_to_text[sid]
        t = eid_to_text[tid]
        if s and t and s != t:
            edges.append((s, t))
    # Stable de-duplication
    dedup: List[Tuple[str, str]] = []
    seen = set()
    for e in edges:
        if e in seen:
            continue
        seen.add(e)
        dedup.append(e)
    return dedup


def document_text(root: ET.Element) -> str:
    words = []
    for tok in root.findall("token"):
        w = (tok.text or "").strip()
        if w:
            words.append(w)
    text = " ".join(words)
    return re.sub(r"\s+([,.;:!?])", r"\1", text).strip()


def infer_domain(filename: str, mode: str) -> str:
    if mode == "news":
        return "news"
    stem = filename.split(".")[0]
    upper = stem.upper()
    if upper.startswith("WSJ"):
        return "wsj"
    if upper.startswith("NYT"):
        return "nyt"
    if upper.startswith("APW") or upper.startswith("AP"):
        return "ap"
    if upper.startswith("VOA"):
        return "voa"
    if upper.startswith("PRI"):
        return "pri"
    if upper.startswith("ABC"):
        return "abc"
    if upper.startswith("CNN"):
        return "cnn"
    if upper.startswith("EA"):
        return "ea"
    if upper.startswith("ED"):
        return "ed"
    return "other_news"


def convert_dir(input_dir: Path, min_edges: int = 1, domain_mode: str = "news") -> List[Dict]:
    out: List[Dict] = []
    for path in sorted(input_dir.glob("*.xml")):
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            continue

        tokens = collect_tokens(root)
        markables = root.find("Markables")
        relations = root.find("Relations")
        eid_to_text = event_spans(markables, tokens)
        edges = clink_edges(relations, eid_to_text)
        if len(edges) < min_edges:
            continue

        out.append(
            {
                "domain": infer_domain(path.name, domain_mode),
                "text": document_text(root),
                "gold_edges": [[a, b] for a, b in edges],
                "source_file": path.name,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Causal-TimeBank CAT XML to Paper A JSON.")
    parser.add_argument(
        "--input-dir",
        default="data/external/Causal-TimeBank",
        help="Directory containing CAT XML files (*.xml).",
    )
    parser.add_argument(
        "--output",
        default="data/paperA_causaltimebank.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--min-edges",
        type=int,
        default=1,
        help="Keep documents with at least this many CLINK edges.",
    )
    parser.add_argument(
        "--domain-mode",
        choices=["news", "source_prefix"],
        default="news",
        help="How to assign domain labels.",
    )
    args = parser.parse_args()

    data = convert_dir(
        Path(args.input_dir),
        min_edges=max(1, args.min_edges),
        domain_mode=args.domain_mode,
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2))

    total_edges = sum(len(x["gold_edges"]) for x in data)
    print(f"saved {len(data)} documents with {total_edges} gold edges -> {out_path}")


if __name__ == "__main__":
    main()
