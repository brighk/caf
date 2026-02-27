#!/usr/bin/env python3
"""
Build Paper A real-domain benchmark from three real corpora:
- medical: PubMed abstracts
- economics: Federal Reserve system sentences
- policy: GovReport reports

Output format matches paperA/run_real_domain_benchmark.py input:
[
  {"domain": "...", "text": "...", "gold_edges": [["a","b"], ...]},
  ...
]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from datasets import load_dataset


def norm(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[,;:\- ]+|[,;:\- ]+$", "", text)
    return text


def short_phrase(text: str, max_words: int = 8) -> str:
    t = norm(text)
    words = t.split()
    if len(words) > max_words:
        return ""
    return t


PATTERNS = [
    ("forward", re.compile(r"\b(.{3,80}?)\s+causes?\s+(.{3,80}?)\b", re.IGNORECASE)),
    ("forward", re.compile(r"\b(.{3,80}?)\s+leads?\s+to\s+(.{3,80}?)\b", re.IGNORECASE)),
    ("forward", re.compile(r"\b(.{3,80}?)\s+results?\s+in\s+(.{3,80}?)\b", re.IGNORECASE)),
    ("reverse", re.compile(r"\b(.{3,80}?)\s+due\s+to\s+(.{3,80}?)\b", re.IGNORECASE)),
    ("reverse", re.compile(r"\b(.{3,80}?)\s+because\s+of\s+(.{3,80}?)\b", re.IGNORECASE)),
    ("forward", re.compile(r"\b(.{3,80}?)\s+increases?\s+(.{3,80}?)\b", re.IGNORECASE)),
    ("forward", re.compile(r"\b(.{3,80}?)\s+decreases?\s+(.{3,80}?)\b", re.IGNORECASE)),
    ("forward", re.compile(r"\b(.{3,80}?)\s+raises?\s+(.{3,80}?)\b", re.IGNORECASE)),
    ("forward", re.compile(r"\b(.{3,80}?)\s+reduces?\s+(.{3,80}?)\b", re.IGNORECASE)),
    ("forward", re.compile(r"\b(.{3,80}?)\s+improves?\s+(.{3,80}?)\b", re.IGNORECASE)),
    ("forward", re.compile(r"\b(.{3,80}?)\s+worsens?\s+(.{3,80}?)\b", re.IGNORECASE)),
]


def extract_edges(text: str) -> List[Tuple[str, str]]:
    edges: List[Tuple[str, str]] = []
    for sent in re.split(r"[.!?]\s+", text):
        s = sent.strip()
        if len(s) < 12:
            continue
        for direction, pat in PATTERNS:
            for a, b in pat.findall(s):
                left = short_phrase(a)
                right = short_phrase(b)
                if not left or not right or left == right:
                    continue
                if direction == "forward":
                    edges.append((left, right))
                else:
                    edges.append((right, left))

    dedup: List[Tuple[str, str]] = []
    seen = set()
    for e in edges:
        if e in seen:
            continue
        seen.add(e)
        dedup.append(e)
    return dedup


def collect_medical(target_n: int, max_scan: int) -> List[Dict]:
    ds = load_dataset("Gaborandi/diabetes_mellitus_type2_pubmed_abstracts", streaming=True)
    out: List[Dict] = []
    scanned = 0
    for row in ds["train"]:
        title = str(row.get("title", "")).strip()
        abstract = str(row.get("abstract", "")).strip()
        text = f"{title}. {abstract}".strip()
        scanned += 1
        edges = extract_edges(text)
        if edges:
            out.append(
                {
                    "domain": "medical",
                    "text": text,
                    "gold_edges": [[a, b] for a, b in edges],
                    "source": "Gaborandi/diabetes_mellitus_type2_pubmed_abstracts",
                    "source_id": str(row.get("pubmed_id", "")),
                }
            )
            if len(out) >= target_n:
                break
        if scanned >= max_scan:
            break
    return out


def collect_economics(target_n: int, max_scan: int) -> List[Dict]:
    ds = load_dataset("gtfintechlab/federal_reserve_system", "5768", streaming=True)
    out: List[Dict] = []
    scanned = 0
    for row in ds["train"]:
        text = str(row.get("sentences", "")).strip()
        scanned += 1
        edges = extract_edges(text)
        # Domain-tailored fallback for economics text with "X and Y" policy effects.
        if not edges:
            m = re.search(
                r"\b(lower|higher|rising|falling|tightening|easing|stronger|weaker)\s+([a-z][a-z0-9 \-]{2,40})\s+(?:and|with)\s+([a-z][a-z0-9 \-]{2,40})",
                text.lower(),
            )
            if m:
                cause = short_phrase(m.group(1) + " " + m.group(2))
                effect = short_phrase(m.group(3))
                if cause and effect and cause != effect:
                    edges = [(cause, effect)]
        if edges:
            out.append(
                {
                    "domain": "economics",
                    "text": text,
                    "gold_edges": [[a, b] for a, b in edges],
                    "source": "gtfintechlab/federal_reserve_system",
                    "source_id": str(row.get("__index_level_0__", "")),
                }
            )
            if len(out) >= target_n:
                break
        if scanned >= max_scan:
            break
    return out


def collect_policy(target_n: int, max_scan: int, max_chars: int) -> List[Dict]:
    ds = load_dataset("ccdv/govreport-summarization", streaming=True)
    out: List[Dict] = []
    scanned = 0
    for row in ds["train"]:
        text = str(row.get("report", "")).strip()
        if len(text) > max_chars:
            text = text[:max_chars]
        scanned += 1
        edges = extract_edges(text)
        if edges:
            out.append(
                {
                    "domain": "policy",
                    "text": text,
                    "gold_edges": [[a, b] for a, b in edges],
                    "source": "ccdv/govreport-summarization",
                }
            )
            if len(out) >= target_n:
                break
        if scanned >= max_scan:
            break
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/paperA_real_med_econ_policy.json")
    p.add_argument("--per-domain", type=int, default=40)
    p.add_argument("--max-scan", type=int, default=30000)
    p.add_argument("--policy-max-chars", type=int, default=1800)
    args = p.parse_args()

    medical = collect_medical(args.per_domain, args.max_scan)
    economics = collect_economics(args.per_domain, args.max_scan)
    policy = collect_policy(args.per_domain, args.max_scan, args.policy_max_chars)

    data = medical + economics + policy
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2))

    print(f"saved {len(data)} samples -> {out}")
    print(f"medical={len(medical)} economics={len(economics)} policy={len(policy)}")


if __name__ == "__main__":
    main()
