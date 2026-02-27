#!/usr/bin/env python3
"""
Convert Alqarni/fincausal-2026-en to Paper A real-domain JSON format.

Each sample is converted to one causal edge: cause -> effect
where:
- cause: answer
- effect: extracted from question ("reason for ...")
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict

from datasets import load_dataset


def norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def extract_effect(question: str) -> str:
    q = (question or "").strip()
    patterns = [
        r"reason for (.+?)[\?\.]?$",
        r"main reason for (.+?)[\?\.]?$",
        r"what caused (.+?)[\?\.]?$",
        r"cause of (.+?)[\?\.]?$",
        r"led to (.+?)[\?\.]?$",
    ]
    ql = q.lower()
    for p in patterns:
        m = re.search(p, ql)
        if m:
            return norm(m.group(1))
    return norm(ql.rstrip(" ?.!"))


def convert(split: str, limit: int) -> List[Dict]:
    ds = load_dataset("Alqarni/fincausal-2026-en", split=split)
    out: List[Dict] = []
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        text = row.get("context", "")
        question = row.get("question", "")
        cause = norm(row.get("answer", ""))
        effect = extract_effect(question)
        if not text or not cause or not effect or cause == effect:
            continue
        out.append(
            {
                "domain": "finance",
                "text": text,
                "gold_edges": [[cause, effect]],
                "source_file": f"fincausal_2026_{split}",
            }
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="train")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--output", default="data/paperA_fincausal2026.json")
    args = p.parse_args()

    rows = convert(args.split, args.limit)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"saved {len(rows)} samples -> {out}")


if __name__ == "__main__":
    main()
