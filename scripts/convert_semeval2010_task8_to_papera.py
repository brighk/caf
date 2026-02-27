#!/usr/bin/env python3
"""
Convert SemEval 2010 Task 8 to Paper A real-domain JSON format.

We keep only Cause-Effect relations and convert each sentence to one gold edge.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset


E1 = re.compile(r"<e1>(.*?)</e1>", re.IGNORECASE | re.DOTALL)
E2 = re.compile(r"<e2>(.*?)</e2>", re.IGNORECASE | re.DOTALL)
TAG = re.compile(r"</?e[12]>", re.IGNORECASE)
WS = re.compile(r"\s+")


def norm(text: str) -> str:
    return WS.sub(" ", text.strip().lower())


def clean_sentence(s: str) -> str:
    return WS.sub(" ", TAG.sub("", s).strip())


def convert(limit: int = 0) -> List[Dict]:
    ds = load_dataset("SemEvalWorkshop/sem_eval_2010_task_8")
    names = ds["train"].features["relation"].names

    out: List[Dict] = []
    count = 0
    for split in ("train", "test"):
        for row in ds[split]:
            rel_id = int(row["relation"])
            rel = names[rel_id]
            if rel not in ("Cause-Effect(e1,e2)", "Cause-Effect(e2,e1)"):
                continue

            sent = str(row["sentence"])
            m1 = E1.search(sent)
            m2 = E2.search(sent)
            if not m1 or not m2:
                continue
            e1 = norm(m1.group(1))
            e2 = norm(m2.group(1))
            if not e1 or not e2 or e1 == e2:
                continue

            if rel == "Cause-Effect(e1,e2)":
                cause, effect = e1, e2
            else:
                cause, effect = e2, e1

            out.append(
                {
                    "domain": "general",
                    "text": clean_sentence(sent),
                    "gold_edges": [[cause, effect]],
                    "source_file": f"semeval2010_task8_{split}",
                }
            )

            count += 1
            if limit and count >= limit:
                return out

    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/paperA_semeval2010_task8_cause_effect.json")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    rows = convert(limit=args.limit)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"saved {len(rows)} samples -> {out}")


if __name__ == "__main__":
    main()
