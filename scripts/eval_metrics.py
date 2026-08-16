#!/usr/bin/env python3
"""Compute corpus-local retrieval metrics from qrels and a ranking file."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def load_qrels(path: Path) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 4:
            qid, docid, rel = parts[0], parts[2], float(parts[3])
        elif len(parts) >= 3:
            qid, docid, rel = parts[0], parts[1], float(parts[2])
        else:
            continue
        if rel > 0:
            qrels.setdefault(qid, set()).add(docid)
    return qrels


def load_ranking(path: Path) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2:
            rows.setdefault(parts[0], []).append(parts[1])
    return rows


def metrics(qrels: dict[str, set[str]], ranking: dict[str, list[str]], k: int = 10):
    recalls, ndcgs, rrs = [], [], []
    for qid, relevant in qrels.items():
        ranked = ranking.get(qid, [])[:k]
        hits = [int(docid in relevant) for docid in ranked]
        recalls.append(len(set(ranked) & relevant) / len(relevant))
        dcg = sum(hit / math.log2(i + 2) for i, hit in enumerate(hits))
        ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant))))
        ndcgs.append(dcg / ideal if ideal else 0.0)
        rrs.append(next((1 / (i + 1) for i, hit in enumerate(hits) if hit), 0.0))
    n = max(1, len(qrels))
    return {"recall_10": sum(recalls) / n, "ndcg_10": sum(ndcgs) / n, "mrr_10": sum(rrs) / n}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qrels", required=True, type=Path)
    parser.add_argument("--ranking", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = metrics(load_qrels(args.qrels), load_ranking(args.ranking))
    row = {"model": args.model, "dataset": args.dataset, **result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if args.output.exists():
        with args.output.open(newline="", encoding="utf-8") as handle:
            existing = [r for r in csv.DictReader(handle) if not (
                r.get("model") == args.model and r.get("dataset") == args.dataset
            )]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(existing + [row])
    print(", ".join([args.model, args.dataset] + [f"{k}={v * 100:.2f}" for k, v in result.items()]))


if __name__ == "__main__":
    main()
