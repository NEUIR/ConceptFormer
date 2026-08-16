#!/usr/bin/env python3
"""Materialize corpus parquets for the three paired Jina evaluation datasets."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Dataset, Features, Image, Value, load_dataset


SOURCES = {
    "tqa": "jinaai/tqa",
    "owid_charts_en": "jinaai/owid_charts_en",
    "wikimedia-commons-maps": "jinaai/wikimedia-commons-maps",
}


def normalized_basename(doc_id: str) -> str:
    name = Path(doc_id).name
    return re.sub(r"^\d{6}_", "", name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", default="data/eval/data", type=Path)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()
    for name, source in SOURCES.items():
        target_dir = args.eval_root / name
        query_path = target_dir / "query.parquet"
        if not query_path.exists():
            raise FileNotFoundError(query_path)
        rows = pq.read_table(query_path, columns=["relevant_doc_ids"]).to_pylist()
        doc_ids = [str(row["relevant_doc_ids"][0]) for row in rows]
        raw = load_dataset(source, split="test", cache_dir=args.cache_dir)
        by_name = {Path(str(row["image_filename"])).name: row for row in raw}
        corpus = []
        for doc_id in doc_ids:
            basename = normalized_basename(doc_id)
            if basename not in by_name:
                raise KeyError(f"Could not match {doc_id} in {source}")
            corpus.append({"doc_id": doc_id, "image": by_name[basename]["image"]})
        dataset = Dataset.from_list(
            corpus,
            features=Features({"doc_id": Value("string"), "image": Image()}),
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        dataset.to_parquet(str(target_dir / "corpus.parquet"))
        print(f"{name}: wrote {len(dataset)} documents")


if __name__ == "__main__":
    main()
