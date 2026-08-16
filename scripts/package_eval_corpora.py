#!/usr/bin/env python3
"""Package all six evaluation corpora with image bytes embedded in Parquet."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


OPENDOC_DATASETS = ("infovqa", "chartqa", "slidevqa")
JINA_DATASETS = ("tqa", "owid_charts_en", "wikimedia-commons-maps")
CORPUS_SCHEMA = pa.schema(
    [
        pa.field("doc_id", pa.string()),
        pa.field(
            "image",
            pa.struct(
                [
                    pa.field("bytes", pa.binary()),
                    pa.field("path", pa.string()),
                ]
            ),
        ),
        pa.field("dataset_name", pa.string()),
    ]
)


def remove_existing_shards(target_dir: Path) -> None:
    for path in target_dir.glob("corpus*.parquet"):
        path.unlink()


def package_opendoc(dataset: str, source_root: Path, target_dir: Path) -> list[Path]:
    sources = sorted((source_root / dataset).glob("*.parquet"))
    if not sources:
        raise FileNotFoundError(f"No OpenDocVQA corpus shards found for {dataset}")
    outputs = []
    total = len(sources)
    for index, source in enumerate(sources):
        schema = pq.ParquetFile(source).schema_arrow
        if "doc_id" not in schema.names or "image" not in schema.names:
            raise ValueError(f"Unexpected corpus schema in {source}: {schema}")
        target = target_dir / f"corpus-{index:05d}-of-{total:05d}.parquet"
        shutil.copy2(source, target)
        outputs.append(target)
    return outputs


def package_jina(dataset: str, source_root: Path, target_dir: Path) -> list[Path]:
    source = source_root / dataset / "corpus.parquet"
    if not source.is_file():
        raise FileNotFoundError(source)
    target = target_dir / "corpus-00000-of-00001.parquet"
    writer = pq.ParquetWriter(target, CORPUS_SCHEMA, compression="zstd")
    try:
        parquet = pq.ParquetFile(source)
        for batch in parquet.iter_batches(batch_size=64):
            output_rows = []
            for row in batch.to_pylist():
                image = row["image"] or {}
                image_bytes = image.get("bytes")
                image_path = image.get("path")
                if image_bytes is None:
                    if not image_path:
                        raise ValueError(f"Missing image bytes and path for {row['doc_id']}")
                    image_file = Path(image_path)
                    if not image_file.is_file():
                        raise FileNotFoundError(image_file)
                    image_bytes = image_file.read_bytes()
                    image_path = image_file.name
                output_rows.append(
                    {
                        "doc_id": str(row["doc_id"]),
                        "image": {"bytes": image_bytes, "path": image_path},
                        "dataset_name": str(row.get("dataset_name") or dataset),
                    }
                )
            writer.write_table(pa.Table.from_pylist(output_rows, schema=CORPUS_SCHEMA))
    finally:
        writer.close()
    return [target]


def positive_doc_ids(qrels_path: Path) -> set[str]:
    doc_ids = set()
    with qrels_path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) >= 4 and float(parts[3]) > 0:
                doc_ids.add(parts[2])
    return doc_ids


def verify_and_update_manifest(dataset: str, target_dir: Path, shards: list[Path]) -> None:
    corpus_ids = set()
    row_count = 0
    for shard in shards:
        table = pq.read_table(shard, columns=["doc_id", "image"])
        row_count += len(table)
        for row in table.to_pylist():
            corpus_ids.add(str(row["doc_id"]))
            image = row["image"] or {}
            if not image.get("bytes"):
                raise ValueError(f"Corpus image bytes are missing for {row['doc_id']}")
    positives = positive_doc_ids(target_dir / "qrels.txt")
    missing = positives - corpus_ids
    if missing:
        preview = sorted(missing)[:10]
        raise ValueError(f"{dataset}: {len(missing)} qrels documents missing: {preview}")

    manifest_path = target_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "corpus_included": True,
            "corpus_rows": row_count,
            "corpus_positive_docs": len(positives),
            "corpus_shards": [path.name for path in shards],
            "corpus_bytes": sum(path.stat().st_size for path in shards),
            "image_storage": "embedded bytes; no external image paths required",
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"{dataset}: {row_count} corpus rows, {len(positives)} positive docs, "
        f"{manifest['corpus_bytes']} bytes across {len(shards)} shard(s)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, default=Path("huggingface/ConceptFormer-Eval/data"))
    parser.add_argument("--opendoc-root", type=Path, required=True)
    parser.add_argument("--jina-root", type=Path, required=True)
    args = parser.parse_args()

    for dataset in (*OPENDOC_DATASETS, *JINA_DATASETS):
        target_dir = args.eval_root / dataset
        target_dir.mkdir(parents=True, exist_ok=True)
        remove_existing_shards(target_dir)
        if dataset in OPENDOC_DATASETS:
            shards = package_opendoc(dataset, args.opendoc_root, target_dir)
        else:
            shards = package_jina(dataset, args.jina_root, target_dir)
        verify_and_update_manifest(dataset, target_dir, shards)


if __name__ == "__main__":
    main()
