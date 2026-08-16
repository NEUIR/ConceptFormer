#!/usr/bin/env python3
"""Validate release schemas, qrels coverage, tokenizers, and model metadata."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import torch
from safetensors import safe_open
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
HF = ROOT / "huggingface"


def validate_data() -> None:
    train = HF / "ConceptFormer-Trainset" / "data" / "train.parquet"
    assert pq.read_metadata(train).num_rows == 37966
    required = {"query_id", "query_text", "relevant_doc_ids", "describe", "bbox_2d"}
    assert required <= set(pq.read_schema(train).names)
    eval_root = HF / "ConceptFormer-Eval" / "data"
    for directory in sorted(path for path in eval_root.iterdir() if path.is_dir()):
        table = pq.read_table(directory / "query.parquet", columns=["query_id"])
        qids = {str(value) for value in table.column("query_id").to_pylist()}
        assert len(qids) == table.num_rows
        qrel_qids = set()
        positive_doc_ids = set()
        for line in (directory / "qrels.txt").read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if parts:
                qrel_qids.add(parts[0])
                if int(parts[3]) > 0:
                    positive_doc_ids.add(parts[2])
        assert qids == qrel_qids, f"qrels mismatch in {directory.name}"

        corpus_shards = sorted(directory.glob("corpus*.parquet"))
        assert corpus_shards, f"missing image corpus in {directory.name}"
        corpus_doc_ids = set()
        corpus_rows = 0
        for shard in corpus_shards:
            parquet = pq.ParquetFile(shard)
            for batch in parquet.iter_batches(columns=["doc_id", "image"], batch_size=256):
                doc_ids = batch.column("doc_id").to_pylist()
                images = batch.column("image").to_pylist()
                assert all(doc_id for doc_id in doc_ids), f"empty doc_id in {shard.name}"
                assert all(image and image.get("bytes") for image in images), (
                    f"missing embedded image bytes in {shard.name}"
                )
                corpus_doc_ids.update(str(doc_id) for doc_id in doc_ids)
                corpus_rows += batch.num_rows
        assert len(corpus_doc_ids) == corpus_rows, f"duplicate corpus doc_id in {directory.name}"
        missing_positive_docs = positive_doc_ids - corpus_doc_ids
        assert not missing_positive_docs, (
            f"{directory.name} is missing {len(missing_positive_docs)} positive documents"
        )

        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["query_rows"] == table.num_rows
        assert manifest["corpus_included"] is True
        assert manifest["corpus_rows"] == corpus_rows
        assert manifest["corpus_positive_docs"] == len(positive_doc_ids)
        assert manifest["corpus_shards"] == [path.name for path in corpus_shards]
        assert manifest["corpus_bytes"] == sum(path.stat().st_size for path in corpus_shards)


def validate_model(name: str) -> None:
    directory = HF / name
    config = json.loads((directory / "adapter_config.json").read_text(encoding="utf-8"))
    assert not config["base_model_name_or_path"].startswith("/")
    tokenizer = AutoTokenizer.from_pretrained(directory, trust_remote_code=True)
    project_tokens = [token for token in tokenizer.additional_special_tokens if "lcon" in token.lower()]
    assert project_tokens == ["<|lcon|>"]
    with safe_open(directory / "adapter_model.safetensors", framework="pt") as handle:
        assert len(list(handle.keys())) > 0
    state = torch.load(directory / "conceptformer_state.pt", map_location="cpu", weights_only=True)
    assert set(state) <= {"latent_proj.weight", "latent_pooler.attn_score.weight"}


def main() -> None:
    validate_data()
    validate_model("ConceptFormer-Qwen")
    validate_model("ConceptFormer-Phi3V")
    print("ConceptFormer release validation passed.")


if __name__ == "__main__":
    main()
