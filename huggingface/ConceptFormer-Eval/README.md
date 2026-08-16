---
language:
- en
license: other
task_categories:
- visual-document-retrieval
pretty_name: ConceptFormer Evaluation Suite
size_categories:
- 10K<n<100K
---

# ConceptFormer Evaluation Suite

Queries, qrels, and source manifests for six visual-document retrieval benchmarks:
InfoVQA, ChartQA, SlideVQA, TQA, OWID Charts, and Wikimedia Maps.

Retrieval is performed separately in each benchmark's own corpus. This repository includes
the complete corpus images for all six benchmarks as embedded bytes in sharded Parquet
files; no external image paths are required at evaluation time.

Each dataset directory contains `query.parquet`, `qrels.txt`, `manifest.json`, and one or
more `corpus-*.parquet` shards. Corpus rows use `doc_id`, `image: {bytes, path}`, and
`dataset_name`. Images are not resized or re-encoded during packaging.

The six corpora contain 80,333 image rows in total. Query and corpus Parquet files use
separate schemas; score retrieval with the `qrels.txt` file in the same dataset directory.

```python
from datasets import load_dataset

corpus = load_dataset(
    "parquet",
    data_files="hf://datasets/hmhm1229/ConceptFormer-Eval/data/infovqa/corpus-*.parquet",
    split="train",
)
```
