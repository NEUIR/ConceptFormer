# Data

## Training Set

`ConceptFormer-Trainset` contains 37,966 query-document pairs. Each row stores the query,
positive document identifiers, a textual evidence description, bounding boxes in `xyxy`
format, original and processed image sizes, answers, and source-dataset labels.

The release does not duplicate upstream document images. Resolve `relevant_doc_ids`
against `NTT-hil-insight/OpenDocVQA-Corpus` or a local copy of that corpus. This keeps the
dataset card explicit about provenance and allows each upstream license to remain intact.

| Column | Type | Meaning |
|---|---|---|
| `query_id` | string | Stable query identifier |
| `query_text` | string | Retrieval instruction and question |
| `relevant_doc_ids` | list[string] | Positive document IDs |
| `describe` | string | Region-level evidence description |
| `bbox_2d` | JSON string | One or more `[x1,y1,x2,y2]` boxes |
| `image_size` | JSON string | Original `[width,height]` |
| `input_size` | JSON string | Annotation input `[width,height]` |
| `answers` | JSON string | Accepted answers when available |
| `dataset_names` | JSON string | Source dataset labels |

## Evaluation Set

Evaluation is performed independently within each dataset corpus.

| Dataset | Queries | Corpus images | Parquet shards |
|---|---:|---:|---:|
| InfoVQA | 1,048 | 5,485 | 5 |
| ChartQA | 150 | 20,882 | 3 |
| SlideVQA | 760 | 52,380 | 14 |
| TQA | 1,000 | 1,000 | 1 |
| OWID Charts | 131 | 131 | 1 |
| Wikimedia Maps | 455 | 455 | 1 |

The evaluation repository is image-complete. Every `corpus-*.parquet` row contains a
document ID and the original image bytes, so evaluation does not depend on local absolute
paths or another corpus download. Images are not resized or re-encoded while being
packaged. No benchmark corpora are mixed during retrieval.

To rebuild the corpus shards from upstream data, run:

```bash
python scripts/package_eval_corpora.py \
  --opendoc-root /path/to/OpenDocVQA-Corpus \
  --jina-root /path/to/jina_vdr_eval
```

## Integrity

Run `python scripts/build_manifest.py huggingface` after preparing a release. The generated
`SHA256SUMS` files cover every uploaded artifact except the checksum file itself.

## Licensing

The code is released under Apache-2.0. Dataset samples and images retain their upstream
licenses and terms. Users must comply with the terms of each source benchmark.
