# ConceptFormer

ConceptFormer is a visual document retriever that learns dynamic latent concepts for
query-aware document representation. Each annotated evidence region contributes one or
more `<|lcon|>` tokens. Their hidden states are pooled into a latent concept embedding,
whose retrieval distribution is aligned with the query distribution.

This repository contains the training and evaluation code, the release manifests for the
training and evaluation datasets, and model cards for the Qwen2.5-VL and Phi-3-Vision
checkpoints.

## News

- **2026-08**: Initial public release with Qwen2.5-VL and Phi-3-Vision checkpoints.

## Released Resources

| Resource | Base model | Description |
|---|---|---|
| `ConceptFormer-Qwen` | `Qwen/Qwen2.5-VL-7B-Instruct` | PEFT adapter and latent state, forward KL weight 0.2 |
| `ConceptFormer-Phi3V` | `Tevatron/dse-phi3-docmatix-v1` | PEFT adapter and latent state, forward KL weight 0.2 |
| `ConceptFormer-Trainset` | OpenDocVQA sources | 37,966 grounded query-document training rows |
| `ConceptFormer-Eval` | Six public benchmarks | Queries, qrels, manifests, and redistribution-safe corpora |

The release cards are under [`huggingface/`](huggingface). Model repositories intentionally
publish the trained PEFT adapters rather than duplicate their 16 GB and 7.8 GB public base
models. The loaders combine each adapter with the `base_model` declared in its model card.

## Environment

```bash
conda create -n conceptformer python=3.10 -y
conda activate conceptformer
pip install -e '.[eval]'
cp configs/paths.example.sh configs/paths.sh
```

FlashAttention is optional. Install the version compatible with the local CUDA and PyTorch
toolchain if memory-efficient attention is required.

## Project Layout

```text
ConceptFormer/
|-- configs/                 # paths and DeepSpeed configuration
|-- docs/                    # data, model, annotation, and reproduction notes
|-- huggingface/             # ready-to-upload dataset and model repositories
|-- scripts/                 # training, download, encoding, search, and evaluation
|-- src/conceptformer/
|   |-- data_construction/   # bounding-box annotation pipeline
|   `-- retriever/           # model, collator, trainer, encoding, and search
`-- tests/                   # unit tests for latent concepts and alignment losses
```

## Data

The paper evaluates each benchmark against **its own corpus**. The selected evaluation
suite is InfoVQA, ChartQA, SlideVQA, TQA, OWID Charts, and Wikimedia Maps.

```bash
bash scripts/download_data.sh hmhm1229 data
```

The training table references upstream images by document ID. `ConceptFormer-Eval` is a
self-contained evaluation release: every benchmark directory includes query/qrels files
and sharded corpus Parquet files with the original image bytes embedded. See
[`docs/DATA.md`](docs/DATA.md) for schemas, source URLs, counts, and licensing notes.

## Evidence Annotation

Bounding boxes are generated from an image-question pair with a JSON-only annotation
prompt. The exact prompt and API command are in
[`docs/ANNOTATION.md`](docs/ANNOTATION.md). API credentials are read only from environment
variables and are never stored in the repository.

## Training

Both released checkpoints use four GPUs, three epochs, per-device batch size 8, gradient
accumulation 4, bfloat16, LoRA rank 8, and forward latent-distribution KL weight 0.2.

```bash
# Qwen2.5-VL
GPU_IDS=0,1,2,3 bash scripts/train_qwen.sh

# Phi-3-Vision
GPU_IDS=0,1,2,3 bash scripts/train_phi3v.sh
```

The special-token vocabulary contains exactly one ConceptFormer token: `<|lcon|>`. A
sample receives a dynamic number of repeated tokens derived from its selected bounding-box
patches. No start/end boundary tokens are required.

## Loading an Adapter

The model repositories contain trained PEFT adapters. The ConceptFormer loader resolves
the public base model, resizes the tokenizer for `<|lcon|>`, and merges the adapter:

```python
import torch
from conceptformer.retriever.modeling import ConceptFormerRetriever

model = ConceptFormerRetriever.load(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    lora_name_or_path="hmhm1229/ConceptFormer-Qwen",
    pooling="eos",
    normalize=True,
    dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()
```

For Phi3V, use `Tevatron/dse-phi3-docmatix-v1` with
`hmhm1229/ConceptFormer-Phi3V`.

## Evaluation

Download a checkpoint, encode each benchmark's corpus and queries, retrieve within that
corpus, and compute Recall@10 and NDCG@10:

```bash
bash scripts/download_models.sh hmhm1229 models
bash scripts/evaluate.sh \
  models/ConceptFormer-Qwen \
  Qwen/Qwen2.5-VL-7B-Instruct \
  0,1,2,3 \
  infovqa,chartqa,slidevqa,tqa,owid_charts_en,wikimedia-commons-maps
```

Expected macro averages on the six benchmarks are:

| Model | Recall@10 | NDCG@10 |
|---|---:|---:|
| ConceptFormer (Phi3V) | 80.08 | 67.00 |
| ConceptFormer (Qwen) | **88.33** | **75.97** |

Per-dataset results and the exact protocol are in
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Reproducibility

- Seeds are fixed through the Transformers training argument `--seed`.
- Dataset rows and qrels are versioned by SHA-256 in the release manifests.
- Evaluation is corpus-local; datasets are never pooled together.
- Checkpoint cards record the base model, adapter initialization, token convention, and
  training objective.

## Acknowledgements

The release layout follows the practical organization of
[OpenBMB/MoRE](https://github.com/OpenBMB/MoRE). ConceptFormer builds on Hugging Face
Transformers, PEFT, Qwen2.5-VL, Phi-3-Vision, OpenDocVQA, and Jina VDR datasets. Please cite
the corresponding upstream work when using these resources.

## Citation

```bibtex
@article{conceptformer2026,
  title   = {ConceptFormer: Latent Concept Learning for Visual Document Retrieval},
  author  = {Anonymous},
  journal = {arXiv preprint},
  year    = {2026}
}
```

Replace the placeholder citation and contact information before the public release.
