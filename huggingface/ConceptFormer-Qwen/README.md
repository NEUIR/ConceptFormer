---
library_name: peft
base_model: Qwen/Qwen2.5-VL-7B-Instruct
license: apache-2.0
pipeline_tag: image-feature-extraction
tags:
- visual-document-retrieval
- conceptformer
- qwen2.5-vl
---

# ConceptFormer-Qwen

PEFT/LoRA checkpoint for ConceptFormer based on `Qwen/Qwen2.5-VL-7B-Instruct`.
This repository contains the trained adapter and ConceptFormer sidecar, not a duplicate of
the approximately 16 GB base model. Loaders resolve the base model from the `base_model`
metadata and merge this adapter at load time.

## Configuration

- Latent concept token: `<|lcon|>`
- Dynamic latent concept length from grounded regions
- Mean latent pooling
- Forward ranking-distribution KL, weight 0.2
- Three epochs, bfloat16, LoRA rank 8 / alpha 64 / dropout 0.1

## Loading

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

The wrapper loads the adapter tokenizer before PEFT, resizes the base embedding table for
`<|lcon|>`, and merges the adapter. The base model is downloaded separately.

## Results

| Metric | InfoVQA | ChartQA | SlideVQA | TQA | OWID Charts | Wikimedia Maps | Average |
|---|---:|---:|---:|---:|---:|---:|---:|
| Recall@10 | 93.03 | 98.67 | 90.88 | 72.20 | 99.24 | 76.04 | 88.33 |
| NDCG@10 | 79.23 | 95.79 | 82.41 | 41.30 | 95.39 | 61.69 | 75.97 |

Use this adapter with the ConceptFormer code repository. `conceptformer_state.pt` stores
the latent projection used by the training objective. Evaluation encodes images and
queries separately and retrieves only inside each dataset corpus.

## Limitations

Performance depends on document rendering, image resizing, and corpus version. This model
is intended for retrieval research and should not be used as a factual QA system without
downstream validation.
