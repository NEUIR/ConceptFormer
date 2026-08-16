---
library_name: peft
base_model: Tevatron/dse-phi3-docmatix-v1
license: apache-2.0
pipeline_tag: image-feature-extraction
tags:
- visual-document-retrieval
- conceptformer
- phi-3-vision
---

# ConceptFormer-Phi3V

PEFT/LoRA checkpoint for ConceptFormer based on `Tevatron/dse-phi3-docmatix-v1`, initialized
from the VDocRetriever Phi-3-Vision adapter. This repository contains the trained adapter
and ConceptFormer sidecar, not a duplicate of the approximately 7.8 GB base model. Loaders
resolve the base model from the `base_model` metadata and merge this adapter at load time.

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
    "Tevatron/dse-phi3-docmatix-v1",
    lora_name_or_path="hmhm1229/ConceptFormer-Phi3V",
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
| Recall@10 | 91.70 | 95.33 | 88.73 | 68.80 | 99.24 | 36.70 | 80.08 |
| NDCG@10 | 76.41 | 88.66 | 78.56 | 39.53 | 92.98 | 25.89 | 67.00 |

Use this adapter with the ConceptFormer code repository. `conceptformer_state.pt` stores
the latent projection used by the training objective.

## Limitations

Performance depends on document rendering, image resizing, and corpus version. This model
is intended for retrieval research and should not be used as a factual QA system without
downstream validation.
