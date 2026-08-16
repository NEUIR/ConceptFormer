# Reproducibility

## Protocol

For every benchmark, encode all query strings and all images in that benchmark's own
corpus. L2-normalize embeddings, retrieve by inner product, and evaluate the top 10. We
report binary Recall@10 and NDCG@10, macro-averaged across queries. The final Average
column is the unweighted mean across the six datasets.

## Expected Results

| Model | Metric | InfoVQA | ChartQA | SlideVQA | TQA | OWID Charts | Wikimedia Maps | Avg. |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Phi3V | R@10 | 91.70 | 95.33 | 88.73 | 68.80 | 99.24 | 36.70 | 80.08 |
| Phi3V | N@10 | 76.41 | 88.66 | 78.56 | 39.53 | 92.98 | 25.89 | 67.00 |
| Qwen | R@10 | 93.03 | 98.67 | 90.88 | 72.20 | 99.24 | 76.04 | 88.33 |
| Qwen | N@10 | 79.23 | 95.79 | 82.41 | 41.30 | 95.39 | 61.69 | 75.97 |

Small numerical differences can arise from CUDA kernels, image decoder versions, or a
different Transformers image-resizing implementation. Use the pinned package versions,
the published query and qrels files, and the manifest checksums when comparing runs.

## Training Configuration

The exact commands are `scripts/train_qwen.sh` and `scripts/train_phi3v.sh`. Both use seed
42, 3 epochs, learning rate `1e-4`, LoRA rank 8, alpha 64, dropout 0.1, batch size 8 per
GPU, gradient accumulation 4, and four GPUs. The released setting uses dynamic latent
concept lengths, mean pooling, `concept2image` forward KL, and `lambda=0.2`.
