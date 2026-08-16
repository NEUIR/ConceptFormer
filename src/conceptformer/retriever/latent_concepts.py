# -*- coding: utf-8 -*-
"""Latent concept helpers for ConceptFormer.

This module is intentionally additive: every helper here is only invoked when
the user explicitly opts in via ``latent_align_mode != 'none'`` or
``latent_mse_weight > 0``. With the default arguments the contrastive
retrieval path stays unchanged.

The latent concept branch introduces one repeatable special token:

    <|lcon|>

A latent input sequence has the LVR-style dynamic form:

    image + query_text + <|lcon|> * num_bbox_patches

where ``num_bbox_patches`` is the number of Qwen visual tokens selected by the
sample's bounding boxes. Multiple boxes are concatenated in box order.

Following the LCON cache objective, the input embeddings at ``<|lcon|>``
positions are teacher-forced with the selected original-image visual patch
embeddings, and the hidden states at those ``<|lcon|>`` positions are trained
to align with those embeddings:

    C = hidden_state[position(<|lcon|>)].

Pooling C -> E_c gives a cache/document-side representation used to compute a
latent ranking distribution. The default q2concept variant computes it for the
same query:

    P_c = softmax(q_reps @ E_c^T / temperature)

The older concept2image variant instead computes:

    P_c = softmax(E_c @ p_reps^T / temperature)

Both variants are aligned with the query-to-image distribution P_q via KL(s).
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Special tokens
# ---------------------------------------------------------------------------
LCON_TOKEN = "<|lcon|>"
LCON_SPECIAL_TOKENS = [LCON_TOKEN]


def get_latent_mse_weight(args) -> float:
    """Return the LCON visual-embedding MSE weight.

    ``latent_gamma_roi_mse`` is kept as a deprecated alias so old shell scripts
    fail soft while we migrate their naming away from ROI terminology.
    """
    new_weight = float(getattr(args, 'latent_mse_weight', 0.0) or 0.0)
    old_weight = float(getattr(args, 'latent_gamma_roi_mse', 0.0) or 0.0)
    return new_weight if new_weight > 0.0 else old_weight


def get_concept_cache_align_weight(args) -> float:
    """Return the recurrent latent concept cache-to-ROI InfoNCE weight."""
    return float(getattr(args, 'concept_cache_align_weight', 0.0) or 0.0)


# ---------------------------------------------------------------------------
# Bounding box helpers
# ---------------------------------------------------------------------------
def parse_bboxes(raw) -> List[List[float]]:
    """Best-effort parsing of a heterogeneous bbox cell.

    Accepts: list[list], JSON string, numpy array, dict with ``area`` keys,
    or ``None``. Returns a possibly empty list of [x1, y1, x2, y2] floats.
    """
    if raw is None:
        return []
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode('utf-8', errors='ignore')
    if isinstance(raw, str):
        s = raw.strip()
        if not s or s.lower() in {'nan', 'none', 'null', '[]'}:
            return []
        try:
            import json as _json
            raw = _json.loads(s)
        except Exception:
            try:
                import ast
                raw = ast.literal_eval(s)
            except Exception:
                return []

    try:
        import numpy as _np
        if isinstance(raw, _np.ndarray):
            raw = raw.tolist()
    except Exception:
        pass

    out: List[List[float]] = []
    if isinstance(raw, dict):
        for key in ('boxes', 'regions', 'bbox', 'bboxes', 'bbox_2d', 'area', 'areas'):
            if key in raw:
                return parse_bboxes(raw[key])
        return []
    if not isinstance(raw, (list, tuple)):
        return []

    for item in raw:
        if isinstance(item, dict):
            xyxy = (
                item.get('area')
                or item.get('bbox')
                or item.get('bbox_2d')
                or item.get('box')
                or item.get('xyxy')
            )
        else:
            xyxy = item
        if xyxy is None:
            continue
        try:
            xyxy = [float(v) for v in xyxy]
        except Exception:
            continue
        if len(xyxy) != 4:
            continue
        out.append(xyxy)
    return out


# ---------------------------------------------------------------------------
# Generic ranking-distribution helper
# ---------------------------------------------------------------------------
def compute_ranking_distribution(
    anchor_emb: torch.Tensor,
    doc_embs: torch.Tensor,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the in-batch ranking distribution induced by ``anchor_emb``."""
    if anchor_emb.dim() != 2 or doc_embs.dim() != 2:
        raise ValueError(
            f"compute_ranking_distribution expects 2D anchor/doc tensors, "
            f"got anchor.dim={anchor_emb.dim()} doc.dim={doc_embs.dim()}"
        )
    scores = torch.matmul(anchor_emb, doc_embs.transpose(0, 1)) / max(float(temperature), 1e-12)
    probs = F.softmax(scores, dim=-1)
    log_probs = F.log_softmax(scores, dim=-1)
    return probs, log_probs


def _kl_div(log_q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Stable KL(p || q) using F.kl_div which expects log_q + p."""
    return F.kl_div(log_q, p, reduction='batchmean')


def compute_kl_alignment(
    log_probs_q: torch.Tensor,
    probs_q: torch.Tensor,
    log_probs_c: torch.Tensor,
    probs_c: torch.Tensor,
    mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute (loss_kl_forward, loss_kl_reverse) for the requested mode.

    - forward: KL(sg[P_q] || P_c), implemented as F.kl_div(log_p_c, p_q.detach()).
    - reverse: KL(sg[P_c] || P_q), implemented as F.kl_div(log_p_q, p_c.detach()).
    """
    device = log_probs_q.device
    zero = torch.zeros((), device=device)
    if mode == 'none':
        return zero, zero
    if mode == 'forward':
        return _kl_div(log_probs_c, probs_q.detach()), zero
    if mode == 'reverse':
        return zero, _kl_div(log_probs_q, probs_c.detach())
    if mode == 'both':
        loss_forward = _kl_div(log_probs_c, probs_q.detach())
        loss_reverse = _kl_div(log_probs_q, probs_c.detach())
        return loss_forward, loss_reverse
    raise ValueError(f"Unknown latent_align_mode: {mode!r}")


# ---------------------------------------------------------------------------
# Latent token pooling
# ---------------------------------------------------------------------------
class LatentTokenPooler(nn.Module):
    """Pool [B, M, H] latent tokens -> [B, H] using the configured strategy."""

    def __init__(self, mode: str = 'mean', hidden_size: Optional[int] = None):
        super().__init__()
        if mode not in {'mean', 'last', 'attention', 'end'}:
            raise ValueError(f"latent_pooling must be one of mean/last/attention/end, got {mode!r}")
        self.mode = mode
        if mode == 'attention':
            assert hidden_size is not None, "hidden_size is required for attention pooling"
            self.attn = nn.Linear(hidden_size, 1, bias=False)
        else:
            self.attn = None

    def forward(self, c: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if c.dim() != 3:
            raise ValueError(f"LatentTokenPooler expects [B, M, H], got shape {tuple(c.shape)}")
        if c.size(1) == 0:
            return c.new_zeros(c.size(0), c.size(2))
        if self.mode == 'end':
            raise RuntimeError(
                "end pooling is gathered from the final <|lcon|> sequence position "
                "inside ConceptFormerRetriever, not from the dynamic token matrix."
            )
        if self.mode == 'last':
            if mask is None:
                return c[:, -1, :]
            lengths = mask.to(dtype=torch.long).sum(dim=1).clamp(min=1) - 1
            return c[torch.arange(c.size(0), device=c.device), lengths]
        if mask is None:
            mask = torch.ones(c.shape[:2], device=c.device, dtype=c.dtype)
        else:
            mask = mask.to(dtype=c.dtype)
        if self.mode == 'mean':
            num = (c * mask.unsqueeze(-1)).sum(dim=1)
            den = mask.sum(dim=1, keepdim=True).clamp(min=1e-6)
            return num / den
        logits = self.attn(c).squeeze(-1)
        logits = logits.masked_fill(mask <= 0, float('-inf'))
        weights = torch.softmax(logits, dim=-1)
        weights = torch.where(torch.isfinite(weights), weights, torch.zeros_like(weights))
        return (c * weights.unsqueeze(-1)).sum(dim=1)


def latent_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    token_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """MSE between predicted LCON states and selected visual patch embeddings.

    pred/target: [B, M, H], where M is the dynamic maximum number of bbox visual
        tokens in the current batch.
    token_mask: [B, M], True for real LCON tokens and False for padding.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"latent_mse_loss expects same shape, got pred={tuple(pred.shape)} "
            f"target={tuple(target.shape)}"
        )
    if pred.dim() != 3:
        raise ValueError(f"latent_mse_loss expects [B, M, H], got {tuple(pred.shape)}")

    if token_mask is None:
        token_mask = torch.ones(pred.shape[:2], device=pred.device, dtype=torch.bool)
    token_mask = token_mask.to(device=pred.device, dtype=torch.bool)
    if token_mask.shape != pred.shape[:2]:
        raise ValueError(
            f"token_mask shape {tuple(token_mask.shape)} does not match pred[:2] "
            f"{tuple(pred.shape[:2])}"
        )

    if int(token_mask.sum().item()) == 0:
        return pred.sum() * 0.0

    pred_valid = pred[token_mask].float()
    target_valid = target.detach()[token_mask].float()
    return (pred_valid - target_valid).pow(2).mean(dim=-1).mean()


def latent_visual_alignment_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    token_mask: Optional[torch.Tensor],
    loss_type: str = "mse",
) -> torch.Tensor:
    """Tokenwise LCON-to-visual alignment loss.

    ``cosine`` follows REPA's representation-alignment form: L2-normalise both
    sides and maximise their dot product. We use ``1 - cos`` so the reported
    scalar is non-negative; this has the same gradient as REPA's ``-cos`` up
    to an additive constant.
    """
    loss_type = (loss_type or "mse").lower()
    if loss_type in {"mse", "l2"}:
        return latent_mse_loss(pred, target, token_mask)

    if pred.shape != target.shape:
        raise ValueError(
            f"latent_visual_alignment_loss expects same shape, got pred={tuple(pred.shape)} "
            f"target={tuple(target.shape)}"
        )
    if pred.dim() != 3:
        raise ValueError(
            f"latent_visual_alignment_loss expects [B, M, H], got {tuple(pred.shape)}"
        )

    if token_mask is None:
        token_mask = torch.ones(pred.shape[:2], device=pred.device, dtype=torch.bool)
    token_mask = token_mask.to(device=pred.device, dtype=torch.bool)
    if token_mask.shape != pred.shape[:2]:
        raise ValueError(
            f"token_mask shape {tuple(token_mask.shape)} does not match pred[:2] "
            f"{tuple(pred.shape[:2])}"
        )
    if int(token_mask.sum().item()) == 0:
        return pred.sum() * 0.0

    if loss_type in {"cos", "cosine", "repa"}:
        pred_valid = pred[token_mask].float()
        target_valid = target.detach()[token_mask].float()
        pred_norm = F.normalize(pred_valid, p=2, dim=-1)
        target_norm = F.normalize(target_valid, p=2, dim=-1)
        return (1.0 - (pred_norm * target_norm).sum(dim=-1)).mean()

    raise ValueError(f"Unknown latent_visual_loss_type: {loss_type!r}")


def masked_mean_pool_tokens(
    tokens: torch.Tensor,
    token_mask: Optional[torch.Tensor],
    detach: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Masked mean-pool variable ROI token targets per sample.

    Returns:
        pooled: [B, H] with zeros for samples that have no valid tokens.
        valid_mask: [B] True when a sample has at least one valid token.
    """
    if tokens.dim() != 3:
        raise ValueError(f"masked_mean_pool_tokens expects [B, R, H], got {tuple(tokens.shape)}")
    if token_mask is None:
        token_mask = torch.ones(tokens.shape[:2], device=tokens.device, dtype=torch.bool)
    token_mask = token_mask.to(device=tokens.device, dtype=torch.bool)
    if token_mask.shape != tokens.shape[:2]:
        raise ValueError(
            f"token_mask shape {tuple(token_mask.shape)} does not match tokens[:2] "
            f"{tuple(tokens.shape[:2])}"
        )
    values = tokens.detach() if detach else tokens
    weights = token_mask.to(dtype=values.dtype)
    denom = weights.sum(dim=1, keepdim=True)
    valid = denom.squeeze(1) > 0
    pooled = (values * weights.unsqueeze(-1)).sum(dim=1) / denom.clamp(min=1.0)
    pooled = pooled * valid.unsqueeze(-1).to(dtype=pooled.dtype)
    return pooled, valid


def cache_align_info_nce_loss(
    projected_cache: torch.Tensor,
    visual_targets: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    tau: float = 0.07,
    symmetric: bool = False,
) -> torch.Tensor:
    """In-batch InfoNCE between projected recurrent cache and ROI visual targets."""
    if projected_cache.shape != visual_targets.shape:
        raise ValueError(
            f"cache_align_info_nce_loss expects same shape, got "
            f"projected_cache={tuple(projected_cache.shape)} visual_targets={tuple(visual_targets.shape)}"
        )
    if projected_cache.dim() != 2:
        raise ValueError(
            f"cache_align_info_nce_loss expects [B, H], got {tuple(projected_cache.shape)}"
        )
    if valid_mask is None:
        valid_mask = torch.ones(projected_cache.size(0), device=projected_cache.device, dtype=torch.bool)
    valid_mask = valid_mask.to(device=projected_cache.device, dtype=torch.bool)
    if valid_mask.shape != projected_cache.shape[:1]:
        raise ValueError(
            f"valid_mask shape {tuple(valid_mask.shape)} does not match batch {projected_cache.size(0)}"
        )
    if int(valid_mask.sum().item()) == 0:
        return projected_cache.sum() * 0.0

    z = F.normalize(projected_cache[valid_mask].float(), p=2, dim=-1)
    v = F.normalize(visual_targets[valid_mask].float(), p=2, dim=-1)
    logits = torch.matmul(z, v.transpose(0, 1)) / max(float(tau), 1e-12)
    labels = torch.arange(logits.size(0), device=logits.device, dtype=torch.long)
    loss = F.cross_entropy(logits, labels)
    if symmetric:
        loss = 0.5 * (loss + F.cross_entropy(logits.transpose(0, 1), labels))
    return loss


# ---------------------------------------------------------------------------
# Qwen bbox -> visual-token index helpers
# ---------------------------------------------------------------------------
def _expected_visual_tokens_qwen(t: int, h: int, w: int, merge_size: int = 2) -> int:
    return int(t * (h // merge_size) * (w // merge_size))


def _bbox_to_grid_indices_qwen(
    bbox_xyxy: Sequence[float],
    h_bar: int,
    w_bar: int,
    merge_size: int = 2,
    patch_size: int = 14,
) -> List[int]:
    """Convert one bbox in resized-image pixels to merged-grid token indices."""
    if h_bar <= 0 or w_bar <= 0:
        return []
    cell = patch_size * merge_size
    grid_h = max(1, h_bar // cell)
    grid_w = max(1, w_bar // cell)
    x1, y1, x2, y2 = bbox_xyxy
    if x2 <= x1 or y2 <= y1:
        return []
    cx1 = max(0, min(grid_w - 1, int(math.floor(x1 / cell))))
    cy1 = max(0, min(grid_h - 1, int(math.floor(y1 / cell))))
    eps = 1e-6
    cx2 = max(0, min(grid_w - 1, int(math.ceil((x2 - eps) / cell)) - 1))
    cy2 = max(0, min(grid_h - 1, int(math.ceil((y2 - eps) / cell)) - 1))
    if cx2 < cx1 or cy2 < cy1:
        return []
    out: List[int] = []
    for r in range(cy1, cy2 + 1):
        base = r * grid_w
        for c in range(cx1, cx2 + 1):
            out.append(base + c)
    return out


def bbox_to_visual_token_indices_qwen(
    image_grid_thw: torch.Tensor,
    bboxes: List[List[List[float]]],
    bbox_image_sizes: Optional[torch.Tensor] = None,
    merge_size: int = 2,
    patch_size: int = 14,
) -> List[torch.Tensor]:
    """Map per-sample bboxes to Qwen visual-token indices without resampling.

    The returned list has length B. Each tensor contains local indices into
    that sample's flattened Qwen visual-token sequence. Multiple bboxes are
    concatenated directly in input order, including duplicates for overlap.
    """
    if image_grid_thw.dim() != 2 or image_grid_thw.size(1) != 3:
        raise ValueError(
            f"image_grid_thw must be [B, 3], got {tuple(image_grid_thw.shape)}"
        )
    bsz = image_grid_thw.size(0)
    if len(bboxes) != bsz:
        raise ValueError(f"bboxes length {len(bboxes)} != batch size {bsz}")

    all_indices: List[torch.Tensor] = []
    for i in range(bsz):
        t, h, w = [int(v) for v in image_grid_thw[i].tolist()]
        n_tokens = _expected_visual_tokens_qwen(t, h, w, merge_size=merge_size)
        h_bar = h * patch_size
        w_bar = w * patch_size

        scale_x = 1.0
        scale_y = 1.0
        if bbox_image_sizes is not None:
            src_w = float(
                bbox_image_sizes[i][0].item()
                if hasattr(bbox_image_sizes[i][0], 'item')
                else bbox_image_sizes[i][0]
            )
            src_h = float(
                bbox_image_sizes[i][1].item()
                if hasattr(bbox_image_sizes[i][1], 'item')
                else bbox_image_sizes[i][1]
            )
            if src_w > 0 and src_h > 0:
                scale_x = float(w_bar) / src_w
                scale_y = float(h_bar) / src_h

        gathered_idx: List[int] = []
        for bb in bboxes[i]:
            if bbox_image_sizes is not None:
                bb = [
                    float(bb[0]) * scale_x,
                    float(bb[1]) * scale_y,
                    float(bb[2]) * scale_x,
                    float(bb[3]) * scale_y,
                ]
            gathered_idx.extend(_bbox_to_grid_indices_qwen(
                bb, h_bar=h_bar, w_bar=w_bar,
                merge_size=merge_size, patch_size=patch_size,
            ))

        concat_idx = [idx for idx in gathered_idx if 0 <= idx < n_tokens]
        all_indices.append(torch.as_tensor(concat_idx, dtype=torch.long))
    return all_indices


# ---------------------------------------------------------------------------
# Phi3V bbox -> dynamic LCON count helpers
# ---------------------------------------------------------------------------
def _phi3v_hd_transform_geometry(
    width: float,
    height: float,
    hd_num: int = 16,
    base_size: int = 336,
) -> Optional[dict]:
    """Mirror Phi3V HD_transform sizing and padding geometry."""
    if width <= 0 or height <= 0:
        return None
    transposed = width < height
    work_w, work_h = (height, width) if transposed else (width, height)
    ratio = work_w / work_h
    scale = 1
    while scale * math.ceil(scale / ratio) <= hd_num:
        scale += 1
    scale = max(1, scale - 1)

    new_w = int(scale * base_size)
    new_h = int(new_w / ratio)
    padded_h = int(math.ceil(new_h / base_size) * base_size)
    top_padding = (padded_h - new_h) // 2
    if transposed:
        final_w, final_h = padded_h, new_w
    else:
        final_w, final_h = new_w, padded_h
    return {
        'transposed': transposed,
        'new_w': new_w,
        'new_h': new_h,
        'top_padding': top_padding,
        'final_w': final_w,
        'final_h': final_h,
    }


def _transform_bbox_phi3v_hd(
    bbox_xyxy: Sequence[float],
    src_w: float,
    src_h: float,
    geom: dict,
) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
    xs: List[float] = []
    ys: List[float] = []
    for x, y in corners:
        if geom['transposed']:
            xt = y * float(geom['new_w']) / src_h
            yt = x * float(geom['new_h']) / src_w + float(geom['top_padding'])
            xf, yf = yt, xt
        else:
            xf = x * float(geom['new_w']) / src_w
            yf = y * float(geom['new_h']) / src_h + float(geom['top_padding'])
        xs.append(xf)
        ys.append(yf)
    return [min(xs), min(ys), max(xs), max(ys)]


def bbox_to_lcon_token_counts_phi3v(
    bboxes: List[List[List[float]]],
    bbox_image_sizes: torch.Tensor,
    num_crops: int = 16,
    merge_size: int = 2,
    patch_size: int = 14,
) -> List[int]:
    """Count Phi3V HD sub-image grid tokens covered by each sample's bboxes.

    Phi3V first applies the HD transform: optional image transpose, resize to
    336-crop layout, height padding to a 336 multiple, and optional transpose
    back. Its sub-image features are then 2x2-merged into a 28px cell grid.
    For KL-only no-vis runs we only need the dynamic number of LCON slots, so
    this deliberately returns counts rather than visual-token indices.
    """
    if bbox_image_sizes.dim() != 2 or bbox_image_sizes.size(1) != 2:
        raise ValueError(
            f"bbox_image_sizes must be [B, 2], got {tuple(bbox_image_sizes.shape)}"
        )
    if len(bboxes) != bbox_image_sizes.size(0):
        raise ValueError(f"bboxes length {len(bboxes)} != batch size {bbox_image_sizes.size(0)}")

    counts: List[int] = []
    for i, sample_boxes in enumerate(bboxes):
        if not sample_boxes:
            counts.append(0)
            continue
        src_w = float(bbox_image_sizes[i][0].item())
        src_h = float(bbox_image_sizes[i][1].item())
        geom = _phi3v_hd_transform_geometry(src_w, src_h, hd_num=num_crops)
        if geom is None:
            counts.append(0)
            continue

        h_bar = int(geom['final_h'])
        w_bar = int(geom['final_w'])
        max_tokens = max(0, (h_bar // (patch_size * merge_size)) * (w_bar // (patch_size * merge_size)))
        gathered_idx: List[int] = []
        for bb in sample_boxes:
            transformed = _transform_bbox_phi3v_hd(bb, src_w, src_h, geom)
            gathered_idx.extend(_bbox_to_grid_indices_qwen(
                transformed,
                h_bar=h_bar,
                w_bar=w_bar,
                merge_size=merge_size,
                patch_size=patch_size,
            ))
        counts.append(len([idx for idx in gathered_idx if 0 <= idx < max_tokens]))
    return counts
