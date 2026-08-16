"""Sanity tests for the dynamic LVR-style latent concept extension."""
from __future__ import annotations

import os
import sys

import pytest
import torch
from types import SimpleNamespace
from torch import nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from conceptformer.retriever.latent_concepts import (  # noqa: E402
    LCON_SPECIAL_TOKENS,
    LatentTokenPooler,
    bbox_to_visual_token_indices_qwen,
    cache_align_info_nce_loss,
    compute_kl_alignment,
    compute_ranking_distribution,
    get_latent_mse_weight,
    latent_visual_alignment_loss,
    latent_mse_loss,
    masked_mean_pool_tokens,
    parse_bboxes,
    _bbox_to_grid_indices_qwen,
    _expected_visual_tokens_qwen,
)
from conceptformer.retriever.modeling.conceptformer import (  # noqa: E402
    LatentConceptConfig,
    ConceptFormerRetriever,
    _flatten_image_features,
)


def _make_reps(b=4, n=4, h=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(b, h, generator=g)
    p = torch.randn(n, h, generator=g)
    c = torch.randn(b, h, generator=g)
    q.requires_grad_(True)
    c.requires_grad_(True)
    return q, p, c


def test_release_registers_exactly_one_project_token():
    assert LCON_SPECIAL_TOKENS == ["<|lcon|>"]


def _kl_components(q, p, c, mode):
    probs_q, log_probs_q = compute_ranking_distribution(q, p, temperature=0.5)
    probs_c, log_probs_c = compute_ranking_distribution(c, p, temperature=0.5)
    return compute_kl_alignment(log_probs_q, probs_q, log_probs_c, probs_c, mode)


def test_kl_forward_only_has_grad_on_c():
    q, p, c = _make_reps(seed=1)
    fwd, rev = _kl_components(q, p, c, mode="forward")
    assert fwd.item() > 0.0
    assert rev.item() == 0.0
    fwd.backward()
    assert q.grad is None or torch.allclose(q.grad, torch.zeros_like(q.grad))
    assert c.grad is not None and c.grad.abs().sum().item() > 0.0


def test_kl_reverse_only_has_grad_on_q():
    q, p, c = _make_reps(seed=2)
    fwd, rev = _kl_components(q, p, c, mode="reverse")
    assert fwd.item() == 0.0
    assert rev.item() > 0.0
    rev.backward()
    assert q.grad is not None and q.grad.abs().sum().item() > 0.0
    assert c.grad is None or torch.allclose(c.grad, torch.zeros_like(c.grad))


def test_kl_both_has_grad_on_both():
    q, p, c = _make_reps(seed=3)
    fwd, rev = _kl_components(q, p, c, mode="both")
    (fwd + rev).backward()
    assert q.grad is not None and q.grad.abs().sum().item() > 0.0
    assert c.grad is not None and c.grad.abs().sum().item() > 0.0


@pytest.mark.parametrize("mode", ["mean", "last", "attention"])
def test_pooler_shapes(mode):
    pooler = LatentTokenPooler(mode=mode, hidden_size=16)
    c = torch.randn(3, 5, 16)
    out = pooler(c)
    assert out.shape == (3, 16)


def test_pooler_handles_empty_dynamic_tokens():
    pooler = LatentTokenPooler(mode="mean")
    c = torch.zeros(2, 0, 8)
    out = pooler(c)
    assert out.shape == (2, 8)
    assert torch.equal(out, torch.zeros_like(out))


def test_latent_mse_loss_masks_padding_and_detaches_target():
    pred = torch.zeros(2, 3, 4, requires_grad=True)
    target = torch.ones(2, 3, 4, requires_grad=True)
    mask = torch.tensor([[True, False, False], [True, True, False]])
    loss = latent_mse_loss(pred, target, mask)
    assert torch.isclose(loss, torch.tensor(1.0))
    loss.backward()
    assert pred.grad is not None and pred.grad.abs().sum().item() > 0.0
    assert target.grad is None


def test_latent_mse_loss_all_padding_is_zero_with_grad_path():
    pred = torch.randn(2, 0, 4, requires_grad=True)
    target = torch.randn(2, 0, 4)
    loss = latent_mse_loss(pred, target, torch.zeros(2, 0, dtype=torch.bool))
    assert loss.item() == 0.0
    loss.backward()
    assert pred.grad is not None


def test_latent_visual_alignment_loss_cosine_repa_style():
    pred = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], requires_grad=True)
    target = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]], requires_grad=True)
    mask = torch.tensor([[True, True]])
    loss = latent_visual_alignment_loss(pred, target, mask, loss_type="cosine")
    assert torch.isclose(loss, torch.tensor(0.5))
    loss.backward()
    assert pred.grad is not None and pred.grad.abs().sum().item() > 0.0
    assert target.grad is None


def test_latent_visual_alignment_loss_cosine_ignores_padding():
    pred = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], requires_grad=True)
    target = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]], requires_grad=True)
    mask = torch.tensor([[True, False]])
    loss = latent_visual_alignment_loss(pred, target, mask, loss_type="cosine")
    assert torch.isclose(loss, torch.tensor(0.0))
    loss.backward()
    assert pred.grad is not None
    assert pred.grad[0, 1].abs().sum().item() == 0.0
    assert target.grad is None


def test_masked_mean_pool_tokens_skips_empty_roi_samples():
    tokens = torch.tensor([
        [[1.0, 3.0], [3.0, 5.0]],
        [[10.0, 10.0], [20.0, 20.0]],
    ])
    mask = torch.tensor([[True, True], [False, False]])
    pooled, valid = masked_mean_pool_tokens(tokens, mask, detach=True)
    assert torch.equal(valid, torch.tensor([True, False]))
    assert torch.allclose(pooled[0], torch.tensor([2.0, 4.0]))
    assert torch.equal(pooled[1], torch.zeros(2))


def test_cache_align_info_nce_zero_when_no_valid_pairs_has_grad_path():
    z = torch.randn(2, 4, requires_grad=True)
    v = torch.randn(2, 4)
    loss = cache_align_info_nce_loss(z, v, torch.zeros(2, dtype=torch.bool))
    assert loss.item() == 0.0
    loss.backward()
    assert z.grad is not None


def test_cache_align_info_nce_symmetric_is_finite():
    z = torch.eye(3, requires_grad=True)
    v = torch.eye(3)
    loss = cache_align_info_nce_loss(z, v, tau=0.07, symmetric=True)
    assert torch.isfinite(loss)
    loss.backward()
    assert z.grad is not None


def test_bbox_to_grid_indices_simple():
    idx = _bbox_to_grid_indices_qwen([0, 0, 28, 28], h_bar=56, w_bar=56)
    assert idx == [0]
    idx_full = _bbox_to_grid_indices_qwen([0, 0, 56, 56], h_bar=56, w_bar=56)
    assert sorted(idx_full) == [0, 1, 2, 3]


def test_bbox_to_visual_token_indices_qwen_scales_and_concats():
    grid = torch.tensor([[1, 4, 4]])
    bboxes = [
        [
            [0, 0, 50, 50],      # original top-left quarter -> token 0
            [0, 0, 50, 50],      # repeated intentionally
            [50, 0, 100, 50],    # original top-right quarter -> token 1
        ],
    ]
    out = bbox_to_visual_token_indices_qwen(
        image_grid_thw=grid,
        bboxes=bboxes,
        bbox_image_sizes=torch.tensor([[100.0, 100.0]]),
    )
    assert len(out) == 1
    assert out[0].tolist() == [0, 0, 1]


def test_gather_qwen_concept_targets_uses_dynamic_indices():
    method = ConceptFormerRetriever._gather_qwen_concept_targets.__get__(object(), object)
    grid = torch.tensor([[1, 4, 4], [1, 6, 4]])
    n_a = _expected_visual_tokens_qwen(1, 4, 4)
    n_b = _expected_visual_tokens_qwen(1, 6, 4)
    projected = torch.arange((n_a + n_b) * 8, dtype=torch.float32).reshape(n_a + n_b, 8)
    indices = torch.tensor([[0, -1, -1], [0, 1, 5]])
    target, mask = method(
        projected_visual_tokens=projected,
        image_grid_thw=grid,
        visual_token_indices=indices,
    )
    assert target.shape == (2, 3, 8)
    assert mask.tolist() == [[True, False, False], [True, True, True]]
    assert torch.allclose(target[0, 0], projected[0])
    assert torch.allclose(target[1, 2], projected[n_a + 5])


def test_gather_lcon_prediction_tokens_uses_lcon_position():
    method = ConceptFormerRetriever._gather_lcon_prediction_tokens.__get__(object(), object)
    last_hidden = torch.arange(1 * 6 * 2, dtype=torch.float32).reshape(1, 6, 2)
    lcon_positions = torch.tensor([[3, 5, -1]])
    out = method(last_hidden, lcon_positions=lcon_positions, input_ids=None)
    assert torch.equal(out[0, 0], last_hidden[0, 3])
    assert torch.equal(out[0, 1], last_hidden[0, 5])
    assert torch.equal(out[0, 2], torch.zeros(2))


def test_gather_single_position_tokens_uses_last_lcon_position():
    method = ConceptFormerRetriever._gather_single_position_tokens.__get__(object(), object)
    last_hidden = torch.arange(2 * 6 * 2, dtype=torch.float32).reshape(2, 6, 2)
    positions = torch.tensor([4, -1])
    out, mask = method(
        last_hidden,
        positions=positions,
        input_ids=None,
        token_id=None,
        token_name="<|lcon|>",
    )
    assert torch.equal(out[0], last_hidden[0, 4])
    assert torch.equal(out[1], torch.zeros(2))
    assert mask.tolist() == [True, False]


def test_gather_single_position_tokens_falls_back_to_last_token_id():
    method = ConceptFormerRetriever._gather_single_position_tokens.__get__(object(), object)
    last_hidden = torch.arange(1 * 7 * 2, dtype=torch.float32).reshape(1, 7, 2)
    input_ids = torch.tensor([[10, 99, 11, 99, 12, 0, 0]])
    out, mask = method(
        last_hidden,
        positions=None,
        input_ids=input_ids,
        token_id=99,
        token_name="<|lcon|>",
    )
    assert torch.equal(out[0], last_hidden[0, 3])
    assert mask.tolist() == [True]


class _TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=2, image_token_id=999, pad_token_id=0)
        self.emb = nn.Embedding(128, 2)

    def get_input_embeddings(self):
        return self.emb

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        return SimpleNamespace(hidden_states=(inputs_embeds.cumsum(dim=1),))


def test_recurrent_rollout_feeds_previous_hidden_state_and_keeps_grad():
    encoder = _TinyEncoder()
    with torch.no_grad():
        encoder.emb.weight.zero_()
        encoder.emb.weight[10] = torch.tensor([1.0, 0.0])
        encoder.emb.weight[20] = torch.tensor([0.0, 2.0])
        encoder.emb.weight[30] = torch.tensor([0.0, 0.5])
    model = ConceptFormerRetriever(
        encoder=encoder,
        latent_cfg=LatentConceptConfig(cache_align_weight=0.2, cache_steps=2),
        lcon_token_id=20,
        image_pad_token_id=999,
    )
    latent = {
        "input_ids": torch.tensor([[10, 20, 20, 30]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
    }
    states, mask = model._rollout_recurrent_concept_cache(
        latent=latent,
        projected_visual_tokens=torch.zeros(0, 2),
        lcon_positions=torch.tensor([[1, 2]]),
        use_cache=False,
    )
    h1 = torch.tensor([1.0, 2.0])
    h2 = torch.tensor([3.0, 4.0])
    assert torch.allclose(states[0, 0], h1)
    assert torch.allclose(states[0, 1], h2)
    assert mask.tolist() == [[True, True]]
    states.sum().backward()
    assert encoder.emb.weight.grad is not None
    assert encoder.emb.weight.grad[20].abs().sum().item() > 0.0


def test_pool_recurrent_cache_mean_and_last():
    model = ConceptFormerRetriever(
        encoder=_TinyEncoder(),
        latent_cfg=LatentConceptConfig(cache_align_weight=0.2, cache_steps=3, cache_pool="mean"),
    )
    states = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [9.0, 9.0]]])
    mask = torch.tensor([[True, True, False]])
    assert torch.allclose(model._pool_recurrent_cache(states, mask), torch.tensor([[2.0, 2.0]]))
    model.latent_cfg.cache_pool = "last"
    assert torch.allclose(model._pool_recurrent_cache(states, mask), torch.tensor([[3.0, 3.0]]))


class _PoolerOutput:
    def __init__(self, tensor):
        self.pooler_output = tensor


def test_flatten_image_features_handles_tuple_like_lvr():
    a = torch.randn(3, 16)
    b = torch.randn(5, 16)
    out = _flatten_image_features((a, b))
    assert out.shape == (8, 16)
    assert torch.equal(out[:3], a)
    assert torch.equal(out[3:], b)


def test_flatten_image_features_handles_pooler_output():
    inner = torch.randn(4, 16)
    out = _flatten_image_features(_PoolerOutput(inner))
    assert torch.equal(out, inner)


def test_parse_bboxes_handles_common_shapes():
    assert parse_bboxes('[[10, 20, 30, 40]]') == [[10, 20, 30, 40]]
    assert parse_bboxes([{"bbox_2d": [1, 2, 3, 4]}]) == [[1, 2, 3, 4]]
    assert parse_bboxes("not-json") == []


def test_data_arguments_default_is_disabled_and_alias_is_supported():
    from conceptformer.retriever.arguments import DataArguments

    args = DataArguments()
    assert args.latent_align_mode == "none"
    assert args.latent_kl_variant == "q2concept"
    assert args.latent_mse_weight == 0.0
    assert args.concept_cache_align_weight == 0.0
    assert args.concept_cache_steps == 8
    assert args.concept_cache_pool == "mean"
    assert get_latent_mse_weight(args) == 0.0

    args.latent_gamma_roi_mse = 0.25
    assert get_latent_mse_weight(args) == 0.25
    args.latent_mse_weight = 0.5
    assert get_latent_mse_weight(args) == 0.5
