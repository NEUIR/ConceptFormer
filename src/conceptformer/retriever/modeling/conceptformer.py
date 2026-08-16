from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn, Tensor
import torch.nn.functional as F

from transformers import (
    PreTrainedModel,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoTokenizer,
)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

from transformers.file_utils import ModelOutput
from conceptformer.retriever.arguments import (
    ModelArguments,
    ConceptFormerTrainingArguments as TrainingArguments,
    DataArguments,
)
from conceptformer.retriever.latent_concepts import (
    LCON_TOKEN,
    LatentTokenPooler,
    compute_ranking_distribution,
    compute_kl_alignment,
    get_latent_mse_weight,
    get_concept_cache_align_weight,
    latent_visual_alignment_loss,
    masked_mean_pool_tokens,
    cache_align_info_nce_loss,
    _expected_visual_tokens_qwen,
)

import logging
logger = logging.getLogger(__name__)


def _load_conceptformer_state(model: nn.Module, adapter_path: Optional[str]) -> None:
    """Load training-only latent projection weights when they are available."""
    if not adapter_path or getattr(model, 'latent_proj', None) is None:
        return
    state_path = None
    local = Path(adapter_path) / 'conceptformer_state.pt'
    if local.is_file():
        state_path = str(local)
    else:
        try:
            from huggingface_hub import hf_hub_download
            state_path = hf_hub_download(adapter_path, 'conceptformer_state.pt')
        except Exception:
            return
    state = torch.load(state_path, map_location='cpu', weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected ConceptFormer state keys: {unexpected}")
    logger.info("Loaded ConceptFormer sidecar from %s; missing wrapper keys=%d", state_path, len(missing))


@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None
    # Per-component diagnostics for the latent concept extension.
    loss_contrast: Optional[Tensor] = None
    loss_text_kl: Optional[Tensor] = None
    loss_kl_forward: Optional[Tensor] = None
    loss_kl_reverse: Optional[Tensor] = None
    loss_mse: Optional[Tensor] = None
    loss_cache_align: Optional[Tensor] = None


# Latent-concept configuration stored on the wrapper module.
@dataclass
class LatentConceptConfig:
    align_mode: str = 'none'             # none / forward / reverse / both
    lambda_forward: float = 0.0
    lambda_reverse: float = 0.0
    mse_weight: float = 0.0
    visual_loss_type: str = 'mse'
    kl_variant: str = 'q2concept'           # q2concept / concept2image / bbox2image / q2concept+concept2image
    pooling: str = 'mean'
    cache_align_weight: float = 0.0
    cache_steps: int = 8
    cache_pool: str = 'mean'
    recurrent_kl: bool = False
    recurrent_impl: str = 'exact'
    cache_align_tau: float = 0.07
    cache_align_symmetric: bool = False
    cache_align_detach_target: bool = True

    @property
    def latent_enabled(self) -> bool:
        return self.align_mode != 'none' or self.mse_weight > 0 or self.cache_align_weight > 0 or self.recurrent_kl

    @property
    def needs_kl(self) -> bool:
        return self.align_mode != 'none' and (self.lambda_forward > 0 or self.lambda_reverse > 0)


class ConceptFormerRetriever(nn.Module):
    TRANSFORMER_CLS = AutoModelForCausalLM

    def __init__(self,
                 encoder: PreTrainedModel,
                 pooling: str = 'cls',
                 normalize: bool = False,
                 temperature: float = 1.0,
                 kl_loss_weight: float = 1.0,
                 is_qwen: bool = False,
                 latent_cfg: Optional[LatentConceptConfig] = None,
                 image_pad_token_id: Optional[int] = None,
                 lcon_token_id: Optional[int] = None,
                 last_lcon_token_id: Optional[int] = None,
                 ):
        super().__init__()
        self.config = encoder.config
        self.encoder = encoder
        self.pooling = pooling
        self.normalize = normalize
        self.temperature = temperature
        self.kl_loss_weight = kl_loss_weight
        self.is_qwen = is_qwen
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.is_ddp = dist.is_initialized()
        if self.is_ddp:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

        # Latent-concept state.
        self.latent_cfg = latent_cfg or LatentConceptConfig()
        if self.latent_cfg.cache_align_weight > 0 and self.latent_cfg.mse_weight > 0:
            raise ValueError(
                "concept_cache_align_weight > 0 replaces latent_mse_weight; set latent_mse_weight=0."
            )
        self.image_pad_token_id = image_pad_token_id
        self.lcon_token_id = lcon_token_id
        # Last-token pooling points at the same single public token.
        self.last_lcon_token_id = lcon_token_id if last_lcon_token_id is None else last_lcon_token_id

        hidden_size = getattr(self.config, 'hidden_size', None)
        if self.latent_cfg.latent_enabled and hidden_size is not None:
            self.latent_pooler = LatentTokenPooler(self.latent_cfg.pooling, hidden_size=hidden_size)
            # Projection W_c applied to pooled latent tokens so they live in the
            # document-embedding space (E_c is used for latent retrieval scoring).
            self.latent_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            with torch.no_grad():
                nn.init.eye_(self.latent_proj.weight)
            # Cache-align now compares the pooled recurrent cache state directly
            # against the pooled ROI visual target; no extra projection head.
            self.concept_cache_align_proj = None
        else:
            self.latent_pooler = None
            self.latent_proj = None
            self.concept_cache_align_proj = None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, query: Dict[str, Tensor] = None,
                document: Dict[str, Tensor] = None,
                pair: Dict[str, Tensor] = None,
                query_describe: Dict[str, Tensor] = None,
                d_exist_ids: Tensor = None,
                latent: Dict[str, Tensor] = None,
                use_cache: bool = True
        ):
        q_reps = self.encode_query(query, use_cache=use_cache) if query else None
        p_reps = self.encode_document(document, use_cache=use_cache) if document else None
        outputs = self.generate_output(pair, use_cache=use_cache) if pair else None

        q_describe_reps = self.encode_query(query_describe, use_cache=use_cache) if query_describe is not None else None

        # Run the latent-concept branch only when explicitly enabled.
        latent_state = None
        if latent is not None and self.latent_cfg.latent_enabled:
            latent_state = self._encode_latent(latent, use_cache=use_cache)

        if q_reps is None or p_reps is None:
            return EncoderOutput(
                q_reps=q_reps,
                p_reps=p_reps
            )

        if self.training:
            if self.is_ddp:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)
                if q_describe_reps is not None:
                    q_describe_reps = self._dist_gather_tensor(q_describe_reps)
                if d_exist_ids is not None:
                    d_exist_ids = self._dist_gather_tensor(d_exist_ids)
                if latent_state is not None and latent_state.get('c_reps') is not None:
                    latent_state['c_reps'] = self._dist_gather_tensor(latent_state['c_reps'])
                if latent_state is not None and latent_state.get('c_valid_mask') is not None:
                    latent_state['c_valid_mask'] = self._dist_gather_tensor(
                        latent_state['c_valid_mask'].to(dtype=torch.long)
                    ).to(dtype=torch.bool)
                if latent_state is not None and latent_state.get('cache_c_pre') is not None:
                    latent_state['cache_c_pre'] = self._dist_gather_tensor(latent_state['cache_c_pre'])
                if latent_state is not None and latent_state.get('cache_visual_target') is not None:
                    latent_state['cache_visual_target'] = self._dist_gather_tensor(
                        latent_state['cache_visual_target']
                    )
                if latent_state is not None and latent_state.get('cache_visual_mask') is not None:
                    latent_state['cache_visual_mask'] = self._dist_gather_tensor(
                        latent_state['cache_visual_mask'].to(dtype=torch.long)
                    ).to(dtype=torch.bool)

            scores = self.compute_similarity(q_reps, p_reps)
            scores = scores.view(q_reps.size(0), -1)
            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            target = target * (p_reps.size(0) // q_reps.size(0))
            loss_contrast = self.compute_loss(scores / self.temperature, target)
            loss = loss_contrast

            if d_exist_ids is not None:
                image_mask = (d_exist_ids == 1)
            else:
                image_mask = torch.zeros(q_reps.size(0), dtype=torch.bool, device=q_reps.device)

            num_image = image_mask.sum().item()
            loss_text_kl = torch.zeros((), device=q_reps.device, dtype=q_reps.dtype)
            if num_image > 0 and q_describe_reps is not None:
                image_q = q_reps[image_mask]
                image_p = p_reps[image_mask]
                image_describe = q_describe_reps[image_mask]
                image_scores = self.compute_similarity(image_q, image_p)
                describe_scores = self.compute_similarity(image_describe, image_p)
                loss_text_kl = self.compute_kl_loss(
                    image_scores / self.temperature,
                    describe_scores / self.temperature
                )
                loss = loss + self.kl_loss_weight * loss_text_kl

            # ---- Latent Concept Alignment ----------------------------
            loss_kl_forward = torch.zeros((), device=q_reps.device, dtype=q_reps.dtype)
            loss_kl_reverse = torch.zeros((), device=q_reps.device, dtype=q_reps.dtype)
            loss_mse = torch.zeros((), device=q_reps.device, dtype=q_reps.dtype)
            loss_cache_align = torch.zeros((), device=q_reps.device, dtype=q_reps.dtype)

            if latent_state is not None and self.latent_cfg.needs_kl:
                c_reps = latent_state['c_reps']
                if c_reps is not None and c_reps.size(0) == q_reps.size(0):
                    c_valid_mask = latent_state.get('c_valid_mask')
                    if c_valid_mask is not None:
                        c_valid_mask = c_valid_mask.to(device=q_reps.device, dtype=torch.bool)
                    else:
                        c_valid_mask = torch.ones(q_reps.size(0), device=q_reps.device, dtype=torch.bool)
                    if bool(c_valid_mask.any().item()):
                        q_for_kl = q_reps[c_valid_mask]
                        p_for_kl = p_reps[c_valid_mask]
                        c_for_kl = c_reps[c_valid_mask]
                    else:
                        q_for_kl = None
                        p_for_kl = None
                        c_for_kl = None
                    if q_for_kl is not None:
                        # Variant q2concept: align q->image with q->latent concept.
                        # Variant concept2image: align q->image with latent concept->image
                        # (the earlier formulation).
                        probs_q, log_probs_q = compute_ranking_distribution(
                            q_for_kl, p_for_kl, temperature=self.temperature)
                        kl_variant = (self.latent_cfg.kl_variant or 'q2concept').lower()
                        if kl_variant == 'q2concept':
                            probs_c, log_probs_c = compute_ranking_distribution(
                                q_for_kl, c_for_kl, temperature=self.temperature)
                        elif kl_variant in {'concept2image', 'bbox2image', 'bbox_image2image', 'old'}:
                            probs_c, log_probs_c = compute_ranking_distribution(
                                c_for_kl, p_for_kl, temperature=self.temperature)
                        elif kl_variant in {
                            'q2concept+concept2image',
                            'q2concept_plus_concept2image',
                            'q2concept-concept2image',
                            'q2concept_concept2image',
                        }:
                            probs_q2c, log_probs_q2c = compute_ranking_distribution(
                                q_for_kl, c_for_kl, temperature=self.temperature)
                            probs_c2p, log_probs_c2p = compute_ranking_distribution(
                                c_for_kl, p_for_kl, temperature=self.temperature)
                            loss_q2c_forward, loss_q2c_reverse = compute_kl_alignment(
                                log_probs_q, probs_q, log_probs_q2c, probs_q2c,
                                mode=self.latent_cfg.align_mode,
                            )
                            loss_c2p_forward, loss_c2p_reverse = compute_kl_alignment(
                                log_probs_q, probs_q, log_probs_c2p, probs_c2p,
                                mode=self.latent_cfg.align_mode,
                            )
                            loss_kl_forward = loss_q2c_forward + loss_c2p_forward
                            loss_kl_reverse = loss_q2c_reverse + loss_c2p_reverse
                            loss = (
                                loss
                                + self.latent_cfg.lambda_forward * loss_kl_forward
                                + self.latent_cfg.lambda_reverse * loss_kl_reverse
                            )
                            probs_c = None
                        else:
                            raise ValueError(f"Unknown latent_kl_variant: {self.latent_cfg.kl_variant!r}")
                        if probs_c is not None:
                            loss_kl_forward, loss_kl_reverse = compute_kl_alignment(
                                log_probs_q, probs_q, log_probs_c, probs_c,
                                mode=self.latent_cfg.align_mode,
                            )
                            loss = (
                                loss
                                + self.latent_cfg.lambda_forward * loss_kl_forward
                                + self.latent_cfg.lambda_reverse * loss_kl_reverse
                            )

            if (latent_state is not None
                    and self.latent_cfg.mse_weight > 0
                    and latent_state.get('mse_target') is not None):
                loss_mse = latent_visual_alignment_loss(
                    latent_state['c_tokens'],
                    latent_state['mse_target'],
                    token_mask=latent_state.get('c_token_mask'),
                    loss_type=self.latent_cfg.visual_loss_type,
                )
                loss = loss + self.latent_cfg.mse_weight * loss_mse

            if (latent_state is not None
                    and self.latent_cfg.cache_align_weight > 0
                    and latent_state.get('cache_c_pre') is not None
                    and latent_state.get('cache_visual_target') is not None):
                loss_cache_align = self._compute_cache_align_loss(
                    latent_state['cache_c_pre'],
                    latent_state['cache_visual_target'],
                    latent_state.get('cache_visual_mask'),
                )
                loss = loss + self.latent_cfg.cache_align_weight * loss_cache_align

            if outputs:
                loss = loss + outputs.loss

            if self.is_ddp:
                loss = loss * self.world_size

            scores = self.compute_similarity(q_reps, p_reps)

        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None
            loss_contrast = None
            loss_text_kl = None
            loss_kl_forward = None
            loss_kl_reverse = None
            loss_mse = None
            loss_cache_align = None

        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
            loss_contrast=loss_contrast if self.training else None,
            loss_text_kl=loss_text_kl if self.training else None,
            loss_kl_forward=loss_kl_forward if self.training else None,
            loss_kl_reverse=loss_kl_reverse if self.training else None,
            loss_mse=loss_mse if self.training else None,
            loss_cache_align=loss_cache_align if self.training else None,
        )

    # ------------------------------------------------------------------
    # Latent Concept helpers
    # ------------------------------------------------------------------
    def _encode_latent(self, latent: Dict[str, Tensor], use_cache: bool = True) -> Dict[str, Tensor]:
        """Run a single LVR-style forward pass on image + query + dynamic LCON.

        Returns a dict with:
            c_tokens: [B, M, H]  - hidden states that predict <|lcon|> tokens
            c_reps:   [B, H]     - pooled + projected anchor for P_c
            mse_target: optional [B, M, H] detached bbox visual patch target
            c_token_mask: [B, M] mask for real dynamic LCON tokens

        Selected image patch embeddings are inserted into the input stream at
        ``<|lcon|>`` positions, and the visual alignment loss is taken on the
        hidden states at those same ``<|lcon|>`` positions.
        """
        # Move side-info out of the kwargs we send to the backbone.
        lcon_positions = latent.pop('lcon_positions') if 'lcon_positions' in latent else None
        last_lcon_positions = latent.pop('last_lcon_positions', None)
        visual_token_indices = latent.pop('visual_token_indices', None)
        visual_token_mask = latent.pop('visual_token_mask', None)
        lcon_token_count = latent.pop('lcon_token_count', None)
        # Retain bbox metadata only for debug/introspection; dynamic token
        # indices have already been computed by the collator.
        bbox = latent.pop('bbox', None)
        bbox_count = latent.pop('bbox_count', None)
        bbox_image_size = latent.pop('bbox_image_size', None)
        bbox_image_size_mask = latent.pop('bbox_image_size_mask', None)

        # ---- Step 1: LVR-style projected visual tokens (frozen teacher) ----
        projected_visual_tokens = None
        mse_target = None
        visual_target_mask = visual_token_mask
        use_recurrent_rollout = bool(self.latent_cfg.recurrent_kl or self.latent_cfg.cache_align_weight > 0)
        needs_projected_visual_tokens = self.is_qwen and (
            visual_token_indices is not None or use_recurrent_rollout
        )
        if needs_projected_visual_tokens:
            projected_visual_tokens = self._get_qwen_projected_visual_tokens(latent)
            if visual_token_indices is not None:
                mse_target, visual_target_mask = self._gather_qwen_concept_targets(
                    projected_visual_tokens=projected_visual_tokens,
                    image_grid_thw=latent.get('image_grid_thw'),
                    visual_token_indices=visual_token_indices,
                )

        kl_variant = (self.latent_cfg.kl_variant or '').lower()
        if kl_variant in {'bbox2image', 'bbox_image2image'}:
            if not self.is_qwen or mse_target is None or visual_target_mask is None:
                raise RuntimeError("bbox2image KL requires Qwen bbox visual-token targets.")
            e_c_pre, c_valid_mask = masked_mean_pool_tokens(
                mse_target,
                visual_target_mask,
                detach=True,
            )
            e_c = self.latent_proj(e_c_pre) if self.latent_proj is not None else e_c_pre
            if self.normalize:
                e_c = F.normalize(e_c, p=2, dim=-1)
            return {
                'c_tokens': mse_target,
                'c_reps': e_c,
                'c_token_mask': visual_target_mask,
                'c_valid_mask': c_valid_mask,
                'mse_target': None,
            }

        # ---- Step 2a: recurrent cache rollout, used as the cache representation
        # for both latent KL and the cache-to-ROI contrastive alignment.
        if use_recurrent_rollout:
            if projected_visual_tokens is None:
                raise RuntimeError("Recurrent latent concept rollout requires Qwen projected visual tokens.")
            recurrent_impl = (self.latent_cfg.recurrent_impl or 'exact').lower()
            if recurrent_impl in {'causal_slots', 'single_forward', 'lvc_style'}:
                latent_states, c_token_mask = self._forward_causal_concept_slots(
                    latent=latent,
                    projected_visual_tokens=projected_visual_tokens,
                    lcon_positions=lcon_positions,
                    use_cache=use_cache,
                )
            elif recurrent_impl in {'exact', 'iterative'}:
                latent_states, c_token_mask = self._rollout_recurrent_concept_cache(
                    latent=latent,
                    projected_visual_tokens=projected_visual_tokens,
                    lcon_positions=lcon_positions,
                    use_cache=use_cache,
                )
            else:
                raise ValueError(f"Unknown concept_recurrent_impl: {self.latent_cfg.recurrent_impl!r}")
            c_valid_mask = c_token_mask.any(dim=1) if c_token_mask is not None else None
            e_c_pre = self._pool_recurrent_cache(latent_states, c_token_mask)
            e_c = self.latent_proj(e_c_pre)
            if self.normalize:
                e_c = F.normalize(e_c, p=2, dim=-1)
            cache_visual_target = None
            cache_visual_mask = None
            if mse_target is not None:
                cache_visual_target, cache_visual_mask = masked_mean_pool_tokens(
                    mse_target,
                    visual_target_mask,
                    detach=bool(self.latent_cfg.cache_align_detach_target),
                )
            visual_alignment_tokens = latent_states
            visual_alignment_target = None
            visual_alignment_mask = c_token_mask
            if self.latent_cfg.mse_weight > 0 and cache_visual_target is not None:
                # Recurrent rollout has a fixed number of latent steps, while
                # ROI visual targets are variable length. Align pooled cache c
                # to pooled ROI v so all recurrent steps still receive gradient
                # through the configured cache pooling operation.
                visual_alignment_tokens = e_c_pre.unsqueeze(1)
                visual_alignment_target = cache_visual_target.unsqueeze(1)
                visual_alignment_mask = (
                    cache_visual_mask.to(device=e_c_pre.device, dtype=torch.bool).unsqueeze(1)
                    if cache_visual_mask is not None
                    else None
                )

            if lcon_positions is not None:
                latent['lcon_positions'] = lcon_positions
            if last_lcon_positions is not None:
                latent['last_lcon_positions'] = last_lcon_positions
            if visual_token_indices is not None:
                latent['visual_token_indices'] = visual_token_indices
            if visual_token_mask is not None:
                latent['visual_token_mask'] = visual_token_mask
            if lcon_token_count is not None:
                latent['lcon_token_count'] = lcon_token_count
            if bbox is not None:
                latent['bbox'] = bbox
            if bbox_count is not None:
                latent['bbox_count'] = bbox_count
            if bbox_image_size is not None:
                latent['bbox_image_size'] = bbox_image_size
            if bbox_image_size_mask is not None:
                latent['bbox_image_size_mask'] = bbox_image_size_mask

            return {
                'c_tokens': visual_alignment_tokens,
                'c_reps': e_c,
                'c_token_mask': visual_alignment_mask,
                'c_valid_mask': c_valid_mask,
                'mse_target': visual_alignment_target,
                'cache_c_pre': e_c_pre,
                'cache_visual_target': cache_visual_target,
                'cache_visual_mask': cache_visual_mask,
            }

        # ---- Step 2b: LM forward over teacher-forced LCON input (legacy MSE/cos path) ----
        c_token_mask = visual_target_mask
        encoder_inputs = {k: v for k, v in latent.items() if v is not None}
        if self.is_qwen and projected_visual_tokens is not None and mse_target is not None:
            encoder_inputs['inputs_embeds'] = self._build_qwen_concept_inputs_embeds(
                latent=latent,
                projected_visual_tokens=projected_visual_tokens,
                lcon_positions=lcon_positions,
                mse_target=mse_target,
                token_mask=c_token_mask,
            )
            # We already scattered both image features and dynamic LCON teacher
            # features into inputs_embeds above. Keep input_ids/image_grid_thw
            # for RoPE computation, but avoid recomputing image features.
            encoder_inputs.pop('pixel_values', None)
            encoder_inputs.pop('pixel_values_videos', None)

        outputs = self._forward_hidden_states(
            **encoder_inputs,
            return_dict=True,
            output_hidden_states=True,
            use_cache=use_cache,
        )
        last_hidden = outputs.hidden_states[-1]  # [B, T, H]

        c_tokens = self._gather_lcon_prediction_tokens(
            last_hidden,
            lcon_positions=lcon_positions,
            input_ids=latent.get('input_ids'),
        )
        if c_token_mask is None and lcon_positions is not None:
            c_token_mask = lcon_positions >= 0
        c_valid_mask = None
        if c_token_mask is not None:
            c_token_mask = c_token_mask.to(device=last_hidden.device, dtype=torch.bool)
            c_valid_mask = c_token_mask.any(dim=1)
        if (self.latent_cfg.pooling or '').lower() == 'end':
            e_c_pre, last_lcon_mask = self._gather_single_position_tokens(
                last_hidden,
                positions=last_lcon_positions,
                input_ids=latent.get('input_ids'),
                token_id=self.last_lcon_token_id,
                token_name=LCON_TOKEN,
            )
            if c_valid_mask is None:
                c_valid_mask = last_lcon_mask
            else:
                c_valid_mask = c_valid_mask & last_lcon_mask
        else:
            e_c_pre = self.latent_pooler(c_tokens, mask=c_token_mask)
        e_c = self.latent_proj(e_c_pre)
        if self.normalize:
            e_c = F.normalize(e_c, p=2, dim=-1)

        # Restore side-info to the dict so it survives subsequent introspection.
        if lcon_positions is not None:
            latent['lcon_positions'] = lcon_positions
        if last_lcon_positions is not None:
            latent['last_lcon_positions'] = last_lcon_positions
        if visual_token_indices is not None:
            latent['visual_token_indices'] = visual_token_indices
        if visual_token_mask is not None:
            latent['visual_token_mask'] = visual_token_mask
        if lcon_token_count is not None:
            latent['lcon_token_count'] = lcon_token_count
        if bbox is not None:
            latent['bbox'] = bbox
        if bbox_count is not None:
            latent['bbox_count'] = bbox_count
        if bbox_image_size is not None:
            latent['bbox_image_size'] = bbox_image_size
        if bbox_image_size_mask is not None:
            latent['bbox_image_size_mask'] = bbox_image_size_mask

        return {
            'c_tokens': c_tokens,
            'c_reps': e_c,
            'c_token_mask': c_token_mask,
            'c_valid_mask': c_valid_mask,
            'mse_target': mse_target.detach() if mse_target is not None else None,
        }

    def _forward_causal_concept_slots(self,
                                   latent: Dict[str, Tensor],
                                   projected_visual_tokens: Tensor,
                                   lcon_positions: Optional[Tensor],
                                   use_cache: bool = True) -> Tuple[Tensor, Tensor]:
        """Latent-VC SFT-style single-forward causal latent concept slots.

        The sequence already contains fixed ``<|lcon|>`` tokens. A single
        causal decoder forward lets later latent slots attend to earlier latent
        slots, then we gather hidden states at those positions. This is much
        faster than the exact h_{s-1}->x_s rollout while keeping the same
        [B, S, H] latent-cache interface for KL/MSE/cos/cache-align losses.
        """
        if lcon_positions is None:
            raise RuntimeError("lcon_positions are required for causal latent concept slots.")
        steps = int(self.latent_cfg.cache_steps or 8)
        if steps <= 0:
            raise ValueError(f"concept_cache_steps must be positive, got {steps}")
        if lcon_positions.size(1) < steps:
            raise RuntimeError(
                f"Causal latent concept slots need {steps} <|lcon|> positions, "
                f"got {lcon_positions.size(1)}."
            )

        positions = lcon_positions[:, :steps]
        inputs_embeds = self._build_qwen_image_inputs_embeds(
            latent=latent,
            projected_visual_tokens=projected_visual_tokens,
        )
        encoder_inputs = {k: v for k, v in latent.items() if v is not None}
        encoder_inputs.pop('pixel_values', None)
        encoder_inputs.pop('pixel_values_videos', None)

        backbone = getattr(self.encoder, 'model', None)
        if backbone is not None:
            outputs = backbone(
                **encoder_inputs,
                inputs_embeds=inputs_embeds,
                return_dict=True,
                output_hidden_states=False,
                use_cache=False if self.training else use_cache,
            )
            last_hidden = getattr(outputs, 'last_hidden_state', None)
            if last_hidden is None:
                last_hidden = outputs[0]
        else:
            outputs = self.encoder(
                **encoder_inputs,
                inputs_embeds=inputs_embeds,
                return_dict=True,
                output_hidden_states=True,
                use_cache=False if self.training else use_cache,
            )
            last_hidden = outputs.hidden_states[-1]

        latent_states = self._gather_lcon_prediction_tokens(
            last_hidden,
            lcon_positions=positions,
            input_ids=latent.get('input_ids'),
        )
        token_mask = positions.to(device=last_hidden.device) >= 0
        return latent_states, token_mask

    def _rollout_recurrent_concept_cache(self,
                                      latent: Dict[str, Tensor],
                                      projected_visual_tokens: Tensor,
                                      lcon_positions: Optional[Tensor],
                                      use_cache: bool = True) -> Tuple[Tensor, Tensor]:
        """Generate recurrent latent concept cache states without ROI teacher forcing.

        Full-sequence v1 implementation: for step s, the current <|lcon|>
        input embedding is replaced by h_{s-1}; previously generated cache
        states are also written into their earlier latent positions so causal
        attention sees the recurrent cache history.
        """
        if lcon_positions is None:
            raise RuntimeError("lcon_positions are required for recurrent latent concept cache rollout.")
        steps = int(self.latent_cfg.cache_steps or 8)
        if steps <= 0:
            raise ValueError(f"concept_cache_steps must be positive, got {steps}")
        if lcon_positions.size(1) < steps:
            raise RuntimeError(
                f"Recurrent latent concept cache rollout needs {steps} <|lcon|> positions, "
                f"got {lcon_positions.size(1)}."
            )

        base_embeds = self._build_qwen_image_inputs_embeds(
            latent=latent,
            projected_visual_tokens=projected_visual_tokens,
        )
        positions = lcon_positions[:, :steps].to(device=base_embeds.device, dtype=torch.long)
        token_mask = positions >= 0
        encoder_inputs = {k: v for k, v in latent.items() if v is not None}
        encoder_inputs.pop('pixel_values', None)
        encoder_inputs.pop('pixel_values_videos', None)

        states: List[Tensor] = []
        for step in range(steps):
            step_embeds = base_embeds.clone()
            for prev_idx, prev_state in enumerate(states):
                step_embeds = self._scatter_sequence_states(
                    step_embeds,
                    positions[:, prev_idx],
                    prev_state,
                )
            if step > 0:
                step_embeds = self._scatter_sequence_states(
                    step_embeds,
                    positions[:, step],
                    states[-1],
                )
            backbone = getattr(self.encoder, 'model', None)
            if backbone is not None:
                outputs = backbone(
                    **encoder_inputs,
                    inputs_embeds=step_embeds,
                    return_dict=True,
                    output_hidden_states=False,
                    use_cache=False if self.training else use_cache,
                )
                last_hidden = outputs.last_hidden_state
            else:
                outputs = self.encoder(
                    **encoder_inputs,
                    inputs_embeds=step_embeds,
                    return_dict=True,
                    output_hidden_states=True,
                    use_cache=False if self.training else use_cache,
                )
                last_hidden = outputs.hidden_states[-1]
            step_state = self._gather_step_state(last_hidden, positions[:, step])
            states.append(step_state)

        return torch.stack(states, dim=1), token_mask

    @staticmethod
    def _scatter_sequence_states(inputs_embeds: Tensor, positions: Tensor, states: Tensor) -> Tensor:
        valid = positions >= 0
        if not bool(valid.any().item()):
            return inputs_embeds
        batch_idx = torch.arange(inputs_embeds.size(0), device=inputs_embeds.device)[valid]
        seq_pos = positions.to(device=inputs_embeds.device)[valid]
        inputs_embeds[batch_idx, seq_pos] = states.to(inputs_embeds.dtype)[valid]
        return inputs_embeds

    @staticmethod
    def _gather_step_state(last_hidden: Tensor, positions: Tensor) -> Tensor:
        bsz, _, hsz = last_hidden.shape
        positions = positions.to(device=last_hidden.device, dtype=torch.long)
        mask = positions >= 0
        gather_pos = positions.clamp(min=0).view(bsz, 1, 1).expand(-1, 1, hsz)
        out = last_hidden.gather(1, gather_pos).squeeze(1)
        return out * mask.unsqueeze(-1).to(dtype=out.dtype)

    def _pool_recurrent_cache(self, latent_states: Tensor, token_mask: Optional[Tensor]) -> Tensor:
        pool = (self.latent_cfg.cache_pool or 'mean').lower()
        if pool == 'last':
            if token_mask is None:
                return latent_states[:, -1, :]
            lengths = token_mask.to(dtype=torch.long).sum(dim=1).clamp(min=1) - 1
            return latent_states[torch.arange(latent_states.size(0), device=latent_states.device), lengths]
        if pool != 'mean':
            raise ValueError(f"Unknown concept_cache_pool: {self.latent_cfg.cache_pool!r}")
        if token_mask is None:
            return latent_states.mean(dim=1)
        weights = token_mask.to(device=latent_states.device, dtype=latent_states.dtype)
        denom = weights.sum(dim=1, keepdim=True).clamp(min=1.0)
        return (latent_states * weights.unsqueeze(-1)).sum(dim=1) / denom

    def _compute_cache_align_loss(self,
                                  cache_c_pre: Tensor,
                                  visual_targets: Optional[Tensor],
                                  visual_mask: Optional[Tensor]) -> Tensor:
        if visual_targets is None:
            return cache_c_pre.sum() * 0.0
        if visual_targets.size(-1) != cache_c_pre.size(-1):
            raise RuntimeError(
                "Recurrent cache dim does not match ROI visual target dim: "
                f"cache={cache_c_pre.size(-1)} target={visual_targets.size(-1)}"
            )
        if visual_mask is None:
            visual_mask = torch.ones(cache_c_pre.size(0), device=cache_c_pre.device, dtype=torch.bool)
        visual_mask = visual_mask.to(device=cache_c_pre.device, dtype=torch.bool)
        if not bool(visual_mask.any().item()):
            return cache_c_pre.sum() * 0.0
        return cache_align_info_nce_loss(
            cache_c_pre,
            visual_targets.to(device=cache_c_pre.device, dtype=cache_c_pre.dtype),
            valid_mask=visual_mask,
            tau=self.latent_cfg.cache_align_tau,
            symmetric=bool(self.latent_cfg.cache_align_symmetric),
        )

    def _gather_single_position_tokens(self,
                                      last_hidden: Tensor,
                                      positions: Optional[Tensor],
                                      input_ids: Optional[Tensor],
                                      token_id: Optional[int],
                                      token_name: str) -> Tuple[Tensor, Tensor]:
        """Gather one sequence hidden state per sample, usually the last <|lcon|>."""
        bsz, _, hsz = last_hidden.shape
        if positions is None:
            if input_ids is None or token_id is None:
                raise RuntimeError(
                    f"Cannot locate {token_name} token: positions and token id are missing."
                )
            positions = torch.full((bsz,), fill_value=-1, device=last_hidden.device, dtype=torch.long)
            token_ids = input_ids.to(device=last_hidden.device)
            for i in range(bsz):
                idxs = (token_ids[i] == int(token_id)).nonzero(as_tuple=False).flatten()
                if idxs.numel() > 0:
                    positions[i] = idxs[-1]
        else:
            positions = positions.to(device=last_hidden.device, dtype=torch.long)
        if positions.dim() != 1 or positions.size(0) != bsz:
            raise ValueError(
                f"{token_name} positions must be [B], got {tuple(positions.shape)} for batch {bsz}."
            )
        mask = positions >= 0
        gather_pos = positions.clamp(min=0).view(bsz, 1, 1).expand(-1, 1, hsz)
        gathered = last_hidden.gather(1, gather_pos).squeeze(1)
        gathered = gathered * mask.unsqueeze(-1).to(dtype=gathered.dtype)
        return gathered, mask

    def _gather_lcon_prediction_tokens(self,
                                       last_hidden: Tensor,
                                       lcon_positions: Optional[Tensor],
                                       input_ids: Optional[Tensor]) -> Tensor:
        """Gather [B, M, H] hidden states that predict the <|lcon|> tokens."""
        bsz, _, hsz = last_hidden.shape
        if lcon_positions is not None:
            pred_positions = lcon_positions.clamp(min=0)
            gathered = last_hidden.gather(
                1, pred_positions.unsqueeze(-1).expand(-1, -1, hsz)
            )
            mask = (lcon_positions >= 0).unsqueeze(-1).to(gathered.dtype)
            return gathered * mask
        # Fall back to scanning input_ids for the known latent concept token id.
        if input_ids is None or self.lcon_token_id is None:
            raise RuntimeError(
                "Cannot locate <|lcon|> tokens: lcon_positions and input_ids are both missing."
            )
        per_sample = []
        max_n = 0
        for i in range(bsz):
            idxs = (input_ids[i] == self.lcon_token_id).nonzero(as_tuple=False).flatten()
            per_sample.append(idxs)
            max_n = max(max_n, int(idxs.numel()))
        out = last_hidden.new_zeros(bsz, max_n, hsz)
        for i, idxs in enumerate(per_sample):
            if idxs.numel() == 0:
                continue
            pred_positions = idxs.clamp(min=0)
            out[i, : idxs.numel()] = last_hidden[i, pred_positions]
        return out

    def _gather_qwen_concept_targets(self,
                                  projected_visual_tokens: Tensor,
                                  image_grid_thw: Tensor,
                                  visual_token_indices: Tensor) -> Tuple[Tensor, Tensor]:
        """Gather dynamic bbox patch embeddings from flat Qwen image features."""
        if image_grid_thw is None:
            raise RuntimeError("image_grid_thw is required to gather Qwen LCON targets.")
        if visual_token_indices.dim() != 2:
            raise ValueError(
                f"visual_token_indices must be [B, M], got {tuple(visual_token_indices.shape)}"
            )
        if projected_visual_tokens.dim() != 2:
            raise ValueError(
                f"projected_visual_tokens must be [N, H], got {tuple(projected_visual_tokens.shape)}"
            )
        bsz, max_m = visual_token_indices.shape
        if image_grid_thw.size(0) != bsz:
            raise ValueError(
                f"image_grid_thw batch {image_grid_thw.size(0)} != visual_token_indices batch {bsz}"
            )

        hidden = projected_visual_tokens.size(-1)
        targets = projected_visual_tokens.new_zeros(bsz, max_m, hidden)
        mask = torch.zeros(bsz, max_m, device=projected_visual_tokens.device, dtype=torch.bool)
        token_indices = visual_token_indices.to(device=projected_visual_tokens.device, dtype=torch.long)

        offset = 0
        for i in range(bsz):
            t, h, w = [int(v) for v in image_grid_thw[i].detach().cpu().tolist()]
            n_tokens = _expected_visual_tokens_qwen(t, h, w)
            if n_tokens <= 0:
                continue
            end = offset + n_tokens
            if end > projected_visual_tokens.size(0):
                raise RuntimeError(
                    "Qwen visual token count mismatch while gathering LCON targets: "
                    f"need slice [{offset}:{end}] from {projected_visual_tokens.size(0)} tokens."
                )
            positions = (token_indices[i] >= 0).nonzero(as_tuple=False).flatten()
            if positions.numel() > 0:
                local_idx = token_indices[i, positions]
                valid = (local_idx >= 0) & (local_idx < n_tokens)
                positions = positions[valid]
                local_idx = local_idx[valid]
                if positions.numel() > 0:
                    targets[i, positions] = projected_visual_tokens.index_select(0, offset + local_idx)
                    mask[i, positions] = True
            offset = end
        return targets.detach(), mask

    def _build_qwen_concept_inputs_embeds(self,
                                       latent: Dict[str, Tensor],
                                       projected_visual_tokens: Tensor,
                                       lcon_positions: Optional[Tensor],
                                       mse_target: Tensor,
                                       token_mask: Optional[Tensor]) -> Tensor:
        """Scatter image features and dynamic LCON teacher features into embeddings."""
        input_ids = latent.get('input_ids')
        if input_ids is None:
            raise RuntimeError("input_ids are required to build Qwen LCON inputs_embeds.")

        inputs_embeds = self._build_qwen_image_inputs_embeds(latent, projected_visual_tokens)

        if lcon_positions is None or token_mask is None or mse_target.numel() == 0:
            return inputs_embeds

        token_mask = token_mask.to(device=input_ids.device, dtype=torch.bool)
        batch_idx, token_idx = torch.nonzero(token_mask, as_tuple=True)
        if batch_idx.numel() == 0:
            return inputs_embeds
        seq_pos = lcon_positions.to(device=input_ids.device)[batch_idx, token_idx]
        valid = seq_pos >= 0
        if bool(valid.any().item()):
            batch_idx = batch_idx[valid]
            token_idx = token_idx[valid]
            seq_pos = seq_pos[valid]
            teacher = mse_target.to(inputs_embeds.device, inputs_embeds.dtype)[batch_idx, token_idx]
            inputs_embeds[batch_idx, seq_pos] = teacher
        return inputs_embeds

    def _build_qwen_image_inputs_embeds(self,
                                       latent: Dict[str, Tensor],
                                       projected_visual_tokens: Tensor) -> Tensor:
        """Scatter projected Qwen image features into the text embedding stream."""
        input_ids = latent.get('input_ids')
        if input_ids is None:
            raise RuntimeError("input_ids are required to build Qwen inputs_embeds.")

        embedder = self.encoder.get_input_embeddings()
        inputs_embeds = embedder(input_ids)

        base = self._vl_base_model()
        image_token_id = getattr(getattr(base, 'config', None), 'image_token_id', None)
        if image_token_id is None:
            image_token_id = self.image_pad_token_id
        if image_token_id is None:
            raise RuntimeError("Could not resolve Qwen image token id for latent inputs.")

        image_mask = input_ids == int(image_token_id)
        n_image_tokens = int(image_mask.sum().item())
        if n_image_tokens != int(projected_visual_tokens.size(0)):
            raise RuntimeError(
                "Image features and image tokens do not match in latent branch: "
                f"tokens={n_image_tokens}, features={projected_visual_tokens.size(0)}"
            )
        image_features = projected_visual_tokens.to(inputs_embeds.device, inputs_embeds.dtype)
        return inputs_embeds.masked_scatter(
            image_mask.unsqueeze(-1).expand_as(inputs_embeds),
            image_features,
        )

    # ------------------------------------------------------------------
    # LVR-aligned projected visual token extraction
    # ------------------------------------------------------------------
    def _vl_base_model(self) -> nn.Module:
        """Walk through PEFT wrapping to the underlying VL transformer.

        PEFT layers a ``PeftModel`` over a ``LoraModel`` over the actual
        backbone, exposed as ``peft_model.base_model.model``. We detect PEFT
        explicitly via ``peft_config`` to avoid clashing with HF's own
        ``PreTrainedModel.base_model`` property which has a different meaning.
        """
        enc = self.encoder
        if hasattr(enc, 'peft_config'):
            base_lora = getattr(enc, 'base_model', None)
            if base_lora is not None and hasattr(base_lora, 'model'):
                return base_lora.model
        return enc

    def _forward_hidden_states(self, **kwargs):
        """Forward retrieval features without materializing unused LM logits."""
        if self.is_qwen:
            return self.encoder(**kwargs)

        base = self._vl_base_model()
        backbone = getattr(base, 'model', None)
        if backbone is None:
            return self.encoder(**kwargs)
        return backbone(**kwargs)

    def _get_qwen_projected_visual_tokens(self, latent: Dict[str, Tensor]) -> Tensor:
        """LVR-style visual-patch source for Qwen2.5-VL.

        Mirrors LVR's ``image_embeds = self.model.get_image_features(...)``
        followed by ``torch.cat(image_embeds, dim=0)`` so we end up with a flat
        ``[N_total_visual_tokens, H]`` tensor of *projected* visual embeddings
        — the same per-token quantity that the LM later masked_scatters into
        the textual hidden stream. Computed under ``no_grad`` and detached;
        the visual encoder is treated as a frozen teacher.
        """
        pixel_values = latent.get('pixel_values')
        image_grid_thw = latent.get('image_grid_thw')
        if pixel_values is None or image_grid_thw is None:
            raise RuntimeError(
                "Qwen LCON MSE requires both `pixel_values` and `image_grid_thw` in the latent batch."
            )

        base = self._vl_base_model()
        getter = getattr(base, 'get_image_features', None)
        if getter is None:
            # Some HF versions expose the method on the wrapper, try once more.
            getter = getattr(self.encoder, 'get_image_features', None)
        if getter is None:
            raise RuntimeError(
                "Could not find `get_image_features` on the Qwen2.5-VL backbone; "
                "please verify the model class exposes this LVR-compatible API."
            )

        with torch.no_grad():
            out = getter(pixel_values=pixel_values, image_grid_thw=image_grid_thw)
        return _flatten_image_features(out).detach()

    # ------------------------------------------------------------------
    # Existing helpers (unchanged behaviour)
    # ------------------------------------------------------------------
    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def compute_kl_loss(self, teacher_scores, student_scores):
        teacher_probs = F.softmax(teacher_scores, dim=-1)
        student_log_probs = F.log_softmax(student_scores, dim=-1)
        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        return kl_loss

    def gradient_checkpointing_enable(self, **kwargs):
        # Recurrent latent concept feeds `inputs_embeds` directly. PyTorch's reentrant
        # checkpoint path drops gradients when none of the tensor inputs require
        # grad, which can silently cut the latent branch off from LoRA weights.
        # Non-reentrant checkpointing keeps the memory benefit while supporting
        # parameter-only gradients through inputs_embeds.
        kwargs = dict(kwargs)
        if kwargs.get("gradient_checkpointing_kwargs") is None:
            kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
        try:
            self.encoder.model.gradient_checkpointing_enable(**kwargs)
        except TypeError:
            self.encoder.model.gradient_checkpointing_enable()

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    @staticmethod
    def _is_qwen(model_name_or_path: str) -> bool:
        return 'qwen' in model_name_or_path.lower()

    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            train_args: TrainingArguments,
            data_args: DataArguments = None,
            tokenizer=None,
            **hf_kwargs,
    ):
        is_qwen = cls._is_qwen(model_args.model_name_or_path)
        hf_kwargs['trust_remote_code'] = True

        if is_qwen:
            transformer_cls = AutoModel
            logger.info(f"Loading Qwen model from {model_args.model_name_or_path}")
        else:
            transformer_cls = AutoModelForCausalLM

        base_model = transformer_cls.from_pretrained(model_args.model_name_or_path, **hf_kwargs)

        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = (
                base_model.config.eos_token_id if is_qwen else 0
            )

        logger.info(f"Model config - hidden_size: {base_model.config.hidden_size}, "
                     f"pad_token_id: {base_model.config.pad_token_id}")

        kl_loss_weight = data_args.kl_loss_weight if data_args else 1.0
        latent_cfg = _build_latent_cfg(data_args)

        # If the tokenizer was extended with latent concept special tokens by the driver,
        # resize the model embeddings before LoRA wraps the encoder. New rows are
        # initialised with the *mean* of the existing embeddings (a standard
        # trick from "Vocabulary Tricks for Efficient Pre-Training") so the
        # latent branch starts in a sensible region of the embedding space
        # rather than from a random Normal init.
        n_new_tokens = 0
        if tokenizer is not None:
            try:
                vocab_now = int(base_model.get_input_embeddings().weight.size(0))
            except Exception:
                vocab_now = None
            if not vocab_now:
                # Under ZeRO-3 initialisation, the local shard can report an
                # embedding row count of 0. The unpartitioned vocabulary size is
                # still available on the config and should drive resize checks.
                vocab_now = getattr(base_model.config, 'vocab_size', None)
                vocab_now = int(vocab_now) if vocab_now is not None else None
            tok_size = len(tokenizer)
            if vocab_now is not None and tok_size > vocab_now:
                n_new_tokens = tok_size - vocab_now
                logger.info(
                    "Resizing token embeddings from %s to %s for latent concept special tokens.",
                    vocab_now, tok_size,
                )
                base_model.resize_token_embeddings(tok_size)
                _meaninit_new_token_rows(base_model, n_new=n_new_tokens)

        image_pad_token_id = _resolve_image_pad_token(tokenizer, is_qwen=is_qwen)
        lcon_token_id = None
        last_lcon_token_id = None
        if tokenizer is not None:
            try:
                tid = tokenizer.convert_tokens_to_ids(LCON_TOKEN)
                if tid != tokenizer.unk_token_id:
                    lcon_token_id = int(tid)
            except Exception:
                lcon_token_id = None
            try:
                tid = tokenizer.convert_tokens_to_ids(LCON_TOKEN)
                if tid != tokenizer.unk_token_id:
                    last_lcon_token_id = int(tid)
            except Exception:
                last_lcon_token_id = None

        if model_args.lora or model_args.lora_name_or_path:
            if train_args.gradient_checkpointing:
                base_model.enable_input_require_grads()
            if model_args.lora_name_or_path:
                logger.info(f"Loading LoRA weights from {model_args.lora_name_or_path}")
                lora_config = LoraConfig.from_pretrained(model_args.lora_name_or_path, **hf_kwargs)
                lora_model = PeftModel.from_pretrained(base_model, model_args.lora_name_or_path, is_trainable=True)
                # The pretrained LoRA was likely trained without latent concept; if we
                # just added new token rows, mark the input embedding (and
                # output embedding when present) trainable so the latent
                # branch can actually learn. This is a no-op when latent concept is off.
                if n_new_tokens > 0:
                    _enable_embedding_grad(lora_model)
            else:
                logger.info(f"Creating new LoRA config with r={model_args.lora_r}, "
                            f"alpha={model_args.lora_alpha}")
                # When latent concept tokens are added we *must* keep the input embedding
                # module trainable so that the new <|lcon|> row can learn.
                # Otherwise LoRA freezes the base model and the latent tokens
                # are stuck at their (mean-initialised) values, which makes the
                # latent branch degenerate.
                modules_to_save = ['embed_tokens'] if n_new_tokens > 0 else None
                lora_config = LoraConfig(
                    base_model_name_or_path=model_args.model_name_or_path,
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=model_args.lora_r,
                    lora_alpha=model_args.lora_alpha,
                    lora_dropout=model_args.lora_dropout,
                    target_modules=model_args.lora_target_modules.split(','),
                    modules_to_save=modules_to_save,
                    inference_mode=False,
                )
                lora_model = get_peft_model(base_model, lora_config)
                lora_model.print_trainable_parameters()
            model = cls(
                encoder=lora_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature,
                kl_loss_weight=kl_loss_weight,
                is_qwen=is_qwen,
                latent_cfg=latent_cfg,
                image_pad_token_id=image_pad_token_id,
                lcon_token_id=lcon_token_id,
                last_lcon_token_id=last_lcon_token_id,
            )
        else:
            model = cls(
                encoder=base_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature,
                kl_loss_weight=kl_loss_weight,
                is_qwen=is_qwen,
                latent_cfg=latent_cfg,
                image_pad_token_id=image_pad_token_id,
                lcon_token_id=lcon_token_id,
                last_lcon_token_id=last_lcon_token_id,
            )
        _load_conceptformer_state(model, model_args.lora_name_or_path)
        return model

    @classmethod
    def load(cls,
             model_name_or_path: str,
             pooling: str = 'cls',
             normalize: bool = False,
             lora_name_or_path: str = None,
             **hf_kwargs):
        is_qwen = cls._is_qwen(model_name_or_path)
        hf_kwargs['trust_remote_code'] = True
        transformer_cls = AutoModel if is_qwen else AutoModelForCausalLM
        base_model = transformer_cls.from_pretrained(model_name_or_path, **hf_kwargs)
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = (
                base_model.config.eos_token_id if is_qwen else 0
            )
        if lora_name_or_path:
            # ConceptFormer adapters are trained with one extra special token saved
            # alongside the adapter tokenizer. Resize the base model before
            # PEFT loads modules_to_save embeddings, otherwise vocab-size
            # mismatch can break evaluation. Plain ConceptFormer adapters are no-op.
            try:
                adapter_tokenizer = AutoTokenizer.from_pretrained(
                    lora_name_or_path,
                    trust_remote_code=True,
                )
                adapter_vocab = len(adapter_tokenizer)
                cur_vocab = int(base_model.get_input_embeddings().weight.size(0))
                if adapter_vocab > cur_vocab:
                    base_model.resize_token_embeddings(adapter_vocab)
                    _meaninit_new_token_rows(base_model, n_new=adapter_vocab - cur_vocab)
                    logger.info(
                        "Resized base embeddings from %s to %s before loading adapter %s.",
                        cur_vocab, adapter_vocab, lora_name_or_path,
                    )
            except Exception as exc:  # pragma: no cover - best-effort compatibility
                logger.info(
                    "Could not inspect adapter tokenizer at %s before PEFT load: %s",
                    lora_name_or_path, exc,
                )
            lora_config = LoraConfig.from_pretrained(lora_name_or_path, **hf_kwargs)
            lora_model = PeftModel.from_pretrained(base_model, lora_name_or_path, config=lora_config)
            lora_model = lora_model.merge_and_unload()
            model = cls(
                encoder=lora_model,
                pooling=pooling,
                normalize=normalize,
                is_qwen=is_qwen,
            )
        else:
            model = cls(
                encoder=base_model,
                pooling=pooling,
                normalize=normalize,
                is_qwen=is_qwen,
            )
        return model

    def save(self, output_dir: str):
        self.encoder.save_pretrained(output_dir)

    def encode_query(self, qry, use_cache=True):
        query_hidden_states = self._forward_hidden_states(
            **qry,
            return_dict=True,
            output_hidden_states=True,
            output_attentions=True,
            use_cache=use_cache,
        )
        query_hidden_states = query_hidden_states.hidden_states[-1]
        return self._pooling(query_hidden_states, qry['attention_mask'])

    def encode_document(self, doc, use_cache=True):
        return self.encode_query(doc, use_cache=use_cache)

    def generate_output(self, pair, use_cache=True):
        return self.encoder(**pair, use_cache=use_cache)

    def _pooling(self, last_hidden_state, attention_mask):
        if self.pooling in ['cls', 'first']:
            reps = last_hidden_state[:, 0]
        elif self.pooling in ['mean', 'avg', 'average']:
            masked_hiddens = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
            reps = masked_hiddens.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.pooling in ['last', 'eos']:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_state.shape[0]
            reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
        else:
            raise ValueError(f'unknown pooling method: {self.pooling}')
        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps


# ---------------------------------------------------------------------------
# Helpers used by build()
# ---------------------------------------------------------------------------
def _build_latent_cfg(data_args: Optional[DataArguments]) -> LatentConceptConfig:
    if data_args is None:
        return LatentConceptConfig()
    return LatentConceptConfig(
        align_mode=getattr(data_args, 'latent_align_mode', 'none'),
        lambda_forward=float(getattr(data_args, 'latent_lambda_forward', 0.0) or 0.0),
        lambda_reverse=float(getattr(data_args, 'latent_lambda_reverse', 0.0) or 0.0),
        mse_weight=get_latent_mse_weight(data_args),
        visual_loss_type=getattr(data_args, 'latent_visual_loss_type', 'mse'),
        kl_variant=getattr(data_args, 'latent_kl_variant', 'q2concept'),
        pooling=getattr(data_args, 'latent_pooling', 'mean'),
        cache_align_weight=get_concept_cache_align_weight(data_args),
        cache_steps=int(getattr(data_args, 'concept_cache_steps', 8) or 8),
        cache_pool=getattr(data_args, 'concept_cache_pool', 'mean'),
        recurrent_kl=bool(getattr(data_args, 'concept_recurrent_kl', False)),
        recurrent_impl=getattr(data_args, 'concept_recurrent_impl', 'exact'),
        cache_align_tau=float(getattr(data_args, 'concept_cache_align_tau', 0.07) or 0.07),
        cache_align_symmetric=bool(getattr(data_args, 'concept_cache_align_symmetric', False)),
        cache_align_detach_target=bool(getattr(data_args, 'concept_cache_align_detach_target', True)),
    )


def _flatten_image_features(out) -> Tensor:
    """Normalise the various return shapes of ``get_image_features`` to a
    flat ``[N_total_visual_tokens, H]`` tensor.

    The HF Qwen2.5-VL family returns a *tuple of per-image tensors* here, which
    LVR concatenates with ``torch.cat(..., dim=0)``. Some other backbones may
    return a ``BaseModelOutput`` wrapping a ``pooler_output`` tensor; we
    handle that too.
    """
    if hasattr(out, 'pooler_output') and out.pooler_output is not None:
        out = out.pooler_output
    if hasattr(out, 'last_hidden_state') and not isinstance(out, torch.Tensor):
        # Common ModelOutput shape: pick the dense tensor.
        out = out.last_hidden_state
    if isinstance(out, (list, tuple)):
        if len(out) == 0:
            raise RuntimeError("get_image_features returned an empty tuple/list.")
        out = torch.cat(out, dim=0)
    if not isinstance(out, torch.Tensor):
        raise RuntimeError(
            f"Unexpected return type from get_image_features: {type(out)}; "
            f"expected Tensor or tuple of Tensors."
        )
    if out.dim() == 3:  # [N, P, H] -> [N*P, H]
        out = out.reshape(-1, out.size(-1))
    elif out.dim() != 2:
        raise RuntimeError(
            f"Unexpected projected visual tokens shape: {tuple(out.shape)}; "
            f"expected 2D [N, H] or 3D [N, P, H]."
        )
    return out


def _meaninit_new_token_rows(model: nn.Module, n_new: int) -> None:
    """Initialise the last ``n_new`` rows of input/output embeddings with the
    column-wise mean of the *existing* rows. Avoids the well-known instability
    that comes from feeding a Transformer a randomly-initialised special token.
    """
    if n_new <= 0:
        return
    try:
        in_emb = model.get_input_embeddings()
        with torch.no_grad():
            old_rows = in_emb.weight.data[:-n_new]
            mean_row = old_rows.mean(dim=0, keepdim=True)
            in_emb.weight.data[-n_new:] = mean_row.expand(n_new, -1).clone()
    except Exception as exc:  # pragma: no cover - logged only
        logger.warning("[latent concept] mean-init of input embeddings failed: %s", exc)
    try:
        out_emb = model.get_output_embeddings()
        if out_emb is not None and out_emb.weight is not in_emb.weight:
            with torch.no_grad():
                old_rows = out_emb.weight.data[:-n_new]
                mean_row = old_rows.mean(dim=0, keepdim=True)
                out_emb.weight.data[-n_new:] = mean_row.expand(n_new, -1).clone()
    except Exception as exc:  # pragma: no cover - logged only
        logger.warning("[latent concept] mean-init of output embeddings failed: %s", exc)


def _enable_embedding_grad(model: nn.Module) -> None:
    """Force the input/output embedding weights to ``requires_grad=True``.

    Used when extending the vocabulary on top of a pretrained PEFT checkpoint
    that did not include the embedding module in ``modules_to_save``. Without
    this, the freshly-added latent concept token embeddings would be frozen and could
    never learn.
    """
    try:
        in_emb = model.get_input_embeddings()
        if in_emb is not None and hasattr(in_emb, 'weight'):
            in_emb.weight.requires_grad_(True)
    except Exception as exc:  # pragma: no cover - logged only
        logger.warning("[latent concept] could not enable input embedding grad: %s", exc)
    try:
        out_emb = model.get_output_embeddings()
        if out_emb is not None and hasattr(out_emb, 'weight'):
            out_emb.weight.requires_grad_(True)
    except Exception as exc:  # pragma: no cover - logged only
        logger.warning("[latent concept] could not enable output embedding grad: %s", exc)


def _resolve_image_pad_token(tokenizer, is_qwen: bool) -> Optional[int]:
    if tokenizer is None:
        return None
    candidates = ['<|image_pad|>'] if is_qwen else ['<|image_1|>']
    for tok in candidates:
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
        except Exception:
            tid = None
        if tid is not None and tid != tokenizer.unk_token_id:
            return int(tid)
    return None
