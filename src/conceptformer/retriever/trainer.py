import os
from typing import Optional

import torch

from transformers.trainer import Trainer, TRAINING_ARGS_NAME
import torch.distributed as dist
from .modeling import ConceptFormerRetriever
from .arguments import DataArguments

import logging
logger = logging.getLogger(__name__)


class ConceptFormerRetrieverTrainer(Trainer):
    def __init__(self, *args, data_args: DataArguments = None, **kwargs):
        super(ConceptFormerRetrieverTrainer, self).__init__(*args, **kwargs)
        self.is_ddp = dist.is_initialized()
        self._dist_loss_scale_factor = dist.get_world_size() if self.is_ddp else 1
        self.data_args = data_args
        # Light-weight running counters for per-component latent concept losses, flushed
        # on every logging call (HF Trainer logs `loss` itself).
        self._concept_loss_buf = {}
        self._concept_loss_count = 0

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (ConceptFormerRetriever,)
        if not isinstance(self.model, supported_classes):
            raise ValueError(f"Unsupported model class {self.model}")
        else:
            if state_dict is None:
                state_dict = self.model.state_dict()
            # Some latent concept layers (latent_pooler, latent_proj/cache head) live on
            # the wrapper itself, not under `encoder.*`. Filter to the encoder
            # prefix so HF/PEFT save still works for the original retriever.
            prefix = 'encoder.'
            encoder_state_dict = {
                k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)
            }
            self.model.encoder.save_pretrained(
                output_dir, state_dict=encoder_state_dict, safe_serialization=self.args.save_safetensors
            )

            # Also persist latent concept-specific weights, if any, to a sidecar file.
            concept_state = {
                k: v.detach().cpu()
                for k, v in state_dict.items()
                if k.startswith(('latent_proj.', 'latent_pooler.'))
            }
            if concept_state:
                torch.save(concept_state, os.path.join(output_dir, 'conceptformer_state.pt'))

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def _unpack_inputs(self, inputs):
        """Backwards-compatible unpacking of (q, d, p, qd, d_exist [, latent])."""
        latent = None
        if isinstance(inputs, (list, tuple)):
            if len(inputs) == 6:
                query, document, pair, query_describe, d_exist_ids, latent = inputs
            elif len(inputs) == 5:
                query, document, pair, query_describe, d_exist_ids = inputs
            elif len(inputs) == 3:
                query, document, pair = inputs
                query_describe, d_exist_ids = None, None
            else:
                raise ValueError(f"Unexpected input format with {len(inputs)} elements")
        else:
            raise ValueError(f"Unexpected inputs of type {type(inputs)}")
        return query, document, pair, query_describe, d_exist_ids, latent

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        query, document, pair, query_describe, d_exist_ids, latent = self._unpack_inputs(inputs)

        outputs = model(
            query=query,
            document=document,
            pair=pair,
            query_describe=query_describe,
            d_exist_ids=d_exist_ids,
            latent=latent,
        )
        loss = outputs.loss

        # Accumulate latent concept per-component diagnostics for logging.
        cfg = getattr(model, 'latent_cfg', None)
        if cfg is not None:
            comp_map = {
                'loss_contrast': outputs.loss_contrast,
                'loss_text_kl': outputs.loss_text_kl,
                'loss_kl_forward': outputs.loss_kl_forward,
                'loss_kl_reverse': outputs.loss_kl_reverse,
                'loss_mse': outputs.loss_mse,
                'loss_cache_align': outputs.loss_cache_align,
            }
            for name, value in comp_map.items():
                if value is None:
                    continue
                v = value.detach().float().item()
                self._concept_loss_buf[name] = self._concept_loss_buf.get(name, 0.0) + v
            self._concept_loss_count += 1

        if return_outputs:
            return loss, outputs
        return loss

    def log(self, logs, *args, **kwargs):  # type: ignore[override]
        """Inject the latent concept per-component running averages into HF Trainer logs."""
        if self._concept_loss_count > 0 and self._concept_loss_buf:
            for name, total in self._concept_loss_buf.items():
                logs[name] = total / max(1, self._concept_loss_count)
            cfg = getattr(getattr(self, 'model', None), 'latent_cfg', None)
            if cfg is not None:
                logs['latent_align_mode'] = cfg.align_mode
                logs['lambda_forward'] = cfg.lambda_forward
                logs['lambda_reverse'] = cfg.lambda_reverse
                logs['mse_weight'] = cfg.mse_weight
                logs['visual_loss_type'] = getattr(cfg, 'visual_loss_type', 'mse')
                logs['latent_kl_variant'] = getattr(cfg, 'kl_variant', 'q2concept')
                logs['cache_align_weight'] = float(getattr(cfg, 'cache_align_weight', 0.0) or 0.0)
                logs['cache_align_tau'] = float(getattr(cfg, 'cache_align_tau', 0.07) or 0.07)
                logs['cache_pool'] = getattr(cfg, 'cache_pool', 'mean')
                logs['recurrent_kl'] = bool(getattr(cfg, 'recurrent_kl', False))
                logs['recurrent_impl'] = getattr(cfg, 'recurrent_impl', 'exact')
                logs['text_kl_weight'] = float(getattr(self.data_args, 'kl_loss_weight', 0.0) or 0.0)
            self._concept_loss_buf = {}
            self._concept_loss_count = 0
        return super(ConceptFormerRetrieverTrainer, self).log(logs, *args, **kwargs)

    def training_step(self, *args):
        return super(ConceptFormerRetrieverTrainer, self).training_step(*args) / self._dist_loss_scale_factor
