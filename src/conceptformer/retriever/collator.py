import torch
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from transformers import PreTrainedTokenizer, ProcessorMixin
from conceptformer.retriever.arguments import DataArguments
from conceptformer.retriever.latent_concepts import (
    LCON_TOKEN,
    bbox_to_lcon_token_counts_phi3v,
    bbox_to_visual_token_indices_qwen,
    get_latent_mse_weight,
    get_concept_cache_align_weight,
)
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class TrainCollator:
    data_args: DataArguments
    tokenizer: PreTrainedTokenizer
    processor: ProcessorMixin

    # Filled in __post_init__:
    is_qwen: bool = field(init=False, default=False)
    lcon_id: Optional[int] = field(init=False, default=None)
    latent_enabled: bool = field(init=False, default=False)

    def __post_init__(self):
        self.is_qwen = 'qwen' in type(self.processor).__name__.lower()

        # Resolve latent concept special-token ids if they are present in the tokenizer.
        # The driver script is responsible for adding these tokens (and
        # resizing model embeddings) when the latent branch is enabled.
        def _id(tok: str) -> Optional[int]:
            try:
                tid = self.tokenizer.convert_tokens_to_ids(tok)
            except Exception:
                tid = None
            unk = getattr(self.tokenizer, 'unk_token_id', None)
            if tid is None or tid == unk:
                return None
            return int(tid)

        self.lcon_id = _id(LCON_TOKEN)

        latent_mode = getattr(self.data_args, 'latent_align_mode', 'none')
        mse_weight = get_latent_mse_weight(self.data_args)
        cache_align_weight = get_concept_cache_align_weight(self.data_args)
        recurrent_kl_enabled = bool(getattr(self.data_args, 'concept_recurrent_kl', False))
        self.latent_enabled = (
            (latent_mode != 'none')
            or (mse_weight > 0)
            or (cache_align_weight > 0)
            or recurrent_kl_enabled
        )
        if self.latent_enabled and self.lcon_id is None:
            raise ValueError(
                "The latent concept branch is enabled but the tokenizer does not contain "
                "the required <|lcon|> special token. "
                "Make sure the training driver registers them and resizes embeddings."
            )

    def _encode_images(self, images, queries: Optional[List[str]] = None,
                        lcon_suffix_ids: Optional[List[int]] = None,
                        lcon_suffix_ids_per_sample: Optional[List[List[int]]] = None):
        """Encode images via the multimodal processor.

        If ``queries`` and a latent concept suffix are provided, the textual prompt
        becomes the user query (rather than the generic OCR prompt) and the
        suffix is appended *after* tokenisation. ``lcon_suffix_ids_per_sample``
        enables LVR-style dynamic counts where each sample gets a different
        number of ``<|lcon|>`` tokens.
        """
        use_query_prompt = queries is not None
        suffix = list(lcon_suffix_ids) if lcon_suffix_ids else []
        per_sample_suffix = lcon_suffix_ids_per_sample

        if self.is_qwen:
            if use_query_prompt:
                messages_list = [[{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": f"Query: {q}\nIdentify the document evidence relevant to the query."},
                    ]
                }] for q, img in zip(queries, images)]
            else:
                messages_list = [[{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": "What is shown in this image?"}
                    ]
                }] for img in images]
            texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                     for msg in messages_list]
            batch_inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
            input_ids_list = [batch_inputs['input_ids'][i].tolist() for i in range(len(images))]
            if suffix or per_sample_suffix is not None:
                # batch_inputs['input_ids'] is right-padded with pad_token_id; trim
                # padding before appending the latent concept suffix to keep the latent
                # tokens contiguous and addressable.
                pad_id = self.tokenizer.pad_token_id
                trimmed: List[List[int]] = []
                for i, ids in enumerate(input_ids_list):
                    while ids and ids[-1] == pad_id:
                        ids = ids[:-1]
                    sample_suffix = per_sample_suffix[i] if per_sample_suffix is not None else suffix
                    trimmed.append(ids + sample_suffix)
                input_ids_list = trimmed
            return input_ids_list, batch_inputs
        else:
            if use_query_prompt:
                prompts = [
                    f"<|image_1|>\nQuery: {q}\nIdentify the document evidence relevant to the query."
                    for q in queries
                ]
            else:
                prompts = ["<|image_1|>\nWhat is shown in this image?" for _ in images]
            collated_list = [self.processor(p, img, return_tensors="pt")
                             for p, img in zip(prompts, images)]
            input_ids_list = []
            for d in collated_list:
                ids = d['input_ids']
                if torch.is_tensor(ids):
                    while ids.dim() > 1:
                        ids = ids[0]
                    ids = ids.tolist()
                else:
                    while isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], list):
                        ids = ids[0]
                input_ids_list.append([int(t) for t in ids])
            if suffix or per_sample_suffix is not None:
                updated = []
                for i, ids in enumerate(input_ids_list):
                    sample_suffix = per_sample_suffix[i] if per_sample_suffix is not None else suffix
                    updated.append(ids + sample_suffix)
                input_ids_list = updated
            return input_ids_list, collated_list

    def _attach_image_tensors(self, target_dict, image_meta):
        if self.is_qwen:
            target_dict['pixel_values'] = image_meta['pixel_values']
            if 'image_grid_thw' in image_meta:
                target_dict['image_grid_thw'] = image_meta['image_grid_thw']
            elif 'image_sizes' in image_meta:
                target_dict['image_sizes'] = image_meta['image_sizes']
        else:
            target_dict['pixel_values'] = torch.stack([d['pixel_values'][0] for d in image_meta], dim=0)
            target_dict['image_sizes'] = torch.stack([d['image_sizes'][0] for d in image_meta], dim=0)

    def build_image_attention_mask(self, seq_len, input_lengths):
        image_attention_masks = []
        for input_len in input_lengths:
            image_attention_mask = torch.tril(torch.ones(seq_len, seq_len), diagonal=0)
            image_attention_mask[input_len:, :input_len-1] = 0 
            image_attention_masks.append(image_attention_mask.unsqueeze(0))
        image_attention_masks = torch.cat(image_attention_masks, dim=0)
        return image_attention_masks

    def _build_lcon_collated(self, all_queries, all_images, all_bboxes, all_bbox_image_sizes):
        """Build the latent-concept input dict when concept learning is enabled."""
        if not self.latent_enabled:
            return None

        input_ids_list, image_meta = self._encode_images(all_images, queries=all_queries)

        bsz = len(all_bboxes)
        safe_bboxes: List[List[List[float]]] = [b if b else [] for b in all_bboxes]

        bbox_image_size = torch.zeros(bsz, 2, dtype=torch.float32)
        bbox_image_size_mask = torch.zeros(bsz, dtype=torch.bool)
        for i, size in enumerate(all_bbox_image_sizes):
            if not size:
                continue
            try:
                w, h = float(size[0]), float(size[1])
            except Exception:
                continue
            if w <= 0 or h <= 0:
                continue
            bbox_image_size[i] = torch.tensor([w, h], dtype=torch.float32)
            bbox_image_size_mask[i] = True

        mse_enabled = get_latent_mse_weight(self.data_args) > 0
        cache_align_enabled = get_concept_cache_align_weight(self.data_args) > 0
        recurrent_kl_enabled = bool(getattr(self.data_args, 'concept_recurrent_kl', False))
        fixed_lcon_tokens = int(getattr(self.data_args, 'latent_num_tokens', 0) or 0)
        if fixed_lcon_tokens < 0:
            raise ValueError(f"latent_num_tokens must be non-negative, got {fixed_lcon_tokens}")
        fixed_kl_novis = (
            fixed_lcon_tokens > 0
            and not mse_enabled
            and not cache_align_enabled
            and not recurrent_kl_enabled
        )
        kl_only_novis = (
            not mse_enabled
            and not cache_align_enabled
            and not recurrent_kl_enabled
        )
        if fixed_lcon_tokens > 0 and not fixed_kl_novis:
            raise ValueError(
                "latent_num_tokens is only supported for KL-only novis runs "
                "(latent_mse_weight=0, concept_cache_align_weight=0, concept_recurrent_kl=false)."
            )
        if not self.is_qwen and not kl_only_novis:
            raise ValueError(
                "Dynamic LCON visual-token supervision is currently implemented for "
                "Qwen2.5-VL batches. Non-Qwen models currently support only KL-only "
                "no-vis LCON slots."
            )

        lcon_count_overrides: Optional[List[int]] = None
        if fixed_kl_novis:
            visual_token_indices_list = [torch.empty(0, dtype=torch.long) for _ in range(bsz)]
        elif self.is_qwen:
            image_grid_thw = image_meta.get('image_grid_thw')
            if image_grid_thw is None:
                raise ValueError("Qwen latent branch requires image_grid_thw from the processor.")
            visual_token_indices_list = bbox_to_visual_token_indices_qwen(
                image_grid_thw=image_grid_thw,
                bboxes=safe_bboxes,
                bbox_image_sizes=bbox_image_size,
            )
        else:
            image_processor = getattr(self.processor, 'image_processor', None)
            num_crops = int(getattr(image_processor, 'num_crops', 16) or 16)
            lcon_count_overrides = bbox_to_lcon_token_counts_phi3v(
                bboxes=safe_bboxes,
                bbox_image_sizes=bbox_image_size,
                num_crops=num_crops,
            )
            visual_token_indices_list = [torch.empty(0, dtype=torch.long) for _ in range(bsz)]

        cache_steps = int(getattr(self.data_args, 'concept_cache_steps', 8) or 8)
        if cache_steps <= 0:
            raise ValueError(f"concept_cache_steps must be positive, got {cache_steps}")

        suffixes: List[List[int]] = []
        lcon_counts: List[int] = []
        for i, idxs in enumerate(visual_token_indices_list):
            if fixed_kl_novis:
                n_tokens = fixed_lcon_tokens
            elif lcon_count_overrides is not None:
                n_tokens = int(lcon_count_overrides[i])
            elif cache_align_enabled or recurrent_kl_enabled:
                n_tokens = cache_steps
            else:
                n_tokens = int(idxs.numel())
            lcon_counts.append(n_tokens)
            suffixes.append([self.lcon_id] * n_tokens)

        pad_id = self.tokenizer.pad_token_id
        input_ids_with_lcon: List[List[int]] = []
        for ids, suffix in zip(input_ids_list, suffixes):
            ids = list(ids)
            while ids and ids[-1] == pad_id:
                ids = ids[:-1]
            input_ids_with_lcon.append(ids + suffix)

        lcon_collated = {'input_ids': input_ids_with_lcon}
        lcon_collated = self.tokenizer.pad(
            lcon_collated,
            padding=True,
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors='pt',
        )
        self._attach_image_tensors(lcon_collated, image_meta)

        # Locate <|lcon|> positions per sample. Fixed KL-only novis runs keep
        # a constant number of latent slots and intentionally do not ship bbox
        # visual-token targets to the model.
        max_lcon = max(lcon_counts, default=0)
        include_visual_targets = self.is_qwen and not fixed_kl_novis and lcon_count_overrides is None
        max_visual = (
            max((int(x.numel()) for x in visual_token_indices_list), default=0)
            if include_visual_targets
            else 0
        )
        lcon_positions = torch.full((bsz, max_lcon), fill_value=-1, dtype=torch.long)
        last_lcon_positions = torch.full((bsz,), fill_value=-1, dtype=torch.long)
        visual_token_indices = (
            torch.full((bsz, max_visual), fill_value=-1, dtype=torch.long)
            if include_visual_targets
            else None
        )
        visual_token_mask = (
            torch.zeros(bsz, max_visual, dtype=torch.bool)
            if include_visual_targets
            else None
        )
        lcon_token_count = torch.zeros(bsz, dtype=torch.long)
        ids_np = lcon_collated['input_ids']
        for i in range(ids_np.size(0)):
            row = ids_np[i].tolist()
            idxs = [j for j, t in enumerate(row) if t == self.lcon_id]
            if idxs:
                last_lcon_positions[i] = int(idxs[-1])
            vt = visual_token_indices_list[i]
            n = min(len(idxs), lcon_counts[i], max_lcon)
            # The only <|lcon|> tokens are appended at the end, but slicing the
            # tail keeps this robust if a tokenizer ever emits the id elsewhere.
            idxs = idxs[-n:]
            if n > 0:
                lcon_positions[i, :n] = torch.tensor(idxs, dtype=torch.long)
            if include_visual_targets:
                v = min(int(vt.numel()), max_visual)
                if v > 0:
                    visual_token_indices[i, :v] = vt[:v].to(dtype=torch.long)
                    visual_token_mask[i, :v] = True
            lcon_token_count[i] = n
        lcon_collated['lcon_positions'] = lcon_positions
        lcon_collated['last_lcon_positions'] = last_lcon_positions
        if include_visual_targets:
            lcon_collated['visual_token_indices'] = visual_token_indices
            lcon_collated['visual_token_mask'] = visual_token_mask
        lcon_collated['lcon_token_count'] = lcon_token_count

        # Pad bbox lists to a uniform shape so we can ship them as tensors.
        max_boxes = 0
        for b in all_bboxes:
            if b:
                max_boxes = max(max_boxes, len(b))
        bsz = len(all_bboxes)
        if max_boxes == 0:
            bbox_tensor = torch.zeros(bsz, 0, 4)
            bbox_count = torch.zeros(bsz, dtype=torch.long)
        else:
            bbox_tensor = torch.zeros(bsz, max_boxes, 4)
            bbox_count = torch.zeros(bsz, dtype=torch.long)
            for i, b in enumerate(all_bboxes):
                if not b:
                    continue
                k = min(len(b), max_boxes)
                bbox_tensor[i, :k] = torch.tensor(b[:k], dtype=torch.float32)
                bbox_count[i] = k
        lcon_collated['bbox'] = bbox_tensor
        lcon_collated['bbox_count'] = bbox_count
        lcon_collated['bbox_image_size'] = bbox_image_size
        lcon_collated['bbox_image_size_mask'] = bbox_image_size_mask

        return lcon_collated

    def __call__(self, features: List[Tuple]):
        if len(features[0]) == 2:
            all_queries = [f[0] for f in features]
            all_images = [f[-1] for f in features]
            d_exists = ['no'] * len(features)
            describes = [''] * len(features)
            all_bboxes: List[List[List[float]]] = [[] for _ in features]
            all_bbox_image_sizes = [None for _ in features]
        else:
            all_queries = [f[0] for f in features]
            all_images = [f[1] for f in features]
            d_exists = [f[2] for f in features]
            describes = [f[3] if len(f) > 3 else '' for f in features]
            all_bboxes = [f[4] if len(f) > 4 else [] for f in features]
            all_bbox_image_sizes = [f[5] if len(f) > 5 else None for f in features]

        d_exist_ids = torch.tensor(
            [1 if de == 'yes' else 0 for de in d_exists],
            dtype=torch.long
        )

        q_collated = self.tokenizer(
            all_queries,
            padding=False, 
            truncation=True,
            max_length=self.data_args.query_max_len-1 if self.data_args.append_eos_token else self.data_args.query_max_len,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=True,
        )

        d_collated = {}
        d_input_ids, d_image_meta = self._encode_images(all_images)
        d_collated['input_ids'] = d_input_ids

        if self.data_args.append_eos_token:
            q_collated['input_ids'] = [q + [self.tokenizer.eos_token_id] for q in q_collated['input_ids']]
            d_collated['input_ids'] = [d + [self.tokenizer.eos_token_id] for d in d_collated['input_ids']]

        if self.data_args.pretrain:
            p_collated = {}
            all_input_ids, all_label_ids, input_lengths = [], [], []

            for i, ocr in enumerate(all_queries):
                prompt_input_ids = torch.tensor(d_collated['input_ids'][i]).unsqueeze(0)
                answer = f'{ocr}<|im_end|>' if self.is_qwen else f'{ocr}<|end|>\n<|endoftext|>'
                answer_input_ids = self.tokenizer(
                    answer, add_special_tokens=False, max_length=self.data_args.answer_max_len, truncation=True, return_tensors='pt')['input_ids']
                input_ids = torch.cat([prompt_input_ids, answer_input_ids], dim=1)
                labels = torch.cat(
                    [
                        torch.tensor([-100] * len(prompt_input_ids[0])).unsqueeze(0),
                        answer_input_ids,
                    ],
                    dim=1,
                )
                all_input_ids.append(input_ids.squeeze(0).unsqueeze(1))
                all_label_ids.append(labels.squeeze(0).unsqueeze(1))
                input_lengths.append(len(prompt_input_ids[0]))

            input_ids = torch._C._nn.pad_sequence(
                all_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            ).squeeze(2)
            labels = torch._C._nn.pad_sequence(
                all_label_ids, batch_first=True, padding_value=-100
            ).squeeze(2)

            p_collated['input_ids'] = input_ids
            p_collated['labels'] = labels

            if self.data_args.image_attention_mask:
                image_attention_mask = self.build_image_attention_mask(input_ids.size()[1], input_lengths)
                p_collated['attention_mask'] = image_attention_mask.unsqueeze(1)
        else:
            p_collated = None

        q_collated = self.tokenizer.pad(
            q_collated,
            padding=True, 
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors='pt',
        )
        d_collated = self.tokenizer.pad(
            d_collated,
            padding=True, 
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors='pt',
        )

        self._attach_image_tensors(d_collated, d_image_meta)
        if self.data_args.pretrain:
            p_collated['pixel_values'] = d_collated['pixel_values']
            if 'image_grid_thw' in d_collated:
                p_collated['image_grid_thw'] = d_collated['image_grid_thw']
            if 'image_sizes' in d_collated:
                p_collated['image_sizes'] = d_collated['image_sizes']

        all_describes_text = []
        for i in range(len(features)):
            if d_exists[i] == 'yes' and describes[i] and describes[i].strip():
                all_describes_text.append(describes[i])
            else:
                all_describes_text.append("N/A")

        q_describe_collated = self.tokenizer(
            all_describes_text,
            padding=False,
            truncation=True,
            max_length=self.data_args.query_max_len-1 if self.data_args.append_eos_token else self.data_args.query_max_len,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=True,
        )

        if self.data_args.append_eos_token:
            q_describe_collated['input_ids'] = [q + [self.tokenizer.eos_token_id] for q in q_describe_collated['input_ids']]

        q_describe_collated = self.tokenizer.pad(
            q_describe_collated,
            padding=True,
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors='pt',
        )

        # Latent-concept branch (only when explicitly enabled).
        lcon_collated = self._build_lcon_collated(
            all_queries, all_images, all_bboxes, all_bbox_image_sizes,
        )

        if lcon_collated is None:
            return q_collated, d_collated, p_collated, q_describe_collated, d_exist_ids
        return q_collated, d_collated, p_collated, q_describe_collated, d_exist_ids, lcon_collated

@dataclass
class EncodeCollator:
    data_args: DataArguments
    tokenizer: PreTrainedTokenizer
    processor: ProcessorMixin

    def __post_init__(self):
        self.is_qwen = 'qwen' in type(self.processor).__name__.lower()

    def _encode_images(self, images):
        if self.is_qwen:
            messages_list = [[{
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "What is shown in this image?"}
                ]
            }] for img in images]
            texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                     for msg in messages_list]
            batch_inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
            input_ids_list = [batch_inputs['input_ids'][i].tolist() for i in range(len(images))]
            return input_ids_list, batch_inputs
        else:
            collated_list = [self.processor("<|image_1|>\nWhat is shown in this image?", img, return_tensors="pt")
                             for img in images]
            input_ids_list = []
            for d in collated_list:
                ids = d['input_ids']
                if torch.is_tensor(ids):
                    while ids.dim() > 1:
                        ids = ids[0]
                    ids = ids.tolist()
                else:
                    while isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], list):
                        ids = ids[0]
                input_ids_list.append([int(t) for t in ids])
            return input_ids_list, collated_list

    def __call__(self, features: List[Tuple[str, str]]):
        text_ids = [x[0] for x in features]
        texts = [x[1] for x in features]
        images = [x[-1] for x in features]

        if self.data_args.encode_is_query:
            collated = self.tokenizer(
                texts,
                padding=False,
                truncation=True,
                max_length=self.data_args.query_max_len-1 if self.data_args.append_eos_token else self.data_args.query_max_len,
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=True,
            )
            image_meta = None
        else:
            collated = {}
            input_ids_list, image_meta = self._encode_images(images)
            collated['input_ids'] = input_ids_list

        if self.data_args.append_eos_token:
            collated['input_ids'] = [x + [self.tokenizer.eos_token_id] for x in collated['input_ids']]

        collated = self.tokenizer.pad(
            collated,
            padding=True,
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors='pt',
        )
        if not self.data_args.encode_is_query and image_meta is not None:
            if self.is_qwen:
                collated['pixel_values'] = image_meta['pixel_values']
                if 'image_grid_thw' in image_meta:
                    collated['image_grid_thw'] = image_meta['image_grid_thw']
                elif 'image_sizes' in image_meta:
                    collated['image_sizes'] = image_meta['image_sizes']
            else:
                collated['pixel_values'] = torch.stack([d['pixel_values'][0] for d in image_meta], dim=0)
                collated['image_sizes'] = torch.stack([d['image_sizes'][0] for d in image_meta], dim=0)

        return text_ids, collated
