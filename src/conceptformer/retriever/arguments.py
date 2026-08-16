import os
from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    pooling: str = field(
        default='cls',
        metadata={"help": "pooling method for query and document encoder"}
    )
    normalize: bool = field(
        default=False,
        metadata={"help": "normalize query and document representations"}
    )

    temperature: float = field(
        default=1.0,
        metadata={"help": "temperature for softmax"}
    )

    lora: bool = field(default=False,
        metadata={"help": "do parameter-efficient fine-tuning with lora"}
    )

    lora_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained lora model or model identifier from huggingface.co/models"}
    )

    lora_r: int = field(
        default=8,
        metadata={"help": "lora r"}
    )

    lora_alpha: int = field(
        default=64,
        metadata={"help": "lora alpha"}
    )

    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "lora dropout"}
    )

    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",
        metadata={"help": "lora target modules"}
    )

    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": "Floating-point format in which the model weights should be initialized and trained. Choose one "
                    "of `[float32, float16, bfloat16]`. "
        },
    )

@dataclass
class DataArguments:
    dataset_name: str = field(
        default='NTT-hil-insight/OpenDocVQA', metadata={"help": "huggingface dataset name"}
    )
    dataset_config: str = field(
        default=None, metadata={"help": "huggingface dataset config, useful for datasets with sub-datasets"}
    )

    dataset_path: str = field(
        default=None, metadata={"help": "Path to local data files or directory"}
    )

    dataset_split: str = field(
        default='train', metadata={"help": "dataset split"}
    )

    dataset_cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the data downloaded from huggingface"}
    )

    corpus_name: str = field(
        default='NTT-hil-insight/OpenDocVQA-Corpus', metadata={"help": "huggingface dataset name"}
    )

    corpus_config: str = field(
        default=None, metadata={"help": "huggingface dataset config, useful for datasets with sub-datasets"}
    )

    corpus_path: str = field(
        default=None, metadata={"help": "Path to local data files or directory"}
    )

    corpus_split: str = field(
        default='train', metadata={"help": "dataset split"}
    )

    dataset_number_of_shards: int = field(
        default=1, metadata={"help": "number of shards to split the dataset into"}
    )

    dataset_shard_index: int = field(
        default=0, metadata={"help": "shard index to use, to be used with dataset_number_of_shards"}
    )

    train_group_size: int = field(
        default=8, metadata={"help": "number of documents used to train for each query"}
    )
    positive_document_no_shuffle: bool = field(
        default=False, metadata={"help": "always use the first positive document for training"})

    image_attention_mask: bool = field(
        default=False, metadata={"help": "custom attention mask for RCG task"})

    pretrain: bool = field(
        default=False, metadata={"help": "whether pre-training is executed or not"})

    encode_is_query: bool = field(default=False)

    encode_output_path: str = field(default=None, metadata={"help": "where to save the encode"})

    query_max_len: Optional[int] = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for query. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    answer_max_len: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for document. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    qwen_min_pixels: int = field(
        default=256 * 28 * 28,
        metadata={"help": "Minimum Qwen image pixels passed to the processor."},
    )

    qwen_max_pixels: int = field(
        default=1280 * 28 * 28,
        metadata={"help": "Maximum Qwen image pixels passed to the processor."},
    )

    append_eos_token: bool = field(
        default=False, metadata={"help": "append eos token to query and document, this is currently used for repllama"}
    )

    pad_to_multiple_of: Optional[int] = field(
        default=16,
        metadata={
            "help": "If set will pad the sequence to a multiple of the provided value. This is especially useful to "
                    "enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta)."
        },
    )

    training_mode: str = field(
        default='mixed',
        metadata={"help": "Training mode: 'text_only' (only text batch), 'image_only' (only image batch), "
                         "'sequential' (text first then image), 'mixed' (mixed training)"}
    )

    # kl_loss_weight controls KL(P_q || P_t), where t is the annotated
    # (a.k.a. text_kl_weight in the spec). It controls KL(P_q || P_t) where t is
    # the reasoning-guided textual description embedding. The new latent
    # evidence KL weights (latent_lambda_forward / latent_lambda_reverse) and
    # visual-patch alignment weight (latent_mse_weight) are **separate** and additive.
    kl_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Text-description KL weight. Set to 0 to disable text KL."}
    )

    image_sample_strategy: str = field(
        default='random',
        metadata={"help": "Strategy to sample images from query_image/relevant_doc_ids list: 'random', 'first', 'all'"}
    )

    # ------------------------------------------------------------------
    # Latent Concept Alignment (latent concept) extension
    # ------------------------------------------------------------------
    # Textual KL alignment stays controlled by
    # `kl_loss_weight` above (a.k.a. text_kl_weight). The fields below add a
    # *separate* latent concept branch that learns dynamic latent tokens c_{q,d}
    # whose induced ranking distribution P_c is aligned with the query
    # distribution P_q via KL(s), and whose token-level states are aligned with
    # original-image bbox visual patch embeddings. All defaults make this a
    # no-op so standard contrastive training behaves identically when nothing
    # is set.
    latent_align_mode: str = field(
        default='none',
        metadata={
            "help": "Latent concept KL mode: 'none' (disabled), 'forward' "
                    "(KL(sg[P_q] || P_c)), 'reverse' (KL(sg[P_c] || P_q)) or 'both'."
        }
    )

    latent_lambda_forward: float = field(
        default=0.0,
        metadata={"help": "Weight for KL(sg[P_q] || P_c). Used when latent_align_mode is 'forward' or 'both'."}
    )

    latent_lambda_reverse: float = field(
        default=0.0,
        metadata={"help": "Weight for KL(sg[P_c] || P_q). Used when latent_align_mode is 'reverse' or 'both'."}
    )

    latent_mse_weight: float = field(
        default=0.0,
        metadata={"help": "Weight for dynamic LCON-to-bbox-visual-patch alignment. 0 disables this branch."}
    )

    latent_visual_loss_type: str = field(
        default='mse',
        metadata={"help": "Token-level LCON visual alignment loss: 'mse' or 'cosine'/'repa'."}
    )

    latent_kl_variant: str = field(
        default='q2concept',
        metadata={
            "help": "Latent KL distribution variant: 'q2concept' aligns q->image with q->latent concept; "
                    "'concept2image' aligns q->image with latent concept->image; "
                    "'q2concept+concept2image' applies both q2concept and concept2image KL terms; "
                    "'bbox2image' aligns q->image with pooled bbox image-token -> image."
        }
    )

    latent_gamma_roi_mse: float = field(
        default=0.0,
        metadata={"help": "Deprecated alias for latent_mse_weight; kept for older scripts."}
    )

    latent_num_tokens: int = field(
        default=0,
        metadata={
            "help": "If >0, use this fixed number of LCON tokens for KL-only novis runs. "
                    "Default 0 keeps dynamic bbox-selected token count."
        }
    )

    latent_roi_loss_type: str = field(
        default='ln_mse',
        metadata={"help": "Deprecated and ignored; use latent_visual_loss_type instead."}
    )

    latent_pooling: str = field(
        default='mean',
        metadata={"help": "How to pool latent states C -> E_c: 'mean', 'last', or 'attention'."}
    )

    concept_cache_align_weight: float = field(
        default=0.0,
        metadata={"help": "Weight for recurrent latent concept cache-to-ROI visual InfoNCE alignment. 0 disables this branch."}
    )

    concept_cache_steps: int = field(
        default=8,
        metadata={"help": "Number of recurrent latent concept cache rollout steps."}
    )

    concept_cache_pool: str = field(
        default='mean',
        metadata={"help": "How to pool recurrent cache states for KL/cache alignment: 'mean' or 'last'."}
    )

    concept_recurrent_kl: bool = field(
        default=False,
        metadata={
            "help": "If true, use recurrent latent concept rollout states for latent KL even when "
                    "cache alignment is disabled."
        }
    )

    concept_recurrent_impl: str = field(
        default='exact',
        metadata={
            "help": "Implementation for recurrent latent concept states: 'exact' feeds h_{s-1} "
                    "as the next latent input via iterative full forwards; "
                    "'causal_slots' uses a single Latent-VC SFT-style causal forward "
                    "over fixed <|lcon|> slots."
        }
    )

    concept_cache_align_tau: float = field(
        default=0.07,
        metadata={"help": "Temperature for recurrent cache-to-ROI visual InfoNCE."}
    )

    concept_cache_align_symmetric: bool = field(
        default=False,
        metadata={"help": "If true, average latent->ROI and ROI->latent InfoNCE losses."}
    )

    concept_cache_align_detach_target: bool = field(
        default=True,
        metadata={"help": "Detach pooled ROI visual targets before cache alignment."}
    )

    bbox_field: Optional[str] = field(
        default=None,
        metadata={
            "help": "Explicit dataset column name that stores per-sample bounding boxes "
                    "(xyxy in original-image pixel space when an image_size-like column is present; "
                    "otherwise xyxy in resized image space). If None, auto-detect among "
                    "['model_boxes', 'bbox', 'bbox_2d', 'bboxes', 'boxes', 'area', 'regions']."
        }
    )


@dataclass
class ConceptFormerTrainingArguments(TrainingArguments):
    warmup_ratio: float = field(default=0.1)
