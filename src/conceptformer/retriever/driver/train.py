import logging
import os
import sys
import torch

from transformers import AutoTokenizer
from transformers import AutoProcessor 

from transformers import (
    HfArgumentParser,
    set_seed,
)

from conceptformer.retriever.arguments import ModelArguments, DataArguments, \
    ConceptFormerTrainingArguments as TrainingArguments
from conceptformer.retriever.dataset import TrainDataset
from conceptformer.retriever.collator import TrainCollator
from conceptformer.retriever.modeling import ConceptFormerRetriever
from conceptformer.retriever.trainer import ConceptFormerRetrieverTrainer as Trainer
from conceptformer.retriever.latent_concepts import (
    LCON_SPECIAL_TOKENS,
    get_latent_mse_weight,
    get_concept_cache_align_weight,
)

logger = logging.getLogger(__name__)


def _maybe_register_latent_tokens(tokenizer, data_args: DataArguments) -> int:
    """Add the single latent concept token to the tokenizer when the latent
    branch is enabled. Returns the number of tokens that were actually added.
    """
    latent_enabled = (
        getattr(data_args, 'latent_align_mode', 'none') != 'none'
        or get_latent_mse_weight(data_args) > 0.0
        or get_concept_cache_align_weight(data_args) > 0.0
    )
    if not latent_enabled:
        return 0
    to_add = []
    for tok in LCON_SPECIAL_TOKENS:
        try:
            existing = tokenizer.convert_tokens_to_ids(tok)
        except Exception:
            existing = None
        if existing is None or existing == tokenizer.unk_token_id:
            to_add.append(tok)
    if not to_add:
        return 0
    n_added = tokenizer.add_special_tokens({'additional_special_tokens': to_add})
    logger.info("[latent concept] Registered %d new special tokens: %s", n_added, to_add)
    return n_added


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)
    logger.info(
        "[latent concept] latent_align_mode=%s lambda_forward=%s lambda_reverse=%s visual_weight=%s "
        "visual_loss=%s kl_variant=%s num_tokens=%s pooling=%s cache_align_weight=%s cache_steps=%s "
        "cache_pool=%s recurrent_kl=%s cache_tau=%s cache_symmetric=%s bbox_field=%s",
        data_args.latent_align_mode,
        data_args.latent_lambda_forward,
        data_args.latent_lambda_reverse,
        get_latent_mse_weight(data_args),
        data_args.latent_visual_loss_type,
        data_args.latent_kl_variant,
        data_args.latent_num_tokens,
        data_args.latent_pooling,
        get_concept_cache_align_weight(data_args),
        data_args.concept_cache_steps,
        data_args.concept_cache_pool,
        data_args.concept_recurrent_kl,
        data_args.concept_cache_align_tau,
        data_args.concept_cache_align_symmetric,
        data_args.bbox_field,
    )

    set_seed(training_args.seed)

    model_name = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
    is_qwen = 'qwen' in model_name.lower()

    processor_kwargs = dict(cache_dir=model_args.cache_dir, trust_remote_code=True)
    if is_qwen:
        processor_kwargs.update(
            min_pixels=data_args.qwen_min_pixels,
            max_pixels=data_args.qwen_max_pixels,
        )
        logger.info(
            "Qwen processor pixels: min=%s max=%s",
            data_args.qwen_min_pixels,
            data_args.qwen_max_pixels,
        )

    processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)
    tokenizer = processor.tokenizer

    if is_qwen and tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info(f"Set Qwen pad_token_id to eos_token_id: {tokenizer.eos_token_id}")

    # Register latent concept tokens before model build so embeddings can be resized.
    _maybe_register_latent_tokens(tokenizer, data_args)

    if training_args.bf16:
        torch_dtype = torch.bfloat16
    elif training_args.fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    attn_implementation = (
        'eager' if data_args.pretrain or torch_dtype == torch.float32 else 'flash_attention_2'
    )
    logger.info(
        "Using torch_dtype=%s, attn_implementation=%s",
        torch_dtype,
        attn_implementation,
    )

    model = ConceptFormerRetriever.build(
        model_args,
        training_args,
        data_args=data_args,
        tokenizer=tokenizer,
        cache_dir=model_args.cache_dir,
        trust_remote_code=True,
        torch_dtype=torch_dtype, 
        _attn_implementation=attn_implementation,
    )

    train_dataset = TrainDataset(data_args)
    collator = TrainCollator(data_args, tokenizer, processor)

    # Sanity check: if LCON visual alignment is requested but the dataset has no bbox column,
    # fail loudly here rather than mid-training.
    if get_latent_mse_weight(data_args) > 0.0 and get_concept_cache_align_weight(data_args) > 0.0:
        raise ValueError(
            "concept_cache_align_weight > 0 replaces the old MSE/cos visual alignment branch; "
            "please set latent_mse_weight=0 when using recurrent cache alignment."
        )

    if (get_latent_mse_weight(data_args) > 0.0 or get_concept_cache_align_weight(data_args) > 0.0) and train_dataset.bbox_field is None:
        raise ValueError(
            "latent concept visual/cache alignment was requested but no bbox column was found in the "
            "training dataset (auto-detection candidates: model_boxes/bbox/bbox_2d/bboxes/boxes/area/regions, "
            "or pass --bbox_field <name>). Either disable LCON visual alignment or use a dataset with bbox supervision."
        )

    trainer_cls = Trainer

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        data_args=data_args
    )
    train_dataset.trainer = trainer

    trainer.train()
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
