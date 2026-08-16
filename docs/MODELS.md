# Models

The released model repositories contain PEFT adapters and ConceptFormer sidecar weights,
not copies of the public base-model weights. This keeps the release compact while retaining
the exact trained parameters. The evaluation loader downloads or opens the declared base
model and merges the adapter at load time.

## ConceptFormer-Qwen

- Base model: `Qwen/Qwen2.5-VL-7B-Instruct`
- Adapter: LoRA rank 8, alpha 64, dropout 0.1
- Pooling: EOS / last valid token
- Latent concept token: `<|lcon|>`
- Objective: contrastive retrieval plus forward distribution KL, weight 0.2
- Training: 3 epochs, bfloat16, effective batch size 128 on four GPUs

## ConceptFormer-Phi3V

- Base model: `Tevatron/dse-phi3-docmatix-v1`
- Initialization adapter: `NTT-hil-insight/VDocRetriever-Phi3-vision-pretrained`
- Adapter: LoRA rank 8, alpha 64, dropout 0.1
- Pooling: EOS / last valid token
- Latent concept token: `<|lcon|>`
- Objective: contrastive retrieval plus forward distribution KL, weight 0.2
- Training: 3 epochs, bfloat16, effective batch size 128 on four GPUs

The `conceptformer_state.pt` sidecar stores the latent projection used during training.
Standard query/document encoding uses the released LoRA adapter. Tokenizer files contain
one ConceptFormer-specific token and no hidden boundary tokens.
