import argparse

import torch
from datasets import load_dataset
from sae_lens import (LoggingConfig, MultiSAETrainingRunner, MultiSAETrainingRunnerConfig,
                      TopKTrainingSAEConfig)
from transformers import AutoConfig

# Training is throughput bound, and TF32 is accurate enough for SAE training.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main():
    """
    Train one TopK SAE per layer on the input to the MLP (up_proj and gate_proj).

    The hook sits on the output of post_attention_layernorm, which is exactly
    the tensor the MLP projections receive, RMSNorm gain included. All layers
    are trained from a single shared forward pass and saved to
    <output_path>/layers.{i}.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="meta-llama/Llama-3.2-1B", help='Path to the model')
    parser.add_argument('--output_path', type=str, required=True, help='Directory to save the trained SAEs to.')
    parser.add_argument('--layers', type=int, nargs='+', default=None, help='Layers to train SAEs for (default: all).')
    parser.add_argument('--d_sae', type=int, default=32768, help='Number of SAE latents.')
    parser.add_argument('--k', type=int, default=64, help='Number of active latents per token.')
    parser.add_argument('--training_tokens', type=int, default=300_000_000, help='Number of training tokens.')
    parser.add_argument('--context_size', type=int, default=1024, help='Tokens per training sequence.')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for SAE initialization and data order.')
    args = parser.parse_args()

    model_config = AutoConfig.from_pretrained(args.model)
    layers = args.layers if args.layers is not None else list(range(model_config.num_hidden_layers))

    # WikiText-103 test is WikiText-2 test, the eval set, so train on the train split only.
    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    dataset = dataset.filter(lambda x: x["text"].strip() != "")

    cfg = MultiSAETrainingRunnerConfig(
        saes={f"layers.{i}": TopKTrainingSAEConfig(d_in=model_config.hidden_size,
                                                   d_sae=args.d_sae, k=args.k)
              for i in layers},
        hook_names={f"layers.{i}": f"model.layers.{i}.post_attention_layernorm" for i in layers},
        model_name=args.model,
        model_class_name="AutoModelForCausalLM",
        # The multi-hook store keeps activations in the model's dtype, so load the
        # model in float32 to match the SAEs, which train under bf16 autocast.
        model_from_pretrained_kwargs={"dtype": "float32"},
        autocast=True,
        context_size=args.context_size,
        training_tokens=args.training_tokens,
        # The buffer holds activations for every hook, so keep it small.
        n_batches_in_buffer=4,
        store_batch_size_prompts=8,
        lr=args.lr,
        lr_warm_up_steps=1000,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=args.seed,
        logger=LoggingConfig(log_to_wandb=False),
        output_path=args.output_path,
    )
    MultiSAETrainingRunner(cfg, override_dataset=dataset).run()


if __name__ == '__main__':
    main()
