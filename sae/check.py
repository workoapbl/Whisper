import argparse

import torch
from transformers import AutoTokenizer

from lib.data import get_loaders
from main import get_llm_from_path
from .prune import load_sae


@torch.no_grad()
def main():
    """
    Report how well each SAE reconstructs the MLP input on WikiText-2 calibration data.

    Run on the dense model before pruning to catch hook or scale mismatches
    between SAE training and the pruning hook.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="meta-llama/Llama-3.2-1B", help='Path to the model')
    parser.add_argument('--sae_path', type=str, required=True, help='Directory of per-layer SAEs (layers.{i}).')
    parser.add_argument('--nsamples', type=int, default=8, help='Number of calibration samples.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    args = parser.parse_args()

    model = get_llm_from_path(args.model)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    device = torch.device("cuda:0")
    dataloader, _ = get_loaders("wikitext2", nsamples=args.nsamples, seed=args.seed,
                                seqlen=model.seqlen, tokenizer=tokenizer)

    for i, layer in enumerate(model.model.layers):
        norm = layer.post_attention_layernorm
        sae = load_sae(args.sae_path, i, norm.weight.device)

        acts = []
        handle = norm.register_forward_hook(lambda _, inp, out: acts.append(out.float()))
        for batch in dataloader:
            model(batch[0].to(device))
        handle.remove()

        x = torch.cat(acts).reshape((-1, model.config.hidden_size))
        err, l0 = 0, 0
        for chunk in x.split(model.seqlen):
            feats = sae.encode(chunk)
            err += (chunk - sae.decode(feats)).pow(2).sum().item()
            l0 += (feats != 0).sum().item()
        fvu = err / (x - x.mean(dim=0)).pow(2).sum().item()
        print(f"layer {i} fvu {fvu:.4f} l0 {l0 / x.shape[0]:.1f}")


if __name__ == '__main__':
    main()
