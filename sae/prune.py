import os

import torch
from sae_lens import SAE

from lib.prune import prune_reconstruct
from lib.sparsegpt import SparseGPT
from .sparsegpt import SAESparseGPT


def load_sae(sae_path, layer_id, device):
    """Load the SAE trained on the input to layer layer_id's MLP."""
    return SAE.load_from_disk(os.path.join(sae_path, f"layers.{layer_id}"),
                              device=str(device), dtype="float32")


def prune_sae_sparsegpt(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0,
                        sae_path=None, starts_with=None, true_sequential=False,
                        baseline_fallback=None, offload_to_cpu=False):
    print('Starting SAE-SPARSEGPT ...')
    print(f"SAE path: {sae_path}")

    # up_proj and gate_proj share an input, so they share one SAE per layer.
    layer_ids = {mod: i for i, layer in enumerate(model.model.layers)
                 for mod in (layer.mlp.up_proj, layer.mlp.gate_proj)}
    cache = {}

    def sae_for(module):
        i = layer_ids[module]
        if i not in cache:
            cache.clear()
            cache[i] = load_sae(sae_path, i, module.weight.device)
        return cache[i]

    def make_collector(name, module):
        if baseline_fallback is not None and name in baseline_fallback:
            return SparseGPT(module)
        # Only the MLP input has an SAE, so other layers always use SparseGPT.
        if module not in layer_ids:
            return SparseGPT(module)
        return SAESparseGPT(module, sae_for(module))

    prune_reconstruct(args, model, tokenizer, device, make_collector,
                      prune_n=prune_n, prune_m=prune_m, starts_with=starts_with,
                      true_sequential=true_sequential, offload_to_cpu=offload_to_cpu)
