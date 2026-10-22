import torch

from lib.sparsegpt import SparseGPT

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class SAESparseGPT(SparseGPT):
    """
    Pruning using the Hessian of SAE reconstructions.

    Each calibration input X is replaced by its reconstruction
    X_hat = W_dec f(X) + b_dec before H is accumulated, so pruning preserves
    the layer's output on the part of the input the SAE explains.

    The fraction of variance the SAE leaves unexplained (FVU) is tracked and
    printed before pruning, since later layers see already-pruned activations.
    """

    def __init__(self, layer, sae):
        super().__init__(layer)
        self.sae = sae
        self.err = 0
        self.total = torch.zeros(self.columns, device=self.dev)
        self.sq = 0
        self.count = 0

    def add_batch(self, inp, out):
        inp = inp.float()
        inp_hat = self.sae.decode(self.sae.encode(inp))
        x = inp.reshape((-1, inp.shape[-1]))
        self.err += (inp - inp_hat).pow(2).sum().item()
        self.total += x.sum(dim=0)
        self.sq += x.pow(2).sum().item()
        self.count += x.shape[0]
        super().add_batch(inp_hat, out)

    def fvu(self):
        var = self.sq - self.total.pow(2).sum().item() / self.count
        return self.err / var

    def fasterprune(self, *args, **kwargs):
        print(f"sae fvu {self.fvu():.4f}")
        super().fasterprune(*args, **kwargs)

    def free(self):
        self.sae = None
        self.total = None
        super().free()
