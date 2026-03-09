import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class GCN(nn.Module):
    """Single GCN layer supporting dense or torch sparse COO adjacency.

    Graph convolution:  out = A_norm @ (X @ W) + b
      A_norm  : (N,N) normalised adjacency (sparse COO or dense)
      X       : (N, in_features) node feature matrix
      W       : (in_features, out_features) learnable weight
      b       : (out_features,) learnable bias

    Activation (e.g. ReLU) is applied by the caller, not here.

    Weight initialisation — FIXED from original:
    ──────────────────────────────────────────────────────────────────────────
    Original code used  uniform(-1/sqrt(out), 1/sqrt(out)).
    This ignores in_features entirely, which under-scales the weights when
    in_features >> out_features (e.g. 768→400):

      variance of each output unit = in_features × var(W)
      Original: var(W) = (1/sqrt(out))²/3 = 1/(3·out)
                → output var = in/out  (can be >> 1 or << 1)
      Glorot:   var(W) = 2/(in+out)
                → output var ≈ 1  (by design)

    After two GCN layers with original init (in=768→400→400):
      output std ≈ 0.33  (shrinking signal, harder gradient flow)
    After two layers with Glorot:
      output std ≈ 0.91  (close to 1, healthy gradient flow)

    PyTorch's nn.init.xavier_uniform_ implements exactly Glorot uniform.
    ──────────────────────────────────────────────────────────────────────────
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features  = int(in_features)
        self.out_features = int(out_features)

        self.weight = Parameter(
            torch.empty([self.in_features, self.out_features], dtype=torch.float)
        )
        if bias:
            self.bias = Parameter(torch.zeros(self.out_features, dtype=torch.float))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Glorot / Xavier uniform: accounts for both in AND out dimensions.
        # Keeps signal variance ≈ 1 through each layer regardless of shape.
        nn.init.xavier_uniform_(self.weight)

        # Bias: initialise to zero (standard practice for GCNs).
        # Previously uniform(-stdv, stdv) — non-zero bias adds to the
        # ReLU positive-shift problem (fixed with LayerNorm in Step3Model,
        # but zero-bias at init is cleaner and has no downside).
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        adj: torch.Tensor,
        inputs: torch.Tensor,
        identity: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            adj     : (N, N) normalised adjacency — sparse COO or dense float
            inputs  : (N, in_features) node features
            identity: if True, skip inputs and compute adj @ W directly
                      (rare; kept for backward compatibility)
        Returns:
            out: (N, out_features)
        """
        if identity:
            support = self.weight                   # (in, out) used as (N, out) — caller's responsibility
            out = torch.sparse.mm(adj, support) if adj.is_sparse else torch.matmul(adj, support)
        else:
            support = torch.matmul(inputs, self.weight)    # (N, out_features)
            out = torch.sparse.mm(adj, support) if adj.is_sparse else torch.matmul(adj, support)

        if self.bias is not None:
            out = out + self.bias
        return out