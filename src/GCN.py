import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class GCN(nn.Module):
    """Single GCN layer supporting dense or torch sparse COO adjacency."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)

        self.weight = Parameter(torch.empty([self.in_features, self.out_features], dtype=torch.float))
        if bias:
            self.bias = Parameter(torch.empty([self.out_features], dtype=torch.float))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, adj: torch.Tensor, inputs: torch.Tensor, identity: bool = False) -> torch.Tensor:
        """
        Args:
            adj: (N,N) torch sparse COO or dense
            inputs: (N,Fin) dense (ignored if identity=True)
            identity: if True, compute adj @ W (rare; kept for compatibility)
        """
        if identity:
            if adj.is_sparse:
                out = torch.sparse.mm(adj, self.weight)
            else:
                out = torch.matmul(adj, self.weight)
        else:
            support = torch.matmul(inputs, self.weight)  # (N, Fout)
            if adj.is_sparse:
                out = torch.sparse.mm(adj, support)
            else:
                out = torch.matmul(adj, support)

        if self.bias is not None:
            out = out + self.bias
        return out
