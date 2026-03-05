# step2_global_encoder_fixed.py
from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import scipy.sparse as sp
except Exception as e:
    raise SystemExit("scipy is required to load sparse adjacencies") from e

GLOBAL_GCN_DIM = 400

def load_pkl(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)

def load_feature(path: Path) -> torch.Tensor:
    """
    Loads .pkl/.npy feature matrices into torch.float32 dense.
    Handles:
      - scipy sparse matrices (e.g., pos_onehot.pkl / empath_onehot.pkl)
      - object arrays / list-of-arrays (e.g., embeddings saved as list)
    """
    if path.suffix == ".npy":
        obj = np.load(path, allow_pickle=True)
    else:
        obj = load_pkl(path)

    # 1) scipy sparse -> dense numpy
    if "sp" in globals() and sp.issparse(obj):
        arr = obj.toarray()
        return torch.tensor(arr, dtype=torch.float32)

    # 2) list/tuple -> stack
    if isinstance(obj, (list, tuple)):
        arr = np.asarray(obj, dtype=object)
        if arr.dtype == object:
            arr = np.stack([np.asarray(x, dtype=np.float32) for x in obj], axis=0)
        else:
            arr = arr.astype(np.float32, copy=False)
        return torch.tensor(arr, dtype=torch.float32)

    # 3) numpy array
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            # common cases: object array of vectors, or array of arrays
            try:
                arr = obj.astype(np.float32)
            except Exception:
                arr = np.stack([np.asarray(x, dtype=np.float32) for x in obj], axis=0)
        else:
            arr = obj.astype(np.float32, copy=False)
        return torch.tensor(arr, dtype=torch.float32)

    # 4) numpy conversion - Others
    arr = np.asarray(obj)
    if arr.dtype == object:
        arr = arr.astype(np.float32)
    else:
        arr = arr.astype(np.float32, copy=False)
    return torch.tensor(arr, dtype=torch.float32)


def scipy_to_torch_coo(A: Any) -> torch.Tensor:
    """Convert scipy sparse (or numpy dense) to torch sparse COO."""
    if sp.issparse(A):
        A = A.tocoo()
        idx = torch.tensor(np.vstack([A.row, A.col]), dtype=torch.long)
        val = torch.tensor(A.data, dtype=torch.float32)
        return torch.sparse_coo_tensor(idx, val, size=A.shape).coalesce()

    A = sp.coo_matrix(np.asarray(A))
    return scipy_to_torch_coo(A)


def normalize_adj_coo(adj: torch.Tensor) -> torch.Tensor:
    """
    Symmetric normalization: D^{-1/2} A D^{-1/2}.
    Assumes adj includes self-loops (your global graphs set diagonal=1).
    """
    adj = adj.coalesce()
    idx = adj.indices()
    val = adj.values()

    deg = torch.sparse.sum(adj, dim=1).to_dense()
    deg_inv_sqrt = torch.pow(deg.clamp(min=1e-12), -0.5)

    val_norm = val * deg_inv_sqrt[idx[0]] * deg_inv_sqrt[idx[1]]
    return torch.sparse_coo_tensor(idx, val_norm, adj.shape).coalesce()


@dataclass
class GlobalEmbeddings:
    word: torch.Tensor
    pos: torch.Tensor
    liwc: torch.Tensor
    entity: torch.Tensor
    text: torch.Tensor


class TwoLayerGCN(nn.Module):
    """2-layer GCN built from a single-layer GCN module."""

    def __init__(self, gcn_layer_cls: type[nn.Module], in_dim: int, hid_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.g1 = gcn_layer_cls(in_dim, hid_dim)
        self.g2 = gcn_layer_cls(hid_dim, out_dim)
        self.dropout = float(dropout)

    def forward(self, adj_norm: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        h = self.g1(adj_norm, x)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.g2(adj_norm, h)
        return h


class Step2GlobalEncoder(nn.Module):
    """
    Step 2: global node encoding for 5 views.
    This matches the paper: 2-layer GCN per global graph view.
    """

    def __init__(
        self,
        gcn_layer_cls: type[nn.Module],
        dims_in: Dict[str, int],
        hid_dim: int,
        out_dim: int,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.word = TwoLayerGCN(gcn_layer_cls, dims_in["word"], hid_dim, out_dim, dropout)
        self.pos = TwoLayerGCN(gcn_layer_cls, dims_in["pos"], hid_dim, out_dim, dropout)
        self.liwc = TwoLayerGCN(gcn_layer_cls, dims_in["liwc"], hid_dim, out_dim, dropout)
        self.entity = TwoLayerGCN(gcn_layer_cls, dims_in["entity"], hid_dim, out_dim, dropout)
        self.text = TwoLayerGCN(gcn_layer_cls, dims_in["text"], hid_dim, out_dim, dropout)

    def forward(self, A_norm: Dict[str, torch.Tensor], X: Dict[str, torch.Tensor]) -> GlobalEmbeddings:
        return GlobalEmbeddings(
            word=self.word(A_norm["word"], X["word"]),
            pos=self.pos(A_norm["pos"], X["pos"]),
            liwc=self.liwc(A_norm["liwc"], X["liwc"]),
            entity=self.entity(A_norm["entity"], X["entity"]),
            text=self.text(A_norm["text"], X["text"]),
        )


def load_step2_inputs(out_dir: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, int]]:
    """
    Load Step-2 inputs from your global graph output folder.

    Returns:
      A_norm: normalized torch sparse adjacencies (dict)
      X:      dense node features (dict)
      dims_in: feature dims for each view
    """
    p = Path(out_dir)

    adj_paths = {
        "word": p / "adj_word.pkl",
        "pos": p / "adj_tag.pkl",
        "liwc": p / "adj_empath.pkl",
        "entity": p / "adj_entity.pkl",
        "text": p / "adj_text.pkl",
    }

    A_norm: Dict[str, torch.Tensor] = {}
    for k, ap in adj_paths.items():
        if not ap.exists():
            raise FileNotFoundError(f"Missing adjacency for view '{k}': {ap}")
        A = load_pkl(ap)
        A_norm[k] = normalize_adj_coo(scipy_to_torch_coo(A))

    feat_paths = {
        "word": p / "word_type_bert_emb.pkl",
        "pos": p / "pos_onehot.pkl",
        "liwc": p / "empath_onehot.pkl",
        "entity": p / "entity_emb.pkl",
        "text": p / "doc_bert_emb.npy",
    }

    X: Dict[str, torch.Tensor] = {}
    for k, fp in feat_paths.items():
        if not fp.exists():
            raise FileNotFoundError(f"Missing feature for view '{k}': {fp}")
        X[k] = load_feature(fp)

    # Basic row-dimension sanity
    for k in adj_paths:
        if A_norm[k].shape[0] != X[k].shape[0]:
            raise ValueError(f"Shape mismatch for view '{k}': A is {tuple(A_norm[k].shape)}, X is {tuple(X[k].shape)}")

    dims_in = {k: int(X[k].shape[1]) for k in X}
    return A_norm, X, dims_in

def save_global_embeddings(out_dir: str, gcn_layer_cls: type[nn.Module], hid_dim: int = GLOBAL_GCN_DIM, out_dim: int = GLOBAL_GCN_DIM) -> None:
    A_norm, X, dims_in = load_step2_inputs(out_dir)
    enc = Step2GlobalEncoder(gcn_layer_cls, dims_in, hid_dim, out_dim, dropout=0.5)
    enc.eval()
    with torch.no_grad():
        H = enc(A_norm, X)

    p = Path(out_dir)
    torch.save(
        {
            "H_word": H.word.cpu(),
            "H_pos": H.pos.cpu(),
            "H_liwc": H.liwc.cpu(),
            "H_entity": H.entity.cpu(),
            "H_text": H.text.cpu(),
        },
        p / "H_views.pt",
    )
    print("Saved:", p / "H_views.pt")


def sanity_forward(out_dir: str, gcn_layer_cls: type[nn.Module], hid_dim: int = GLOBAL_GCN_DIM, out_dim: int = GLOBAL_GCN_DIM) -> None:
    """One-time check that Step 2 runs and shapes match."""
    A_norm, X, dims_in = load_step2_inputs(out_dir)
    enc = Step2GlobalEncoder(gcn_layer_cls, dims_in, hid_dim, out_dim, dropout=0.5)
    enc.eval()
    with torch.no_grad():
        H = enc(A_norm, X)
    print("H_word:", tuple(H.word.shape))
    print("H_pos:", tuple(H.pos.shape))
    print("H_liwc:", tuple(H.liwc.shape))
    print("H_entity:", tuple(H.entity.shape))
    print("H_text:", tuple(H.text.shape))
