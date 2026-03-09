# file: src/train_end2end.py
from __future__ import annotations

import collections
import csv
import json
import pickle
import random
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union, Optional, Callable

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from GCN import GCN
from local_graph import LocalGraphBuilder, build_X_local_from_Hviews

try:
    from global_encoder import GLOBAL_GCN_DIM
except Exception:
    GLOBAL_GCN_DIM = 400


USER_COL = "ID_COL"
TRAIT_NAMES = ["OPEN", "CON", "EXT", "AGR", "NEU"]


# -------------------------
# utils
# -------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _f1_per_label(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12):
    """Per-label precision, recall, F1. Single unified path — no sklearn dependency.
    All three returned arrays are guaranteed consistent: 2*p*r/(p+r) == f1.
    """
    tp = (y_true * y_pred).sum(axis=0).astype(np.float64)
    fp = ((1 - y_true) * y_pred).sum(axis=0).astype(np.float64)
    fn = (y_true * (1 - y_pred)).sum(axis=0).astype(np.float64)
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    f1 = np.nan_to_num(f1, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return precision.astype(np.float32), recall.astype(np.float32), f1


def label_cols_from_method(method: str) -> List[str]:
    m = str(method).strip().lower()
    if m not in {"mean", "median", "kmeans"}:
        raise ValueError(f"Unknown label_method='{method}'. Expected one of: mean, median, kmeans")
    return [
        f"OPEN_bin_{m}",
        f"CONSICEN_bin_{m}",
        f"EXTRO_bin_{m}",
        f"AGREE_bin_{m}",
        f"NEURO_bin_{m}",
    ]


def load_split_doc_indices(out_dir: Union[str, Path], split: str) -> List[int]:
    fp = Path(out_dir) / f"{split}_idx.json"
    if not fp.exists():
        raise FileNotFoundError(f"Missing split index file: {fp}")
    return list(map(int, json.loads(fp.read_text(encoding="utf-8"))))


def _read_csv_rows(csv_path: Union[str, Path]) -> List[Dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            raise ValueError(f"Empty CSV or missing header: {csv_path}")
        return list(r)


def load_labels_users_aligned(
    csv_path: Union[str, Path],
    doc_indices: Sequence[int],
    label_cols: Sequence[str],
    user_col: str = USER_COL,
) -> Tuple[Dict[int, np.ndarray], Dict[int, int]]:
    rows = _read_csv_rows(csv_path)
    if len(rows) != len(doc_indices):
        raise ValueError(
            f"Row count mismatch for {Path(csv_path).name}: csv_rows={len(rows)} but split_indices={len(doc_indices)}. "
            f"Ensure CSV has exactly the same rows in the same order as doc_indices."
        )

    missing = [c for c in label_cols if c not in rows[0]]
    if missing:
        raise KeyError(f"{Path(csv_path).name} missing label columns: {missing}. Found: {list(rows[0].keys())}")
    if user_col not in rows[0]:
        raise KeyError(f"{Path(csv_path).name} missing user id column '{user_col}'. Found: {list(rows[0].keys())}")

    labels: Dict[int, np.ndarray] = {}
    users: Dict[int, int] = {}
    invalid_label_count = 0

    for i, doc_idx in enumerate(doc_indices):
        row = rows[i]
        try:
            y = np.array([float(row[c]) for c in label_cols], dtype=np.float32)
        except (ValueError, KeyError) as e:
            raise ValueError(f"Failed to parse labels for row {i} (doc_idx={doc_idx}): {e}")
        
        if not np.isfinite(y).all():
            invalid_label_count += 1
            # Skip this row with invalid labels
            continue
        
        labels[int(doc_idx)] = y

        u_raw = row[user_col]
        try:
            u = int(float(u_raw))
        except (ValueError, TypeError):
            # Hash string user IDs consistently
            u = abs(hash(u_raw)) % (2**31 - 1)
        users[int(doc_idx)] = u

    if invalid_label_count > 0:
        print(f"[WARNING] {Path(csv_path).name}: Skipped {invalid_label_count} rows with non-finite labels", flush=True)
    
    if len(labels) == 0:
        raise ValueError(f"No valid labels loaded from {csv_path}")
    
    return labels, users


def _safe_item(x: torch.Tensor) -> float:
    return float(torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).item())


def compute_pos_rates_from_labels_dict(labels_dict: Dict[int, np.ndarray]) -> np.ndarray:
    Y = np.stack(list(labels_dict.values()), axis=0).astype(np.float32)
    return (Y > 0.5).mean(axis=0)


def sanity_label_rates_from_dict(labels_dict: Dict[int, np.ndarray], name: str) -> None:
    rates = compute_pos_rates_from_labels_dict(labels_dict)
    print(f"[SANITY] {name} label pos-rates: " + ", ".join([f"{TRAIT_NAMES[i]}={rates[i]:.3f}" for i in range(5)]), flush=True)


# -------------------------
# sparse helpers
# -------------------------
def scipy_to_torch_coo(A: sp.spmatrix) -> torch.Tensor:
    A = A.tocoo()
    idx = torch.tensor(np.vstack([A.row, A.col]), dtype=torch.long)
    val = torch.tensor(A.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, size=A.shape).coalesce()


def normalize_adj_coo(adj: torch.Tensor) -> torch.Tensor:
    adj = adj.coalesce()
    idx = adj.indices()
    val = adj.values()
    deg = torch.sparse.sum(adj, dim=1).to_dense()
    deg_inv_sqrt = torch.pow(deg.clamp(min=1e-12), -0.5)
    val_norm = val * deg_inv_sqrt[idx[0]] * deg_inv_sqrt[idx[1]]
    return torch.sparse_coo_tensor(idx, val_norm, adj.shape).coalesce()


def block_diag_sparse(adjs: List[torch.Tensor]) -> torch.Tensor:
    rows: List[torch.Tensor] = []
    cols: List[torch.Tensor] = []
    vals: List[torch.Tensor] = []
    offset = 0
    total_n = 0

    for a in adjs:
        a = a.coalesce()
        ij = a.indices()
        vv = a.values()
        rows.append(ij[0] + offset)
        cols.append(ij[1] + offset)
        vals.append(vv)
        n = int(a.shape[0])
        offset += n
        total_n += n

    if total_n == 0:
        return torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros((0,), dtype=torch.float32),
            (0, 0),
        )

    IJ = torch.stack([torch.cat(rows), torch.cat(cols)], dim=0)
    V = torch.cat(vals)
    return torch.sparse_coo_tensor(IJ, V, (total_n, total_n)).coalesce()


# -------------------------
# step2 loader
# -------------------------
def _load_pkl(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_feature(path: Path) -> torch.Tensor:
    """Load feature matrix from pickle or npy, handling edge cases."""
    if path.suffix == ".npy":
        obj = np.load(path, allow_pickle=True)
    else:
        obj = _load_pkl(path)

    if sp.issparse(obj):
        result = torch.tensor(obj.toarray(), dtype=torch.float32)
    elif isinstance(obj, (list, tuple)):
        arr = np.asarray(obj, dtype=object)
        if arr.dtype == object:
            arr = np.stack([np.asarray(x, dtype=np.float32) for x in obj], axis=0)
        else:
            arr = arr.astype(np.float32, copy=False)
        result = torch.tensor(arr, dtype=torch.float32)
    elif isinstance(obj, np.ndarray):
        if obj.dtype == object:
            try:
                arr = obj.astype(np.float32)
            except Exception:
                arr = np.stack([np.asarray(x, dtype=np.float32) for x in obj], axis=0)
        else:
            arr = obj.astype(np.float32, copy=False)
        result = torch.tensor(arr, dtype=torch.float32)
    else:
        arr = np.asarray(obj, dtype=np.float32)
        result = torch.tensor(arr, dtype=torch.float32)
    
    # Validate result is 2D
    if result.dim() != 2:
        raise RuntimeError(f"Expected 2D feature tensor, got {result.shape} from {path.name}")
    
    return result


def load_step2_inputs_liwc(
    out_dir: Union[str, Path],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, int]]:
    p = Path(out_dir)

    adj_paths: Dict[str, Path] = {
        "word": p / "adj_word.pkl",
        "pos": p / "adj_tag.pkl",
        "entity": p / "adj_entity.pkl",
        "text": p / "adj_text.pkl",
    }
    liwc_adj = p / "adj_liwc.pkl"
    if not liwc_adj.exists():
        liwc_adj = p / "adj_empath.pkl"
    adj_paths["liwc"] = liwc_adj

    feat_paths: Dict[str, Path] = {
        "word": p / "word_type_bert_emb.pkl",
        "pos": p / "pos_onehot.pkl",
        "entity": p / "entity_emb.pkl",
        "text": p / "doc_bert_emb.npy",
    }
    liwc_feat = p / "liwc_onehot.pkl"
    if not liwc_feat.exists():
        liwc_feat = p / "empath_onehot.pkl"
    feat_paths["liwc"] = liwc_feat

    # Validate all files exist
    for k, ap in adj_paths.items():
        if not ap.exists():
            raise FileNotFoundError(f"Missing adjacency for view '{k}': {ap}")
    for k, fp in feat_paths.items():
        if not fp.exists():
            raise FileNotFoundError(f"Missing feature for view '{k}': {fp}")

    A_norm: Dict[str, torch.Tensor] = {}
    for k, ap in adj_paths.items():
        A = _load_pkl(ap)
        if not sp.issparse(A):
            A = sp.coo_matrix(np.asarray(A))
        A_norm[k] = normalize_adj_coo(scipy_to_torch_coo(A))
        print(f"[LOAD] Adjacency '{k}': shape={tuple(A_norm[k].shape)}, nnz={A_norm[k].coalesce()._nnz()}", flush=True)

    X: Dict[str, torch.Tensor] = {}
    for k, fp in feat_paths.items():
        X[k] = _load_feature(fp)
        print(f"[LOAD] Feature '{k}': shape={tuple(X[k].shape)}", flush=True)

    # Validate shapes match
    for k in adj_paths:
        if A_norm[k].shape[0] != X[k].shape[0]:
            raise ValueError(
                f"Shape mismatch for view '{k}': adjacency {tuple(A_norm[k].shape)} vs feature {tuple(X[k].shape)}. "
                f"Number of nodes must match."
            )

    dims_in = {k: int(X[k].shape[1]) for k in X}
    print(f"[LOAD] Feature dimensions: {dims_in}", flush=True)
    return A_norm, X, dims_in


# -------------------------
# models
# -------------------------
class TwoLayerGCN(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.g1 = GCN(in_dim, hid_dim)
        self.g2 = GCN(hid_dim, out_dim)
        self.dropout = float(dropout)

    def forward(self, A_norm: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        h = self.g1(A_norm, X)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.g2(A_norm, h)
        return h


class GlobalEncoder(nn.Module):
    def __init__(self, dims_in: Dict[str, int], hid_dim: int = GLOBAL_GCN_DIM, out_dim: int = GLOBAL_GCN_DIM, dropout: float = 0.3):
        super().__init__()
        self.word = TwoLayerGCN(dims_in["word"], hid_dim, out_dim, dropout)
        self.pos = TwoLayerGCN(dims_in["pos"], hid_dim, out_dim, dropout)
        self.liwc = TwoLayerGCN(dims_in["liwc"], hid_dim, out_dim, dropout)
        self.entity = TwoLayerGCN(dims_in["entity"], hid_dim, out_dim, dropout)
        self.text = TwoLayerGCN(dims_in["text"], hid_dim, out_dim, dropout)

    def forward(self, A_norm: Dict[str, torch.Tensor], X: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            "H_word": self.word(A_norm["word"], X["word"]),
            "H_pos": self.pos(A_norm["pos"], X["pos"]),
            "H_liwc": self.liwc(A_norm["liwc"], X["liwc"]),
            "H_entity": self.entity(A_norm["entity"], X["entity"]),
            "H_text": self.text(A_norm["text"], X["text"]),
        }


# -------------------------
# losses
# -------------------------
class BalancedBCELoss(nn.Module):
    """Standard BCE loss — correct choice for our balanced 50/50 OCEAN data.

    WHY NOT ASL (AsymmetricLoss) for this dataset:
    ─────────────────────────────────────────────
    ASL (Ridnik 2021) was designed for SPARSE multi-label data (MS-COCO, ImageNet)
    where positive labels are rare (~1–10% per class). Its key asymmetry:
      gamma_pos=1, gamma_neg=2 → negative focal weight is stronger

    On BALANCED data (50/50) this asymmetry is HARMFUL:
      At p=0.5, balanced batch:
        Positive gradient: (1-p)^gp / p × p(1-p) = 0.423  (pushes logit UP for y=1)
        Negative gradient: p_m^gn / (1-p_m) × p(1-p) = 0.227  (pushes logit DOWN for y=0)
        Net gradient = (−0.423 + 0.227) / 2 = −0.098  → SYSTEMATICALLY increases logit

    ASL's positive gradient is 1.87× stronger than its negative gradient on our data.
    Result: all logits drift positive within 25 training steps → PosRate=1.0 collapse.

    BCE has perfectly symmetric gradients on balanced data:
      d(−log p)/d(logit) = −(1−p),   d(−log(1−p))/d(logit) = +p
      At p=0.5: −0.5 and +0.5 → net gradient = 0 on balanced batch  ✓

    Switch back to ASL only if/when traits become severely imbalanced (pos_rate < 20%).
    """
    def __init__(self, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        # pos_weight allows per-trait re-weighting if imbalance emerges later
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pw = self.pos_weight
        if pw is not None:
            pw = pw.to(logits.device)
        out = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)
        if not torch.isfinite(out):
            print(f"[BCE WARNING] non-finite loss, returning 0. "
                  f"logit range=[{logits.min().item():.2f}, {logits.max().item():.2f}]",
                  flush=True)
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        return out


class UserLevelAggregator(nn.Module):
    def forward(self, z: torch.Tensor, user_ids: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        unique_users = torch.unique(user_ids, sorted=True)
        agg_z, agg_y = [], []
        for uid in unique_users:
            mask = user_ids == uid
            agg_z.append(z[mask].mean(dim=0))
            agg_y.append((y[mask].mean(dim=0) >= 0.5).float())
        return torch.stack(agg_z, dim=0), torch.stack(agg_y, dim=0)


class LocalGCNEncoder(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int = 400, out_dim: int = 400, dropout: float = 0.3):
        super().__init__()
        self.g1 = GCN(in_dim, hid_dim)
        self.g2 = GCN(hid_dim, out_dim)
        self.dropout = float(dropout)

    def forward(self, adj_norm: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        h = self.g1(adj_norm, x)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.g2(adj_norm, h)


class UserSelfAttention(nn.Module):
    """Scaled dot-product attention with a same-user bias.

    same_user_bias adds a fixed value to attention logits for posts from the
    same user, so they attend more to each other.

    CAUTION on same_user_bias magnitude:
      With 32 posts/batch (16 users × 2), bias=5.0 gives same-user attention
      weight ≈ 83% — the model almost completely ignores cross-user context.
      bias=2.0 gives ≈ 20%, keeping meaningful cross-user attention.
      Default changed from 5.0 → 2.0.  Override via SAME_USER_BIAS env var.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.q     = nn.Linear(dim, dim, bias=False)
        self.k     = nn.Linear(dim, dim, bias=False)
        self.v     = nn.Linear(dim, dim, bias=False)
        self.scale = dim ** 0.5

    @staticmethod
    def build_S(user_ids: torch.Tensor) -> torch.Tensor:
        uid = user_ids.view(-1, 1)
        return (uid == uid.t()).float()

    def forward(self, x: torch.Tensor, user_ids: torch.Tensor,
                same_user_bias: float = 2.0) -> torch.Tensor:
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)
        logits = (Q @ K.t()) / self.scale
        S      = self.build_S(user_ids).to(logits.device)
        logits = logits + float(same_user_bias) * S
        A      = F.softmax(logits, dim=1)
        return A @ V


class Step3Model(nn.Module):
    """Local GCN → user self-attention → LayerNorm → classifier.

    WHY LayerNorm before cls:
      The GCN uses ReLU activations, so all intermediate features are >= 0.
      After user-level attention pooling, z_user is a non-negative vector.
      A random linear classifier applied to a non-negative input produces
      logits that are biased positive (half positive, half negative weights,
      but all-positive inputs → positive dot products dominate).
      This causes PosRate≈1.0 from step 1, before any learning.

      LayerNorm centres and scales z_user to mean=0, std=1 per sample,
      which ensures the classifier sees zero-mean input at initialisation,
      giving P(logit>0) ≈ 0.5 per label — the correct starting point.
    """
    def __init__(self, in_dim: int, hid_dim: int = 400, out_dim: int = 400,
                 dropout: float = 0.3, num_labels: int = 5):
        super().__init__()
        self.enc       = LocalGCNEncoder(in_dim, hid_dim=hid_dim, out_dim=out_dim, dropout=dropout)
        self.user_attn = UserSelfAttention(out_dim)
        self.norm      = nn.LayerNorm(out_dim)   # ← centres ReLU-biased embeddings
        self.cls       = nn.Linear(out_dim, num_labels)

    def forward(self, adj_norm: torch.Tensor, x: torch.Tensor,
                post_node_index: torch.Tensor, user_ids: torch.Tensor,
                same_user_bias: float) -> Tuple[torch.Tensor, torch.Tensor]:
        h       = self.enc(adj_norm, x)
        z       = h[post_node_index]
        z_user  = self.user_attn(z, user_ids, same_user_bias=same_user_bias)
        z_user  = self.norm(z_user)              # normalise before classifier
        return self.cls(z_user), z_user

    def cls_only(self, z_user: torch.Tensor) -> torch.Tensor:
        # Always re-normalise before cls: mean(LN(x1),LN(x2)) has std≈0.707,
        # not 1.0 — the aggregation undoes part of the LayerNorm scaling.
        return self.cls(self.norm(z_user))


def init_classifier_bias_from_priors(model: Step3Model, train_labels: Dict[int, np.ndarray]) -> None:
    priors = compute_pos_rates_from_labels_dict(train_labels)
    priors = np.clip(priors, 1e-4, 1 - 1e-4)
    bias = np.log(priors / (1 - priors)).astype(np.float32)
    with torch.no_grad():
        model.cls.bias.copy_(torch.tensor(bias, device=model.cls.bias.device))
    print("[init] cls.bias=logit(train_priors): " + ", ".join([f"{TRAIT_NAMES[i]}={bias[i]:+.3f}" for i in range(5)]), flush=True)


# -------------------------
# data
# -------------------------
class UserBatchSampler(torch.utils.data.Sampler[List[int]]):
    def __init__(self, doc_indices, users, batch_size, n_per_user=2, shuffle=True):
        self.batch_size = int(batch_size)
        self.n_per_user = int(max(1, n_per_user))
        self.shuffle = bool(shuffle)

        self.user_to_idx = collections.defaultdict(list)
        for idx, doc_idx in enumerate(doc_indices):
            u = users.get(int(doc_idx))
            if u is not None:
                self.user_to_idx[int(u)].append(int(idx))

    def __iter__(self):
        user_lists = {u: list(idx_list) for u, idx_list in self.user_to_idx.items()}
        if self.shuffle:
            for u in user_lists:
                random.shuffle(user_lists[u])

        active_users = [u for u in user_lists if len(user_lists[u]) > 0]
        while len(active_users) > 0:
            batch = []
            if self.shuffle:
                random.shuffle(active_users)

            users_per_batch = max(1, self.batch_size // self.n_per_user)
            users_to_pick = active_users[:users_per_batch]

            for u in users_to_pick:
                chunk = user_lists[u][: self.n_per_user]
                user_lists[u] = user_lists[u][self.n_per_user :]
                batch.extend(chunk)

            active_users = [u for u in active_users if len(user_lists[u]) > 0]
            if len(batch) > 0:
                yield batch

    def __len__(self) -> int:
        users_per_batch = max(1, self.batch_size // self.n_per_user)
        total_chunks = 0
        for idxs in self.user_to_idx.values():
            n = len(idxs)
            total_chunks += (n + self.n_per_user - 1) // self.n_per_user
        return max(1, (total_chunks + users_per_batch - 1) // users_per_batch)


class SplitDocDataset(Dataset):
    def __init__(self, out_dir: Union[str, Path], split: str, builder: LocalGraphBuilder):
        self.outp = Path(out_dir)
        self.builder = builder
        self.doc_indices = load_split_doc_indices(self.outp, split)

    def __len__(self) -> int:
        return len(self.doc_indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        doc_idx = int(self.doc_indices[idx])
        nodes, edges, adj = self.builder.build(doc_idx)
        return {"doc_idx": doc_idx, "nodes": nodes, "adj": adj}


def _get_post_lid(nodes: Any) -> int:
    post_node = None
    for n in nodes:
        ntype = None
        for attr in ("ntype", "type", "node_type"):
            v = getattr(n, attr, None)
            if v is not None:
                ntype = v
                break
        if ntype in ("post", "text", "doc"):
            post_node = n
            break
    if post_node is None:
        raise RuntimeError("No post/text/doc node found in local graph nodes.")
    for attr in ("local_id", "local", "lid", "id"):
        lid = getattr(post_node, attr, None)
        if lid is not None:
            return int(lid)
    raise RuntimeError("Post node has no local_id/local/lid/id attribute.")


@dataclass
class Batched:
    doc_idx: torch.Tensor
    user_id: torch.Tensor
    y: torch.Tensor
    x: torch.Tensor
    adj_norm: torch.Tensor
    post_node_index: torch.Tensor


def make_collate_fn(
    *,
    split: str,
    labels: Dict[int, np.ndarray],
    users: Dict[int, int],
    global_encoder: GlobalEncoder,
    A_norm: Dict[str, torch.Tensor],
    X: Dict[str, torch.Tensor],
    device: torch.device,
    H_views_override: Optional[Dict[str, torch.Tensor]] = None,
) -> Callable[[List[Dict[str, Any]]], Batched]:
    def collate_fn(items: List[Dict[str, Any]]) -> Batched:
        H_views = H_views_override
        if H_views is None:
            H_views = global_encoder(A_norm, X)

        doc_ids: List[int] = []
        user_ids: List[int] = []
        ys: List[np.ndarray] = []
        xs: List[torch.Tensor] = []
        adjs: List[torch.Tensor] = []
        post_idx: List[int] = []

        node_offset = 0
        for it in items:
            doc_idx = int(it["doc_idx"])
            if doc_idx not in labels or doc_idx not in users:
                raise KeyError(f"Missing label/user for doc_idx={doc_idx} in split={split}")

            nodes = it["nodes"]
            adj: sp.csr_matrix = it["adj"]
            post_lid = _get_post_lid(nodes)

            x_local = build_X_local_from_Hviews(nodes, doc_idx=doc_idx, H_views=H_views).to(device).float()
            adj_t = normalize_adj_coo(scipy_to_torch_coo(adj)).to(device)

            doc_ids.append(doc_idx)
            user_ids.append(int(users[doc_idx]))
            ys.append(labels[doc_idx])
            xs.append(x_local)
            adjs.append(adj_t)
            post_idx.append(node_offset + post_lid)

            node_offset += int(x_local.shape[0])

        Xb = torch.cat(xs, dim=0)
        Ab = block_diag_sparse(adjs)
        yb = torch.tensor(np.stack(ys, axis=0), dtype=torch.float32, device=device)
        doc_t = torch.tensor(doc_ids, dtype=torch.long, device=device)
        user_t = torch.tensor(user_ids, dtype=torch.long, device=device)
        post_t = torch.tensor(post_idx, dtype=torch.long, device=device)

        return Batched(doc_idx=doc_t, user_id=user_t, y=yb, x=Xb, adj_norm=Ab, post_node_index=post_t)

    return collate_fn


# -------------------------
# metrics
# -------------------------
@torch.no_grad()
def compute_f1_metrics(logits: torch.Tensor, y: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """Compute F1 metrics at sentence/batch level. Returns consistent dict."""
    probs = torch.sigmoid(logits)
    probs = torch.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
    
    y_np = y.detach().cpu().numpy().astype(np.int32)
    y_pred = (probs >= threshold).float().detach().cpu().numpy().astype(np.int32)
    
    _, _, f1_per_trait = _f1_per_label(y_np, y_pred)
    
    return {
        "macro_f1": float(f1_per_trait.mean()),
        "f1_open": float(f1_per_trait[0]),
        "f1_con": float(f1_per_trait[1]),
        "f1_ext": float(f1_per_trait[2]),
        "f1_agr": float(f1_per_trait[3]),
        "f1_neu": float(f1_per_trait[4]),
        "pos_rate": float((probs >= threshold).float().mean().item()),
        "true_rate": float(y.float().mean().item()),
    }


@torch.no_grad()
def metrics_from_logits(logits: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    """DEPRECATED: Use compute_f1_metrics instead. Kept for backwards compatibility."""
    return compute_f1_metrics(logits, y, threshold=0.5)


@torch.no_grad()
def run_eval(
    *,
    loader: DataLoader,
    model: Step3Model,
    bce: BalancedBCELoss,
    agg: UserLevelAggregator,
    same_user_bias: float,
) -> Dict[str, float]:
    """Evaluate model on user-level aggregated predictions."""
    tot_loss = tot_bce = 0.0
    nb = 0
    logits_all: List[torch.Tensor] = []
    y_all: List[torch.Tensor] = []

    for b in loader:
        # Forward pass at sentence level
        logits_sent, z_sent = model(b.adj_norm, b.x, b.post_node_index, b.user_id, same_user_bias=same_user_bias)
        
        # Aggregate to user level
        z_user, y_user = agg(z_sent, b.user_id, b.y)
        logits_user = model.cls_only(z_user)

        # Compute losses at user level
        l_bce = bce(logits_user, y_user)
        loss = l_bce

        # Accumulate
        tot_loss += _safe_item(loss)
        tot_bce += _safe_item(l_bce)
        nb += 1

        logits_all.append(logits_user.detach().cpu())
        y_all.append(y_user.detach().cpu())

    # Compute metrics on all accumulated user-level predictions
    if logits_all:
        logits_combined = torch.cat(logits_all, dim=0)
        y_combined = torch.cat(y_all, dim=0)
        
        probs = torch.sigmoid(logits_combined).numpy()
        y_true = y_combined.numpy().astype(np.int32)
        y_pred = (probs >= 0.5).astype(np.int32)
        
        _, _, f1_per_trait = _f1_per_label(y_true, y_pred)
        
        out = {
            "loss": tot_loss / max(nb, 1),
            "bce": tot_bce / max(nb, 1),
            "macro_f1": float(np.clip(f1_per_trait.mean(), 0.0, 1.0)),
        }
        for i, name in enumerate(TRAIT_NAMES):
            out[f"f1_{name}"] = float(np.clip(f1_per_trait[i], 0.0, 1.0))
    else:
        out = {
            "loss": 0.0,
            "bce": 0.0,
            "macro_f1": 0.0,
        }
        for name in TRAIT_NAMES:
            out[f"f1_{name}"] = 0.0
    
    return out


# -------------------------
# train
# -------------------------
def train(
    out_dir: str,
    train_csv: str,
    val_csv: str,
    test_csv: str,
    *,
    seed: int = 1,
    label_method: str = "mean",
    max_epochs: int = 1000,
    patience: int = 25,
    lr_local: float = 1e-4,
    lr_global: float = 1e-5,   # 10× lower than lr_local: global encoder on 53k-node graphs
    weight_decay: float = 5e-4,
    use_scheduler: bool = True,
    scheduler_eta_min: float = 1e-6,
    batch_size: int = 32,
    same_user_bias: float = 2.0,    # reduced from 5.0: see UserSelfAttention note
    grad_print_every: int = 25,
    max_grad_norm: float = 1.0,
    global_gcn_dim: int = GLOBAL_GCN_DIM,
    n_per_user: int = 2,
    local_dropout: float = 0.3,
    smooth_k: int = 3,  # rolling average window
) -> None:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outp = Path(out_dir)
    label_cols = label_cols_from_method(label_method)
    
    # ====== VALIDATION: Check all files exist ======
    for path_str, name in [(train_csv, "TRAIN_CSV"), (val_csv, "VAL_CSV"), (test_csv, "TEST_CSV")]:
        p = Path(path_str)
        if not p.exists():
            raise FileNotFoundError(f"{name} not found: {path_str}")
    
    if not outp.exists():
        raise FileNotFoundError(f"Output directory not found: {out_dir}")
    
    # ====== VALIDATION: Load indices ======
    try:
        train_idx = load_split_doc_indices(outp, "train")
        val_idx = load_split_doc_indices(outp, "val")
        test_idx = load_split_doc_indices(outp, "test")
        print(f"[VALIDATION] Loaded split indices: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}", flush=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load split indices from {outp}: {e}")
    
    # ====== VALIDATION: Load and align data ======
    try:
        train_labels, train_users = load_labels_users_aligned(train_csv, train_idx, label_cols=label_cols)
        val_labels, val_users = load_labels_users_aligned(val_csv, val_idx, label_cols=label_cols)
        test_labels, test_users = load_labels_users_aligned(test_csv, test_idx, label_cols=label_cols)
        print(f"[VALIDATION] Labels loaded: train={len(train_labels)}, val={len(val_labels)}, test={len(test_labels)}", flush=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load labels: {e}")
    
    if len(train_labels) == 0 or len(val_labels) == 0 or len(test_labels) == 0:
        raise RuntimeError("Empty label dictionary detected after loading")

    builder = LocalGraphBuilder(outp, entity_col="entity_lid", lru_cache_size=2000)

    A_norm, X, dims_in = load_step2_inputs_liwc(outp)
    A_norm = {k: v.to(device) for k, v in A_norm.items()}
    X = {k: v.to(device) for k, v in X.items()}

    # global_encoder dropout set to 0.0:
    # TwoLayerGCN has dropout INSIDE the encoder (between GCN layers).
    # When dropout > 0, H_views in training mode differ from eval mode
    # (train: 30% features zeroed + scaled 1.43×; eval: full features).
    # The local model trains on train-mode H_views → catastrophic mismatch at val.
    # Setting dropout=0 makes H_views IDENTICAL in train and eval → gap disappears.
    # Regularisation for the global encoder comes from weight decay (wd=5e-4).
    global_encoder = GlobalEncoder(dims_in, hid_dim=global_gcn_dim, out_dim=global_gcn_dim, dropout=0.0).to(device)

    # warm-start
    _pretrained_path = outp / "global_encoder_pretrained.pt"
    if _pretrained_path.exists():
        _ckpt = torch.load(str(_pretrained_path), map_location=device)
        global_encoder.load_state_dict(_ckpt, strict=False)
        print(f"[warm-start] Loaded global encoder weights from {_pretrained_path.name}", flush=True)
    else:
        print(f"[warm-start] No pretrained checkpoint found at {_pretrained_path.name} — training from scratch", flush=True)

    # infer local in_dim
    global_encoder.train()
    tmp_ds = SplitDocDataset(outp, "train", builder)
    tmp_sampler = UserBatchSampler(tmp_ds.doc_indices, train_users, batch_size=1, n_per_user=1, shuffle=False)
    tmp_loader = DataLoader(
        tmp_ds,
        batch_sampler=tmp_sampler,
        num_workers=0,
        collate_fn=make_collate_fn(
            split="train",
            labels=train_labels,
            users=train_users,
            global_encoder=global_encoder,
            A_norm=A_norm,
            X=X,
            device=device,
            H_views_override=None,
        ),
    )
    tmp_batch = next(iter(tmp_loader))

    model = Step3Model(in_dim=int(tmp_batch.x.shape[1]), num_labels=len(TRAIT_NAMES), dropout=local_dropout).to(device)
    init_classifier_bias_from_priors(model, train_labels)

    print("\n" + "=" * 80)
    print("SHINE END-TO-END TRAINING INITIALIZATION")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    if device.type == "cuda":
        print(f"[Device] cuda | {torch.cuda.get_device_name(0)} | {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
    else:
        print("[Device] cpu")
    print(f"[Splits] train_docs={len(train_labels)} val_docs={len(val_labels)} test_docs={len(test_labels)}")
    print(f"[Batch] docs={batch_size} n_per_user={n_per_user} approx_users/batch={max(1, batch_size//max(1,n_per_user))}")
    print(f"[Dropout] local_dropout={local_dropout} global_dropout=0.0 (zero to eliminate train/val H_views mismatch)")
    print(f"[Loss] BCE(balanced) — SupCon removed")
    print(f"[Opt] lr_local={lr_local:.1e} lr_global={lr_global:.1e} (10× lower) wd={weight_decay:.1e}")
    print(f"[Attn] same_user_bias={same_user_bias} (2.0=default; 5.0 makes model ignore cross-user context)")
    print("=" * 80 + "\n", flush=True)

    sanity_label_rates_from_dict(train_labels, "TRAIN")
    sanity_label_rates_from_dict(val_labels, "VAL")
    sanity_label_rates_from_dict(test_labels, "TEST")

    opt = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": lr_local, "weight_decay": weight_decay},
            {"params": global_encoder.parameters(), "lr": lr_global, "weight_decay": weight_decay},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=float(scheduler_eta_min)) if use_scheduler else None

    bce_loss = BalancedBCELoss()
    agg_train = UserLevelAggregator()
    agg_eval  = UserLevelAggregator()

    # datasets created once (avoid JSON re-reads)
    train_ds = SplitDocDataset(outp, "train", builder)
    val_ds = SplitDocDataset(outp, "val", builder)
    test_ds = SplitDocDataset(outp, "test", builder)

    train_sampler = UserBatchSampler(train_ds.doc_indices, train_users, batch_size=batch_size, n_per_user=n_per_user, shuffle=True)
    val_sampler   = UserBatchSampler(val_ds.doc_indices,   val_users,   batch_size=batch_size, n_per_user=n_per_user, shuffle=False)
    test_sampler  = UserBatchSampler(test_ds.doc_indices,  test_users,  batch_size=batch_size, n_per_user=n_per_user, shuffle=False)

    best_val = float("inf")
    bad = 0
    best_path = outp / "end2end_shine_best.pt"
    step = 0

    recent_val: List[float] = []

    for ep in range(1, max_epochs + 1):
        model.train()
        global_encoder.train()

        # ── Cache H_views ONCE per epoch ────────────────────────────────────
        # Previously H_views_override=None meant global_encoder was called
        # every batch (217× per epoch), each time with SLIGHTLY different
        # weights (because opt.step() ran between calls).  The local model
        # trained on 217 different H_views within one epoch, then faced yet
        # another H_views at epoch N+1 start → the loss spike pattern.
        #
        # Fix: compute H_views once, freeze for the entire epoch.  Effects:
        #   • H_views are CONSISTENT within each epoch (no intra-epoch drift)
        #   • Training is ~217× faster for the global encoder forward pass
        #   • Epoch-start spike eliminated (local model only adjusts once/epoch)
        #   • Train and val now use identically-structured cached H_views
        with torch.no_grad():
            H_views_train = global_encoder(A_norm, X)

        train_loader = DataLoader(
            train_ds,
            batch_sampler=UserBatchSampler(
                train_ds.doc_indices, train_users,
                batch_size=batch_size, n_per_user=n_per_user, shuffle=True,
            ),
            num_workers=0,
            collate_fn=make_collate_fn(
                split="train",
                labels=train_labels,
                users=train_users,
                global_encoder=global_encoder,
                A_norm=A_norm,
                X=X,
                device=device,
                H_views_override=H_views_train,   # ← cached, consistent
            ),
        )
        # ─────────────────────────────────────────────────────────────────────

        total = total_b = 0.0
        n_batches = 0
        train_logits_all: List[torch.Tensor] = []
        train_y_all: List[torch.Tensor] = []

        for bi, b in enumerate(train_loader, start=1):
            step += 1
            opt.zero_grad(set_to_none=True)

            logits_sent, z_sent = model(b.adj_norm, b.x, b.post_node_index, b.user_id, same_user_bias=same_user_bias)
            z_user, y_user = agg_train(z_sent, b.user_id, b.y)
            logits_user = model.cls_only(z_user)

            l_b = bce_loss(logits_user, y_user)
            loss = torch.nan_to_num(l_b, nan=0.0, posinf=0.0, neginf=0.0)

            loss.backward()
            if max_grad_norm is not None and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(global_encoder.parameters()),
                    max_norm=float(max_grad_norm),
                )

            opt.step()

            total += _safe_item(loss)
            total_b += _safe_item(l_b)
            n_batches += 1

            train_logits_all.append(logits_user.detach().cpu())
            train_y_all.append(y_user.detach().cpu())

            if step % grad_print_every == 0:
                stats         = compute_f1_metrics(logits_user, y_user, threshold=0.5)
                lr_local_now  = opt.param_groups[0]["lr"]
                lr_global_now = opt.param_groups[1]["lr"]
                # With LayerNorm+correct ASL, PosRate should never reach 1.0.
                # If this flag appears, check model architecture changes.
                collapse_flag = "  [!COLLAPSE — check BCE/LayerNorm]" if stats["pos_rate"] >= 0.95 else ""
                print(
                    f"[TRAIN] Ep={ep:03d} Batch={bi:04d}/{len(train_loader):04d} Step={step:06d} | "
                    f"B={int(y_user.shape[0]):02d} | "
                    f"Loss(avg)={total/max(n_batches,1):.4f} "
                    f"BCE(avg)={total_b/max(n_batches,1):.4f} "
                    f"MacroF1={stats['macro_f1']:.4f} | "
                    f"PosRate={stats['pos_rate']:.4f} TrueRate={stats['true_rate']:.4f}"
                    f"{collapse_flag} | "
                    f"LR(L/G)=({lr_local_now:.2e}/{lr_global_now:.2e})",
                    flush=True,
                )

        # ── epoch-level train metrics (user-level, full epoch) ───────────────
        train_probs     = torch.sigmoid(torch.cat(train_logits_all, dim=0)).numpy()
        train_true      = torch.cat(train_y_all, dim=0).numpy().astype(np.int32)
        train_pred      = (train_probs >= 0.5).astype(np.int32)
        _, _, train_f1s = _f1_per_label(train_true, train_pred)
        train_macro_f1  = float(train_f1s.mean())

        # ── eval (val only; test evaluated ONCE at end on best checkpoint) ───
        # Previously a test_loader was rebuilt here every epoch and never used —
        # that wasted one full global-encoder GCN pass per epoch.
        model.eval()
        global_encoder.eval()

        with torch.no_grad():
            H_views_eval = global_encoder(A_norm, X)

        val_loader = DataLoader(
            val_ds,
            batch_sampler=val_sampler,
            num_workers=0,
            collate_fn=make_collate_fn(
                split="val",
                labels=val_labels,
                users=val_users,
                global_encoder=global_encoder,
                A_norm=A_norm,
                X=X,
                device=device,
                H_views_override=H_views_eval,
            ),
        )

        va = run_eval(
            loader=val_loader, model=model,
            bce=bce_loss, agg=agg_eval,
            same_user_bias=same_user_bias,
        )

        model.train()
        global_encoder.train()

        if scheduler is not None:
            scheduler.step()

        recent_val.append(float(va["loss"]))
        smoothed = float(np.mean(recent_val[-max(1, int(smooth_k)):]))

        print(
            f"[EPOCH] Ep={ep:03d} | "
            f"Train Loss={total/max(n_batches,1):.4f} "
            f"MacroF1={train_macro_f1:.4f} "
            f"F1(O={train_f1s[0]:.3f} C={train_f1s[1]:.3f} E={train_f1s[2]:.3f} "
            f"A={train_f1s[3]:.3f} N={train_f1s[4]:.3f}) | "
            f"Val Loss={va['loss']:.4f} (smooth{smooth_k}={smoothed:.4f}) "
            f"MacroF1={va['macro_f1']:.4f} "
            f"F1(O={va['f1_OPEN']:.3f} C={va['f1_CON']:.3f} E={va['f1_EXT']:.3f} "
            f"A={va['f1_AGR']:.3f} N={va['f1_NEU']:.3f})",
            flush=True,
        )

        if math.isfinite(smoothed) and (smoothed + 1e-9 < best_val):
            prev = best_val
            best_val = smoothed
            bad = 0
            torch.save(
                {"local": model.state_dict(), "global": global_encoder.state_dict(), "epoch": ep, "best_val_smooth": best_val, "best_val_raw": float(va["loss"])},
                best_path,
            )
            print(f"[CHECKPOINT] Ep={ep:03d} NEW BEST | smooth_val {prev:.4f}->{best_val:.4f} saved {best_path.name}", flush=True)
        else:
            bad += 1
            print(f"[CHECKPOINT] Ep={ep:03d} no improve | best_smooth={best_val:.4f} patience {bad}/{patience}", flush=True)
            if bad >= patience:
                print(f"[EARLY STOP] stopped at ep={ep:03d} best_smooth_val={best_val:.4f}", flush=True)
                break

    print(f"\n[DONE] Best checkpoint: {best_path}", flush=True)

    # final test
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["local"])
    global_encoder.load_state_dict(ckpt["global"])

    model.eval()
    global_encoder.eval()

    with torch.no_grad():
        H_views_eval = global_encoder(A_norm, X)

    test_loader = DataLoader(
        test_ds,
        batch_sampler=test_sampler,
        num_workers=0,
        collate_fn=make_collate_fn(
            split="test",
            labels=test_labels,
            users=test_users,
            global_encoder=global_encoder,
            A_norm=A_norm,
            X=X,
            device=device,
            H_views_override=H_views_eval,
        ),
    )

    te = run_eval(loader=test_loader, model=model, bce=bce_loss, agg=agg_eval, same_user_bias=same_user_bias)
    print(
        f"\n[TEST] Loss={te['loss']:.4f} MacroF1={te['macro_f1']:.4f} | "
        f"F1(OPEN={te['f1_OPEN']:.4f}, CON={te['f1_CON']:.4f}, EXT={te['f1_EXT']:.4f}, AGR={te['f1_AGR']:.4f}, NEU={te['f1_NEU']:.4f})",
        flush=True,
    )


if __name__ == "__main__":
    import os

    ROOT = Path(__file__).resolve().parent
    OUT_DIR_DEFAULT = ROOT / "outputs" / "global_graph_output"
    DATA_PROCESSED_DEFAULT = ROOT / "data" / "processed" / "preprocess_check_out"

    OUT_DIR       = Path(os.getenv("GLOBAL_GRAPH_OUT", str(OUT_DIR_DEFAULT)))
    DATA_PROCESSED = Path(os.getenv("DATA_PROCESSED", str(DATA_PROCESSED_DEFAULT)))

    TRAIN_CSV = Path(os.getenv("TRAIN_CSV", str(DATA_PROCESSED / "final_train_preprocessed.csv")))
    VAL_CSV   = Path(os.getenv("VAL_CSV",   str(DATA_PROCESSED / "final_val_preprocessed.csv")))
    TEST_CSV  = Path(os.getenv("TEST_CSV",  str(DATA_PROCESSED / "final_test_preprocessed.csv")))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train(
        out_dir=str(OUT_DIR),
        train_csv=str(TRAIN_CSV),
        val_csv=str(VAL_CSV),
        test_csv=str(TEST_CSV),
        seed=int(os.getenv("SEED", "1")),
        label_method=os.getenv("LABEL_METHOD", "mean"),
        max_epochs=int(os.getenv("MAX_EPOCHS", "1000")),
        patience=int(os.getenv("PATIENCE", "25")),
        lr_local=float(os.getenv("LR_LOCAL", "1e-4")),
        lr_global=float(os.getenv("LR_GLOBAL", "1e-5")),
        weight_decay=float(os.getenv("WEIGHT_DECAY", "5e-4")),
        batch_size=int(os.getenv("BATCH_SIZE", "32")),
        same_user_bias=float(os.getenv("SAME_USER_BIAS", "2.0")),
        grad_print_every=int(os.getenv("GRAD_PRINT_EVERY", "25")),
        use_scheduler=os.getenv("USE_SCHEDULER", "1") not in {"0", "false", "False"},
        scheduler_eta_min=float(os.getenv("SCHED_ETA_MIN", "1e-6")),
        max_grad_norm=float(os.getenv("MAX_GRAD_NORM", "1.0")),
        global_gcn_dim=int(os.getenv("GLOBAL_GCN_DIM", str(GLOBAL_GCN_DIM))),
        n_per_user=int(os.getenv("N_PER_USER", "2")),
        local_dropout=float(os.getenv("LOCAL_DROPOUT", "0.3")),
        smooth_k=int(os.getenv("SMOOTH_K", "3")),
    )