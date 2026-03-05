# file: src/train_end2end.py
"""
SHINE-style training loop (max_epoch=1000, early stop on val loss patience=10)
with TRUE end-to-end trainability of Global Encoder parameters by recomputing H_views
PER BATCH (no detach), while keeping split isolation (no label leakage) and
transductive global graphs (built from all docs like SHINE/TextGCN).

Prereqs:
- Step1 artifacts already exist in OUT_DIR (global_graph_output):
  adj_word.pkl, adj_tag.pkl, adj_liwc.pkl, adj_entity.pkl, adj_text.pkl
  word_type_bert_emb.pkl, pos_onehot.pkl, liwc_onehot.pkl, entity_emb.pkl, doc_bert_emb.npy
  doc_word_seq.jsonl, word_pos_edges.csv, word_entity_edges.csv, liwc_word2cats.json, etc.

This script:
- Builds local graphs using LocalGraphBuilder (CSV/JSONL artifacts unchanged)
- Builds local node features using build_X_local_from_Hviews (unchanged)
- Trains:
  - Global Encoder parameters W1_tau/W2_tau (trainable, per batch)
  - local encoder parameters (local GCN)
  - user self-attention parameters (Wq/Wk/Wv)
  - classifier parameters
- Uses 2 LR groups (local faster than global)
- Logs rich training/validation metrics
"""


from __future__ import annotations

import collections
import csv
import json
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union, Optional

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datetime import datetime

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
    tp = (y_true * y_pred).sum(axis=0)
    fp = ((1 - y_true) * y_pred).sum(axis=0)
    fn = (y_true * (1 - y_pred)).sum(axis=0)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return precision, recall, f1


def label_cols_from_method(method: str) -> List[str]:
    """
    ISSUE 2 fix:
    build label columns dynamically (mean/median/kmeans)
    """
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
            f"Row count mismatch for {Path(csv_path).name}: csv_rows={len(rows)} but split_indices={len(doc_indices)}."
        )

    missing = [c for c in label_cols if c not in rows[0]]
    if missing:
        raise KeyError(f"{Path(csv_path).name} missing label columns: {missing}.")
    if user_col not in rows[0]:
        raise KeyError(f"{Path(csv_path).name} missing user id column '{user_col}'.")

    labels: Dict[int, np.ndarray] = {}
    users: Dict[int, int] = {}

    for i, doc_idx in enumerate(doc_indices):
        row = rows[i]
        y = np.array([float(row[c]) for c in label_cols], dtype=np.float32)
        labels[int(doc_idx)] = y

        u_raw = row[user_col]
        try:
            u = int(float(u_raw))
        except Exception:
            u = abs(hash(u_raw)) % (2**31 - 1)
        users[int(doc_idx)] = u

    return labels, users


def compute_class_distribution(labels_dict: Dict[int, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """Compute class distribution statistics for each trait"""
    if not labels_dict:
        return {}
    
    labels_array = np.array(list(labels_dict.values()))
    n_samples = len(labels_array)
    
    dist = {}
    for i, trait in enumerate(TRAIT_NAMES):
        trait_labels = labels_array[:, i]
        pos_rate = float((trait_labels > 0.5).mean())
        neg_rate = 1.0 - pos_rate
        dist[trait] = {
            "positive_rate": pos_rate,
            "negative_rate": neg_rate,
            "positive_count": int((trait_labels > 0.5).sum()),
            "negative_count": int((trait_labels <= 0.5).sum()),
            "mean": float(trait_labels.mean()),
            "std": float(trait_labels.std())
        }
    
    return dist


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
    """
    Robust feature loader: .pkl / .npy -> torch.float32 dense
    Handles scipy sparse + list/tuple + object arrays.
    """
    if path.suffix == ".npy":
        obj = np.load(path, allow_pickle=True)
    else:
        obj = _load_pkl(path)

    if sp.issparse(obj):
        return torch.tensor(obj.toarray(), dtype=torch.float32)

    if isinstance(obj, (list, tuple)):
        arr = np.asarray(obj, dtype=object)
        if arr.dtype == object:
            arr = np.stack([np.asarray(x, dtype=np.float32) for x in obj], axis=0)
        else:
            arr = arr.astype(np.float32, copy=False)
        return torch.tensor(arr, dtype=torch.float32)

    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            try:
                arr = obj.astype(np.float32)
            except Exception:
                arr = np.stack([np.asarray(x, dtype=np.float32) for x in obj], axis=0)
        else:
            arr = obj.astype(np.float32, copy=False)
        return torch.tensor(arr, dtype=torch.float32)

    arr = np.asarray(obj)
    if arr.dtype == object:
        arr = arr.astype(np.float32)
    else:
        arr = arr.astype(np.float32, copy=False)
    return torch.tensor(arr, dtype=torch.float32)


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

    X: Dict[str, torch.Tensor] = {k: _load_feature(fp) for k, fp in feat_paths.items()}

    for k in adj_paths:
        if A_norm[k].shape[0] != X[k].shape[0]:
            raise ValueError(f"Shape mismatch for '{k}': A {tuple(A_norm[k].shape)} vs X {tuple(X[k].shape)}")

    dims_in = {k: int(X[k].shape[1]) for k in X}
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
    def __init__(
        self,
        dims_in: Dict[str, int],
        hid_dim: int = GLOBAL_GCN_DIM,
        out_dim: int = GLOBAL_GCN_DIM,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.word = TwoLayerGCN(dims_in["word"], hid_dim, out_dim, dropout)
        self.pos = TwoLayerGCN(dims_in["pos"], hid_dim, out_dim, dropout)
        self.liwc = TwoLayerGCN(dims_in["liwc"], hid_dim, out_dim, dropout)
        self.entity = TwoLayerGCN(dims_in["entity"], hid_dim, out_dim, dropout)
        self.text = TwoLayerGCN(dims_in["text"], hid_dim, out_dim, dropout)
        
        # Store architecture info
        self.arch_info = {
            "input_dims": dims_in,
            "hidden_dim": hid_dim,
            "output_dim": out_dim,
            "dropout": dropout
        }

    def forward(self, A_norm: Dict[str, torch.Tensor], X: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            "H_word": self.word(A_norm["word"], X["word"]),
            "H_pos": self.pos(A_norm["pos"], X["pos"]),
            "H_liwc": self.liwc(A_norm["liwc"], X["liwc"]),
            "H_entity": self.entity(A_norm["entity"], X["entity"]),
            "H_text": self.text(A_norm["text"], X["text"]),
        }


def paper_beta(y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    y = (y > 0.5).float()
    dots = y @ y.t()
    denom = dots.sum(dim=1, keepdim=True).clamp_min(eps)
    return dots / denom


def paper_contrastive_loss(x_user: torch.Tensor, y: torch.Tensor, theta: float = 10.0, eps: float = 1e-12) -> torch.Tensor:
    B = int(x_user.shape[0])
    if B <= 1:
        return torch.zeros((), dtype=x_user.dtype, device=x_user.device)
    dist = torch.cdist(x_user, x_user, p=2)
    logits = -dist / max(float(theta), eps)
    log_p = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    beta = paper_beta(y, eps=eps).to(x_user.device)
    return -(beta * log_p).sum() / float(B)


class LocalGCNEncoder(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int = 400, out_dim: int = 400, dropout: float = 0.5):
        super().__init__()
        self.g1 = GCN(in_dim, hid_dim)
        self.g2 = GCN(hid_dim, out_dim)
        self.dropout = float(dropout)
        
        self.arch_info = {
            "input_dim": in_dim,
            "hidden_dim": hid_dim,
            "output_dim": out_dim,
            "dropout": dropout
        }

    def forward(self, adj_norm: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        h = self.g1(adj_norm, x)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.g2(adj_norm, h)


class UserSelfAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.scale = dim ** 0.5
        
        self.arch_info = {
            "dim": dim,
            "scale": float(self.scale)
        }

    @staticmethod
    def build_S(user_ids: torch.Tensor) -> torch.Tensor:
        uid = user_ids.view(-1, 1)
        return (uid == uid.t()).float()

    def forward(self, x: torch.Tensor, user_ids: torch.Tensor, same_user_bias: float = 5.0) -> torch.Tensor:
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)
        logits = (Q @ K.t()) / self.scale
        S = self.build_S(user_ids).to(logits.device)
        logits = logits + float(same_user_bias) * S
        A = F.softmax(logits, dim=1)
        return A @ V


class Step3Model(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int = 400, out_dim: int = 400, dropout: float = 0.5, num_labels: int = 5):
        super().__init__()
        self.enc = LocalGCNEncoder(in_dim, hid_dim=hid_dim, out_dim=out_dim, dropout=dropout)
        self.user_attn = UserSelfAttention(out_dim)
        self.cls = nn.Linear(out_dim, num_labels)
        
        self.arch_info = {
            "local_encoder": self.enc.arch_info,
            "attention": self.user_attn.arch_info,
            "classifier": {
                "input_dim": out_dim,
                "output_dim": num_labels
            }
        }

    def forward(
        self,
        adj_norm: torch.Tensor,
        x: torch.Tensor,
        post_node_index: torch.Tensor,
        user_ids: torch.Tensor,
        same_user_bias: float = 5.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(adj_norm, x)
        z = h[post_node_index]
        z_user = self.user_attn(z, user_ids, same_user_bias=same_user_bias)
        return self.cls(z_user), z_user


# -------------------------
# data
# -------------------------
class UserBatchSampler(torch.utils.data.Sampler[List[int]]):
    """
    Samples batches by grouping multiple docs per user so contrastive loss has positives.

    __len__ improved by estimating number of "chunks" per user (ceil(n_docs_u/n_per_user)),
    then estimating batches as ceil(total_chunks / users_per_batch).
    """

    def __init__(self, doc_indices, users, batch_size, n_per_user=4, shuffle=True):
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
        # chunk-based estimate (much closer to reality than ceil(total_docs/batch_size))
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
    # First find the post/text/doc node
    post_node = None
    for n in nodes:
        # Safe attribute retrieval without 'or' chaining
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
        raise RuntimeError(
            f"No post/text/doc node found. Node types present: "
            f"{[getattr(n, 'ntype', '?') for n in nodes]}"
        )
    
    # Get local_id - explicitly check each attribute, don't use 'or' chaining
    for attr in ("local_id", "local", "lid", "id"):
        lid = getattr(post_node, attr, None)
        if lid is not None:
            return int(lid)
    
    raise RuntimeError(f"Post node has no local_id attribute. Available: {dir(post_node)}")


@dataclass
class Batched:
    doc_idx: torch.Tensor
    user_id: torch.Tensor
    y: torch.Tensor
    x: torch.Tensor
    adj_norm: torch.Tensor
    post_node_index: torch.Tensor


def make_loader(
    out_dir: Union[str, Path],
    split: str,
    labels: Dict[int, np.ndarray],
    users: Dict[int, int],
    builder: LocalGraphBuilder,
    global_encoder: GlobalEncoder,
    A_norm: Dict[str, torch.Tensor],
    X: Dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
    shuffle: bool,
    h_views_cache: Optional[Dict[str, Dict[str, Any]]] = None,
) -> DataLoader:
    ds = SplitDocDataset(out_dir, split=split, builder=builder)

    def collate_fn(items: List[Dict[str, Any]]) -> Batched:
        # Cache only when no_grad (eval). Never cache during train because autograd graph can't be reused across batches.
        if (not torch.is_grad_enabled()) and (h_views_cache is not None) and ("H" in h_views_cache):
            H_views = h_views_cache["H"].get("views")
            if H_views is None:
                H_views = global_encoder(A_norm, X)
                h_views_cache["H"]["views"] = H_views
        else:
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

    sampler = UserBatchSampler(
        ds.doc_indices,
        users,
        batch_size=batch_size,
        n_per_user=max(1, batch_size // 4),
        shuffle=shuffle,
    )
    return DataLoader(ds, batch_sampler=sampler, num_workers=0, collate_fn=collate_fn)


# -------------------------
# eval
# -------------------------
@torch.no_grad()
def evaluate(
    model: Step3Model,
    global_encoder: GlobalEncoder,
    loader: DataLoader,
    theta: float,
    lam: float,
    same_user_bias: float,
) -> Dict[str, float]:
    # BUG 2 fix: toggle BOTH modules
    model_was_training = model.training
    global_was_training = global_encoder.training

    model.eval()
    global_encoder.eval()

    bce = nn.BCEWithLogitsLoss()
    total = total_bce = total_cl = 0.0
    n_batches = 0
    logits_all: List[torch.Tensor] = []
    y_all: List[torch.Tensor] = []

    for b in loader:
        logits, z = model(b.adj_norm, b.x, b.post_node_index, b.user_id, same_user_bias=same_user_bias)
        y = b.y

        logits_all.append(logits.detach().cpu())
        y_all.append(y.detach().cpu())

        l_b = bce(logits, y)
        l_c = paper_contrastive_loss(z, y, theta=theta)
        loss = l_b + lam * l_c

        total += float(loss.item())
        total_bce += float(l_b.item())
        total_cl += float(l_c.item())
        n_batches += 1

    probs = torch.sigmoid(torch.cat(logits_all, dim=0)).numpy()
    y_true = torch.cat(y_all, dim=0).numpy().astype(np.int32)
    y_pred = (probs >= 0.5).astype(np.int32)

    _, _, f1 = _f1_per_label(y_true, y_pred)

    out = {
        "loss": total / max(n_batches, 1),
        "bce": total_bce / max(n_batches, 1),
        "cl": total_cl / max(n_batches, 1),
        "macro_f1": float(f1.mean()),
    }
    for i, name in enumerate(TRAIT_NAMES):
        out[f"f1_{name}"] = float(f1[i])

    # restore modes
    model.train(model_was_training)
    global_encoder.train(global_was_training)
    return out


# -------------------------
# grad stats
# -------------------------
def grad_mean_abs(module: nn.Module) -> float:
    tot = 0.0
    n = 0
    for p in module.parameters():
        if p.grad is None:
            continue
        tot += float(p.grad.detach().abs().mean().cpu())
        n += 1
    return tot / max(n, 1)


def grad_l2_norm(module: nn.Module) -> float:
    total_sq = 0.0
    for p in module.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        total_sq += float((g * g).sum().item())
    return total_sq ** 0.5


@torch.no_grad()
def batch_metrics_from_logits(logits: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    y_pred = (probs >= 0.5).float()

    y_np = y.detach().cpu().numpy().astype(np.int32)
    y_pred_np = y_pred.detach().cpu().numpy().astype(np.int32)

    _, _, f1 = _f1_per_label(y_np, y_pred_np)
    return {
        "macro_f1": float(f1.mean()),
        "prob_mean": float(probs.mean().item()),
        "prob_std": float(probs.std(unbiased=False).item()),
        "pred_pos_rate": float(y_pred.mean().item()),
        "true_pos_rate": float(y.mean().item()),
    }


# -------------------------
# initial logging
# -------------------------
def log_initial_info(
    out_dir: str,
    train_csv: str,
    val_csv: str,
    test_csv: str,
    label_method: str,
    max_epochs: int,
    patience: int,
    lr_local: float,
    lr_global: float,
    weight_decay: float,
    use_scheduler: bool,
    scheduler_eta_min: float,
    batch_size: int,
    theta: float,
    lam: float,
    same_user_bias: float,
    global_gcn_dim: int,
    train_labels: Dict[int, np.ndarray],
    val_labels: Dict[int, np.ndarray],
    test_labels: Dict[int, np.ndarray],
    train_users: Dict[int, int],
    val_users: Dict[int, int],
    test_users: Dict[int, int],
    A_norm: Dict[str, torch.Tensor],
    X: Dict[str, torch.Tensor],
    dims_in: Dict[str, int],
    global_encoder: GlobalEncoder,
    model: Step3Model,
    device: torch.device,
) -> None:
    """Log comprehensive initial information about data, models, and hyperparameters"""
    
    print("\n" + "="*80)
    print(f"SHINE END-TO-END TRAINING INITIALIZATION")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Paths
    print(f"\n[Paths]")
    print(f"  Output directory: {out_dir}")
    print(f"  Train CSV: {train_csv}")
    print(f"  Validation CSV: {val_csv}")
    print(f"  Test CSV: {test_csv}")
    print(f"  Label method: {label_method}")
    
    # Device info
    print(f"\n[Device]")
    print(f"  Using device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Data splits info
    print(f"\n[Data Splits]")
    print(f"  Train: {len(train_labels)} documents, {len(set(train_users.values()))} unique users")
    print(f"  Validation: {len(val_labels)} documents, {len(set(val_users.values()))} unique users")
    print(f"  Test: {len(test_labels)} documents, {len(set(test_users.values()))} unique users")
    
    # Class distribution - Train
    print(f"\n[Class Distribution - Train Set]")
    train_dist = compute_class_distribution(train_labels)
    for trait, stats in train_dist.items():
        print(f"  {trait}:")
        print(f"    Positive rate: {stats['positive_rate']:.4f} ({stats['positive_count']}/{stats['positive_count'] + stats['negative_count']})")
        print(f"    Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
    
    # Class distribution - Validation
    print(f"\n[Class Distribution - Validation Set]")
    val_dist = compute_class_distribution(val_labels)
    for trait, stats in val_dist.items():
        print(f"  {trait}:")
        print(f"    Positive rate: {stats['positive_rate']:.4f} ({stats['positive_count']}/{stats['positive_count'] + stats['negative_count']})")
        print(f"    Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
    
    # Class distribution - Test
    print(f"\n[Class Distribution - Test Set]")
    test_dist = compute_class_distribution(test_labels)
    for trait, stats in test_dist.items():
        print(f"  {trait}:")
        print(f"    Positive rate: {stats['positive_rate']:.4f} ({stats['positive_count']}/{stats['positive_count'] + stats['negative_count']})")
        print(f"    Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
    
    # Global graph info
    print(f"\n[Global Graphs]")
    for view_name in A_norm.keys():
        adj = A_norm[view_name]
        feat = X[view_name]
        # Safe nnz calculation
        if adj.is_sparse:
            nnz = int(adj._nnz())
        else:
            nnz = int((adj != 0).sum().item())
        print(f"  {view_name.capitalize()}:")
        print(f"    Adjacency: {adj.shape[0]} nodes, {nnz} edges")
        print(f"    Features: {feat.shape[0]} nodes, {feat.shape[1]} dims")
    
    # Global encoder architecture
    print(f"\n[Global Encoder]")
    for view_name, in_dim in dims_in.items():
        print(f"  {view_name.capitalize()} GCN: {in_dim} -> {global_gcn_dim} -> {global_gcn_dim} (dropout=0.5)")
    
    # Local model architecture
    print(f"\n[Local Model]")
    print(f"  Local GCN Encoder:")
    print(f"    Input dim: {model.enc.arch_info['input_dim']}")
    print(f"    Hidden dim: {model.enc.arch_info['hidden_dim']}")
    print(f"    Output dim: {model.enc.arch_info['output_dim']}")
    print(f"    Dropout: {model.enc.arch_info['dropout']}")
    print(f"  User Self-Attention:")
    print(f"    Dim: {model.user_attn.arch_info['dim']}")
    print(f"    Scale: {model.user_attn.arch_info['scale']:.2f}")
    print(f"  Classifier:")
    print(f"    Input dim: {model.cls.in_features}, Output dim: {model.cls.out_features}")
    
    # Parameter counts
    global_params = sum(p.numel() for p in global_encoder.parameters())
    local_params = sum(p.numel() for p in model.parameters())
    print(f"\n[Model Parameters]")
    print(f"  Global encoder: {global_params:,} trainable parameters")
    print(f"  Local model: {local_params:,} trainable parameters")
    print(f"  Total: {global_params + local_params:,} trainable parameters")
    
    # Hyperparameters
    print(f"\n[Hyperparameters]")
    print(f"  Training:")
    print(f"    Max epochs: {max_epochs}")
    print(f"    Early stopping patience: {patience}")
    print(f"    Batch size: {batch_size}")
    print(f"    Label method: {label_method}")
    print(f"  Optimization:")
    print(f"    Learning rate (local): {lr_local:.2e}")
    print(f"    Learning rate (global): {lr_global:.2e}")
    print(f"    Weight decay: {weight_decay:.2e}")
    print(f"    Use scheduler: {use_scheduler}")
    if use_scheduler:
        print(f"    Scheduler eta min: {scheduler_eta_min:.2e}")
    print(f"  Loss functions:")
    print(f"    Contrastive loss theta: {theta:.2f}")
    print(f"    Contrastive loss lambda: {lam:.4f}")
    print(f"    Same user bias: {same_user_bias:.2f}")
    
    print("\n" + "="*80 + "\n")


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
    patience: int = 10,
    lr_local: float = 1e-4,
    lr_global: float = 2e-5,
    weight_decay: float = 5e-4,  
    use_scheduler: bool = True,  
    scheduler_eta_min: float = 1e-6, 
    batch_size: int = 32,
    theta: float = 10.0,
    lam: float = 0.01,
    same_user_bias: float = 5.0,
    grad_print_every: int = 25,
    max_grad_norm: float = 1.0,
    global_gcn_dim: int = GLOBAL_GCN_DIM,  
) -> None:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outp = Path(out_dir)

    label_cols = label_cols_from_method(label_method)

    train_idx = load_split_doc_indices(outp, "train")
    val_idx = load_split_doc_indices(outp, "val")
    test_idx = load_split_doc_indices(outp, "test")

    train_labels, train_users = load_labels_users_aligned(train_csv, train_idx, label_cols=label_cols)
    val_labels, val_users = load_labels_users_aligned(val_csv, val_idx, label_cols=label_cols)
    test_labels, test_users = load_labels_users_aligned(test_csv, test_idx, label_cols=label_cols)

    builder = LocalGraphBuilder(outp, entity_col="entity_lid", lru_cache_size=2000)

    A_norm, X, dims_in = load_step2_inputs_liwc(outp)
    A_norm = {k: v.to(device) for k, v in A_norm.items()}
    X = {k: v.to(device) for k, v in X.items()}

    global_encoder = GlobalEncoder(dims_in, hid_dim=global_gcn_dim, out_dim=global_gcn_dim).to(device)
    global_encoder.train()

    # Build one batch to infer local feature dimension
    tmp_loader = make_loader(
        outp,
        "train",
        train_labels,
        train_users,
        builder,
        global_encoder,
        A_norm,
        X,
        device,
        batch_size=1,
        shuffle=False,
    )
    tmp_batch = next(iter(tmp_loader))
    model = Step3Model(in_dim=int(tmp_batch.x.shape[1]), num_labels=len(TRAIT_NAMES)).to(device)
    
    # Log initial information
    log_initial_info(
        out_dir, train_csv, val_csv, test_csv,
        label_method, max_epochs, patience, lr_local, lr_global,
        weight_decay, use_scheduler, scheduler_eta_min, batch_size,
        theta, lam, same_user_bias, global_gcn_dim,
        train_labels, val_labels, test_labels,
        train_users, val_users, test_users,
        A_norm, X, dims_in,
        global_encoder, model, device
    )

    opt = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": lr_local, "weight_decay": weight_decay},
            {"params": global_encoder.parameters(), "lr": lr_global, "weight_decay": weight_decay},
        ]
    )

    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max_epochs, eta_min=float(scheduler_eta_min)
        )

    bce = nn.BCEWithLogitsLoss()

    # Eval caches
    val_hcache: Dict[str, Dict[str, Any]] = {"H": {"views": None}}
    test_hcache: Dict[str, Dict[str, Any]] = {"H": {"views": None}}

    train_loader = make_loader(
        outp,
        "train",
        train_labels,
        train_users,
        builder,
        global_encoder,
        A_norm,
        X,
        device,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = make_loader(
        outp,
        "val",
        val_labels,
        val_users,
        builder,
        global_encoder,
        A_norm,
        X,
        device,
        batch_size=batch_size,
        shuffle=False,
        h_views_cache=val_hcache,
    )
    test_loader = make_loader(
        outp,
        "test",
        test_labels,
        test_users,
        builder,
        global_encoder,
        A_norm,
        X,
        device,
        batch_size=batch_size,
        shuffle=False,
        h_views_cache=test_hcache,
    )

    best_val = float("inf")
    bad = 0
    best_path = outp / "end2end_shine_best.pt"
    step = 0

    for ep in range(1, max_epochs + 1):
        model.train()
        global_encoder.train()

        total = total_b = total_c = 0.0
        n_batches = 0
        train_logits_all: List[torch.Tensor] = []
        train_y_all: List[torch.Tensor] = []

        for bi, b in enumerate(train_loader, start=1):
            step += 1
            opt.zero_grad(set_to_none=True)

            logits, z = model(
                b.adj_norm,
                b.x,
                b.post_node_index,
                b.user_id,
                same_user_bias=same_user_bias,
            )
            y = b.y

            l_b = bce(logits, y)
            l_c = paper_contrastive_loss(z, y, theta=theta)
            loss = l_b + lam * l_c

            loss.backward()

            # gradient clipping
            if max_grad_norm is not None and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(global_encoder.parameters()),
                    max_norm=float(max_grad_norm),
                )

            g_mean_global = grad_mean_abs(global_encoder)
            g_mean_local = grad_mean_abs(model)
            g_norm_global = grad_l2_norm(global_encoder)
            g_norm_local = grad_l2_norm(model)

            batch_stats = batch_metrics_from_logits(logits, y)

            opt.step()

            total += float(loss.item())
            total_b += float(l_b.item())
            total_c += float(l_c.item())
            n_batches += 1

            train_logits_all.append(logits.detach().cpu())
            train_y_all.append(y.detach().cpu())

            if step % grad_print_every == 0:
                lr_local_now = opt.param_groups[0]["lr"]
                lr_global_now = opt.param_groups[1]["lr"]
                
                # Get actual batch size
                B = int(b.y.shape[0])

                avg_loss = total / max(n_batches, 1)
                avg_bce = total_b / max(n_batches, 1)
                avg_cl = total_c / max(n_batches, 1)

                print(
                    f"[TRAIN] "
                    f"Epoch={ep:03d}/{max_epochs:03d} "
                    f"Batch={bi:04d}/{len(train_loader):04d} "
                    f"Step={step:06d} | "
                    f"B={B:02d} | "
                    f"Loss={loss.item():.4f} (Avg={avg_loss:.4f}) | "
                    f"BCE={l_b.item():.4f} (Avg={avg_bce:.4f}) | "
                    f"CL={l_c.item():.4f} (Avg={avg_cl:.4f}) | "
                    f"MacroF1={batch_stats['macro_f1']:.4f} | "
                    f"Prob(μ={batch_stats['prob_mean']:.4f}, σ={batch_stats['prob_std']:.4f}) | "
                    f"PosRate(Pred={batch_stats['pred_pos_rate']:.4f}, True={batch_stats['true_pos_rate']:.4f}) | "
                    f"Grad(Global/Local) μ_abs=({g_mean_global:.3e}/{g_mean_local:.3e}) L2=({g_norm_global:.3e}/{g_norm_local:.3e}) | "
                    f"LR(Global/Local)=({lr_global_now:.2e}/{lr_local_now:.2e})",
                    flush=True,
                )

        tr_loss = total / max(n_batches, 1)
        tr_b = total_b / max(n_batches, 1)
        tr_c = total_c / max(n_batches, 1)

        train_probs = torch.sigmoid(torch.cat(train_logits_all, dim=0)).numpy()
        train_true = torch.cat(train_y_all, dim=0).numpy().astype(np.int32)
        train_pred = (train_probs >= 0.5).astype(np.int32)
        _, _, train_f1 = _f1_per_label(train_true, train_pred)
        train_macro_f1 = float(train_f1.mean())

        # reset eval cache so val uses current encoder weights
        val_hcache["H"]["views"] = None
        va = evaluate(model, global_encoder, val_loader, theta=theta, lam=lam, same_user_bias=same_user_bias)

        # scheduler step after validation
        if scheduler is not None:
            scheduler.step()

        print(
            f"[EPOCH] Epoch={ep:03d}/{max_epochs:03d} | "
            f"Train: Loss={tr_loss:.4f} BCE={tr_b:.4f} CL={tr_c:.4f} MacroF1={train_macro_f1:.4f} | "
            f"Valid: Loss={va['loss']:.4f} BCE={va['bce']:.4f} CL={va['cl']:.4f} MacroF1={va['macro_f1']:.4f} | "
            f"F1(OPEN={va['f1_OPEN']:.4f}, CON={va['f1_CON']:.4f}, EXT={va['f1_EXT']:.4f}, "
            f"AGR={va['f1_AGR']:.4f}, NEU={va['f1_NEU']:.4f})",
            flush=True,
        )

        if va["loss"] + 1e-9 < best_val:
            prev_best = best_val
            best_val = va["loss"]
            bad = 0
            torch.save(
                {
                    "local": model.state_dict(),
                    "global": global_encoder.state_dict(),
                    "epoch": ep,
                    "best_val": best_val,
                    "best_val_macroF1": va["macro_f1"],
                    "label_method": label_method,
                    "global_gcn_dim": global_gcn_dim,
                },
                best_path,
            )
            print(
                f"[CHECKPOINT] Epoch={ep:03d} NEW BEST | "
                f"Val Loss: {prev_best:.4f} → {best_val:.4f} | "
                f"Val MacroF1={va['macro_f1']:.4f} | "
                f"Saved to {best_path.name}",
                flush=True,
            )
        else:
            bad += 1
            print(
                f"[CHECKPOINT] Epoch={ep:03d} No improvement | "
                f"Best Val Loss={best_val:.4f} | "
                f"Patience {bad}/{patience}",
                flush=True,
            )
            if bad >= patience:
                print(
                    f"[EARLY STOP] Stopped at epoch {ep:03d} | "
                    f"Best Val Loss={best_val:.4f} | "
                    f"Patience={patience}",
                    flush=True,
                )
                break

    print(f"\n[DONE] Best checkpoint: {best_path}", flush=True)

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["local"])
    global_encoder.load_state_dict(ckpt["global"])

    test_hcache["H"]["views"] = None
    te = evaluate(model, global_encoder, test_loader, theta=theta, lam=lam, same_user_bias=same_user_bias)
    print(
        f"\n[TEST] Final Results | "
        f"Loss={te['loss']:.4f} BCE={te['bce']:.4f} CL={te['cl']:.4f} "
        f"MacroF1={te['macro_f1']:.4f} | "
        f"F1(OPEN={te['f1_OPEN']:.4f}, CON={te['f1_CON']:.4f}, EXT={te['f1_EXT']:.4f}, "
        f"AGR={te['f1_AGR']:.4f}, NEU={te['f1_NEU']:.4f})",
        flush=True,
    )


if __name__ == "__main__":
    import os

    ROOT = Path(__file__).resolve().parent

    OUT_DIR_DEFAULT = ROOT / "outputs" / "global_graph_output"
    DATA_PROCESSED_DEFAULT = ROOT / "data" / "processed" / "preprocess_check_out"

    OUT_DIR = Path(os.getenv("GLOBAL_GRAPH_OUT", str(OUT_DIR_DEFAULT)))
    DATA_PROCESSED = Path(os.getenv("DATA_PROCESSED", str(DATA_PROCESSED_DEFAULT)))

    TRAIN_CSV = Path(os.getenv("TRAIN_CSV", str(DATA_PROCESSED / "final_train_preprocessed.csv")))
    VAL_CSV = Path(os.getenv("VAL_CSV", str(DATA_PROCESSED / "final_val_preprocessed.csv")))
    TEST_CSV = Path(os.getenv("TEST_CSV", str(DATA_PROCESSED / "final_test_preprocessed.csv")))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train(
        out_dir=str(OUT_DIR),
        train_csv=str(TRAIN_CSV),
        val_csv=str(VAL_CSV),
        test_csv=str(TEST_CSV),
        seed=1,
        label_method=os.getenv("LABEL_METHOD", "mean"),
        max_epochs=int(os.getenv("MAX_EPOCHS", "1000")),
        patience=int(os.getenv("PATIENCE", "10")),
        lr_local=float(os.getenv("LR_LOCAL", "1e-4")),
        lr_global=float(os.getenv("LR_GLOBAL", "2e-5")),
        weight_decay=float(os.getenv("WEIGHT_DECAY", "5e-4")),
        batch_size=int(os.getenv("BATCH_SIZE", "32")),
        theta=float(os.getenv("THETA", "10.0")),
        lam=float(os.getenv("LAMBDA", "0.01")),
        same_user_bias=float(os.getenv("SAME_USER_BIAS", "5.0")),
        grad_print_every=int(os.getenv("GRAD_PRINT_EVERY", "25")),
        use_scheduler=os.getenv("USE_SCHEDULER", "1") not in {"0", "false", "False"},
        scheduler_eta_min=float(os.getenv("SCHED_ETA_MIN", "1e-6")),
        max_grad_norm=float(os.getenv("MAX_GRAD_NORM", "1.0")),
        global_gcn_dim=int(os.getenv("GLOBAL_GCN_DIM", str(GLOBAL_GCN_DIM))),
    )