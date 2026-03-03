# file: train_local_gcn_step3_end2end_shine_batch.py
"""
SHINE-style training loop (max_epoch=1000, early stop on val loss patience=10)
with TRUE end-to-end trainability of global Eq.(1) parameters by recomputing H_views
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
  - global encoder (Eq.1) parameters W1_tau/W2_tau (trainable, per batch)
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
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from GCN import GCN
from local_graph import LocalGraphBuilder, build_X_local_from_Hviews


LABEL_COLS = [
    "OPEN_bin_mean",
    "CONSICEN_bin_mean",
    "EXTRO_bin_mean",
    "AGREE_bin_mean",
    "NEURO_bin_mean",
]
USER_COL = "ID_COL"
TRAIT_NAMES = ["OPEN", "CON", "EXT", "AGR", "NEU"]


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
    user_col: str = USER_COL,
) -> Tuple[Dict[int, np.ndarray], Dict[int, int]]:
    rows = _read_csv_rows(csv_path)
    if len(rows) != len(doc_indices):
        raise ValueError(
            f"Row count mismatch for {Path(csv_path).name}: csv_rows={len(rows)} but split_indices={len(doc_indices)}."
        )

    missing = [c for c in LABEL_COLS if c not in rows[0]]
    if missing:
        raise KeyError(f"{Path(csv_path).name} missing label columns: {missing}.")
    if user_col not in rows[0]:
        raise KeyError(f"{Path(csv_path).name} missing user id column '{user_col}'.")

    labels: Dict[int, np.ndarray] = {}
    users: Dict[int, int] = {}

    for i, doc_idx in enumerate(doc_indices):
        row = rows[i]
        y = np.array([float(row[c]) for c in LABEL_COLS], dtype=np.float32)
        labels[int(doc_idx)] = y

        u_raw = row[user_col]
        try:
            u = int(float(u_raw))
        except Exception:
            u = abs(hash(u_raw)) % (2**31 - 1)
        users[int(doc_idx)] = u

    return labels, users


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


def _load_pkl(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_feature(path: Path) -> torch.Tensor:
    if path.suffix == ".npy":
        arr = np.load(path, allow_pickle=True)
        if arr.dtype == object:
            arr = np.stack([np.asarray(x, dtype=np.float32) for x in arr], axis=0)
        else:
            arr = arr.astype(np.float32, copy=False)
        return torch.tensor(arr, dtype=torch.float32)

    obj = _load_pkl(path)
    if sp.issparse(obj):
        return torch.tensor(obj.toarray(), dtype=torch.float32)
    arr = np.asarray(obj)
    if arr.dtype == object:
        arr = np.stack([np.asarray(x, dtype=np.float32) for x in obj], axis=0)
    else:
        arr = arr.astype(np.float32, copy=False)
    return torch.tensor(arr, dtype=torch.float32)


def load_step2_inputs_liwc(
    out_dir: Union[str, Path]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, int]]:
    p = Path(out_dir)
    adj_paths: Dict[str, Path] = {}
    feat_paths: Dict[str, Path] = {}

    adj_paths["word"] = p / "adj_word.pkl"
    adj_paths["pos"] = p / "adj_tag.pkl"
    liwc_adj_path = p / "adj_liwc.pkl"
    if not liwc_adj_path.exists():
        liwc_adj_path = p / "adj_empath.pkl"
    adj_paths["liwc"] = liwc_adj_path
    adj_paths["entity"] = p / "adj_entity.pkl"
    adj_paths["text"] = p / "adj_text.pkl"

    feat_paths["word"] = p / "word_type_bert_emb.pkl"
    feat_paths["pos"] = p / "pos_onehot.pkl"
    liwc_feat_path = p / "liwc_onehot.pkl"
    if not liwc_feat_path.exists():
        liwc_feat_path = p / "empath_onehot.pkl"
    feat_paths["liwc"] = liwc_feat_path
    feat_paths["entity"] = p / "entity_emb.pkl"
    feat_paths["text"] = p / "doc_bert_emb.npy"

    for k, ap in adj_paths.items():
        if not ap.exists():
            raise FileNotFoundError(f"Missing adjacency for view '{k}': {ap}")
    for k, fp in feat_paths.items():
        if not fp.exists():
            raise FileNotFoundError(f"Missing features for view '{k}': {fp}")

    A_norm: Dict[str, torch.Tensor] = {}
    for k, ap in adj_paths.items():
        A = _load_pkl(ap)
        A_t = scipy_to_torch_coo(A if sp.issparse(A) else sp.coo_matrix(np.asarray(A)))
        A_norm[k] = normalize_adj_coo(A_t)

    X: Dict[str, torch.Tensor] = {k: _load_feature(fp) for k, fp in feat_paths.items()}

    for k in adj_paths:
        if A_norm[k].shape[0] != X[k].shape[0]:
            raise ValueError(f"Shape mismatch for '{k}': A {tuple(A_norm[k].shape)} vs X {tuple(X[k].shape)}")

    dims_in = {k: int(X[k].shape[1]) for k in X}
    return A_norm, X, dims_in


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


class GlobalEncoderEq1(nn.Module):
    def __init__(self, dims_in: Dict[str, int], hid_dim: int = 256, out_dim: int = 256, dropout: float = 0.5):
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


class UserBatchSampler(torch.utils.data.Sampler):
    def __init__(self, doc_indices, users, batch_size, n_per_user=4, shuffle=True):
        self.batch_size = batch_size
        self.n_per_user = n_per_user
        self.shuffle = shuffle
        self.user_to_idx = collections.defaultdict(list)
        for idx, doc_idx in enumerate(doc_indices):
            u = users.get(doc_idx)
            if u is not None:
                self.user_to_idx[u].append(idx)

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

            users_to_pick = active_users[:max(1, self.batch_size // self.n_per_user)]
            for u in users_to_pick:
                chunk = user_lists[u][:self.n_per_user]
                user_lists[u] = user_lists[u][self.n_per_user:]
                batch.extend(chunk)

            active_users = [u for u in active_users if len(user_lists[u]) > 0]
            if len(batch) > 0:
                yield batch

    def __len__(self):
        total_len = sum(len(v) for v in self.user_to_idx.values())
        return max(1, (total_len + self.batch_size - 1) // self.batch_size)


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
    for n in nodes:
        ntype = getattr(n, "ntype", None) or getattr(n, "type", None) or getattr(n, "node_type", None)
        lid = getattr(n, "local_id", None) or getattr(n, "local", None) or getattr(n, "lid", None) or getattr(n, "id", None)
        if ntype in ("post", "text", "doc") and lid is not None:
            return int(lid)
    return 0


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
    global_encoder: GlobalEncoderEq1,
    A_norm: Dict[str, torch.Tensor],
    X: Dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    ds = SplitDocDataset(out_dir, split=split, builder=builder)

    def collate_fn(items: List[Dict[str, Any]]) -> Batched:
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

        return Batched(
            doc_idx=doc_t,
            user_id=user_t,
            y=yb,
            x=Xb,
            adj_norm=Ab,
            post_node_index=post_t,
        )

    sampler = UserBatchSampler(
        ds.doc_indices,
        users,
        batch_size=batch_size,
        n_per_user=max(1, batch_size // 4),
        shuffle=shuffle,
    )
    return DataLoader(ds, batch_sampler=sampler, num_workers=0, collate_fn=collate_fn)


@torch.no_grad()
def evaluate(model: Step3Model, loader: DataLoader, theta: float, lam: float, same_user_bias: float) -> Dict[str, float]:
    model.eval()
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
    return out


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


def train(
    out_dir: str,
    train_csv: str,
    val_csv: str,
    test_csv: str,
    *,
    seed: int = 1,
    max_epochs: int = 5,
    patience: int = 10,
    lr_local: float = 1e-4,
    lr_global: float = 2e-5,
    weight_decay: float = 1e-2,
    batch_size: int = 8,
    theta: float = 10.0,
    lam: float = 0.01,
    same_user_bias: float = 5.0,
    grad_print_every: int = 25,
) -> None:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outp = Path(out_dir)

    train_idx = load_split_doc_indices(outp, "train")
    val_idx = load_split_doc_indices(outp, "val")
    test_idx = load_split_doc_indices(outp, "test")

    train_labels, train_users = load_labels_users_aligned(train_csv, train_idx)
    val_labels, val_users = load_labels_users_aligned(val_csv, val_idx)
    test_labels, test_users = load_labels_users_aligned(test_csv, test_idx)

    builder = LocalGraphBuilder(outp, entity_col="entity_lid", lru_cache_size=2000)

    A_norm, X, dims_in = load_step2_inputs_liwc(outp)
    A_norm = {k: v.to(device) for k, v in A_norm.items()}
    X = {k: v.to(device) for k, v in X.items()}

    global_encoder = GlobalEncoderEq1(dims_in).to(device)
    global_encoder.train()

    tmp_loader = make_loader(
        outp, "train", train_labels, train_users,
        builder, global_encoder, A_norm, X, device,
        batch_size=1, shuffle=False
    )
    tmp_batch = next(iter(tmp_loader))
    model = Step3Model(in_dim=int(tmp_batch.x.shape[1])).to(device)

    opt = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": lr_local, "weight_decay": weight_decay},
            {"params": global_encoder.parameters(), "lr": lr_global, "weight_decay": weight_decay},
        ]
    )
    bce = nn.BCEWithLogitsLoss()

    train_loader = make_loader(
        outp, "train", train_labels, train_users,
        builder, global_encoder, A_norm, X, device,
        batch_size=batch_size, shuffle=True
    )
    val_loader = make_loader(
        outp, "val", val_labels, val_users,
        builder, global_encoder, A_norm, X, device,
        batch_size=batch_size, shuffle=False
    )
    test_loader = make_loader(
        outp, "test", test_labels, test_users,
        builder, global_encoder, A_norm, X, device,
        batch_size=batch_size, shuffle=False
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

                avg_loss = total / max(n_batches, 1)
                avg_bce = total_b / max(n_batches, 1)
                avg_cl = total_c / max(n_batches, 1)

                print(
                    f"[train] "
                    f"epoch={ep:03d}/{max_epochs:03d} "
                    f"batch={bi:04d}/{len(train_loader):04d} "
                    f"step={step:06d} | "
                    f"loss={loss.item():.4f} avg={avg_loss:.4f} | "
                    f"bce={l_b.item():.4f} avg_bce={avg_bce:.4f} | "
                    f"cl={l_c.item():.4f} avg_cl={avg_cl:.4f} | "
                    f"batch_macroF1={batch_stats['macro_f1']:.4f} | "
                    f"prob_mean={batch_stats['prob_mean']:.4f} "
                    f"prob_std={batch_stats['prob_std']:.4f} | "
                    f"pred_pos={batch_stats['pred_pos_rate']:.4f} "
                    f"true_pos={batch_stats['true_pos_rate']:.4f} | "
                    f"global_mean_abs={g_mean_global:.3e} local_mean_abs={g_mean_local:.3e} | "
                    f"g_l2(global/local)=({g_norm_global:.3e}/{g_norm_local:.3e}) | "
                    f"lr(global/local)=({lr_global_now:.2e}/{lr_local_now:.2e})",
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

        va = evaluate(model, val_loader, theta=theta, lam=lam, same_user_bias=same_user_bias)

        print(
            f"[epoch-summary] "
            f"epoch={ep:03d}/{max_epochs:03d} | "
            f"train_loss={tr_loss:.4f} train_bce={tr_b:.4f} train_cl={tr_c:.4f} "
            f"train_macroF1={train_macro_f1:.4f} | "
            f"val_loss={va['loss']:.4f} val_bce={va['bce']:.4f} val_cl={va['cl']:.4f} "
            f"val_macroF1={va['macro_f1']:.4f} | "
            f"val_F1[OPEN={va['f1_OPEN']:.4f}, CON={va['f1_CON']:.4f}, EXT={va['f1_EXT']:.4f}, "
            f"AGR={va['f1_AGR']:.4f}, NEU={va['f1_NEU']:.4f}]",
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
                },
                best_path,
            )
            print(
                f"[checkpoint] epoch={ep:03d} NEW BEST | "
                f"val_loss: {prev_best:.4f} -> {best_val:.4f} | "
                f"val_macroF1={va['macro_f1']:.4f} | "
                f"saved_to={best_path.name}",
                flush=True,
            )
        else:
            bad += 1
            print(
                f"[checkpoint] epoch={ep:03d} no improvement | "
                f"best_val={best_val:.4f} | "
                f"epochs_without_improvement={bad}/{patience}",
                flush=True,
            )
            if bad >= patience:
                print(
                    f"[EARLY STOP] stopped at epoch={ep:03d} | "
                    f"best_val={best_val:.4f} | "
                    f"patience={patience}",
                    flush=True,
                )
                break

    print(f"[done] best_checkpoint={best_path}", flush=True)

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["local"])
    global_encoder.load_state_dict(ckpt["global"])

    te = evaluate(model, test_loader, theta=theta, lam=lam, same_user_bias=same_user_bias)
    print(
        f"[test-summary] "
        f"loss={te['loss']:.4f} bce={te['bce']:.4f} cl={te['cl']:.4f} "
        f"macroF1={te['macro_f1']:.4f} | "
        f"F1[OPEN={te['f1_OPEN']:.4f}, CON={te['f1_CON']:.4f}, EXT={te['f1_EXT']:.4f}, "
        f"AGR={te['f1_AGR']:.4f}, NEU={te['f1_NEU']:.4f}]",
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
        max_epochs=1000,
        patience=10,
        lr_local=1e-4,
        lr_global=2e-5,
        weight_decay=1e-2,
        batch_size=32,
        theta=10.0,
        lam=0.01,
        same_user_bias=5.0,
        grad_print_every=25,
    )