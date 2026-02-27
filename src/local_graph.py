# local_graph_onfly.py
from __future__ import annotations

import csv
import json
import pickle
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
# -----------------------
# H_views loader (lazy torch import)
# -----------------------

class HViewsProvider:
    """Lazy loader for H_views.pt so Dataset can return X_local for every doc.

    This avoids importing torch at module import time on Windows.
    """

    def __init__(self, out_dir: Union[str, Path], filename: str = "H_views.pt", map_location: str = "cpu"):
        self.out_dir = Path(out_dir)
        self.filename = filename
        self.map_location = map_location
        self._h: Optional[Dict[str, "torch.Tensor"]] = None

    def get(self) -> Dict[str, "torch.Tensor"]:
        if self._h is None:
            import torch  # lazy import
            self._h = torch.load(str(self.out_dir / self.filename), map_location=self.map_location)
        return self._h


import scipy.sparse as sp
try:
    from torch.utils.data import Dataset
except Exception:
    class Dataset:  # fallback so the file can run without importing torch
        pass



# -----------------------
# Helpers: load artifacts
# -----------------------

def _load_pkl(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_doc_word_seq(path: Path) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            out[int(obj["doc_idx"])] = list(map(int, obj["word_ids"]))
    return out


def _load_edges_by_doc(path: Path, cols: Tuple[str, str]) -> Dict[int, List[Tuple[int, int]]]:
    out: Dict[int, List[Tuple[int, int]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            d = int(row["doc_idx"])
            a = int(row[cols[0]])
            b = int(row[cols[1]])
            out.setdefault(d, []).append((a, b))
    return out


def _load_word_id2_list(outp: Path) -> List[str]:
    json_path = outp / "word_id2_list.json"
    pkl_path = outp / "word_id2_list.pkl"
    if json_path.exists():
        return _load_json(json_path)
    if pkl_path.exists():
        return _load_pkl(pkl_path)
    raise FileNotFoundError(
        f"Missing word_id2_list.json or word_id2_list.pkl in {outp} "
        "(needed to map word_id -> token for Empath lookup)."
    )


def _load_split_indices(outp: Path, split: str) -> List[int]:
    """
    Expects files: train_idx.json / val_idx.json / test_idx.json
    Each file should be a JSON list of doc_idx integers.
    """
    fname = {
        "train": "train_idx.json",
        "val": "val_idx.json",
        "test": "test_idx.json",
    }.get(split)
    if fname is None:
        raise ValueError("split must be one of: train, val, test")
    p = outp / fname
    if not p.exists():
        raise FileNotFoundError(f"Missing split file: {p}")
    arr = _load_json(p)
    return [int(x) for x in arr]

def plot_doc0_topology(
    nodes: List[LocalNode],
    edges: List[Tuple[int, int, str, float]],
    out_png: Union[str, Path],
) -> None:
    import matplotlib.pyplot as plt
    import networkx as nx

    type_to_color = {
        "post": "gray",
        "word": "yellow",
        "pos": "green",
        "entity": "blue",
        "liwc": "orange",
    }

    G = nx.DiGraph()
    for n in nodes:
        G.add_node(n.local_id, ntype=n.ntype, gid=n.global_id)

    for s, t, rel, w in edges:
        G.add_edge(int(s), int(t), rel=rel, weight=float(w))

    pos = nx.spring_layout(G, seed=7)

    plt.figure(figsize=(12, 8))

    for t, c in type_to_color.items():
        nodelist = [nid for nid in G.nodes if G.nodes[nid].get("ntype") == t]
        if nodelist:
            nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_size=280, alpha=0.9, node_color=c, label=t)

    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.4)
    # keep labels short: local_id:type
    labels = {nid: f"{nid}:{G.nodes[nid]['ntype']}" for nid in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7)

    plt.legend(loc="best", fontsize=8, frameon=True)
    plt.title("Doc0 Local Graph Topology")
    plt.axis("off")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def plot_doc0_hview_embeddings(
    nodes: List[LocalNode],
    X_local,  # torch.Tensor [num_nodes, d]
    out_png: Union[str, Path],
) -> None:
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    type_to_color = {
        "post": "gray",
        "word": "yellow",
        "pos": "green",
        "entity": "blue",
        "liwc": "orange",
    }

    X = X_local.detach().cpu().numpy()
    Z = PCA(n_components=2, random_state=7).fit_transform(X)

    plt.figure(figsize=(10, 7))
    for t, c in type_to_color.items():
        idx = [i for i, n in enumerate(nodes) if n.ntype == t]
        if idx:
            pts = Z[idx]
            plt.scatter(pts[:, 0], pts[:, 1], s=60, alpha=0.85, c=c, label=t)

    for i, n in enumerate(nodes):
        plt.text(Z[i, 0], Z[i, 1], str(n.local_id), fontsize=7)

    plt.legend(loc="best", fontsize=8, frameon=True)
    plt.title("Doc0 Node Embeddings from H_views (PCA 2D)")
    plt.axis("off")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()



# -----------------------
# Local graph builder
# -----------------------

@dataclass(frozen=True)
class LocalNode:
    local_id: int
    ntype: str  # "post" | "word" | "pos" | "entity" | "liwc"
    global_id: int


class _LRUCache:
    def __init__(self, max_items: int):
        self.max_items = int(max_items)
        self._d: "OrderedDict[int, Any]" = OrderedDict()

    def get(self, k: int) -> Optional[Any]:
        if k not in self._d:
            return None
        v = self._d.pop(k)
        self._d[k] = v
        return v

    def put(self, k: int, v: Any) -> None:
        if self.max_items <= 0:
            return
        if k in self._d:
            self._d.pop(k)
        self._d[k] = v
        while len(self._d) > self.max_items:
            self._d.popitem(last=False)


class LocalGraphBuilder:
    """
    Builds the per-post (local) graph in-memory (no disk cache).
    Uses the same logic as your debug cache script, but returns adjacency directly.
    """

    def __init__(
        self,
        out_dir: Union[str, Path],
        entity_col: str = "entity_lid",
        add_word_word_adjacent: bool = True,
        lru_cache_size: int = 0,
    ) -> None:
        self.outp = Path(out_dir)

        # Load once
        self.doc_word_seq = _load_doc_word_seq(self.outp / "doc_word_seq.jsonl")
        self.word_pos_by_doc = _load_edges_by_doc(self.outp / "word_pos_edges.csv", cols=("word_id", "pos_id"))

        ent_path = self.outp / "word_entity_edges.csv"
        if not ent_path.exists():
            raise FileNotFoundError(f"Missing {ent_path}")
        self.word_ent_by_doc = _load_edges_by_doc(ent_path, cols=("word_id", entity_col))

        self.word_id2_list = _load_word_id2_list(self.outp)

        # Empath mapping: token -> [cats] where cats may be names ("family") or ids
        self.liwc_token2cats: Dict[str, List[Any]] = _load_json(self.outp / "liwc_word2cats.json")

        # Empath id2 list for name->id mapping (optional but needed when cats are names)
        self.liwc_name2id: Optional[Dict[str, int]] = None
        liwc_id2_path = self.outp / "liwc_id2_list.json"
        if liwc_id2_path.exists():
            liwc_id2 = _load_json(liwc_id2_path)
            self.liwc_name2id = {str(name).strip().lower(): i for i, name in enumerate(liwc_id2)}

        # Optional: global word PMI adjacency for paper-faithful local word-word weights (Eq. 3)
        # If present, we will weight adjacent word-word edges by PMI(w_i, w_{i+1}) looked up from this matrix.
        self.adj_word_pmi: Optional[sp.csr_matrix] = None
        adj_word_path = self.outp / "adj_word.pkl"
        if adj_word_path.exists():
            A = _load_pkl(adj_word_path)
            if not sp.issparse(A):
                A = sp.csr_matrix(A)
            self.adj_word_pmi = A.tocsr()

        self.add_word_word_adjacent = bool(add_word_word_adjacent)
        self._cache = _LRUCache(lru_cache_size)

    def build(self, doc_idx: int) -> Tuple[List[LocalNode], List[Tuple[int, int, str, float]], sp.csr_matrix]:
        cached = self._cache.get(int(doc_idx))
        if cached is not None:
            return cached

        doc_idx = int(doc_idx)
        word_seq = self.doc_word_seq.get(doc_idx, [])
        word_pos = self.word_pos_by_doc.get(doc_idx, [])
        word_ent = self.word_ent_by_doc.get(doc_idx, [])

        nodes: List[LocalNode] = []
        edges: List[Tuple[int, int, str, float]] = []
        local_map: Dict[Tuple[str, int], int] = {}

        def _get(ntype: str, gid: int) -> int:
            key = (ntype, gid)
            if key in local_map:
                return local_map[key]
            lid = len(local_map)
            local_map[key] = lid
            nodes.append(LocalNode(local_id=lid, ntype=ntype, global_id=gid))
            return lid

        # post node
        post_lid = _get("post", doc_idx)

        # unique word nodes
        uniq_words = list(dict.fromkeys(word_seq))
        for wid in uniq_words:
            w_lid = _get("word", int(wid))
            edges.append((post_lid, w_lid, "contain_post_word", 1.0))
            edges.append((w_lid, post_lid, "contain_word_post", 1.0))

        # word-word adjacent edges (paper Eq. 3 uses PMI of adjacent words)
        if self.add_word_word_adjacent and len(word_seq) >= 2:
            for a, b in zip(word_seq[:-1], word_seq[1:]):
                if a == b:
                    continue
                a = int(a); b = int(b)
                a_lid = _get("word", a)
                b_lid = _get("word", b)

                # Default edge weight if PMI matrix is unavailable
                w = 1.0

                # If global PMI adjacency is available, use it as the edge weight.
                # If PMI is missing / zero, skip this edge to stay paper-faithful.
                if self.adj_word_pmi is not None:
                    w = float(self.adj_word_pmi[a, b])
                    if w <= 0:
                        w = float(self.adj_word_pmi[b, a])
                    if w <= 0:
                        w = 1e-3  # fallback to keep adjacency chain connected when PPMI missing/<=0

                edges.append((a_lid, b_lid, "adjacent_word_word", w))
                edges.append((b_lid, a_lid, "adjacent_word_word", w))

        # word-pos edges
        for wid, pid in word_pos:
            w_lid = _get("word", int(wid))
            p_lid = _get("pos", int(pid))
            edges.append((w_lid, p_lid, "contain_word_pos", 1.0))
            edges.append((p_lid, w_lid, "contain_pos_word", 1.0))

        # word-entity edges
        for wid, eid in word_ent:
            w_lid = _get("word", int(wid))
            e_lid = _get("entity", int(eid))
            edges.append((w_lid, e_lid, "contain_word_entity", 1.0))
            edges.append((e_lid, w_lid, "contain_entity_word", 1.0))

        # word-empath edges (token lookup, cats may be names or ids)
        for wid in uniq_words:
            wid = int(wid)
            if wid < 0 or wid >= len(self.word_id2_list):
                continue
            token = str(self.word_id2_list[wid]).strip().lower()
            if not token:
                continue
            cats = self.liwc_token2cats.get(token, [])
            if not cats:
                continue

            w_lid = _get("word", wid)
            for cid in cats:
                if isinstance(cid, int):
                    cat_id = cid
                else:
                    s = str(cid).strip().lower()
                    if not s:
                        continue
                    if s.isdigit():
                        cat_id = int(s)
                    else:
                        if self.liwc_name2id is None or s not in self.liwc_name2id:
                            continue
                        cat_id = self.liwc_name2id[s]

                c_lid = _get("liwc", int(cat_id))
                edges.append((w_lid, c_lid, "contain_word_liwc", 1.0))
                edges.append((c_lid, w_lid, "contain_liwc_word", 1.0))

        # adjacency
        adj = self._edges_to_adj(len(nodes), edges, self_loop=True)

        out = (nodes, edges, adj)
        self._cache.put(doc_idx, out)
        return out

    @staticmethod
    def _edges_to_adj(num_nodes: int, edges: List[Tuple[int, int, str, float]], self_loop: bool = True) -> sp.csr_matrix:
        rows, cols, data = [], [], []
        for s, t, _rel, w in edges:
            rows.append(int(s))
            cols.append(int(t))
            data.append(float(w))
        A = sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes)).tocsr()
        A.sum_duplicates()
        if self_loop:
            A = A.tolil()
            A.setdiag(1.0)
            A = A.tocsr()
        A.eliminate_zeros()
        return A

def build_X_local_from_Hviews(
    nodes: List[LocalNode],
    doc_idx: int,
    H_views: Dict[str, "torch.Tensor"],
) -> "torch.Tensor":
    """
    Create local node feature matrix X_local where each row is taken from global H_views.

    Mapping:
      post   -> H_text[doc_idx]
      word   -> H_word[word_id]
      pos    -> H_pos[pos_id]
      liwc -> H_liwc[liwc_id]
      entity -> H_entity[entity_id]   # IMPORTANT: must match H_entity indexing (yours is entity_lid)
    """
    import torch

    H_word = H_views["H_word"]
    H_pos = H_views["H_pos"]
    H_liwc = H_views["H_liwc"]
    H_entity = H_views["H_entity"]
    H_text = H_views["H_text"]

    d = H_word.shape[1]
    X = torch.zeros((len(nodes), d), dtype=H_word.dtype)

    # Safety check
    if not (0 <= doc_idx < H_text.shape[0]):
        raise IndexError(f"doc_idx={doc_idx} out of range for H_text with shape={tuple(H_text.shape)}")

    for n in nodes:
        i = int(n.local_id)
        gid = int(n.global_id)

        if n.ntype == "post":
            X[i] = H_text[doc_idx]

        elif n.ntype == "word":
            if not (0 <= gid < H_word.shape[0]):
                raise IndexError(f"word_id={gid} out of range for H_word shape={tuple(H_word.shape)}")
            X[i] = H_word[gid]

        elif n.ntype == "pos":
            if not (0 <= gid < H_pos.shape[0]):
                raise IndexError(f"pos_id={gid} out of range for H_pos shape={tuple(H_pos.shape)}")
            X[i] = H_pos[gid]

        elif n.ntype == "liwc":
            if not (0 <= gid < H_liwc.shape[0]):
                raise IndexError(f"liwc_id={gid} out of range for H_liwc shape={tuple(H_liwc.shape)}")
            X[i] = H_liwc[gid]

        elif n.ntype == "entity":
            # Your H_entity is (7548,256) so gid must be entity_lid (0..7547)
            if not (0 <= gid < H_entity.shape[0]):
                raise IndexError(f"entity_id={gid} out of range for H_entity shape={tuple(H_entity.shape)}")
            X[i] = H_entity[gid]

        else:
            raise ValueError(f"Unknown node type: {n.ntype}")

    return X



# -----------------------
# Debug helpers (adjacency)
# -----------------------

def inspect_adj(adj: "sp.csr_matrix", topk: int = 40) -> None:
    """Print useful stats about a local adjacency matrix."""
    import numpy as np

    A = adj.tocsr()
    print("adj shape:", A.shape, "nnz:", A.nnz)
    d = A.diagonal()
    print("diag all ones:", bool(np.allclose(d, 1.0)))
    print("symmetric:", bool((A != A.T).nnz == 0))
    if A.nnz:
        print("min/max weight:", float(A.data.min()), float(A.data.max()))

    coo = A.tocoo()
    entries = [(int(r), int(c), float(v)) for r, c, v in zip(coo.row, coo.col, coo.data) if r != c]
    entries.sort(key=lambda x: abs(x[2]), reverse=True)
    print(f"top {min(topk, len(entries))} non-diagonal entries (row,col,val):")
    for r, c, v in entries[:topk]:
        print(r, c, v)


def adjacency_to_csv(adj: "sp.csr_matrix", nodes: List[LocalNode], out_csv: Union[str, Path]) -> None:
    """Export non-zero adjacency entries as an edge list CSV for easy inspection."""
    out_csv = Path(out_csv)
    coo = adj.tocoo()
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["src_lid", "dst_lid", "weight", "src_type", "src_gid", "dst_type", "dst_gid"])
        for r, c, v in zip(coo.row, coo.col, coo.data):
            sr = nodes[int(r)]
            sc = nodes[int(c)]
            w.writerow([int(r), int(c), float(v), sr.ntype, int(sr.global_id), sc.ntype, int(sc.global_id)])
    print("saved:", out_csv)

# -----------------------
# Dataset per split (train/val/test)
# -----------------------

class SplitDataset(Dataset):
    """
    Returns per item:
      doc_idx, nodes, edges, adj
    You can extend it to also return labels/user_id once you wire your label CSV.
    """

    def __init__(
        self,
        out_dir: Union[str, Path],
        split: str,
        builder: Optional[LocalGraphBuilder] = None,
        hviews_provider: Optional[HViewsProvider] = None,
        doc_indices: Optional[Sequence[int]] = None,
        entity_col: str = "entity_lid",
        add_word_word_adjacent: bool = True,
        lru_cache_size: int = 0,
    ) -> None:
        self.outp = Path(out_dir)

        self.builder = builder or LocalGraphBuilder(
            out_dir=self.outp,
            entity_col=entity_col,
            add_word_word_adjacent=add_word_word_adjacent,
            lru_cache_size=lru_cache_size,
        )


        self.hviews_provider = hviews_provider
        if doc_indices is not None:
            self.doc_indices = [int(x) for x in doc_indices]
        else:
            self.doc_indices = _load_split_indices(self.outp, split)

    def __len__(self) -> int:
        return len(self.doc_indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        doc_idx = int(self.doc_indices[idx])
        nodes, edges, adj = self.builder.build(doc_idx)
        X_local = None
        if self.hviews_provider is not None:
            H_views = self.hviews_provider.get()
            X_local = build_X_local_from_Hviews(nodes, doc_idx=doc_idx, H_views=H_views)
        return {
            "doc_idx": doc_idx,
            "nodes": nodes,
            "edges": edges,
            "adj": adj,
            "X_local": X_local,
        }


# -----------------------
# Tiny demo run (VS Code friendly)
# -----------------------

if __name__ == "__main__":
    from pathlib import Path

    OUT_DIR = str((Path(__file__).resolve().parent / "outputs" / "global_graph_output"))

    builder = LocalGraphBuilder(OUT_DIR, entity_col="entity_lid", lru_cache_size=128)


    hviews = HViewsProvider(OUT_DIR, filename="H_views.pt", map_location="cpu")
    # optional: keep your split-length prints here if you like
    for split in ["train", "val", "test"]:
        try:
            ds = SplitDataset(OUT_DIR, split=split, builder=builder, hviews_provider=hviews)
            print(split, "len=", len(ds), "first_doc_idx=", ds.doc_indices[0], "last_doc_idx=", ds.doc_indices[-1])
        except FileNotFoundError as e:
            print(f"[{split}] missing split file, skipping: {e}")
            continue

        item = ds[0]
        print(
            f"[{split}] doc={item['doc_idx']} nodes={len(item['nodes'])} edges={len(item['edges'])} "
            f"adj_shape={item['adj'].shape} nnz={item['adj'].nnz}"
        )
    H_views = hviews.get()
    nodes, edges, adj = builder.build(0)
    X_local = build_X_local_from_Hviews(nodes, doc_idx=0, H_views=H_views)

    print("doc0:", "nodes=", len(nodes), "edges=", len(edges), "adj=", adj.shape, "nnz=", adj.nnz)
    print("X_local:", tuple(X_local.shape))

    topo_png = Path(OUT_DIR) / "doc0_topology.png"
    emb_png = Path(OUT_DIR) / "doc0_hviews_pca.png"
    adj_csv = Path(OUT_DIR) / "doc0_adj_nonzeros.csv"

    # Inspect adjacency for doc0
    inspect_adj(adj, topk=50)
    adjacency_to_csv(adj, nodes, adj_csv)

    # Plots
    plot_doc0_topology(nodes, edges, topo_png)
    plot_doc0_hview_embeddings(nodes, X_local, emb_png)

    print("Saved:", topo_png)
    print("Saved:", emb_png)
    print("Saved:", adj_csv)