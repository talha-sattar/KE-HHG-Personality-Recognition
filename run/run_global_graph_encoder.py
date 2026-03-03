# file: run_global_graph_encoder_repo_relative.py
"""
Runner for Step 2: Global graph encoder (Eq.1) -> saves H_views.pt (repo-relative).

Expected repo layout (relative to this file):
  outputs/global_graph_output/
    adj_word.pkl, adj_tag.pkl, adj_liwc.pkl, adj_entity.pkl, adj_text.pkl
    word_type_bert_emb.pkl, pos_onehot.pkl, liwc_onehot.pkl, entity_emb.pkl, doc_bert_emb.npy

This runner writes:
  outputs/global_graph_output/H_views.pt

Notes:
- This runner does NOT require any absolute paths.
- It matches your previous runner behavior but uses repo-relative output directory.
"""

from __future__ import annotations

from pathlib import Path

from GCN import GCN
from global_encoder import sanity_forward, save_global_embeddings


def main() -> None:
    root = Path(__file__).resolve().parent
    out_dir = root.parent / "src" / "outputs" / "global_graph_output"

    if not out_dir.exists():
        raise FileNotFoundError(
            f"Missing output directory: {out_dir}\n"
            "Run Step 1 (global graph construction) first so adjacencies/features exist."
        )

    # Keep dims consistent with your current setting.
    # If you want paper-matching dims, set hid_dim=400, out_dim=400.
    hid_dim = 400
    out_dim = 400

    save_global_embeddings(str(out_dir), GCN, hid_dim=hid_dim, out_dim=out_dim)
    sanity_forward(str(out_dir), GCN, hid_dim=hid_dim, out_dim=out_dim)


if __name__ == "__main__":
    main()
