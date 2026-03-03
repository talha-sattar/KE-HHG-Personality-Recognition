# file: run_global_graph.py


from __future__ import annotations

import os
from pathlib import Path

from Global_graph import build_personality_graphs


def _require_file(p: Path, env_name: str) -> str:
    if not p.exists():
        raise FileNotFoundError(
            f"Missing file: {p}\n"
            f"Either place it at the repo-relative location above or set env var {env_name} to the correct path."
        )
    return str(p)


def main() -> None:
    root = Path(__file__).resolve().parent

    data_processed = root.parent / "src" / "data" / "processed" / "preprocess_check_out"
    out_dir = root.parent / "src" / "outputs" / "global_graph_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Required CSVs
    train_csv = data_processed / "final_train_preprocessed.csv"
    val_csv = data_processed / "final_val_preprocessed.csv"
    test_csv = data_processed / "final_test_preprocessed.csv"

    for p in (train_csv, val_csv, test_csv):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing processed CSV: {p}\n"
                "Put your processed split CSVs in data/processed/ (repo-relative), or update filenames in this runner."
            )

    # Resource defaults (repo-relative)
    nell_ent2ids_default = root.parent / "data" / "assets" / "entity2id.json"
    transe_vec_default = root.parent / "data" / "assets" / "entity2vec.bern"
    nltk_data_default = root.parent / "data" / "assets" / "nltk_data"

    # Allow overrides via env vars (recommended for big/private files)
    nell_ent2ids = Path(os.getenv("NELL_ENT2ID", str(nell_ent2ids_default)))
    transe_vec = Path(os.getenv("TRANSE_VEC", str(transe_vec_default)))
    nltk_data_dir = Path(os.getenv("NLTK_DATA_DIR", str(nltk_data_default)))

    out = build_personality_graphs(
        dataset="personality",
        train_csv=str(train_csv),
        val_csv=str(val_csv),
        test_csv=str(test_csv),

        text_col="prep_text_strict",
        el_text_col="prep_text_strict",
        pos_tagged_col="pos_raw_word_pos",

        out_dir=str(out_dir),

        nell_ent2ids=_require_file(nell_ent2ids, "NELL_ENT2ID"),
        transe_vec=_require_file(transe_vec, "TRANSE_VEC"),

        pos_backend="nltk",
        auto_download_nltk=True,
        nltk_data_dir=str(nltk_data_dir),

        remove_lowfreq=True,
        min_count=5,
        window_size=5,

        # NEL runtime options
        st_device="cpu",
        use_minilm=True,
        max_candidates=20,
        min_ctx_cos=0.12,
        min_margin=0.06,

        # LIWC ONLY (Empath off)
        use_liwc=False,
        liwc_dic="",
        liwc_min_hits=1,
        liwc_binary_edges=True,
        liwc_keep_numbers=True,
        use_empath=True,

        # Doc-BERT
        build_doc_bert=True,
        doc_bert_model="bert-base-uncased",
        doc_bert_device="cuda",
    )

    print("Artifacts at:", out)


if __name__ == "__main__":
    main()
