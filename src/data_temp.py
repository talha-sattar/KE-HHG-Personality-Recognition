# file: src/audit_splits.py
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np

USER_COL = "ID_COL"
TRAIT_NAMES = ["OPEN", "CON", "EXT", "AGR", "NEU"]


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


def read_csv_rows(csv_path: Union[str, Path]) -> List[Dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            raise ValueError(f"Empty CSV or missing header: {csv_path}")
        return list(r)


def parse_split_labels_users(
    csv_path: Union[str, Path],
    doc_indices: Sequence[int],
    label_cols: Sequence[str],
    user_col: str = USER_COL,
) -> Tuple[np.ndarray, np.ndarray]:
    rows = read_csv_rows(csv_path)
    if len(rows) != len(doc_indices):
        raise ValueError(
            f"[ALIGNMENT ERROR] {Path(csv_path).name}: csv_rows={len(rows)} but idx_count={len(doc_indices)}"
        )

    # column existence check
    missing = [c for c in label_cols if c not in rows[0]]
    if missing:
        raise KeyError(f"[COLUMN ERROR] {Path(csv_path).name} missing label columns: {missing}")
    if user_col not in rows[0]:
        raise KeyError(f"[COLUMN ERROR] {Path(csv_path).name} missing user id column '{user_col}'")

    Y = np.zeros((len(rows), len(label_cols)), dtype=np.float32)
    U = np.zeros((len(rows),), dtype=np.int64)

    for i, row in enumerate(rows):
        try:
            y = np.array([float(row[c]) for c in label_cols], dtype=np.float32)
        except Exception as e:
            raise ValueError(f"[PARSE ERROR] Failed parsing labels at row {i} in {csv_path}: {e}")
        Y[i] = y

        u_raw = row[user_col]
        try:
            U[i] = int(float(u_raw))
        except Exception:
            U[i] = abs(hash(u_raw)) % (2**31 - 1)

    return Y, U


def describe_split(name: str, Y: np.ndarray, U: np.ndarray) -> None:
    print(f"\n=== {name} ===")
    print(f"rows={len(Y):,} | users={len(np.unique(U)):,}")

    # NaN / inf checks
    nan_count = int(np.isnan(Y).sum())
    inf_count = int(np.isinf(Y).sum())
    print(f"NaNs={nan_count} | Infs={inf_count}")

    # Basic stats per label
    Yb = (Y > 0.5).astype(np.int32)
    pos_rates = Yb.mean(axis=0)
    means = Y.mean(axis=0)
    stds = Y.std(axis=0)

    for i, t in enumerate(TRAIT_NAMES):
        pos = int(Yb[:, i].sum())
        print(
            f"{t}: pos_rate={pos_rates[i]:.4f} "
            f"pos={pos}/{len(Y):,} "
            f"mean={means[i]:.4f} std={stds[i]:.4f} "
            f"min={Y[:, i].min():.3f} max={Y[:, i].max():.3f}"
        )

    # Flag suspicious patterns
    for i, t in enumerate(TRAIT_NAMES):
        uniq = np.unique(Yb[:, i])
        if len(uniq) == 1:
            print(f"[FLAG] {name} label {t} is constant: {uniq[0]} (this often causes F1=0).")

    # User-doc distribution quick check
    counts = np.bincount((U - U.min()).astype(np.int64))
    counts = counts[counts > 0]
    print(f"user_docs: min={counts.min()} median={int(np.median(counts))} max={counts.max()}")


def main():
    import os

    ROOT = Path(__file__).resolve().parent
    OUT_DIR_DEFAULT = ROOT / "outputs" / "global_graph_output"
    DATA_PROCESSED_DEFAULT = ROOT / "data" / "processed" / "preprocess_check_out"

    out_dir = Path(os.getenv("GLOBAL_GRAPH_OUT", str(OUT_DIR_DEFAULT)))
    data_dir = Path(os.getenv("DATA_PROCESSED", str(DATA_PROCESSED_DEFAULT)))

    train_csv = Path(os.getenv("TRAIN_CSV", str(data_dir / "final_train_preprocessed.csv")))
    val_csv = Path(os.getenv("VAL_CSV", str(data_dir / "final_val_preprocessed.csv")))
    test_csv = Path(os.getenv("TEST_CSV", str(data_dir / "final_test_preprocessed.csv")))

    label_method = os.getenv("LABEL_METHOD", "mean")
    label_cols = label_cols_from_method(label_method)

    print(f"OUT_DIR={out_dir}")
    print(f"TRAIN={train_csv}")
    print(f"VAL={val_csv}")
    print(f"TEST={test_csv}")
    print(f"LABEL_METHOD={label_method}")
    print(f"LABEL_COLS={label_cols}")
    print(f"USER_COL={USER_COL}")

    train_idx = load_split_doc_indices(out_dir, "train")
    val_idx = load_split_doc_indices(out_dir, "val")
    test_idx = load_split_doc_indices(out_dir, "test")

    Ytr, Utr = parse_split_labels_users(train_csv, train_idx, label_cols, USER_COL)
    Yva, Uva = parse_split_labels_users(val_csv, val_idx, label_cols, USER_COL)
    Yte, Ute = parse_split_labels_users(test_csv, test_idx, label_cols, USER_COL)

    describe_split("TRAIN", Ytr, Utr)
    describe_split("VAL", Yva, Uva)
    describe_split("TEST", Yte, Ute)

    # Quick cross-split comparison flags
    print("\n=== CROSS-SPLIT CHECKS ===")
    tr_rates = (Ytr > 0.5).mean(axis=0)
    va_rates = (Yva > 0.5).mean(axis=0)
    te_rates = (Yte > 0.5).mean(axis=0)
    for i, t in enumerate(TRAIT_NAMES):
        print(f"{t}: train={tr_rates[i]:.4f} val={va_rates[i]:.4f} test={te_rates[i]:.4f}")
        if abs(tr_rates[i] - va_rates[i]) > 0.30:
            print(f"[FLAG] {t}: big train-val shift (>0.30). Check column mapping / preprocessing.")

    print("\nIf VAL shows any label with constant 0 or 1, fix the CSV columns/values first.")


if __name__ == "__main__":
    main()