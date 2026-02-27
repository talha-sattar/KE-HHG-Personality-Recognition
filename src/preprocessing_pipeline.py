# file: preprocessing_pipeline.py
"""
One-file preprocessing pipeline (end-to-end):

Step A) Sentence split (SaT): raw chats -> sentence rows CSV
Step B) Binarize Big5 traits using TRAIN-only thresholds (mean/median/kmeans)
Step C) Train/Val split by USER (no leakage by user)
Step D) Text preprocessing: produce prep_text_strict + pos_raw_word_pos

Outputs:
  - labels_binarized/thresholds_mean_median_kmeans.json
  - labels_binarized/train_binarized_<METHOD>.csv, test_binarized_<METHOD>.csv
  - train.csv, val.csv, test.csv
  - preprocess_check_out/train_preprocessed.csv, val_preprocessed.csv, test_preprocessed.csv

Dependencies:
  pip install pandas numpy scikit-learn matplotlib wtpsplit nltk contractions openpyxl
"""

from __future__ import annotations

import csv
import json
import random
import re
import string
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Optional plotting (can disable)
import matplotlib.pyplot as plt

# Sentence splitter
from wtpsplit import SaT

# Text normalization
import contractions


# =============================================================================
# CONFIG (EDIT ONLY THIS)
# =============================================================================

SEED = 42

# Repo-relative folders (portable for GitHub)
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ASSETS_DIR = ROOT / "assets"
OUTPUTS_DIR = ROOT / "outputs"

# ---- Step A: Sentence split ----
RAW_TRAIN_CSV = RAW_DIR / "Training Sample.csv"
RAW_TEST_CSV = RAW_DIR / "Test Sample.csv"
RAW_TEXT_COL = "chat text"  # column to split into sentences
SAT_MODEL_NAME = "sat-3l-sm"
NEWLINE_FIRST = True
SENTENCE_ROWS_TRAIN = PROCESSED_DIR / "train_split_sentences_rows.csv"
SENTENCE_ROWS_TEST = PROCESSED_DIR / "test_split_sentences_rows.csv"

# ---- Step B: Binarization ----
TRAIT_COLS = ["OPEN", "CONSICEN", "EXTRO", "AGREE", "NEURO"]
METHOD = "mean"  # "mean" | "median" | "kmeans"
LABELS_OUT_DIR = OUTPUTS_DIR / "labels_binarized"

# ---- Step C: Train/Val split ----
USER_COL = "ID_COL"
VAL_USER_FRAC = 0.1  # 10% users for val
TRAIN_CSV = PROCESSED_DIR / "train.csv"
VAL_CSV = PROCESSED_DIR / "val.csv"
TEST_CSV = PROCESSED_DIR / "test.csv"

# ---- Step D: Preprocessing ----
TEXT_COL_FOR_PREP = "sentence_text"
SLANG_CSV = ASSETS_DIR / "slangs_meaning.csv"

PREP_OUT_DIR = PROCESSED_DIR / "preprocess_check_out"
OUT_TRAIN_PREP = PREP_OUT_DIR / "final_train_preprocessed.csv"
OUT_VAL_PREP = PREP_OUT_DIR / "final_val_preprocessed.csv"
OUT_TEST_PREP = PREP_OUT_DIR / "final_test_preprocessed.csv"

DROP_SHORT_SENTENCES = True
MIN_WORDS = 6
COUNT_AFTER_STOPWORDS = True

# Toggle plots for binarization
MAKE_PLOTS = True
BINS_PER_TRAIT = 20
DOT_SAMPLE_CAP = 5000

# =============================================================================
# End CONFIG
# =============================================================================



# =============================================================================
# Step A: Sentence splitting (SaT)
# =============================================================================

WS = re.compile(r"[ \t]+")
NL = re.compile(r"\n+")
HAS_PUNCT = re.compile(r"[.!?]")


def normalize_text(text: str) -> str:
    text = "" if text is None else str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = WS.sub(" ", text)
    return text.strip()


def split_newlines(text: str) -> List[str]:
    return [x.strip() for x in NL.split(text) if x.strip()]


def should_segment_inside(chunk: str, long_threshold: int = 220) -> bool:
    return bool(HAS_PUNCT.search(chunk) or len(chunk) >= long_threshold)


def segment_text(sat: SaT, text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []

    if not NEWLINE_FIRST:
        return [s.strip() for s in sat.split(text) if s and s.strip()]

    out: List[str] = []
    for line in split_newlines(text):
        if not line:
            continue
        if not should_segment_inside(line):
            out.append(line)
        else:
            out.extend([s.strip() for s in sat.split(line) if s and s.strip()])
    return out


def explode_sentences(input_csv: Path, text_col: str, output_csv: Path) -> None:
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {input_csv}")

    df = pd.read_csv(input_csv)
    if text_col not in df.columns:
        raise KeyError(f"Column '{text_col}' not found. Available: {df.columns.tolist()}")

    sat = SaT(SAT_MODEL_NAME)

    out_rows: List[Dict[str, Any]] = []
    cols = df.columns.tolist()
    text_idx = cols.index(text_col)

    for row_tuple in df.itertuples(index=False, name=None):
        base = dict(zip(cols, row_tuple))
        sentences = segment_text(sat, row_tuple[text_idx])

        if not sentences:
            r = base.copy()
            r["sentence_index"] = 0
            r["sentence_text"] = ""
            r["splitter_used"] = f"wtpsplit:SaT:{SAT_MODEL_NAME}"
            out_rows.append(r)
            continue

        for s_idx, sent in enumerate(sentences):
            r = base.copy()
            r["sentence_index"] = s_idx
            r["sentence_text"] = sent
            r["splitter_used"] = f"wtpsplit:SaT:{SAT_MODEL_NAME}"
            out_rows.append(r)

    out_df = pd.DataFrame(out_rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print("✅ Step A saved:", output_csv, "rows=", len(out_df))


# =============================================================================
# Step B: Binarization (TRAIN thresholds only)
# =============================================================================

def _safe_to_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _to_float(x: pd.Series) -> np.ndarray:
    return pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)


def compute_thresholds_on_train(train_df: pd.DataFrame, trait_cols: List[str], seed: int) -> Dict[str, Dict[str, float]]:
    thr: Dict[str, Dict[str, float]] = {}
    for col in trait_cols:
        x = _to_float(train_df[col])
        x = x[~np.isnan(x)]
        if x.size == 0:
            thr[col] = {"mean": np.nan, "median": np.nan, "kmeans": np.nan}
            continue
        mu = float(np.mean(x))
        md = float(np.median(x))
        km = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(x.reshape(-1, 1))
        centers = np.sort(km.cluster_centers_.ravel())
        km_mid = float((centers[0] + centers[1]) / 2.0)
        thr[col] = {"mean": mu, "median": md, "kmeans": km_mid}
    return thr


def _compute_threshold(train_vals: np.ndarray, method: str, seed: int) -> float:
    x = train_vals[~np.isnan(train_vals)]
    if x.size == 0:
        return np.nan
    if method == "mean":
        return float(np.mean(x))
    if method == "median":
        return float(np.median(x))
    if method == "kmeans":
        km = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(x.reshape(-1, 1))
        c = np.sort(km.cluster_centers_.ravel())
        return float((c[0] + c[1]) / 2.0)
    raise ValueError(f"Unknown method: {method}")


def _binarize_with_threshold(vals: np.ndarray, thr: float) -> np.ndarray:
    if np.isnan(thr):
        return np.full_like(vals, fill_value=np.nan, dtype=float)
    return np.where(np.isnan(vals), np.nan, (vals >= thr).astype(float))


def plot_hist_panels(train_df: pd.DataFrame, trait_cols: List[str], thresholds: Dict[str, Dict[str, float]], out_png: Path) -> None:
    n = len(trait_cols)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(8, 1.9 * n), constrained_layout=True)
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, trait_cols):
        x = _to_float(train_df[col])
        x = x[~np.isnan(x)]
        ax.hist(x, bins=BINS_PER_TRAIT, edgecolor="k", alpha=0.65)
        ax.axvline(thresholds[col]["mean"], linestyle="--", linewidth=2.5, label="Mean")
        ax.axvline(thresholds[col]["median"], linestyle=":", linewidth=2.5, label="Median")
        ax.axvline(thresholds[col]["kmeans"], linestyle="-.", linewidth=2.5, label="kMeans")
        ax.set_title(col)
        ax.legend(fontsize=8, frameon=False)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def binarize_traits(train_csv: Path, test_csv: Path, out_dir: Path, method: str, seed: int) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    missing_tr = [c for c in TRAIT_COLS if c not in train_df.columns]
    missing_te = [c for c in TRAIT_COLS if c not in test_df.columns]
    if missing_tr:
        raise ValueError(f"Missing TRAIN columns: {missing_tr}")
    if missing_te:
        raise ValueError(f"Missing TEST columns: {missing_te}")

    thresholds = compute_thresholds_on_train(train_df, TRAIT_COLS, seed=seed)
    (out_dir / "thresholds_mean_median_kmeans.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")

    train_out = train_df.copy()
    test_out = test_df.copy()
    for col in TRAIT_COLS:
        tr = _safe_to_float(train_df[col]).to_numpy()
        te = _safe_to_float(test_df[col]).to_numpy()
        thr = _compute_threshold(tr, method, seed)
        train_out[f"{col}_bin_{method}"] = _binarize_with_threshold(tr, thr)
        test_out[f"{col}_bin_{method}"] = _binarize_with_threshold(te, thr)

    train_bin = out_dir / f"train_binarized_{method}.csv"
    test_bin = out_dir / f"test_binarized_{method}.csv"
    train_out.to_csv(train_bin, index=False)
    test_out.to_csv(test_bin, index=False)

    if MAKE_PLOTS:
        plot_hist_panels(train_df, TRAIT_COLS, thresholds, out_png=out_dir / "hist_panels_train.png")

    print("✅ Step B saved:", train_bin)
    print("✅ Step B saved:", test_bin)
    return train_bin, test_bin


# =============================================================================
# Step C: Train/Val split by user
# =============================================================================

def add_split_columns(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    df = df.copy()
    df["split"] = split_name
    df["ID_COL_split"] = split_name + "_" + df[USER_COL].astype(str)
    return df


def make_train_val_test(trainval_csv: Path, test_csv: Path, train_out: Path, val_out: Path, test_out: Path) -> None:
    df = pd.read_csv(trainval_csv)
    if USER_COL not in df.columns:
        raise KeyError(f"USER_COL='{USER_COL}' not found. Columns: {df.columns.tolist()}")

    by_user = defaultdict(list)
    for i, u in enumerate(df[USER_COL].astype(str).tolist()):
        by_user[u].append(i)

    users = list(by_user.keys())
    random.seed(SEED)
    random.shuffle(users)

    val_users = set(users[: max(1, int(len(users) * VAL_USER_FRAC))])

    train_idx, val_idx = [], []
    for u, rows in by_user.items():
        (val_idx if u in val_users else train_idx).extend(rows)

    train_df = add_split_columns(df.iloc[sorted(train_idx)].reset_index(drop=True), "train")
    val_df = add_split_columns(df.iloc[sorted(val_idx)].reset_index(drop=True), "val")

    train_df.to_csv(train_out, index=False)
    val_df.to_csv(val_out, index=False)
    print("✅ Step C saved:", train_out)
    print("✅ Step C saved:", val_out)

    te = pd.read_csv(test_csv)
    if USER_COL not in te.columns:
        raise KeyError(f"USER_COL='{USER_COL}' not found in test. Columns: {te.columns.tolist()}")
    te = add_split_columns(te, "test")
    te.to_csv(test_out, index=False)
    print("✅ Step C saved:", test_out)


# =============================================================================
# Step D: Text preprocessing (strict/check) + POS
# =============================================================================

PAPER_KEEP_STOPWORDS = {
    "a","an","the","about","above","across","after","against","along","among","around","at","before","behind","below",
    "beneath","beside","between","beyond","by","during","except","for","from","in","inside","into","near","of","off",
    "on","onto","out","outside","over","past","since","through","throughout","to","toward","towards","under",
    "underneath","until","up","upon","with","within","without","i","me","my","myself","we","our","ours","ourselves",
    "you","your","yours","yourself","yourselves","he","him","his","himself","she","her","hers","herself","it","its",
    "itself","they","them","their","theirs","themselves",
}

_PUNCT_TABLE_ALL = str.maketrans({p: " " for p in string.punctuation})
_PUNCT_TABLE_KEEP_APOS = str.maketrans({p: " " for p in string.punctuation if p != "'"})
_DIGITS_RE = re.compile(r"\d+")
_SPACES_RE = re.compile(r"\s+")
_POSSESSIVE_S_RE = re.compile(r"\b([A-Za-z]+)\'s\b")
_POSSESSIVE_PL_RE = re.compile(r"\b([A-Za-z]+)s\'\b")


@dataclass(frozen=True)
class SlangMap:
    mapping: Dict[str, str]


@dataclass(frozen=True)
class PipelineConfig:
    keep_punct: bool
    keep_numbers: bool
    expand_contractions: bool
    slang_action: str              # none|detect|tag|replace
    keep_apostrophe: bool
    possessive_mode: str           # keep|strip|normalize
    letters_only: bool
    stopwords_enabled: bool
    paper_stopwords: bool


def ensure_nltk() -> None:
    import nltk
    for res, pkg in [
        ("tokenizers/punkt", "punkt"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("corpora/stopwords", "stopwords"),
    ]:
        try:
            nltk.data.find(res)
        except LookupError:
            nltk.download(pkg)


def load_slang_map(path: Path) -> SlangMap:
    if not path.exists():
        return SlangMap(mapping={})
    slang_map: Dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "slang" not in reader.fieldnames or "meaning" not in reader.fieldnames:
            return SlangMap(mapping={})
        for row in reader:
            k = (row.get("slang") or "").strip().lower()
            v = (row.get("meaning") or "").strip()
            if k and v:
                slang_map[k] = v
    return SlangMap(mapping=slang_map)


def normalize_spaces(s: str) -> str:
    return _SPACES_RE.sub(" ", (s or "").strip())


def paper_style_stopwords_to_remove() -> Set[str]:
    from nltk.corpus import stopwords as _nltk_stopwords
    nltk_sw = set(_nltk_stopwords.words("english"))
    keep = {w for w in PAPER_KEEP_STOPWORDS if w in nltk_sw}
    remove_set = nltk_sw - keep
    remove_set.add("")
    return remove_set


def stopwords_for(cfg: PipelineConfig) -> Set[str]:
    if not cfg.stopwords_enabled:
        return {""}
    if cfg.paper_stopwords:
        return paper_style_stopwords_to_remove()
    return {""}


def apply_possessives(text: str, mode: str) -> str:
    if mode == "keep":
        return text
    if mode == "strip":
        text = _POSSESSIVE_S_RE.sub(r"\1", text)
        return _POSSESSIVE_PL_RE.sub(r"\1s", text)
    if mode == "normalize":
        text = _POSSESSIVE_S_RE.sub(r"\1s", text)
        return _POSSESSIVE_PL_RE.sub(r"\1s", text)
    raise ValueError(f"Unknown possessive_mode: {mode}")


def tokenize(text: str) -> List[str]:
    from nltk import word_tokenize
    return [t for t in word_tokenize(text or "") if t]


def pos_raw_word_pos(raw: str) -> str:
    from nltk import pos_tag, word_tokenize
    toks = word_tokenize(normalize_spaces(raw))
    pairs = [(w, p.lower()) for w, p in pos_tag(toks)]
    return " ".join([f"{w}/{p}" for w, p in pairs])


def pipeline_final(raw: str, slang: SlangMap, cfg: PipelineConfig) -> str:
    s = "" if raw is None else str(raw)
    s = s.strip().lower()
    s = apply_possessives(s, cfg.possessive_mode)

    if cfg.expand_contractions:
        s = contractions.fix(s)

    # slang (optional)
    if cfg.slang_action != "none" and slang.mapping:
        detected: List[str] = []
        out_tokens: List[str] = []
        punct_chars = set(string.punctuation)
        for tok in s.split():
            low = tok.lower()
            if low in slang.mapping:
                detected.append(low)
                if cfg.slang_action == "replace":
                    tok = slang.mapping[low]
                elif cfg.slang_action == "tag":
                    tok = f"<SLANG:{tok}>"
            else:
                # try stripping punctuation around token
                left, right = 0, len(tok)
                while left < right and tok[left] in punct_chars:
                    left += 1
                while right > left and tok[right - 1] in punct_chars:
                    right -= 1
                core = tok[left:right].lower()
                if core and core in slang.mapping:
                    detected.append(core)
                    if cfg.slang_action == "replace":
                        tok = tok[:left] + slang.mapping[core] + tok[right:]
                    elif cfg.slang_action == "tag":
                        tok = tok[:left] + f"<SLANG:{tok[left:right]}>" + tok[right:]
            out_tokens.append(tok)
        s = " ".join(out_tokens)

    if not cfg.keep_numbers:
        s = _DIGITS_RE.sub(" ", s)

    if not cfg.keep_punct:
        s = s.translate(_PUNCT_TABLE_KEEP_APOS if cfg.keep_apostrophe else _PUNCT_TABLE_ALL)

    if cfg.letters_only:
        s = re.sub(r"[^A-Za-z\s']", " ", s)
        s = re.sub(r"\s{2,}", " ", s)

    return normalize_spaces(s)


def preprocess_split_csv(in_csv: Path, out_csv: Path) -> None:
    df = pd.read_csv(in_csv)
    if TEXT_COL_FOR_PREP not in df.columns:
        raise KeyError(f"Missing '{TEXT_COL_FOR_PREP}' in {in_csv.name}. Columns={df.columns.tolist()}")

    ensure_nltk()
    slang = load_slang_map(SLANG_CSV)

    # CHECK (loose)
    check_cfg = PipelineConfig(
        keep_punct=True,
        keep_numbers=True,
        expand_contractions=True,
        slang_action="detect",
        keep_apostrophe=False,
        possessive_mode="keep",
        letters_only=False,
        stopwords_enabled=False,
        paper_stopwords=False,
    )

    # STRICT (paper-ish, your requirements)
    strict_cfg = PipelineConfig(
        keep_punct=False,
        keep_numbers=False,
        expand_contractions=False,
        slang_action="none",
        keep_apostrophe=True,      # keep apostrophes for don't / dad's
        possessive_mode="keep",
        letters_only=True,
        stopwords_enabled=True,
        paper_stopwords=True,
    )

    sw_check = stopwords_for(check_cfg)
    sw_strict = stopwords_for(strict_cfg)

    raw = df[TEXT_COL_FOR_PREP].fillna("").astype(str)

    strict_texts: List[str] = []
    strict_tokens_no_sw: List[str] = []
    strict_num_tokens: List[int] = []
    pos_col: List[str] = []

    for s in raw.tolist():
        t_strict = pipeline_final(s, slang, strict_cfg)
        toks = tokenize(t_strict)
        toks_no_sw = [t for t in toks if t.lower() not in sw_strict]
        toks_used = toks_no_sw if COUNT_AFTER_STOPWORDS else toks

        strict_texts.append(t_strict)
        strict_tokens_no_sw.append(" ".join(toks_no_sw))
        strict_num_tokens.append(len(toks_used))
        pos_col.append(pos_raw_word_pos(s))

    out = df.copy()
    out["prep_text_strict"] = strict_texts
    out["prep_tokens_strict_no_stopwords"] = strict_tokens_no_sw
    out["prep_num_tokens_strict"] = strict_num_tokens
    out["pos_raw_word_pos"] = pos_col

    if DROP_SHORT_SENTENCES:
        out = out[out["prep_num_tokens_strict"] >= MIN_WORDS].reset_index(drop=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False, encoding="utf-8")
    print("✅ Step D saved:", out_csv, "rows=", len(out))


# =============================================================================
# Main pipeline
# =============================================================================

def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    # A) Sentence split
    explode_sentences(RAW_TRAIN_CSV, RAW_TEXT_COL, SENTENCE_ROWS_TRAIN)
    explode_sentences(RAW_TEST_CSV, RAW_TEXT_COL, SENTENCE_ROWS_TEST)

    # B) Binarize traits (using sentence-rows files)
    train_bin, test_bin = binarize_traits(SENTENCE_ROWS_TRAIN, SENTENCE_ROWS_TEST, LABELS_OUT_DIR, METHOD, SEED)

    # C) Train/Val split by user
    make_train_val_test(train_bin, test_bin, TRAIN_CSV, VAL_CSV, TEST_CSV)

    # D) Preprocess for model input (strict + POS)
    PREP_OUT_DIR.mkdir(parents=True, exist_ok=True)
    preprocess_split_csv(TRAIN_CSV, OUT_TRAIN_PREP)
    preprocess_split_csv(VAL_CSV, OUT_VAL_PREP)
    preprocess_split_csv(TEST_CSV, OUT_TEST_PREP)

    print("\n✅ Pipeline finished.")
    print("Train:", OUT_TRAIN_PREP)
    print("Val  :", OUT_VAL_PREP)
    print("Test :", OUT_TEST_PREP)


if __name__ == "__main__":
    main()