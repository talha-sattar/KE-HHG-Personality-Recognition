# file: Global_graph_empath_4.py
# Adds POS & Empath one-hot feature matrices (sparse identity) saved to:
#   - pos_onehot.pkl
#   - empath_onehot.pkl
#
# Everything else is your existing pipeline for Word/POS/Empath/Entity + Doc-BERT + Ae + WordBERT + At.
# Source baseline referenced from your current file layout. (SHAPES/IO unchanged for other artifacts)
#
# NEW (additive only): dumps per-doc ordered word-id sequences (post-stopwords) to:
#   - doc_word_seq.jsonl
# Each line: {"doc_idx": <int>, "word_ids": [<int>, ...]} (keeps order + duplicates)

import os, re, json, math, hashlib, pickle as pkl
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
from collections import defaultdict


import numpy as np
from scipy.sparse import coo_matrix, eye as speye  # <-- add speye for one-hot identities

# --- POS (NLTK) ---
import nltk
from nltk import pos_tag, word_tokenize

# --- Empath ---
try:
    from empath import Empath
except Exception:
    Empath = None

# ================== Utils ==================
def _debug_write(out_path: Path, name: str, text: str):
    (out_path/name).write_text(text, encoding="utf-8", errors="ignore")

_CLEAN_STEPS = [
]
def _clean(s: str) -> str:
    s = (s or "").strip().lower()
    for pat, rep in _CLEAN_STEPS: s = re.sub(pat, rep, s)
    s = s.replace("_", " ")
    return re.sub(r"\s+", " ", s).strip()

def _normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def parse_token_tag_string(s: str):
    """Parse pre-tagged strings like: 'i/prp was/vbd born/vbn ...' into (token, tag) pairs."""
    out = []
    for part in (s or "").split():
        if "/" not in part:
            continue
        tok, tag = part.rsplit("/", 1)
        tok = tok.strip()
        tag = tag.strip().lower()
        if tok and tag:
            out.append((tok, tag))
    return out

def _norm_surface(s: str) -> str:
    return " ".join((s or "").strip().lower().split())



# ================== IO ==================
def _read_texts_pandas(path: Union[str, Path], text_col: str, el_text_col: Optional[str], pos_col: Optional[str], chunksize: int, max_rows: Optional[int], encoding: str, on_bad_lines: str) -> List[Dict]:
    import pandas as pd
    rows=[]; total=0
    usecols = [text_col] + ([el_text_col] if el_text_col else []) + ([pos_col] if pos_col else [])
    dtype = {c: "string" for c in usecols}
    for ch in pd.read_csv(path, usecols=usecols, chunksize=chunksize, dtype=dtype,
                          encoding=encoding, on_bad_lines=on_bad_lines, engine="python"):
        col_text = ch[text_col].fillna("").astype(str).tolist()
        col_pos = ch[pos_col].fillna("").astype(str).tolist() if pos_col else [""] * len(col_text)
        col_el = ch[el_text_col].fillna("").astype(str).tolist() if el_text_col else col_text
        for v, p, e in zip(col_text, col_pos, col_el):
            rows.append({"text": v, "el_text": e, "pos": p}); total+=1
            if max_rows is not None and total>=max_rows: return rows
    return rows

def _read_texts_csvfallback(path: Union[str, Path], text_col: str, el_text_col: Optional[str], pos_col: Optional[str], max_rows: Optional[int], encoding: str) -> List[Dict]:
    import csv
    try: csv.field_size_limit(2**31-1)
    except OverflowError: csv.field_size_limit(2**30)
    rows=[]; total=0
    with open(path, "r", encoding=encoding, errors="ignore", newline="") as f:
        r = csv.reader(f)
        hdr = next(r, None)
        if hdr is None: return rows
        try: idx = hdr.index(text_col)
        except ValueError: raise KeyError(f"Column '{text_col}' not in header: {hdr}")
        eidx = None
        if el_text_col:
            try:
                eidx = hdr.index(el_text_col)
            except ValueError:
                raise KeyError(f"Column '{el_text_col}' not in header: {hdr}")

        pidx = None
        if pos_col:
                try:
                    pidx = hdr.index(pos_col)
                except ValueError:
                    raise KeyError(f"Column '{pos_col}' not in header: {hdr}")

        for i, row in enumerate(r, 1):
            txt = row[idx] if idx < len(row) else ""
            ptxt = row[pidx] if (pidx is not None and pidx < len(row)) else ""
            etxt = row[eidx] if (eidx is not None and eidx < len(row)) else txt
            rows.append({"text": txt, "el_text": etxt, "pos": ptxt}); total+=1
            if (i % 200000)==0: print(f"[CSV] {i} …")
            if max_rows is not None and total>=max_rows: break
    return rows

def _ensure_nltk(pos_backend: str, auto_download: bool, nltk_data_dir: Optional[Union[str, Path]]) -> None:
    if pos_backend == "skip": return
    if nltk_data_dir: nltk.data.path.insert(0, str(Path(nltk_data_dir)))
    need=[]
    for res in ["tokenizers/punkt", "taggers/averaged_perceptron_tagger_eng"]:
        try: nltk.data.find(res)
        except LookupError: need.append(res.split("/",1)[1])
    if need and auto_download:
        for pkg in need: nltk.download(pkg, quiet=False)
    else:
        for res in ["tokenizers/punkt", "taggers/averaged_perceptron_tagger_eng"]:
            nltk.data.find(res)

# ================== Graph math ==================
def _incidence_matrix(docs: List[str], vocab_map: Dict[str,int], sparse: bool=True):
    if not vocab_map:
        return coo_matrix((len(docs), 0)) if sparse else np.zeros((len(docs), 0), dtype=np.float32)
    rows, cols = [], []
    for i, doc in enumerate(docs):
        seen=set()
        for tok in doc.split():
            j=vocab_map.get(tok)
            if j is not None and j not in seen:
                seen.add(j); rows.append(i); cols.append(j)
    data=np.ones(len(rows), dtype=np.float32)
    M=coo_matrix((data,(rows,cols)), shape=(len(docs), len(vocab_map)))
    return M if sparse else M.toarray()

def _PMI(seqs: List[str], vocab_map: Dict[str, int], window_size: int, sparse: bool):
    V = len(vocab_map)
    if V == 0:
        return coo_matrix((0, 0)) if sparse else np.zeros((0, 0), dtype=np.float64)

    W_ij = np.zeros((V, V), dtype=np.float64)
    W_i = np.zeros(V, dtype=np.float64)
    Wc = 0.0

    for doc in seqs:
        # Shine: split(' ') + remove '' from context
        word_list = doc.split(" ")
        if len(word_list) - window_size < 0:
            window_num = 1
        else:
            window_num = len(word_list) - window_size + 1

        for i in range(window_num):
            Wc += 1.0
            context = list(set(word_list[i : i + window_size]))
            while "" in context:
                context.remove("")

            ids = [vocab_map[t] for t in context if t in vocab_map]
            for a in ids:
                W_i[a] += 1.0

            for x in range(len(ids)):
                ax = ids[x]
                for y in range(x + 1, len(ids)):
                    ay = ids[y]
                    W_ij[ax, ay] += 1.0
                    W_ij[ay, ax] += 1.0

    if sparse:
        rows, cols, data = [], [], []
        for i in range(V):
            rows.append(i)
            cols.append(i)
            data.append(1.0)
            nz = [j for j in np.nonzero(W_ij[i])[0] if j > i]
            for j in nz:
                den = W_i[i] * W_i[j]
                if den > 0 and W_ij[i, j] > 0:
                    val = math.log((W_ij[i, j] * Wc) / den)
                    if val > 0:
                        rows += [i, j]
                        cols += [j, i]
                        data += [val, val]
        return coo_matrix((data, (rows, cols)), shape=(V, V))
    else:
        A = np.zeros((V, V), dtype=np.float64)
        for i in range(V):
            A[i, i] = 1.0
            nz = [j for j in np.nonzero(W_ij[i])[0] if j > i]
            for j in nz:
                den = W_i[i] * W_i[j]
                if den > 0 and W_ij[i, j] > 0:
                    val = math.log((W_ij[i, j] * Wc) / den)
                    if val > 0:
                        A[i, j] = A[j, i] = val
        return A

# ================== NELL helpers ==================
def _load_ent2id_any(path: str) -> Dict[str, Any]:
    p=Path(path)
    if not p.exists(): raise FileNotFoundError(path)
    if p.suffix.lower()==".json":
        return json.loads(p.read_text(encoding="utf-8"))
    raw={}
    with open(p,"r",encoding="utf-8") as f:
        for line in f:
            line=line.rstrip("\n")
            if not line or "\t" not in line: continue
            k,v=line.split("\t",1)
            if v.strip().isdigit(): raw[k]=int(v)
    return raw

def _parse_concept_and_surface(key: str)->Tuple[str,str]:
    if key.startswith("concept_"):
        parts=key.split("_",2)
        if len(parts)>=3: return parts[1], parts[2]
    if key.startswith("concept:"):
        parts=key.split(":",2)
        if len(parts)>=3: return parts[1], parts[2]
    return "unknown", key

def _build_surf2gids_from_raw(raw_ent2id_path: str, add_space_alias: bool=True)->Dict[str,List[int]]:
    raw=_load_ent2id_any(raw_ent2id_path)
    def last_name(k: str)->str:
        if k.startswith("concept_"):
            parts=k.split("_",2); return parts[2] if len(parts)>=3 else k
        if k.startswith("concept:"):
            parts=k.split(":",2); return parts[2] if len(parts)>=3 else k
        return k
    surf2gids={}
    for k,gid in raw.items():
        name=last_name(str(k))
        surf=_norm_surface(name)
        L=surf2gids.setdefault(surf,[])
        g=int(gid)
        if g not in L: L.append(g)
    if add_space_alias:
        adds={}
        for surf,gids in surf2gids.items():
            if "_" in surf:
                alias=_norm_surface(surf.replace("_"," "))
                if alias and alias!=surf:
                    dst=adds.setdefault(alias,[])
                    for g in gids:
                        if g not in dst: dst.append(g)
        for k,v in adds.items():
            dst=surf2gids.setdefault(k,[])
            for g in v:
                if g not in dst: dst.append(g)
    return surf2gids

def _build_gid2concept(raw_ent2id_path: str)->Dict[int,str]:
    raw=_load_ent2id_any(raw_ent2id_path); out={}
    for k,gid in raw.items():
        cpt,_=_parse_concept_and_surface(str(k))
        out[int(gid)]=cpt
    return out

def _build_gid2desc(raw_ent2id_path: str)->Dict[int,str]:
    raw=_load_ent2id_any(raw_ent2id_path)
    out={}
    for k,gid in raw.items():
        cpt,surf_raw=_parse_concept_and_surface(str(k))
        surf=" ".join(surf_raw.replace("_"," ").split())
        out[int(gid)] = f"{surf} — {cpt}"
    return out

# ================== MiniLM gloss scorer (cache) ==================
ST_MODEL_NAME = "all-MiniLM-L6-v2"
ST_DEVICE     = "cpu"   # or "cpu"
ST_BATCH_SIZE = 1024

try:
    from sentence_transformers import SentenceTransformer
    _ST_MODEL = SentenceTransformer(ST_MODEL_NAME, device=ST_DEVICE)
except Exception as e:
    _ST_MODEL = None
    print(f"[WARN] sentence-transformers unavailable ({e}); context scoring off.")

def _file_hash(path: str) -> str:
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""): h.update(chunk)
    return h.hexdigest()[:12]

def _safe_norm(v: np.ndarray) -> np.ndarray:
    d=np.linalg.norm(v, axis=-1, keepdims=True); d=np.maximum(d,1e-8); return v/d

def precompute_gid2desc_embeddings(gid2desc: Dict[int,str], *, batch_size:int, device:str, model_name:str,
                                   raw_ent2id_path:str, cache_root:Path) -> Tuple[np.ndarray, List[int], Dict[int,int], Path]:
    if _ST_MODEL is None: raise RuntimeError("SentenceTransformer not available.")
    ent_hash=_file_hash(raw_ent2id_path)
    model_key=model_name.replace("/","_")
    cache_dir=cache_root / model_key / ent_hash
    cache_dir.mkdir(parents=True, exist_ok=True)
    emb_path=cache_dir/"gid2desc_emb.npy"; gids_path=cache_dir/"gid_order.json"
    if emb_path.exists() and gids_path.exists():
        try:
            gid_order=[int(x) for x in json.loads(gids_path.read_text(encoding="utf-8"))]
            if set(gid_order)==set(gid2desc.keys()):
                emb=np.load(emb_path, mmap_mode="r")
                gid2row={g:i for i,g in enumerate(gid_order)}
                print(f"[cache] gid2desc: {emb.shape}"); return emb, gid_order, gid2row, cache_dir
            else:
                print("[cache] gid set changed; rebuilding …")
        except Exception: print("[cache] corrupt; rebuilding …")
    gid_order=sorted(gid2desc.keys()); texts=[gid2desc[g] for g in gid_order]
    print(f"[precompute] encoding {len(texts):,} glosses on {device} …")
    dim=_ST_MODEL.get_sentence_embedding_dimension()
    embs=np.zeros((len(texts), dim), dtype=np.float32)
    for i in range(0,len(texts), batch_size):
        batch=texts[i:i+batch_size]
        vecs=_ST_MODEL.encode(batch, convert_to_numpy=True, device=device, show_progress_bar=False)
        embs[i:i+len(batch)] = _safe_norm(vecs)
        if (i//batch_size)%20==0: print(f"[precompute] {i+len(batch):,}/{len(texts):,}")
    np.save(emb_path, embs); gids_path.write_text(json.dumps(gid_order), encoding="utf-8")
    gid2row={g:i for i,g in enumerate(gid_order)}
    emb=np.load(emb_path, mmap_mode="r")
    print(f"[cache] saved - {emb_path}")
    return emb, gid_order, gid2row, cache_dir

_last_ctx_text=None
_last_ctx_emb=None
def _ctx_embed(text: str) -> np.ndarray:
    global _last_ctx_text,_last_ctx_emb
    if _ST_MODEL is None: return np.zeros((384,), dtype=np.float32)
    if _last_ctx_text==text and _last_ctx_emb is not None: return _last_ctx_emb
    vec=_ST_MODEL.encode([text], convert_to_numpy=True, device=ST_DEVICE)[0]
    _last_ctx_text=text; _last_ctx_emb=_safe_norm(vec); return _last_ctx_emb

def score_with_cached_matrix(s: str, cands: List[int], emb_mmap: np.ndarray, gid2row: Dict[int,int]) -> np.ndarray:
    if not cands or emb_mmap is None or emb_mmap.size==0: return np.zeros((len(cands),), dtype=np.float32)
    ctx=_ctx_embed(s)
    rows=[]; valid=[]
    for k,g in enumerate(cands):
        r=gid2row.get(g,-1)
        if r>=0: rows.append(r); valid.append(k)
    if not rows: return np.zeros((len(cands),), dtype=np.float32)
    ent=emb_mmap[rows]
    vals=ent @ ctx
    out=np.zeros((len(cands),), dtype=np.float32); out[valid]=vals.astype(np.float32)
    return out

# ================== Trie linker ==================
class _TrieNode:
    __slots__=("ch","gids","s")
    def __init__(self): self.ch={}; self.gids=[]; self.s=""
class _Linker:
    def __init__(self, surf2gids: Dict[str, List[int]]):
        self.root=_TrieNode(); self.surf2gids=surf2gids
        for s,gids in surf2gids.items():
            node=self.root
            for c in s: node=node.ch.setdefault(c,_TrieNode())
            node.gids=list(gids); node.s=s
    def link(self, text: str)->List[Dict[str,Any]]:
        t=_clean(text); n=len(t); i=0; hits=[]
        while i<n:
            node=self.root; j=i; last=(-1,[], "")
            while j<n and t[j] in node.ch:
                node=node.ch[t[j]]
                if node.gids: last=(j,node.gids,node.s)
                j+=1
            if last[0]!=-1:
                end,gids,s_norm=last
                if self._wb(t,i,end): hits.append((s_norm,gids))
                i=end+1
            else: i+=1
        out=[]; seen=set()
        for s_norm,gids in hits:
            sig=(s_norm, tuple(sorted(gids)))
            if sig not in seen: seen.add(sig); out.append((s_norm,gids))
        res=[]; taken=[]; low=text.lower()
        def overlaps(a,b): return not (a[1]<=b[0] or b[1]<=a[0])
        for s_norm,gids in out:
            pattern=r"\b"+re.escape(s_norm).replace(r"\ ", r"\s+")+r"\b"
            for m in re.finditer(pattern, low, flags=re.IGNORECASE):
                span=(m.start(), m.end())
                if any(overlaps(span,t) for t in taken): continue
                taken.append(span)
                res.append({"candidates":list(gids),"surface_norm":s_norm,"start":m.start(),"end":m.end(),"matched_text":text[m.start():m.end()]})
                break
        res.sort(key=lambda x:x["start"]); return res
    @staticmethod
    def _wb(t: str, a: int, b: int)->bool:
        return (a==0 or t[a-1]==" ") and (b==len(t)-1 or t[b+1]==" ")

# ================== TransE helpers (comma seeds) ==================
def _normalize_rows(X: np.ndarray)->np.ndarray:
    n=np.linalg.norm(X, axis=1, keepdims=True); n=np.maximum(n,1e-8); return X/n
def _load_transe_auto(path: str, ent_count: Optional[int]) -> np.ndarray:
    try:
        arr=np.loadtxt(path).astype(np.float32)
        if arr.ndim==1: arr=arr.reshape(1,-1)
        return arr
    except Exception: pass
    if ent_count is None: raise ValueError("Binary TransE but ent_count unknown.")
    filesize=os.path.getsize(path); total_floats=filesize//4
    if total_floats % ent_count != 0: raise ValueError("Cannot infer dim for binary vec.")
    dim=total_floats//ent_count
    return np.fromfile(path, dtype=np.float32).reshape(ent_count, dim)
def _comma_mult(m_i,m_j,full)->float:
    a,b=m_i["end"],m_j["start"];
    if b<=a: a,b=m_j["end"],m_i["start"]
    gap=full[a:b].strip().lower()
    if gap.startswith(",") and len(gap)<=3: return 2.5
    if len(gap)<=2: return 1.5
    return 1.0
def _seed_comma_pairs(mentions,text,E):
    n=len(mentions); chosen=[-1]*n
    if not isinstance(E,np.ndarray) or E.size==0: return chosen
    i=0
    while i+1<n:
        mi,mj=mentions[i],mentions[i+1]
        mult=_comma_mult(mi,mj,text)
        if mult>1.0 and mi["candidates"] and mj["candidates"]:
            Ci=_normalize_rows(E[np.array(mi["candidates"],dtype=int)])
            Cj=_normalize_rows(E[np.array(mj["candidates"],dtype=int)])
            S=Ci@Cj.T
            best_val=-1e9; best=None
            for ii,gi in enumerate(mi["candidates"]):
                for jj,gj in enumerate(mj["candidates"]):
                    val=mult*float(S[ii,jj])
                    if val>best_val: best_val,best=val,(gi,gj)
            if best and best_val>=0.02:
                chosen[i],chosen[i+1]=int(best[0]),int(best[1]); i+=2; continue
        i+=1
    return chosen

# ================== LIWC (.dic) ==================
# IMPORTANT:
# The official LIWC app and LIWC API apply proprietary tokenization / rule logic.
# Here we intentionally use the same parsing approach as your standalone check:
#   from liwc import load_token_parser
# so the pipeline and your check produce consistent category mappings.

from collections import defaultdict as _dd  # local alias to avoid shadowing above
import re as _re


_LIWC_WORD_RE = _re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?")

def _liwc_tokenize(text: str, *, keep_numbers: bool = True) -> List[str]:
    """Tokenize LIWC-style (lowercase; keeps numbers by default)."""
    toks = _LIWC_WORD_RE.findall((text or "").lower())
    if not keep_numbers:
        toks = [t for t in toks if not t.replace(".", "", 1).isdigit()]
    return toks


def _liwc_analyze_docs(
    docs: List[str],
    *,
    liwc_dic: Union[str, Path],
    min_hits: int = 1,
    binary_edges: bool = True,
    keep_numbers: bool = True,
) -> Tuple[List[str], List[int], List[int], List[float], List[str], List[Dict[str, Any]]]:
    """
    Build doc→LIWC incidence and LIWC PMI input strings.

    Uses `liwc.load_token_parser()` to avoid category-id/name mismatches that occur
    with ad-hoc .dic parsing on some LIWC2015 dictionaries.
    """
    try:
        from liwc import load_token_parser
    except Exception as e:
        raise ImportError(
            "Missing dependency 'liwc'. Install in your env: pip install liwc"
        ) from e

    parse_token, category_names = load_token_parser(str(liwc_dic))

    # stable order for matrices
    liwc_id2_list = sorted(list(category_names))
    cat2col = {c: i for i, c in enumerate(liwc_id2_list)}

    liwc_docs_str: List[str] = []
    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    debug_rows: List[Dict[str, Any]] = []

    for i, raw in enumerate(docs):
        toks = _liwc_tokenize(raw or "", keep_numbers=keep_numbers)

        counts: Dict[str, int] = _dd(int)
        token_hits: Dict[str, List[str]] = _dd(list)

        for t in toks:
            cats = list(parse_token(t))
            for c in cats:
                counts[c] += 1
                token_hits[c].append(t)

        present = [c for c, k in counts.items() if k >= min_hits]
        present_names = sorted(set(present))
        liwc_docs_str.append(" ".join(present_names))

        if binary_edges:
            for c in present_names:
                rows.append(i)
                cols.append(cat2col[c])
                vals.append(1.0)
        else:
            for c, k in counts.items():
                if k <= 0:
                    continue
                rows.append(i)
                cols.append(cat2col[c])
                vals.append(float(k))

        top = sorted(((k, c) for c, k in counts.items()), reverse=True)[:25]
        debug_rows.append(
            {
                "doc_idx": i,
                "top_liwc": "; ".join(f"{c}:{k}" for k, c in top),
                "verb_tokens": " ".join(token_hits.get("verb", [])),
                "posemo_tokens": " ".join(token_hits.get("posemo", [])),
                "text": (raw or "")[:200].replace("\n", " "),
            }
        )

    return liwc_docs_str, rows, cols, vals, liwc_id2_list, debug_rows


def _build_liwc_word2cats(
    word_id2_list: List[str], *, liwc_dic: Union[str, Path]
) -> Dict[str, List[str]]:
    """Map each vocab word -> [LIWC categories] via `liwc.load_token_parser()`."""
    try:
        from liwc import load_token_parser
    except Exception as e:
        raise ImportError(
            "Missing dependency 'liwc'. Install in your env: pip install liwc"
        ) from e

    parse_token, _ = load_token_parser(str(liwc_dic))
    out: Dict[str, List[str]] = {}
    for w in word_id2_list:
        if not w:
            continue
        cats = list(parse_token(w.lower()))
        if not cats:
            continue
        # stable unique
        seen = set()
        ordered = []
        for c in cats:
            if c not in seen:
                seen.add(c)
                ordered.append(c)
        out[w] = ordered

    print(f"[liwc_word2cats] vocab words: {len(word_id2_list)}, overlap with LIWC: {len(out)}")
    return out


# ================== Empath ==================

def _empath_analyze_docs(docs: List[str], *, categories: Optional[List[str]], min_hits: int, binary_edges: bool):
    if Empath is None:
        raise ImportError("Empath is not installed. Please run: pip install empath")
    lex = Empath()
    all_cats = sorted(lex.cats.keys())
    if categories is None:
        use_cats = all_cats
    else:
        bad = [c for c in categories if c not in all_cats]
        if bad: raise ValueError(f"Unknown Empath categories: {bad[:8]}")
        use_cats = list(categories)
    cat2id = {c:i for i,c in enumerate(use_cats)}

    empath_docs_str: List[str] = []
    rows, cols, vals = [], [], []
    debug_rows: List[Dict[str, Any]] = []

    for i, raw in enumerate(docs):
        text = raw or ""
        scores = lex.analyze(text, categories=use_cats, normalize=False)
        present = [c for c in use_cats if int(scores.get(c,0)) >= min_hits]
        empath_docs_str.append(" ".join(present))
        if binary_edges:
            for c in present:
                rows.append(i); cols.append(cat2id[c]); vals.append(1.0)
        else:
            for c in use_cats:
                k = int(scores.get(c,0))
                if k>0:
                    rows.append(i); cols.append(cat2id[c]); vals.append(float(k))
                # collect ALL categories with count > 0 for debugging
        all_nonzero = [(int(scores.get(c, 0)), c) for c in use_cats if int(scores.get(c, 0)) > 0]
        all_nonzero.sort(reverse=True)  # sort by count desc

        debug_rows.append({
            "doc_idx": i,
            "top_empath": "; ".join(f"{c}:{k}" for k, c in all_nonzero),
            "text": text[:200].replace("\n"," ")
        })
    return empath_docs_str, rows, cols, vals, use_cats, debug_rows
# ============================================
# NEW: build word -> [empath categories] map
# ============================================
def _build_empath_word2cats_from_lexicon(word_id2_list):
    """
    Build a mapping word -> [empath_categories] using Empath's own lexicon.

    - word_id2_list: list of vocabulary words (index -> word), as dumped to word_id2_list.json
    - returns: dict {word (exact string from vocab) -> [category_name, ...]}
    """
    if Empath is None:
        raise ImportError("Empath is not installed. Please run: pip install empath")

    lex = Empath()

    # Step 1: build a lowercased word -> [cats] map from Empath's internal lexicon
    from collections import defaultdict
    word2cats_full = defaultdict(list)
    for cat, words in lex.cats.items():          # lex.cats: {category: [word1, word2, ...]}
        for w in words:
            wl = w.lower()
            if cat not in word2cats_full[wl]:
                word2cats_full[wl].append(cat)

    # Step 2: align with your vocab (word_id2_list)
    empath_word2cats = {}
    for w in word_id2_list:
        # skip empty / None vocab entries if any
        if not w:
            continue
        wl = w.lower()
        if wl in word2cats_full:
            empath_word2cats[w] = word2cats_full[wl]

    print(f"[empath_word2cats] vocab words: {len(word_id2_list)}, "
          f"overlap with Empath lexicon: {len(empath_word2cats)}")
    return empath_word2cats


# ================== Doc-BERT (RAW) ==================
def _build_doc_bert_embeddings(
    texts: List[str],
    out_path: Path,
    model_name: str = "bert-base-uncased",
    batch_size: int = 64,
    max_length: int = 256,
    device: str = "cpu",
    l2_normalize: bool = True,
    use_pooled_output: bool = True
) -> None:
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
    except Exception as e:
        print(f"[WARN] transformers not available ({e}); skipping Doc-BERT."); return
    if not texts:
        np.save(out_path/"doc_bert_emb.npy", np.zeros((0,0), np.float32)); return
    if device == "cuda":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(device); mdl.eval()
    vecs=[]
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch=texts[i:i+batch_size]
            enc = tok(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            enc = {k:v.to(device) for k,v in enc.items()}
            out = mdl(**enc)
            if use_pooled_output and getattr(out, "pooler_output", None) is not None:
                H = out.pooler_output
            else:
                H = out.last_hidden_state[:, 0, :]
            vecs.append(H.detach().cpu().numpy())
    X = np.vstack(vecs).astype(np.float32)
    if l2_normalize:
        n=np.linalg.norm(X, axis=1, keepdims=True); X = X/np.maximum(n,1e-8)
    np.save(out_path/"doc_bert_emb.npy", X)
    (out_path/"doc_bert_meta.json").write_text(
        json.dumps({"model":model_name,"rows":int(X.shape[0]),"dim":int(X.shape[1])}, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"[Doc-BERT] {X.shape} -> {out_path/'doc_bert_emb.npy'}")

# ================== Word-type BERT (cleaned, lowercased) ==================
def _build_wordtype_bert_embeddings(
    word_nodes: List[str],
    model_name: str = "bert-base-uncased",
    device: str = "cpu"
) -> np.ndarray:
    """Average input-embedding rows for each word type's WordPiece tokens (cleaned, lowercased)."""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
    except Exception as e:
        print(f"[WARN] transformers not available ({e}); skipping word-type BERT.")
        return np.zeros((len(word_nodes), 0), dtype=np.float32)

    if device == "cuda":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name)  # CPU ok; we only read embeddings
    emb_table: "torch.nn.Embedding" = mdl.embeddings.word_embeddings

    dim = emb_table.weight.shape[1]
    W = np.zeros((len(word_nodes), dim), dtype=np.float32)
    unk_id = tok.unk_token_id

    with torch.no_grad():
        for i, w in enumerate(word_nodes):
            ids = tok.encode(w, add_special_tokens=False)
            if not ids: ids = [unk_id]
            vecs = emb_table.weight[ids, :]
            W[i] = vecs.mean(dim=0).cpu().numpy().astype(np.float32)

    n = np.linalg.norm(W, axis=1, keepdims=True)
    W = W / np.maximum(n, 1e-8)
    print(f"[Word-BERT] word_type_emb: {W.shape} from {model_name}")
    return W

# ================== Cosine top-k adjacency (generic, for At/Ae) ==================
def _build_cosine_topk_adj(X: np.ndarray, topk: int = 50, min_sim: float = 0.10, self_loop: bool = True) -> coo_matrix:
    if not isinstance(X, np.ndarray) or X.size == 0:
        return coo_matrix((0,0), dtype=np.float32)
    X = X.astype(np.float32)
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.maximum(nrm, 1e-8)
    n = X.shape[0]

    rows, cols, vals = [], [], []
    block = 8192
    for a in range(0, n, block):
        b = min(a + block, n)
        S = X[a:b] @ X.T
        for i in range(b - a):
            S[i, a + i] = -1.0
        k = min(topk, n - 1)
        if k <= 0: continue
        idx = np.argpartition(S, -k, axis=1)[:, -k:]
        row_idx = np.arange(S.shape[0])[:, None]
        top_scores = S[row_idx, idx]
        mask = top_scores >= min_sim
        sel_rows = np.repeat(np.arange(a, b), mask.sum(axis=1))
        sel_cols = idx[mask]
        sel_vals = top_scores[mask]
        rows.extend(sel_rows.tolist())
        cols.extend(sel_cols.tolist())
        vals.extend(sel_vals.astype(np.float32).tolist())

    if self_loop:
        rows.extend(range(n)); cols.extend(range(n)); vals.extend([1.0]*n)

    M = coo_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32)
    return M.maximum(M.transpose()).tocoo()

def _build_entity_adj_from_embeddings(E_local: np.ndarray, topk: int=50, min_sim: float=0.10, self_loop: bool=True)->coo_matrix:
    return _build_cosine_topk_adj(E_local, topk=topk, min_sim=min_sim, self_loop=self_loop)

# ================== MAIN ==================
def build_personality_graphs(
    *,
    dataset: str,
    train_csv: Optional[Union[str, Path]] = None,
    val_csv:  Optional[Union[str, Path]] = None,
    test_csv:  Optional[Union[str, Path]] = None,
    text_col:  str = "text",
    el_text_col: Optional[str] = None,
    pos_tagged_col: Optional[str] = None,
    split_json: Optional[Union[str, Path]] = None,
    out_dir: Optional[Union[str, Path]] = None,
    # NELL / TransE
    nell_ent2ids: Union[str, Path] = None,
    transe_vec:   Optional[Union[str, Path]] = None,
    # options
    pos_backend: str = "nltk",
    auto_download_nltk: bool = False,
    nltk_data_dir: Optional[Union[str, Path]] = None,
    window_size: int = 5,
    remove_lowfreq: bool = False,
    min_count: int = 5,
    # NEL scoring / acceptance
    use_minilm: bool = True,
    st_device: str = ST_DEVICE,
    st_batch_size: int = ST_BATCH_SIZE,
    cache_root: Optional[Union[str, Path]] = None,
    require_capitalized: bool = False,
    max_candidates: int = 20,
    min_ctx_cos: float = 0.12,
    min_margin: float  = 0.06,
    allowlist_concepts: Optional[List[str]] = None,
    # LIWC
    use_liwc: bool = False,
    liwc_dic: Optional[Union[str, Path]] = None,
    liwc_min_hits: int = 1,
    liwc_binary_edges: bool = True,
    liwc_keep_numbers: bool = True,
    liwc_dic_encoding: str = "utf-8",
    # Empath
    use_empath: bool = False,
    empath_categories_json: Optional[Union[str, Path]] = None,
    empath_min_hits: int = 1,
    empath_binary_edges: bool = True,
    # Doc-BERT (RAW)
    build_doc_bert: bool = True,
    doc_bert_model: str = "bert-base-uncased",
    doc_bert_batch: int = 64,
    doc_bert_maxlen: int = 256,
    doc_bert_device: str = "cpu",
    doc_bert_use_pooled: bool = True,
    # Ae knobs
    ee_build: bool = True,
    ee_topk: int = 50,
    ee_min_sim: float = 0.10,
    ee_self_loop: bool = True,
    # Word-type BERT knobs
    build_word_bert: bool = True,
    word_bert_model: str = "bert-base-uncased",
    word_bert_device: str = "cpu",
    # At (text–text) knobs
    build_text_adj: bool = True,
    at_topk: int = 50,
    at_min_sim: float = 0.10,
    at_self_loop: bool = True,
    # CSV
    csv_chunksize: int = 100_000,
    max_rows: Optional[int] = None,
    csv_encoding: str = "utf-8",
    csv_on_bad_lines: str = "skip",
) -> str:
    print(">>> ENTER build_personality_graphs (Word/POS/Empath/Entity + Doc-BERT + Ae + WordBERT + At)")
    _ensure_nltk(pos_backend, auto_download_nltk, nltk_data_dir)

    # ---- read ----
    if split_json:
        obj=json.loads(Path(split_json).read_text(encoding="utf-8"))
        train=[{"text":v.get("text",""), "el_text": v.get("el_text", v.get("text","")), "pos": v.get("pos","")} for _,v in obj.get("train",{}).items()]
        val=[{"text":v.get("text",""), "el_text": v.get("el_text", v.get("text","")), "pos": v.get("pos","")} for _,v in obj.get("val",{}).items()]
        test =[{"text":v.get("text",""), "el_text": v.get("el_text", v.get("text","")), "pos": v.get("pos","")} for _,v in obj.get("test",{}).items()]
    else:
        if not (train_csv and val_csv and test_csv): raise ValueError("Provide split_json OR (train_csv & val_csv & test_csv).")
        try:
            train=_read_texts_pandas(train_csv, text_col, el_text_col, pos_tagged_col, csv_chunksize, max_rows, csv_encoding, csv_on_bad_lines)
            val=_read_texts_pandas(val_csv, text_col, el_text_col, pos_tagged_col, csv_chunksize, max_rows, csv_encoding, csv_on_bad_lines)
            test =_read_texts_pandas(test_csv,  text_col, el_text_col, pos_tagged_col, csv_chunksize, max_rows, csv_encoding, csv_on_bad_lines)
        except Exception as e:
            print(f"[WARN] pandas read_csv failed ({e}); fallback csv module.")
            train=_read_texts_csvfallback(train_csv, text_col, el_text_col, pos_tagged_col, max_rows, csv_encoding)
            val=_read_texts_csvfallback(val_csv, text_col, el_text_col, pos_tagged_col, max_rows, csv_encoding)
            test =_read_texts_csvfallback(test_csv,  text_col, el_text_col, pos_tagged_col, max_rows, csv_encoding)

    out_path=Path(out_dir) if out_dir else Path(f"./{dataset}_data")
    out_path.mkdir(parents=True, exist_ok=True)
    _debug_write(out_path,"00_inputs.txt", f"train={len(train)} test={len(test)}")


    
    
    # ---- KB, linker, gloss cache ----
    surf2gids=_build_surf2gids_from_raw(str(nell_ent2ids), add_space_alias=True)
    gid2cpt  =_build_gid2concept(str(nell_ent2ids))
    gid2desc =_build_gid2desc(str(nell_ent2ids))
    linker=_Linker(surf2gids)

    emb_mmap=None; gid2row={}
    if use_minilm:
        cache_root = Path(cache_root) if cache_root else (out_path/"_minilm_cache")
        emb_mmap, _, gid2row, _ = precompute_gid2desc_embeddings(
            gid2desc, batch_size=st_batch_size, device=st_device,
            model_name=ST_MODEL_NAME, raw_ent2id_path=str(nell_ent2ids),
            cache_root=cache_root
        )

    # TransE matrix (for comma seed + Xe)
    E = None
    if transe_vec:
        try:
            ent_count = (max(g for L in surf2gids.values() for g in L)+1) if surf2gids else None
            E=_load_transe_auto(str(transe_vec), ent_count=ent_count)
            E=_normalize_rows(E)
        except Exception as e:
            print(f"[WARN] TransE load failed ({e}); disabling comma seeding.")
            E=None

    allow_set=set(a.lower() for a in allowlist_concepts) if allowlist_concepts else None

    # ---- accumulators ----
    tag_set, word_set = set(), set()
    tag_docs: List[str]=[]; word_docs: List[str]=[]
    query_nodes: List[str]=[]               # cleaned doc strings for word graph
    raw_docs: List[str] = []                # RAW for Doc-BERT & NEL

    # NEW (additive only): per-doc ordered tokens after stopword removal (keeps order + duplicates)
    doc_word_seq_tokens: List[List[str]] = []

    entity_nodes: List[int]=[]
    global2local: Dict[int,int] = {}
    rows_idx: List[int]=[]; cols_idx: List[int]=[]
    mentions_rows: List[Dict[str,str]]=[]
    # NEW: collect sparse word-token ↔ entity_lid links during processing.
    # We'll convert tokens -> word_id after word2id is built.
    word_ent_raw: List[Tuple[int, str, int]] = []  # (doc_idx, word_token, entity_lid)

    word_pos_raw: List[Tuple[int, str, str]] = []  # (doc_idx, word_token, pos_tag)
    # ---- scoring & acceptance ----
    def _score_disambig(s: str, cands: List[int]) -> Tuple[int, float, float, np.ndarray]:
        if not cands: return -1, 0.0, 0.0, np.zeros((0,),dtype=np.float32)
        if use_minilm:
            scores = score_with_cached_matrix(s, cands, emb_mmap, gid2row)
        else:
            scores = np.zeros((len(cands),), dtype=np.float32)
        if scores.size==0:
            return -1, 0.0, 0.0, scores
        k=int(np.argmax(scores)); gid=int(cands[k]); top=float(scores[k])
        margin = float((np.sort(scores)[-1]-np.sort(scores)[-2])) if len(scores)>=2 else 1.0
        return gid, top, margin, scores

    def _accept(gid:int, ctx_cos:float, margin:float, used_ctx:bool)->Tuple[bool,str]:
        if gid<0: return False, "no_gid"
        if allow_set is not None:
            cpt = gid2cpt.get(gid,"unknown").lower()
            if cpt not in allow_set: return False, f"concept_not_allowed:{cpt}"
        if used_ctx:
            if ctx_cos < min_ctx_cos: return False, f"low_ctx_cos:{ctx_cos:.3f}"
            if margin  < min_margin:  return False, f"small_margin:{margin:.3f}"
        return True, "minilm_ctx" if used_ctx else "noctx"

    # ---- per-doc processing ----
    def _proc_doc(raw: str, doc_idx: int, pretagged: str = "", raw_for_el: str = ""):
        raw_docs.append(raw or "")
        # WORD graph: cleaned lower
        q_clean = _clean(raw)
        # POS: prefer pre-tagged column (e.g., 'i/prp was/vbd ...'); fallback to NLTK tagging.
        toks_pos: List[str] = []
        tags: List[str] = []
        if pretagged:
            pairs = parse_token_tag_string(pretagged)
            toks_pos = [t for t, _ in pairs]
            tags = [p for _, p in pairs]
        else:
            raw_pos = _normalize_spaces(raw)
            if pos_backend == "nltk":
                toks_pos = word_tokenize(raw_pos)
                tags = [p.lower() for _, p in pos_tag(toks_pos)]
            else:
                tags = []

        toks_word = word_tokenize(q_clean)
        words = [w for w in toks_word if w]

        doc_word_set = set(words)

        # NEW (additive only): store per-doc ordered words (post-stopwords), preserving duplicates
        doc_word_seq_tokens.append(list(words))

        # NEW: collect sparse word↔POS links (doc_idx, cleaned_word_token, pos_tag).
        # We DO NOT re-tag after stopword removal (that would lose context).
        # Instead we map raw tokens → cleaned tokens and keep only those surviving into doc_word_set.
        if pos_backend == "nltk" and tags:
            for tok_raw, tag_raw in zip(toks_pos, tags):
                tok_clean = _clean(tok_raw)
                if not tok_clean:
                    continue
                for t in tok_clean.split():
                    if t and t in doc_word_set:
                        word_pos_raw.append((doc_idx, t, tag_raw))
        tag_set.update(tags); word_set.update(words)
        tag_docs.append(" ".join(tags))
        word_docs.append(" ".join(words))
        query_nodes.append(q_clean)

        # mentions on RAW
        mlist=linker.link(raw_for_el or raw)
        if not mlist: return

        # prune
        pruned=[]
        for m in mlist:
            if len(m["matched_text"]) < 3: continue
            if len(m["candidates"]) > max_candidates: continue
            if require_capitalized and " " not in m["matched_text"].strip():
                if not raw[m["start"]:m["end"]][:1].isupper(): continue
            pruned.append(m)
        if not pruned: return

        # comma seeds via TransE (optional)
        seeds=_seed_comma_pairs(pruned, raw, E) if E is not None else [-1]*len(pruned)

        # decide
        for j,m in enumerate(pruned):
            cands=list(m["candidates"])
            if not cands:
                mentions_rows.append({"doc_idx":str(doc_idx),"start":str(m["start"]),"end":str(m["end"]),
                                      "matched_text":m["matched_text"],"surface_norm":m["surface_norm"],
                                      "chosen_gid":"","candidates":"", "via":"no_candidates",
                                      "ctx_cos":"0.000","margin":"0.000","text":raw})
                continue
            if seeds[j] != -1:
                gid=int(seeds[j]); ctx_cos=1.0; margin=1.0; used_ctx=True
            else:
                gid, ctx_cos, margin, _ = _score_disambig(raw, cands)
                used_ctx = use_minilm
            ok, reason = _accept(gid, ctx_cos, margin, used_ctx)
            if ok:
                lid=global2local.setdefault(gid, len(entity_nodes))
                if lid==len(entity_nodes): entity_nodes.append(gid)
                rows_idx.append(doc_idx); cols_idx.append(lid)
                # NEW: collect sparse word↔entity links from the mention surface text.
                # This does NOT change any existing outputs; it only enables dumping an extra CSV later.
                mention_text = m.get("surface_norm") or m.get("matched_text") or ""
                mention_clean = _clean(mention_text)
                mention_toks = [t for t in word_tokenize(mention_clean) if t and t in doc_word_set]
                for t in mention_toks:
                    word_ent_raw.append((doc_idx, t, int(lid)))
                mentions_rows.append({"doc_idx":str(doc_idx),"start":str(m["start"]),"end":str(m["end"]),
                                      "matched_text":m["matched_text"],"surface_norm":m["surface_norm"],
                                      "chosen_gid":str(gid),"candidates":";".join(str(x) for x in cands),
                                      "via":reason,"ctx_cos":f"{ctx_cos:.3f}","margin":f"{margin:.3f}","text":raw})
            else:
                mentions_rows.append({"doc_idx":str(doc_idx),"start":str(m["start"]),"end":str(m["end"]),
                                      "matched_text":m["matched_text"],"surface_norm":m["surface_norm"],
                                      "chosen_gid":"","candidates":";".join(str(x) for x in cands),
                                      "via":reason,"ctx_cos":f"{ctx_cos:.3f}","margin":f"{margin:.3f}","text":raw})

    # run
    all_rows = (train + val + test)
    all_docs = [d.get("text", "") for d in all_rows]
    for i, row in enumerate(all_rows):
        if (i % 5000)==0 and i>0: print(f"[PROC] {i}/{len(all_rows)}")
        _proc_doc(row.get("text", ""), i, row.get("pos", "") if pos_tagged_col else "", row.get("el_text", row.get("text","")))

    # ---- vocab & matrices ----
    tag_nodes=sorted(tag_set); word_nodes=sorted(word_set)
    tag2id={t:i for i,t in enumerate(tag_nodes)}
    word2id={w:i for i,w in enumerate(word_nodes)}

    # NEW (additive only): dump per-doc ordered word-id sequence (post-stopwords)
    # This does NOT change any existing outputs; it only adds doc_word_seq.jsonl.
    seq_path = out_path / "doc_word_seq.jsonl"
    with open(seq_path, "w", encoding="utf-8") as f:
        for d, toks in enumerate(doc_word_seq_tokens):
            ids = [int(word2id[w]) for w in toks if w in word2id]
            f.write(json.dumps({"doc_idx": int(d), "word_ids": ids}, ensure_ascii=False) + "\n")
    print(f"[OUT] doc_word_seq.jsonl: {len(doc_word_seq_tokens):,} docs -> {seq_path}")

    # NEW: dump word↔entity edges as (doc_idx, word_id, entity_lid) for later local-graph building.
    # This is an extra artifact only; it does NOT change the existing pipeline outputs.
    import csv
    seen = set()
    edges = []
    for d, wtok, elid in word_ent_raw:
        wid = word2id.get(wtok)
        if wid is None:
            continue
        key = (int(d), int(wid), int(elid))
        if key in seen:
            continue
        seen.add(key)
        edges.append(key)

    with open(out_path / "word_entity_edges.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["doc_idx", "word_id", "entity_lid"])
        w.writerows(edges)
    print(f"[OUT] word_entity_edges.csv: {len(edges):,} edges -> {out_path/'word_entity_edges.csv'}")


    # NEW: dump word↔POS edges as (doc_idx, word_id, pos_id) for later local-graph building.
    # This is an extra artifact only; it does NOT change the existing pipeline outputs.
    seen_wp = set()
    wp_edges = []
    for d, wtok, ptag in word_pos_raw:
        wid = word2id.get(wtok)
        pid = tag2id.get(ptag)
        if wid is None or pid is None:
            continue
        key = (int(d), int(wid), int(pid))
        if key in seen_wp:
            continue
        seen_wp.add(key)
        wp_edges.append(key)

    with open(out_path / "word_pos_edges.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["doc_idx", "word_id", "pos_id"])
        w.writerows(wp_edges)
    print(f"[OUT] word_pos_edges.csv: {len(wp_edges):,} edges -> {out_path/'word_pos_edges.csv'}")
    adj_query2tag  = _incidence_matrix(tag_docs,  tag2id, sparse=True) if pos_backend=="nltk" else coo_matrix((len(all_docs),0))
    adj_tag        = _PMI(tag_docs,  tag2id, window_size=window_size, sparse=True) if pos_backend=="nltk" else coo_matrix((0,0))
    adj_query2word = _incidence_matrix(word_docs, word2id, sparse=True)
    adj_word       = _PMI(word_docs, word2id, window_size=window_size, sparse=True)

    # doc–entity incidence
    rows=np.array(rows_idx, dtype=int); cols=np.array(cols_idx, dtype=int); data=np.ones(len(rows_idx), dtype=np.float32)
    adj_query2entity = coo_matrix((data,(rows,cols)), shape=(len(all_docs), len(entity_nodes)))

    # TransE slice (Xe)
    ent_emb = np.zeros((0, 0), dtype=np.float32)
    if entity_nodes and transe_vec:
        try:
            full = _load_transe_auto(str(transe_vec), ent_count=max(entity_nodes)+1)
            full = _normalize_rows(full)
            ent_emb = full[np.array(entity_nodes, dtype=int)]
        except Exception as e:
            print(f"[WARN] Could not slice TransE for local entities ({e}); writing empty emb.")

    # Ae from Xe
    adj_entity = coo_matrix((0,0), dtype=np.float32)
    if ent_emb.size > 0 and ee_build:
        print(f"[Ae] building (n={ent_emb.shape[0]}, d={ent_emb.shape[1]}) topk={ee_topk}, min_sim={ee_min_sim}")
        adj_entity = _build_entity_adj_from_embeddings(ent_emb, topk=ee_topk, min_sim=ee_min_sim, self_loop=ee_self_loop)
    else:
        print("[Ae] skipped (no entity emb or ee_build=False)")

    # ---- LIWC ----
    adj_query2liwc = coo_matrix((len(all_docs), 0))
    adj_liwc       = coo_matrix((0, 0))
    liwc_id2_list: List[str] = []
    liwc_debug_rows: List[Dict[str, Any]] = []

    if use_liwc:
        if not liwc_dic:
            raise ValueError("use_liwc=True but liwc_dic is not provided.")
        (liwc_docs_str, r, c, v, liwc_id2_list, liwc_debug_rows) = _liwc_analyze_docs(
            all_docs,
            liwc_dic=liwc_dic,
            min_hits=liwc_min_hits,
            binary_edges=liwc_binary_edges,
            keep_numbers=liwc_keep_numbers,
        )
        if liwc_id2_list:
            adj_query2liwc = coo_matrix(
                (np.array(v, np.float32), (np.array(r), np.array(c))),
                shape=(len(all_docs), len(liwc_id2_list)),
            )
            liwc2id = {cat: i for i, cat in enumerate(liwc_id2_list)}
            adj_liwc = _PMI(liwc_docs_str, liwc2id, window_size=window_size, sparse=True)

    # ---- Empath ----
    adj_query2empath = coo_matrix((len(all_docs),0))
    adj_empath       = coo_matrix((0,0))
    empath_id2_list: List[str] = []
    empath_debug_rows: List[Dict[str, Any]] = []
    if use_empath:
        cat_list=None
        if empath_categories_json:
            obj=json.loads(Path(empath_categories_json).read_text(encoding="utf-8"))
            if isinstance(obj, dict) and "categories" in obj: obj = obj["categories"]
            if not isinstance(obj, list): raise ValueError("empath_categories_json must be a list or {'categories':[...]} dict")
            cat_list=[str(x) for x in obj]
        (empath_docs_str, r, c, v, empath_id2_list, empath_debug_rows) = _empath_analyze_docs(
            all_docs, categories=cat_list, min_hits=empath_min_hits, binary_edges=empath_binary_edges
        )
        if empath_id2_list:
            adj_query2empath = coo_matrix((np.array(v,np.float32),(np.array(r),np.array(c))),
                                          shape=(len(all_docs), len(empath_id2_list)))
            empath2id={cat:i for i,cat in enumerate(empath_id2_list)}
            # PMI over doc-level empath strings (binary presence); window = |cats| to capture co-doc co-occur
            adj_empath = _PMI(empath_docs_str, empath2id, window_size=len(empath2id) or 1, sparse=True)

    # ---- Doc-BERT on RAW (for At by default) ----
    if build_doc_bert:
        _build_doc_bert_embeddings(
            texts=all_docs, out_path=out_path,
            model_name=doc_bert_model, batch_size=doc_bert_batch,
            max_length=doc_bert_maxlen, device=doc_bert_device,
            l2_normalize=True, use_pooled_output=doc_bert_use_pooled
        )

    # ---- Word-type BERT embeddings (clean, lowercased) ----
    word_type_emb = np.zeros((len(word_nodes), 0), dtype=np.float32)
    if build_word_bert and word_nodes:
        word_type_emb = _build_wordtype_bert_embeddings(
            word_nodes, model_name=word_bert_model, device=word_bert_device
        )

    # ---- At (text–text cosine) ----
    adj_text = coo_matrix((len(all_docs), len(all_docs)), dtype=np.float32)
    if build_text_adj and len(all_docs) > 0:
        X_doc = None
        doc_bert_path = out_path / "doc_bert_emb.npy"
        if doc_bert_path.exists():
            X_doc = np.load(doc_bert_path)
            print(f"[At] using Doc-BERT embeddings: {X_doc.shape}")
        else:
            if word_type_emb.size == 0:
                print("[At] no Doc-BERT and no word-type BERT → skipping At.")
                X_doc = None
            else:
                print("[At] building doc vecs from word-type BERT …")
                dim = word_type_emb.shape[1]
                X_doc = np.zeros((len(all_docs), dim), dtype=np.float32)
                for i, doc in enumerate(word_docs):
                    ids = [word2id.get(w) for w in doc.split() if w in word2id]
                    if not ids: continue
                    X_doc[i] = word_type_emb[np.array(ids)].mean(axis=0)
                nrm = np.linalg.norm(X_doc, axis=1, keepdims=True)
                X_doc = X_doc / np.maximum(nrm, 1e-8)
        if X_doc is not None and X_doc.size > 0:
            adj_text = _build_cosine_topk_adj(X_doc, topk=at_topk, min_sim=at_min_sim, self_loop=at_self_loop)
        else:
            adj_text = coo_matrix((len(all_docs), len(all_docs)), dtype=np.float32)
    else:
        print("[At] disabled.")

    # ---- ONE-HOT FEATURES (NEW) ----
    # POS one-hot (identity |POS|). Why: paper specifies one-hot embeddings for POS.
    pos_onehot = speye(len(tag_nodes), dtype=np.float32, format="coo") if len(tag_nodes) > 0 else coo_matrix((0,0), dtype=np.float32)
    # LIWC one-hot (identity |LIWC|) if LIWC is enabled and categories exist.
    liwc_onehot = coo_matrix((0,0), dtype=np.float32)
    if use_liwc and len(liwc_id2_list) > 0:
        liwc_onehot = speye(len(liwc_id2_list), dtype=np.float32, format="coo")

    # Empath one-hot (identity |Empath|) if Empath is enabled and categories exist.
    empath_onehot = coo_matrix((0,0), dtype=np.float32)
    if use_empath and len(empath_id2_list) > 0:
        empath_onehot = speye(len(empath_id2_list), dtype=np.float32, format="coo")

    # ---- write ----
    def _dump(obj,name): pkl.dump(obj, open(out_path/name,"wb"))
    def _dump_json(obj,name): (out_path/name).write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")

    _dump(adj_query2word,"adj_query2word.pkl"); _dump(adj_word,"adj_word.pkl")
    _dump(adj_query2tag, "adj_query2tag.pkl");  _dump(adj_tag,"adj_tag.pkl")
    _dump(adj_query2entity,"adj_query2entity.pkl")
    _dump(ent_emb,"entity_emb.pkl")
    _dump(_build_entity_adj_from_embeddings(ent_emb, ee_topk, ee_min_sim, ee_self_loop) if (ent_emb.size>0 and ee_build) else coo_matrix((0,0)), "adj_entity.pkl")
    _dump(adj_text, "adj_text.pkl")  # At
    if word_type_emb.size>0:
        _dump(word_type_emb, "word_type_bert_emb.pkl")

        # NEW: dump POS & Empath one-hot feature matrices
    _dump(pos_onehot, "pos_onehot.pkl")
    if use_liwc:
        _dump(liwc_onehot, "liwc_onehot.pkl")

    if use_empath:
        _dump(empath_onehot, "empath_onehot.pkl")

    # ID2 lists
    query_id2_list  = [_clean(d["text"]) for d in (train + val + test)]
    word_id2_list   = sorted(word_set)
    tag_id2_list    = sorted(tag_set)
    entity_id2_list = [str(g) for g in entity_nodes]

    _dump_json(query_id2_list,  "query_id2_list.json")
    _dump_json(word_id2_list,   "word_id2_list.json")
    _dump_json(tag_id2_list,    "tag_id2_list.json")
    _dump_json(entity_id2_list, "entity_id2_list.json")
    _dump_json(list(range(len(train))), "train_idx.json")
    _dump_json(list(range(len(train), len(train) + len(val))), "val_idx.json")
    _dump_json(list(range(len(train) + len(val), len(train) + len(val) + len(test))), "test_idx.json")



    # NEW: word -> [LIWC categories] mapping (for local graph later)
    if use_liwc:
        liwc_word2cats = _build_liwc_word2cats(word_id2_list, liwc_dic=liwc_dic)
        _dump_json(liwc_word2cats, "liwc_word2cats.json")

    # NEW: word -> [Empath categories] mapping (for local graph later)
    if use_empath:
        empath_word2cats = _build_empath_word2cats_from_lexicon(word_id2_list)
        _dump_json(empath_word2cats, "empath_word2cats.json")


    # mentions audit & empath debug
    import csv
    with open(out_path/"entities_mentions.csv","w",encoding="utf-8",newline="") as f:
        H=["doc_idx","start","end","matched_text","surface_norm","chosen_gid","candidates","via","ctx_cos","margin","text"]
        w=csv.DictWriter(f, fieldnames=H); w.writeheader()
        for r in mentions_rows: w.writerow(r)


    if use_liwc:
        _dump(adj_query2liwc, "adj_query2liwc.pkl")
        _dump(adj_liwc,       "adj_liwc.pkl")
        _dump_json(liwc_id2_list, "liwc_id2_list.json")
        with open(out_path/"liwc_doc_scores.csv","w",encoding="utf-8",newline="") as f:
            w = csv.DictWriter(f, fieldnames=["doc_idx", "top_liwc", "verb_tokens", "posemo_tokens", "text"])
            w.writeheader()
            for row in liwc_debug_rows: w.writerow(row)

    if use_empath:
        _dump(adj_query2empath, "adj_query2empath.pkl")
        _dump(adj_empath,       "adj_empath.pkl")
        _dump_json(empath_id2_list, "empath_id2_list.json")
        with open(out_path/"empath_doc_scores.csv","w",encoding="utf-8",newline="") as f:
            w=csv.DictWriter(f, fieldnames=["doc_idx","top_empath","text"])
            w.writeheader()
            for row in empath_debug_rows: w.writerow(row)

    # tiny previews
    try:
        ent_by_doc = defaultdict(list)
        for r in mentions_rows:
            if r.get("chosen_gid"):
                d = int(r["doc_idx"])
                ent_str = f'{r["matched_text"]}(gid={r["chosen_gid"]},via={r["via"]})'
                ent_by_doc[d].append(ent_str)
        N = min(10, len(all_docs)); lines=[]
        for i in range(N):
            raw  = all_docs[i] or ""
            clean= query_nodes[i] if i < len(query_nodes) else ""
            words= word_docs[i] if i < len(word_docs) else ""
            tags = tag_docs[i] if i < len(tag_docs) else ""
            ents = "; ".join(ent_by_doc.get(i, [])) or "(none)"
            lines += [f"[doc {i}] RAW: {raw}",
                      f"         CLEAN: {clean}",
                      f"         WORDS: {words}",
                      f"         POS:   {tags}",
                      f"         ENTS:  {ents}",
                      ""]
        _debug_write(out_path, "01_views_sample.txt", "\n".join(lines))
    except Exception as e:
        print(f"[WARN] debug sample failed: {e}")

    print(f"[OK] docs={len(all_docs)} words={len(word_set)} tags={len(tag_set)} entities={len(entity_nodes)} "
          f"LIWC={'on' if use_liwc else 'off'} Empath={'on' if use_empath else 'off'} Ae={'on' if ent_emb.size>0 and ee_build else 'off'} "
          f"WordBERT={'on' if word_type_emb.size>0 else 'off'} At={'on' if adj_text.shape[0]>0 else 'off'}")
    print(f"[OUT] {out_path}")
    return str(out_path.resolve())
