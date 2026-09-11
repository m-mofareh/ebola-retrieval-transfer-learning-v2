"""
Two-Stage SWAT Transfer Learning for Ebola pAffinity Prediction
Models: DeepDTA (CNN-based) and GraphDTA (GCN-based)

Usage:
  # Auto Stage 1 + 2 (recommended)
  python deepdta_graphdta_protien4.py \
    --model deepdta --stage 12 --seed 42 \
    --output_dir swat_results_deepdta \
    --bindingdb_samples 300000

  # Multi-seed sweep
  bash run_swat.sh
"""

import argparse
import gc
import time
import os
import random
import zipfile
import sys
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# Required by CUDA/cuBLAS for deterministic matrix operations.
# Must be set before CUDA work begins; child Stage-1/Stage-2 processes inherit it.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold


# ============================================================================
# CONSTANTS
# ============================================================================
import warnings

warnings.filterwarnings(
    "ignore",
    message="adaptive_max_pool2d_backward_cuda does not have a deterministic implementation"
)
BASELINE_RMSE = 0.5513

SMILES_CHARS  = list("#%()+-./0123456789=@ABCFGHIKLMNOPRSTUVY[\\]abcdefghiklmnoprstuy")
PROTEIN_CHARS = list("ACDEFGHIKLMNPQRSTVWY")

SMILES_VOCAB  = {c: i+1 for i, c in enumerate(SMILES_CHARS)}
PROTEIN_VOCAB = {c: i+1 for i, c in enumerate(PROTEIN_CHARS)}

SMILES_MAX_LEN  = 150
PROTEIN_MAX_LEN = 1000


# ============================================================================
# UTILS
# ============================================================================
def set_seed(seed):
    import random
    import os

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    # Important:
    # Some CUDA operations used by DeepDTA do not have a fully
    # deterministic implementation. Warn instead of crashing.
    torch.use_deterministic_algorithms(True, warn_only=True)


def calc_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    pearson = float(pearsonr(y_true, y_pred)[0]) \
        if np.std(y_true) > 0 and np.std(y_pred) > 0 else np.nan
    return {
        "RMSE":    float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE":     float(mean_absolute_error(y_true, y_pred)),
        "R2":      float(r2_score(y_true, y_pred)),
        "Pearson": pearson,
    }


def canonicalize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) if mol else None
    except Exception:
        return None


def bemis_murcko_scaffold(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return f"INVALID::{smiles}"
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=True)
        return scaf if scaf else f"ACYCLIC::{smiles}"
    except Exception:
        return f"INVALID::{smiles}"


# ============================================================================
# SCAFFOLD SPLIT — identical to ML baselines and single-stage scripts
# ============================================================================

def make_grouped_split(df, seed, mode="scaffold"):
    """
    Leakage-resistant 70/15/15 scaffold split.
    Identical logic and seeding formula to:
      - ebola_ml_baselines.py
      - single_stage_deepdta_graphdta.py
    So all three pipelines share the same train/val/test rows at each seed.
    """
    if mode == "scaffold":
        groups = df["SMILES"].map(bemis_murcko_scaffold).to_numpy()
    elif mode == "cold_drug":
        groups = df["SMILES"].to_numpy()
    else:
        raise ValueError("mode must be 'scaffold' or 'cold_drug'")

    idx = np.arange(len(df))
    outer = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    trainval_idx, test_idx = next(outer.split(idx, groups=groups))

    inner_groups = groups[trainval_idx]
    inner = GroupShuffleSplit(
        n_splits=1, test_size=0.15 / 0.85, random_state=seed + 1000)
    train_rel, val_rel = next(inner.split(trainval_idx, groups=inner_groups))
    train_idx = trainval_idx[train_rel]
    val_idx   = trainval_idx[val_rel]

    def aset(v): return set(map(str, v))
    for name, a, b in [("train/val",  train_idx, val_idx),
                       ("train/test", train_idx, test_idx),
                       ("val/test",   val_idx,   test_idx)]:
        assert aset(groups[a]).isdisjoint(aset(groups[b])), \
            f"Scaffold group overlap: {name}"
        assert aset(df.iloc[a]["SMILES"]).isdisjoint(
            aset(df.iloc[b]["SMILES"])), f"Compound overlap: {name}"
        pa = set(zip(df.iloc[a]["SMILES"], df.iloc[a]["Sequence"]))
        pb = set(zip(df.iloc[b]["SMILES"], df.iloc[b]["Sequence"]))
        assert pa.isdisjoint(pb), f"Pair overlap: {name}"

    return train_idx, val_idx, test_idx, groups


# ============================================================================
# DATA LOADING — BINDINGDB (IC50 + Ki + Kd as pAffinity)
# ============================================================================

def load_bindingdb(out_dir="bindingdb_data", max_rows=100_000, seed=3):
    """
    Load BindingDB with IC50 + Ki + Kd pooled as pAffinity.

    WHY: Original code loaded IC50 only (mean pAffinity ~6.834).
    Ebola dataset pools IC50+EC50+Ki+Kd (mean pAffinity ~5.825).
    The 1.0 unit gap forced Stage 2 calibration to do heavy lifting.

    FIX: Load IC50+Ki+Kd from BindingDB too, so both databases use
    the same pAffinity formula:
        pAffinity = -log10(value_nM * 1e-9) = 9 - log10(value_nM)

    This reduces the domain gap and makes Stage 1 training more
    consistent with the Ebola target distribution.
    """
    out = Path(out_dir)
    tsv_files = sorted(out.glob("*.tsv"))
    if not tsv_files:
        raise FileNotFoundError(f"No TSV in {out}/")

    print(f"Loading BindingDB (max={max_rows}, assay types: IC50+Ki+Kd) ...")

    reader = pd.read_csv(tsv_files[0], sep="\t", dtype=str,
                         low_memory=False, chunksize=50_000)

    ligand_col = seq_col = None
    ic50_col = ki_col = kd_col = ec50_col = None
    assay_cols = {}
    chunks, total = [], 0

    for i, chunk in enumerate(reader):
        # Detect columns on first chunk
        if i == 0:
            for col in chunk.columns:
                cu = col.upper().strip()
                if "SMILES" in cu and not ligand_col:
                    ligand_col = col
                if "SEQUENCE" in cu and not seq_col:
                    seq_col = col
                if "IC50" in cu and "NM" in cu and not ic50_col:
                    ic50_col = col
                if "KI" in cu and "NM" in cu and "IC50" not in cu and not ki_col:
                    ki_col = col
                if "KD" in cu and "NM" in cu and "IC50" not in cu and not kd_col:
                    kd_col = col
                if "EC50" in cu and "NM" in cu and not ec50_col:
                    ec50_col = col

            assay_cols = {k: v for k, v in {
                "IC50": ic50_col,
                "Ki":   ki_col,
                "Kd":   kd_col,
                "EC50": ec50_col,
            }.items() if v is not None}

            print(f"  Detected columns:")
            print(f"    SMILES  : {ligand_col}")
            print(f"    Sequence: {seq_col}")
            for atype, col in assay_cols.items():
                print(f"    {atype:<6}  : {col}")

        # Build base dataframe for this chunk
        base_cols    = [ligand_col, seq_col]
        avail_assay  = [v for v in assay_cols.values() if v in chunk.columns]
        sub = chunk[base_cols + avail_assay].copy()

        col_rename = {"SMILES": ligand_col, "Sequence": seq_col}
        col_rename.update({k: v for k, v in assay_cols.items() if v in chunk.columns})
        sub.columns = ["SMILES", "Sequence"] + \
                      [k for k, v in assay_cols.items() if v in chunk.columns]

        # Melt all assay types to long format
        val_vars = [k for k in assay_cols.keys() if k in sub.columns]
        melted   = sub.melt(id_vars=["SMILES", "Sequence"],
                            value_vars=val_vars,
                            var_name="assay_type",
                            value_name="value_nM")

        melted = melted.dropna(subset=["value_nM", "SMILES", "Sequence"])
        melted["value_nM"] = pd.to_numeric(melted["value_nM"], errors="coerce")
        melted = melted.dropna(subset=["value_nM"])
        melted = melted[melted["value_nM"] > 0]

        # pAffinity — same formula as Ebola dataset
        melted["pIC50"] = 9.0 - np.log10(melted["value_nM"].astype(float))

        # Median-aggregate duplicate (SMILES, Sequence) pairs
        melted = (melted.groupby(["SMILES", "Sequence"], as_index=False)["pIC50"]
                        .median())

        chunks.append(melted[["SMILES", "Sequence", "pIC50"]])
        total += len(melted)
        if max_rows and total >= max_rows:
            break

    df = pd.concat(chunks, ignore_index=True)
    df = df.drop_duplicates(subset=["SMILES", "Sequence", "pIC50"])
    del chunks; gc.collect()

    if max_rows:
        df = df.sample(n=min(max_rows, len(df)),
                       random_state=seed).reset_index(drop=True)

    print(f"\n  BindingDB loaded:")
    print(f"    Total rows      : {len(df)}")
    print(f"    pAffinity mean  : {df['pIC50'].mean():.3f}")
    print(f"    pAffinity std   : {df['pIC50'].std():.3f}")
    print(f"    pAffinity range : "
          f"{df['pIC50'].min():.3f} — {df['pIC50'].max():.3f}")
    print(f"    Unique SMILES   : {df['SMILES'].nunique()}")
    print(f"    Unique seqs     : {df['Sequence'].nunique()}")
    return df


# ============================================================================
# DATA LOADING — EBOLA (all assay types as pAffinity)
# ============================================================================

def load_ebola_data(zip_path="ebola.zip", out_dir="ebola_data"):
    """
    Load Ebola bioactivity data (IC50+EC50+Ki+Kd) as pAffinity.
    Reads from ebola_dti_ready.csv (produced by build_ebola_dataset.py).
    Falls back to individual split files for backward compatibility.
    Column name pIC50 kept for model code compatibility.
    """
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out)

    # Primary: unified DTI-ready file
    dti_files = list(out.rglob("ebola_dti_ready.csv"))
    if dti_files:
        print(f"  Loading: {dti_files[0].name}")
        df = pd.read_csv(dti_files[0])
        df = df[["canonical_smiles", "sequence", "pAffinity"]].copy()
        df.columns = ["SMILES", "Sequence", "pIC50"]
    else:
        # Fallback: load all split files
        print("  ebola_dti_ready.csv not found — loading split files ...")
        all_dfs = []
        for unit, factor in [("nM", 1e-9), ("uM", 1e-6)]:
            for assay in ["ic50", "ec50", "ki", "kd"]:
                files = list(out.rglob(f"ebola_{unit}_{assay}_seq.csv"))
                if not files:
                    continue
                sub = pd.read_csv(files[0])
                if sub.empty:
                    continue
                sub = sub[["canonical_smiles", "sequence", "value"]].dropna()
                sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
                sub = sub[sub["value"] > 0].dropna()
                sub["pIC50"] = -np.log10(sub["value"] * factor)
                sub = sub.rename(columns={"canonical_smiles": "SMILES",
                                          "sequence":         "Sequence"})
                all_dfs.append(sub[["SMILES", "Sequence", "pIC50"]])
        if not all_dfs:
            raise FileNotFoundError(
                f"No Ebola data files in {out}. "
                "Run build_ebola_dataset.py first.")
        df = pd.concat(all_dfs, ignore_index=True)

    df["SMILES"]   = df["SMILES"].map(canonicalize_smiles)
    df["Sequence"] = df["Sequence"].astype(str).str.upper().str.strip()
    df = df.dropna(subset=["SMILES", "Sequence", "pIC50"])
    df = df[np.isfinite(df["pIC50"])]
    df = (df.groupby(["SMILES", "Sequence"], as_index=False)["pIC50"]
            .median()
            .reset_index(drop=True))

    n_low = int((df["pIC50"] < 4.0).sum())
    print(f"  Ebola: {len(df)} rows (IC50+EC50+Ki+Kd → pAffinity, no cutoff)")
    print(f"  pIC50: min={df['pIC50'].min():.3f}, max={df['pIC50'].max():.3f}, "
          f"mean={df['pIC50'].mean():.3f}, std={df['pIC50'].std():.3f}")
    print(f"  Rare pIC50 < 4.0: {n_low} rows kept")
    return df


# ============================================================================
# PROTEIN SIMILARITY RETRIEVAL
# ============================================================================

def protein_kmer_set(seq, k=3, max_len=1000):
    seq = ''.join([c for c in str(seq).upper()[:max_len] if c in PROTEIN_CHARS])
    if len(seq) < k:
        return set([seq]) if seq else set()
    return set(seq[i:i+k] for i in range(len(seq)-k+1))


def jaccard_similarity(a, b):
    if not a or not b:
        return 0.0
    inter = len(a.intersection(b))
    union = len(a.union(b))
    return inter / union if union else 0.0


def _max_similarity_to_ebola(seq, ebola_sets, kmer, max_len):
    q = protein_kmer_set(seq, k=kmer, max_len=max_len)
    if not q:
        return -1.0
    return max(jaccard_similarity(q, e) for e in ebola_sets)


def _score_sequence_chunk(args):
    seqs, ebola_sets, kmer, max_len = args
    return [_max_similarity_to_ebola(s, ebola_sets, kmer, max_len)
            for s in seqs]


def retrieve_bindingdb_by_protein_similarity(
        df_bdb, df_ebola, top_k,
        kmer=3, max_len=1000, out_dir=None, num_workers=1):

    print("\n" + "="*70)
    print("PROTEIN SIMILARITY RETRIEVAL")
    print("="*70)
    print(f"Candidate BindingDB rows : {len(df_bdb)}")
    print(f"Target Ebola rows        : {len(df_ebola)}")
    print(f"Requested retrieved rows : {top_k}")

    ebola_sets = [s for seq in df_ebola["Sequence"].dropna().unique()
                  if (s := protein_kmer_set(seq, k=kmer, max_len=max_len))]
    if not ebola_sets:
        raise ValueError("No valid Ebola protein sequences for retrieval.")
    print(f"Valid unique Ebola proteins: {len(ebola_sets)}")

    bdb_seqs_all = df_bdb["Sequence"].astype(str).tolist()
    unique_seqs  = list(dict.fromkeys(bdb_seqs_all))
    dedup_factor = len(bdb_seqs_all) / max(1, len(unique_seqs))
    print(f"Unique BindingDB seqs: {len(unique_seqs):,} "
          f"(from {len(bdb_seqs_all):,} rows, {dedup_factor:.1f}x duplication)")

    if num_workers and num_workers > 1:
        chunk_size = max(1, (len(unique_seqs) + num_workers - 1) // num_workers)
        chunks     = [unique_seqs[i:i+chunk_size]
                      for i in range(0, len(unique_seqs), chunk_size)]
        args_list  = [(c, ebola_sets, kmer, max_len) for c in chunks]
        unique_scores = []
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            for cs in tqdm(ex.map(_score_sequence_chunk, args_list),
                           total=len(chunks),
                           desc=f"Protein retrieval ({num_workers} workers)"):
                unique_scores.extend(cs)
    else:
        unique_scores = [
            _max_similarity_to_ebola(s, ebola_sets, kmer, max_len)
            for s in tqdm(unique_seqs, desc="Protein retrieval")
        ]

    seq_to_score = dict(zip(unique_seqs, unique_scores))
    df_ranked = df_bdb.copy()
    df_ranked["protein_similarity"] = [seq_to_score[s] for s in bdb_seqs_all]
    df_ranked = df_ranked[df_ranked["protein_similarity"] >= 0]
    df_ranked = df_ranked.sort_values(
        "protein_similarity", ascending=False).reset_index(drop=True)
    retrieved = df_ranked.head(min(top_k, len(df_ranked))).copy()

    print(f"\nRetrieved: {len(retrieved)} rows")
    print(f"  Similarity max/mean/min: "
          f"{retrieved['protein_similarity'].max():.4f} / "
          f"{retrieved['protein_similarity'].mean():.4f} / "
          f"{retrieved['protein_similarity'].min():.4f}")

    if out_dir is not None:
        retrieved.to_csv(
            Path(out_dir) / "protein_retrieved_bindingdb.csv", index=False)
        print(f"  Saved retrieval table")

    return retrieved



# ============================================================================
# COMPOUND FINGERPRINT HELPER
# ============================================================================

def compound_fingerprint(smiles, radius=2, n_bits=2048):
    """Morgan fingerprint as RDKit ExplicitBitVect."""
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(
            mol, radius, nBits=n_bits)
    except Exception:
        return None


def _max_compound_similarity(smiles, ebola_fps, radius, n_bits):
    fp = compound_fingerprint(smiles, radius=radius, n_bits=n_bits)
    if fp is None or not ebola_fps:
        return 0.0
    from rdkit.Chem import DataStructs
    return max(DataStructs.BulkTanimotoSimilarity(fp, ebola_fps))


def _score_compound_chunk(args):
    smiles_list, ebola_fps, radius, n_bits = args
    return [_max_compound_similarity(s, ebola_fps, radius, n_bits)
            for s in smiles_list]


# ============================================================================
# COMPOUND SIMILARITY RETRIEVAL
# ============================================================================

def retrieve_bindingdb_by_compound_similarity(
        df_bdb, df_ebola, top_k,
        radius=2, n_bits=2048,
        out_dir=None, num_workers=1):
    """
    Keep top_k BindingDB rows whose compound is most similar
    (Morgan Tanimoto) to any Ebola compound.
    Scores each unique SMILES once then maps back to all rows.
    """
    print("\n" + "="*70)
    print("COMPOUND SIMILARITY RETRIEVAL")
    print("="*70)
    print(f"Candidate BindingDB rows : {len(df_bdb)}")
    print(f"Target Ebola rows        : {len(df_ebola)}")
    print(f"Requested retrieved rows : {top_k}")

    ebola_fps = [fp for smi in df_ebola["SMILES"].dropna().unique()
                 if (fp := compound_fingerprint(smi, radius, n_bits)) is not None]
    if not ebola_fps:
        raise ValueError("No valid Ebola compounds for compound retrieval.")
    print(f"Valid unique Ebola compounds: {len(ebola_fps)}")

    bdb_smiles_all = df_bdb["SMILES"].astype(str).tolist()
    unique_smiles  = list(dict.fromkeys(bdb_smiles_all))
    dedup = len(bdb_smiles_all) / max(1, len(unique_smiles))
    print(f"Unique BindingDB compounds: {len(unique_smiles):,} "
          f"(from {len(bdb_smiles_all):,} rows, {dedup:.1f}x duplication)")

    if num_workers and num_workers > 1:
        chunk_size = max(1, (len(unique_smiles) + num_workers - 1) // num_workers)
        chunks     = [unique_smiles[i:i+chunk_size]
                      for i in range(0, len(unique_smiles), chunk_size)]
        args_list  = [(c, ebola_fps, radius, n_bits) for c in chunks]
        unique_scores = []
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            for cs in tqdm(ex.map(_score_compound_chunk, args_list),
                           total=len(chunks),
                           desc=f"Compound retrieval ({num_workers} workers)"):
                unique_scores.extend(cs)
    else:
        unique_scores = [
            _max_compound_similarity(s, ebola_fps, radius, n_bits)
            for s in tqdm(unique_smiles, desc="Compound retrieval")]

    smi_to_score = dict(zip(unique_smiles, unique_scores))
    df_ranked = df_bdb.copy()
    df_ranked["compound_similarity"] = [smi_to_score[s] for s in bdb_smiles_all]
    df_ranked["protein_similarity"]  = 0.0
    df_ranked = df_ranked.sort_values(
        "compound_similarity", ascending=False).reset_index(drop=True)
    retrieved = df_ranked.head(min(top_k, len(df_ranked))).copy()

    print(f"\nRetrieved: {len(retrieved)} rows")
    print(f"  Compound sim max/mean/min: "
          f"{retrieved['compound_similarity'].max():.4f} / "
          f"{retrieved['compound_similarity'].mean():.4f} / "
          f"{retrieved['compound_similarity'].min():.4f}")

    if out_dir:
        retrieved.to_csv(
            Path(out_dir) / "compound_retrieved_bindingdb.csv", index=False)
        print("  Saved retrieval table")
    return retrieved


# ============================================================================
# HYBRID SIMILARITY RETRIEVAL
# ============================================================================

def retrieve_bindingdb_by_hybrid_similarity(
        df_bdb, df_ebola, top_k,
        radius=2, n_bits=2048,
        kmer=3, max_len=1000,
        hybrid_alpha=0.5,
        out_dir=None, num_workers=1):
    """
    Hybrid retrieval: combines compound AND protein similarity.
    hybrid_similarity = alpha * compound_sim + (1-alpha) * protein_sim

    Scores each unique SMILES once and each unique protein once,
    then maps both scores back to all rows and combines.
    alpha=0.5 gives equal weight (default).
    """
    print("\n" + "="*70)
    print("HYBRID (COMPOUND + PROTEIN) SIMILARITY RETRIEVAL")
    print("="*70)
    print(f"Candidate BindingDB rows : {len(df_bdb)}")
    print(f"Target Ebola rows        : {len(df_ebola)}")
    print(f"Requested retrieved rows : {top_k}")
    print(f"Hybrid alpha (compound weight): {hybrid_alpha} "
          f"(protein weight: {1-hybrid_alpha})")

    # ---- Compound scores ----
    ebola_fps = [fp for smi in df_ebola["SMILES"].dropna().unique()
                 if (fp := compound_fingerprint(smi, radius, n_bits)) is not None]
    if not ebola_fps:
        raise ValueError("No valid Ebola compounds for hybrid retrieval.")
    print(f"Valid unique Ebola compounds: {len(ebola_fps)}")

    bdb_smiles_all = df_bdb["SMILES"].astype(str).tolist()
    unique_smiles  = list(dict.fromkeys(bdb_smiles_all))
    print(f"Unique BindingDB compounds: {len(unique_smiles):,}")

    if num_workers and num_workers > 1:
        chunk_size = max(1, (len(unique_smiles) + num_workers - 1) // num_workers)
        chunks     = [unique_smiles[i:i+chunk_size]
                      for i in range(0, len(unique_smiles), chunk_size)]
        args_list  = [(c, ebola_fps, radius, n_bits) for c in chunks]
        comp_scores_unique = []
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            for cs in tqdm(ex.map(_score_compound_chunk, args_list),
                           total=len(chunks),
                           desc="Hybrid: compound scores"):
                comp_scores_unique.extend(cs)
    else:
        comp_scores_unique = [
            _max_compound_similarity(s, ebola_fps, radius, n_bits)
            for s in tqdm(unique_smiles, desc="Hybrid: compound scores")]

    smi_to_comp = dict(zip(unique_smiles, comp_scores_unique))

    # ---- Protein scores ----
    ebola_sets = [s for seq in df_ebola["Sequence"].dropna().unique()
                  if (s := protein_kmer_set(seq, k=kmer, max_len=max_len))]
    if not ebola_sets:
        raise ValueError("No valid Ebola sequences for hybrid retrieval.")
    print(f"Valid unique Ebola proteins: {len(ebola_sets)}")

    bdb_seqs_all  = df_bdb["Sequence"].astype(str).tolist()
    unique_seqs   = list(dict.fromkeys(bdb_seqs_all))
    print(f"Unique BindingDB sequences: {len(unique_seqs):,}")

    if num_workers and num_workers > 1:
        chunk_size = max(1, (len(unique_seqs) + num_workers - 1) // num_workers)
        chunks     = [unique_seqs[i:i+chunk_size]
                      for i in range(0, len(unique_seqs), chunk_size)]
        args_list  = [(c, ebola_sets, kmer, max_len) for c in chunks]
        prot_scores_unique = []
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            for cs in tqdm(ex.map(_score_sequence_chunk, args_list),
                           total=len(chunks),
                           desc="Hybrid: protein scores"):
                prot_scores_unique.extend(cs)
    else:
        prot_scores_unique = [
            _max_similarity_to_ebola(s, ebola_sets, kmer, max_len)
            for s in tqdm(unique_seqs, desc="Hybrid: protein scores")]

    seq_to_prot = dict(zip(unique_seqs, prot_scores_unique))

    # ---- Combine scores ----
    df_ranked = df_bdb.copy()
    df_ranked["compound_similarity"] = [smi_to_comp.get(s, 0.0)
                                         for s in bdb_smiles_all]
    df_ranked["protein_similarity"]  = [seq_to_prot.get(s, 0.0)
                                         for s in bdb_seqs_all]
    df_ranked["hybrid_similarity"]   = (
        hybrid_alpha * df_ranked["compound_similarity"] +
        (1 - hybrid_alpha) * df_ranked["protein_similarity"])

    df_ranked = df_ranked[df_ranked["protein_similarity"] >= 0]
    df_ranked = df_ranked.sort_values(
        "hybrid_similarity", ascending=False).reset_index(drop=True)
    retrieved = df_ranked.head(min(top_k, len(df_ranked))).copy()

    print(f"\nRetrieved: {len(retrieved)} rows")
    print(f"  Compound sim max/mean : "
          f"{retrieved['compound_similarity'].max():.4f} / "
          f"{retrieved['compound_similarity'].mean():.4f}")
    print(f"  Protein  sim max/mean : "
          f"{retrieved['protein_similarity'].max():.4f} / "
          f"{retrieved['protein_similarity'].mean():.4f}")
    print(f"  Hybrid   sim max/mean : "
          f"{retrieved['hybrid_similarity'].max():.4f} / "
          f"{retrieved['hybrid_similarity'].mean():.4f}")

    if out_dir:
        retrieved.to_csv(
            Path(out_dir) / "hybrid_retrieved_bindingdb.csv", index=False)
        print("  Saved retrieval table")
    return retrieved


# ============================================================================
# RANDOM RETRIEVAL
# ============================================================================

def retrieve_bindingdb_random(df_bdb, df_ebola, top_k,
                               seed=42, out_dir=None):
    """
    Random retrieval baseline — uniform sample with no similarity.
    Kept for ablation comparison with protein/compound/hybrid methods.
    """
    print("\n" + "="*70)
    print("RANDOM RETRIEVAL")
    print("="*70)
    print(f"Candidate BindingDB rows : {len(df_bdb)}")
    print(f"Requested retrieved rows : {top_k}")
    print(f"Random seed              : {seed}")

    rng = np.random.RandomState(seed)
    k   = min(top_k, len(df_bdb))
    idx = rng.choice(len(df_bdb), size=k, replace=False)
    retrieved = df_bdb.iloc[idx].copy().reset_index(drop=True)
    retrieved["compound_similarity"] = 0.0
    retrieved["protein_similarity"]  = 0.0
    retrieved["hybrid_similarity"]   = 0.0

    print(f"Retrieved: {len(retrieved)} rows (random)")
    if out_dir:
        retrieved.to_csv(
            Path(out_dir) / "random_retrieved_bindingdb.csv", index=False)
        print("  Saved retrieval table")
    return retrieved

# ============================================================================
# ENCODING
# ============================================================================

def encode_smiles(smiles, max_len=SMILES_MAX_LEN):
    enc = [SMILES_VOCAB.get(c, 0) for c in smiles[:max_len]]
    enc += [0] * (max_len - len(enc))
    return enc


def encode_protein(seq, max_len=PROTEIN_MAX_LEN):
    enc = [PROTEIN_VOCAB.get(c, 0) for c in seq.upper()[:max_len]]
    enc += [0] * (max_len - len(enc))
    return enc


if hasattr(Chem, "ValenceType"):
    def _get_implicit_valence(atom):
        return atom.GetValence(Chem.ValenceType.IMPLICIT)
else:
    def _get_implicit_valence(atom):
        return atom.GetImplicitValence()


def smiles_to_graph(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        def one_of_k(x, allowable):
            if x not in allowable: x = allowable[-1]
            return [x == v for v in allowable]

        def atom_features(atom):
            return np.array(
                one_of_k(atom.GetSymbol(),
                    ['C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca',
                     'Fe','As','Al','I','B','V','K','Tl','Yb','Sb','Sn',
                     'Ag','Pd','Co','Se','Ti','Zn','H','Li','Ge','Cu','Au',
                     'Ni','Cd','In','Mn','Zr','Cr','Pt','Hg','Pb','other']) +
                one_of_k(atom.GetDegree(), [0,1,2,3,4,5,6,7,8,9,10]) +
                one_of_k(atom.GetTotalNumHs(), [0,1,2,3,4,5,6,7,8,9,10]) +
                one_of_k(_get_implicit_valence(atom),
                         [0,1,2,3,4,5,6,7,8,9,10]) +
                [atom.GetIsAromatic()],
                dtype=np.float32)

        node_f = np.array([atom_features(a) for a in mol.GetAtoms()],
                          dtype=np.float32)
        edges = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edges += [[i, j], [j, i]]
        if not edges:
            edges = [[0, 0]]
        return node_f, np.array(edges, dtype=np.int64).T
    except Exception:
        return None


# ============================================================================
# DATASETS
# ============================================================================

class DeepDTADataset(Dataset):
    def __init__(self, df, y_mean=0., y_std=1.):
        self.smiles_enc  = [encode_smiles(s)  for s in df["SMILES"].tolist()]
        self.protein_enc = [encode_protein(p) for p in df["Sequence"].tolist()]
        y = df["pIC50"].values.astype(np.float32)
        self.y = ((y - y_mean) / y_std).astype(np.float32)

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        return (torch.tensor(self.smiles_enc[i],  dtype=torch.long),
                torch.tensor(self.protein_enc[i], dtype=torch.long),
                torch.tensor(self.y[i],           dtype=torch.float32))


def _smiles_to_graph_batch(smiles_list):
    return [smiles_to_graph(s) for s in smiles_list]


class GraphDTADataset(Dataset):
    def __init__(self, df, y_mean=0., y_std=1.,
                 node_mean=None, node_std=None, num_workers=1):
        y_raw       = df["pIC50"].values.astype(np.float32)
        smiles_list = df["SMILES"].tolist()
        seq_list    = df["Sequence"].tolist()

        if num_workers and num_workers > 1:
            chunk_size = max(1, (len(smiles_list) + num_workers - 1) // num_workers)
            chunks = [smiles_list[i:i+chunk_size]
                      for i in range(0, len(smiles_list), chunk_size)]
            graphs_flat = []
            with ProcessPoolExecutor(max_workers=num_workers) as ex:
                for cr in ex.map(_smiles_to_graph_batch, chunks):
                    graphs_flat.extend(cr)
        else:
            graphs_flat = [smiles_to_graph(s) for s in smiles_list]

        self.graphs, self.protein_enc, self.y = [], [], []
        for i, g in enumerate(graphs_flat):
            if g is None: continue
            self.graphs.append(g)
            self.protein_enc.append(encode_protein(seq_list[i]))
            self.y.append((y_raw[i] - y_mean) / y_std)
        self.y = np.array(self.y, dtype=np.float32)

        if node_mean is None:
            all_nodes      = np.concatenate(
                [nf for nf, _ in self.graphs], axis=0)
            self.node_mean = all_nodes.mean(0, keepdims=True)
            self.node_std  = all_nodes.std(0,  keepdims=True) + 1e-8
        else:
            self.node_mean, self.node_std = node_mean, node_std

        self.graphs = [((nf - self.node_mean) / self.node_std, ei)
                       for nf, ei in self.graphs]
        dropped = len(smiles_list) - len(self.y)
        print(f"  GraphDTA: {len(self.y)} valid / {len(smiles_list)} input "
              f"({dropped} dropped, node features z-scored)")

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        node_f, edge_index = self.graphs[i]
        return (torch.tensor(node_f,              dtype=torch.float32),
                torch.tensor(edge_index,          dtype=torch.long),
                torch.tensor(self.protein_enc[i], dtype=torch.long),
                torch.tensor(self.y[i],           dtype=torch.float32))


def graph_collate(batch):
    node_fs, edge_indices, proteins, ys = zip(*batch)
    offset, batch_node_f, batch_edge_idx, batch_assign = 0, [], [], []
    for b, (nf, ei) in enumerate(zip(node_fs, edge_indices)):
        nf_t = nf.float() if isinstance(nf, torch.Tensor) \
               else torch.tensor(nf, dtype=torch.float32)
        ei_t = ei.long()  if isinstance(ei, torch.Tensor) \
               else torch.tensor(ei, dtype=torch.long)
        batch_node_f.append(nf_t)
        batch_edge_idx.append(ei_t + offset)
        batch_assign.extend([b] * len(nf_t))
        offset += len(nf_t)
    return (torch.cat(batch_node_f,   dim=0),
            torch.cat(batch_edge_idx, dim=1),
            torch.tensor(batch_assign, dtype=torch.long),
            torch.stack([p.clone().detach().to(torch.long)    for p in proteins]),
            torch.stack([y.clone().detach().to(torch.float32) for y in ys]))


# ============================================================================
# MODELS
# ============================================================================

class DeepDTA(nn.Module):
    def __init__(self,
                 smiles_vocab_size  = len(SMILES_CHARS) + 1,
                 protein_vocab_size = len(PROTEIN_CHARS) + 1,
                 smiles_embed_dim   = 128, protein_embed_dim = 128,
                 num_filters = 32, smiles_kernel = 8, protein_kernel = 12,
                 fc_dim = 1024, dropout = 0.1):
        super().__init__()
        self.smiles_embed = nn.Embedding(
            smiles_vocab_size, smiles_embed_dim, padding_idx=0)
        self.smiles_norm  = nn.LayerNorm(smiles_embed_dim)
        self.smiles_conv  = nn.Sequential(
            nn.Conv1d(smiles_embed_dim, num_filters,     smiles_kernel),
            nn.BatchNorm1d(num_filters), nn.ReLU(),
            nn.Conv1d(num_filters,      num_filters * 2, smiles_kernel),
            nn.BatchNorm1d(num_filters * 2), nn.ReLU(),
            nn.Conv1d(num_filters * 2,  num_filters * 3, smiles_kernel),
            nn.BatchNorm1d(num_filters * 3), nn.ReLU())
        self.protein_embed = nn.Embedding(
            protein_vocab_size, protein_embed_dim, padding_idx=0)
        self.protein_norm  = nn.LayerNorm(protein_embed_dim)
        self.protein_conv  = nn.Sequential(
            nn.Conv1d(protein_embed_dim, num_filters,     protein_kernel),
            nn.BatchNorm1d(num_filters), nn.ReLU(),
            nn.Conv1d(num_filters,       num_filters * 2, protein_kernel),
            nn.BatchNorm1d(num_filters * 2), nn.ReLU(),
            nn.Conv1d(num_filters * 2,   num_filters * 3, protein_kernel),
            nn.BatchNorm1d(num_filters * 3), nn.ReLU())
        fused = num_filters * 3 * 2
        self.fc = nn.Sequential(
            nn.LayerNorm(fused),
            nn.Linear(fused, fc_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(fc_dim, fc_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(fc_dim // 2, 1))
        self.calibration = nn.Linear(1, 1)
        nn.init.ones_(self.calibration.weight)
        nn.init.zeros_(self.calibration.bias)

    def forward(self, smiles, protein):
        d = self.smiles_norm(self.smiles_embed(smiles))
        d = F.adaptive_max_pool1d(
            self.smiles_conv(d.permute(0,2,1)), 1).squeeze(-1)
        p = self.protein_norm(self.protein_embed(protein))
        p = F.adaptive_max_pool1d(
            self.protein_conv(p.permute(0,2,1)), 1).squeeze(-1)
        return self.calibration(self.fc(torch.cat([d, p], dim=1)))

    def freeze_backbone(self):
        for name, param in self.named_parameters():
            param.requires_grad = (
                "calibration" in name or name.startswith("fc."))
        self._frozen_modules = [
            self.smiles_embed, self.smiles_norm, self.smiles_conv,
            self.protein_embed, self.protein_norm, self.protein_conv]
        self._freeze_bn_eval = True
        self._apply_frozen_bn_eval()

    def _apply_frozen_bn_eval(self):
        for m in getattr(self, "_frozen_modules", []):
            for sub in m.modules():
                if isinstance(sub, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    sub.eval()

    def train(self, mode=True):
        super().train(mode)
        if mode and getattr(self, "_freeze_bn_eval", False):
            self._apply_frozen_bn_eval()
        return self

    def unfreeze_all(self):
        self._freeze_bn_eval = False
        for p in self.parameters(): p.requires_grad = True


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn     = nn.BatchNorm1d(out_dim)

    def forward(self, x, edge_index, batch):
        row, col = edge_index
        agg = torch.zeros_like(x)
        agg.index_add_(0, row, x[col])
        deg = torch.zeros(x.size(0), device=x.device)
        deg.index_add_(0, row, torch.ones(row.size(0), device=x.device))
        return F.relu(self.bn(
            self.linear((x + agg) / deg.clamp(min=1).unsqueeze(-1))))


_HAS_SCATTER_REDUCE = hasattr(torch.Tensor, "scatter_reduce_")


def graph_max_pool(x, batch, num_graphs):
    dim = x.size(1)
    if _HAS_SCATTER_REDUCE:
        out   = torch.zeros(num_graphs, dim, device=x.device, dtype=x.dtype)
        index = batch.unsqueeze(-1).expand(-1, dim)
        out.scatter_reduce_(0, index, x, reduce="amax", include_self=False)
        return out
    out = torch.zeros(num_graphs, dim, device=x.device)
    for g in range(num_graphs):
        mask = (batch == g)
        if mask.any():
            out[g] = x[mask].max(dim=0).values
    return out


class GraphDTA(nn.Module):
    def __init__(self,
                 node_feat_dim = 78, gcn_hidden = 64, gcn_out = 128,
                 protein_vocab_size = len(PROTEIN_CHARS) + 1,
                 protein_embed_dim = 128, num_filters = 32,
                 protein_kernel = 12, fc_dim = 1024, dropout = 0.1):
        super().__init__()
        self.gcn1    = GCNLayer(node_feat_dim, gcn_hidden)
        self.gcn2    = GCNLayer(gcn_hidden,    gcn_hidden)
        self.gcn3    = GCNLayer(gcn_hidden,    gcn_out)
        self.drug_fc = nn.Linear(gcn_out * 2,  gcn_out)
        self.drug_bn = nn.BatchNorm1d(gcn_out)
        self.protein_embed = nn.Embedding(
            protein_vocab_size, protein_embed_dim, padding_idx=0)
        self.protein_conv  = nn.Sequential(
            nn.Conv1d(protein_embed_dim, num_filters,     protein_kernel),
            nn.BatchNorm1d(num_filters), nn.ReLU(),
            nn.Conv1d(num_filters,       num_filters * 2, protein_kernel),
            nn.BatchNorm1d(num_filters * 2), nn.ReLU(),
            nn.Conv1d(num_filters * 2,   num_filters * 3, protein_kernel),
            nn.BatchNorm1d(num_filters * 3), nn.ReLU())
        fused = gcn_out + num_filters * 3
        self.fc = nn.Sequential(
            nn.LayerNorm(fused),
            nn.Linear(fused, fc_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(fc_dim, fc_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(fc_dim // 2, 1))
        self.calibration = nn.Linear(1, 1)
        nn.init.ones_(self.calibration.weight)
        nn.init.zeros_(self.calibration.bias)

    def forward(self, node_f, edge_index, batch, protein):
        x  = self.gcn3(self.gcn2(
            self.gcn1(node_f, edge_index, batch),
            edge_index, batch), edge_index, batch)
        ng = batch.max().item() + 1
        sp = torch.zeros(ng, x.size(1), device=x.device)
        ct = torch.zeros(ng, 1,         device=x.device)
        sp.index_add_(0, batch, x)
        ct.index_add_(0, batch,
                      torch.ones(batch.size(0), 1, device=x.device))
        d  = torch.cat([sp / ct.clamp(min=1),
                        graph_max_pool(x, batch, ng)], dim=1)
        d  = F.relu(self.drug_bn(self.drug_fc(d)))
        p  = F.adaptive_max_pool1d(
            self.protein_conv(
                self.protein_embed(protein).permute(0,2,1)), 1).squeeze(-1)
        return self.calibration(self.fc(torch.cat([d, p], dim=1)))

    def freeze_backbone(self):
        for name, param in self.named_parameters():
            param.requires_grad = (
                "calibration" in name or name.startswith("fc."))
        self._frozen_modules = [
            self.gcn1, self.gcn2, self.gcn3,
            self.drug_fc, self.drug_bn,
            self.protein_embed, self.protein_conv]
        self._freeze_bn_eval = True
        self._apply_frozen_bn_eval()

    def _apply_frozen_bn_eval(self):
        for m in getattr(self, "_frozen_modules", []):
            for sub in m.modules():
                if isinstance(sub, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    sub.eval()

    def train(self, mode=True):
        super().train(mode)
        if mode and getattr(self, "_freeze_bn_eval", False):
            self._apply_frozen_bn_eval()
        return self

    def unfreeze_all(self):
        self._freeze_bn_eval = False
        for p in self.parameters(): p.requires_grad = True


def build_model(model_type, device):
    m = DeepDTA() if model_type == "deepdta" else GraphDTA()
    return m.to(device)


# ============================================================================
# TRAINING
# ============================================================================

def eval_model(model, loader, device, model_type, y_mean=0., y_std=1.):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            if model_type == "deepdta":
                smiles, protein, y = batch
                pred = model(smiles.to(device), protein.to(device))
            else:
                node_f, ei, ba, protein, y = batch
                pred = model(node_f.to(device), ei.to(device),
                             ba.to(device), protein.to(device))
            ps.extend(
                (pred.cpu().numpy().ravel() * y_std + y_mean).tolist())
            ys.extend((y.numpy().ravel() * y_std + y_mean).tolist())
    return calc_metrics(ys, ps)


def run_epochs(model, tr_dl, va_dl, device, model_type,
               epochs, lr, wd, patience, label, y_mean=0., y_std=1.):
    opt     = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=wd)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=lr/20)
    loss_fn = nn.MSELoss()
    best_val, best_state, bad = float("inf"), None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        tl = 0.
        for batch in tr_dl:
            if model_type == "deepdta":
                smiles, protein, y = batch
                pred = model(smiles.to(device), protein.to(device))
                y    = y.to(device).view(-1, 1)
            else:
                node_f, ei, ba, protein, y = batch
                pred = model(node_f.to(device), ei.to(device),
                             ba.to(device), protein.to(device))
                y = y.to(device).view(-1, 1)
            opt.zero_grad()
            loss = loss_fn(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item()
        sched.step()
        val_m = eval_model(model, va_dl, device, model_type, y_mean, y_std)
        vr    = val_m["RMSE"]
        if epoch % 20 == 0 or epoch <= 3:
            print(f"  [{label}] ep{epoch:3d} | "
                  f"loss={tl/len(tr_dl):.4f} | val_RMSE={vr:.4f}")
        if vr < best_val:
            best_val, bad = vr, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f"  [{label}] early stop @ ep{epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)
    return best_val


def make_loaders(df_tr, df_va, df_te, model_type,
                 y_mean, y_std, batch_size, num_workers=1, seed=42):
    pin = torch.cuda.is_available()

    # Explicit generator makes shuffle order reproducible for each seed.
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)

    if model_type == "deepdta":
        kw = dict(num_workers=0, pin_memory=pin)
        return (
            DataLoader(DeepDTADataset(df_tr, y_mean, y_std),
                       batch_size=batch_size, shuffle=True,
                       generator=train_generator, **kw),
            DataLoader(DeepDTADataset(df_va, y_mean, y_std),
                       batch_size=batch_size, shuffle=False, **kw),
            DataLoader(DeepDTADataset(df_te, y_mean, y_std),
                       batch_size=batch_size, shuffle=False, **kw))
    else:
        kw = dict(num_workers=0, pin_memory=pin, collate_fn=graph_collate)
        tr_ds = GraphDTADataset(df_tr, y_mean, y_std,
                                num_workers=num_workers)
        va_ds = GraphDTADataset(df_va, y_mean, y_std,
                                node_mean=tr_ds.node_mean,
                                node_std=tr_ds.node_std,
                                num_workers=num_workers)
        te_ds = GraphDTADataset(df_te, y_mean, y_std,
                                node_mean=tr_ds.node_mean,
                                node_std=tr_ds.node_std,
                                num_workers=num_workers)
        return (DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                           generator=train_generator, **kw),
                DataLoader(va_ds, batch_size=batch_size, shuffle=False, **kw),
                DataLoader(te_ds, batch_size=batch_size, shuffle=False, **kw))


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SWAT DeepDTA/GraphDTA for Ebola pAffinity")

    parser.add_argument("--model",  required=True,
                        choices=["deepdta", "graphdta"])
    parser.add_argument("--stage",  required=True, type=int,
                        choices=[1, 2, 12],
                        help="1=Stage1, 2=Stage2, 12=auto both")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained_checkpoint",
                        help="Stage 1 output dir (Stage 2 only)")

    # Seed — primary arg for shell script sweeps
    parser.add_argument("--seed", type=int, default=3,
                        help="Random seed. Sweep via run_swat.sh")

    # Data
    parser.add_argument("--ebola_zip",              default="ebola.zip")
    parser.add_argument("--bindingdb_dir",          default="bindingdb_data")
    parser.add_argument("--bindingdb_samples",      type=int, default=100_000)
    parser.add_argument("--bindingdb_pool_samples", type=int, default=None)
    parser.add_argument("--split_mode",             default="scaffold",
                        choices=["scaffold", "cold_drug"])
    parser.add_argument("--protein_kmer",           type=int, default=3)
    parser.add_argument("--retrieval_type",         default="protein",
                        choices=["protein","compound","hybrid","random"],
                        help="Retrieval strategy: protein (default), "
                             "compound, hybrid, or random")
    parser.add_argument("--hybrid_alpha",           type=float, default=0.5,
                        help="Compound weight in hybrid retrieval (default 0.5)")
    parser.add_argument("--retrieval_workers",      type=int, default=None)

    # Stage 1
    parser.add_argument("--stage1_lr",       type=float, default=1e-3)
    parser.add_argument("--stage1_wd",       type=float, default=1e-4)
    parser.add_argument("--stage1_epochs",   type=int,   default=100)
    parser.add_argument("--stage1_patience", type=int,   default=20)
    parser.add_argument("--stage1_batch",    type=int,   default=256)

    # Stage 2
    parser.add_argument("--stage2_lr",       type=float, default=1e-4)
    parser.add_argument("--stage2_wd",       type=float, default=1e-4)
    parser.add_argument("--stage2_epochs",   type=int,   default=200)
    parser.add_argument("--stage2_patience", type=int,   default=40)
    parser.add_argument("--stage2_batch",    type=int,   default=64)
    parser.add_argument("--probe_lr",        type=float, default=5e-4)
    parser.add_argument("--probe_epochs",    type=int,   default=50)
    parser.add_argument("--probe_patience",  type=int,   default=15)

    args = parser.parse_args()

    if args.retrieval_workers is None:
        args.retrieval_workers = os.cpu_count() or 1
        print(f"retrieval_workers: auto-detected {args.retrieval_workers} cores")

    set_seed(args.seed)
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    print(f"\nDevice    : {device}")
    print(f"Model     : {args.model.upper()}")
    print(f"Stage     : {args.stage}")
    print(f"Seed      : {args.seed}")
    print(f"Split     : {args.split_mode}")
    print(f"Retrieval : {args.retrieval_type}")
    print(f"Hybrid α  : {args.hybrid_alpha}")

    # ---------------------------------------------------------------- AUTO 12
    if args.stage == 12:
        print("\n" + "="*70)
        print("AUTO MODE: Stage 1 → Stage 2")
        print("="*70)

        s1_dir = out_dir / f"stage1_{args.model}_seed{args.seed}"
        s2_dir = out_dir / f"stage2_{args.model}_seed{args.seed}"

        # IMPORTANT: propagate every experiment-defining argument to the
        # child processes. The old version omitted retrieval_type, so every
        # Stage-1 child silently fell back to the default "protein" method.
        base = [sys.executable, __file__,
                "--model",    args.model,
                "--seed",     str(args.seed),
                "--ebola_zip",     args.ebola_zip,
                "--bindingdb_dir", args.bindingdb_dir,
                "--split_mode",    args.split_mode,
                "--retrieval_type", args.retrieval_type,
                "--hybrid_alpha", str(args.hybrid_alpha),
                "--retrieval_workers", str(args.retrieval_workers)]

        s1_cmd = base + [
            "--stage", "1", "--output_dir", str(s1_dir),
            "--bindingdb_samples", str(args.bindingdb_samples),
            "--protein_kmer",      str(args.protein_kmer),
            "--stage1_lr",         str(args.stage1_lr),
            "--stage1_wd",         str(args.stage1_wd),
            "--stage1_epochs",     str(args.stage1_epochs),
            "--stage1_patience",   str(args.stage1_patience),
            "--stage1_batch",      str(args.stage1_batch),
        ]
        if args.bindingdb_pool_samples:
            s1_cmd += ["--bindingdb_pool_samples",
                       str(args.bindingdb_pool_samples)]

        s2_cmd = base + [
            "--stage", "2", "--output_dir", str(s2_dir),
            "--pretrained_checkpoint", str(s1_dir),
            "--stage2_lr",       str(args.stage2_lr),
            "--stage2_wd",       str(args.stage2_wd),
            "--stage2_epochs",   str(args.stage2_epochs),
            "--stage2_patience", str(args.stage2_patience),
            "--stage2_batch",    str(args.stage2_batch),
            "--probe_lr",        str(args.probe_lr),
            "--probe_epochs",    str(args.probe_epochs),
            "--probe_patience",  str(args.probe_patience),
        ]

        # PYTHONHASHSEED is only fully effective when set before a Python
        # interpreter starts. Setting it here makes both child processes inherit it.
        child_env = os.environ.copy()
        child_env["PYTHONHASHSEED"] = str(args.seed)
        child_env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        print("Running Stage 1 ...")
        subprocess.run(s1_cmd, check=True, env=child_env)
        print("Running Stage 2 ...")
        subprocess.run(s2_cmd, check=True, env=child_env)
        print(f"\n✓ AUTO COMPLETE  seed={args.seed}")
        return

    # ---------------------------------------------------------------- STAGE 1
    if args.stage == 1:
        print("\n" + "="*70)
        print(f"STAGE 1: {args.model.upper()} on BindingDB (retrieved) + Ebola")
        print(f"BindingDB: IC50+Ki+Kd → pAffinity (consistent with Ebola)")
        print("="*70)

        df_bdb_pool = load_bindingdb(args.bindingdb_dir,
                                     args.bindingdb_pool_samples,
                                     seed=args.seed)
        df_ebola    = load_ebola_data(args.ebola_zip)

        # Split Ebola BEFORE retrieval
        tr_idx, va_idx, te_idx, groups = make_grouped_split(
            df_ebola, args.seed, args.split_mode)
        df_ebola_tr = df_ebola.iloc[tr_idx].reset_index(drop=True)

        manifest = df_ebola.copy()
        manifest["split"] = "train"
        manifest.loc[va_idx, "split"] = "validation"
        manifest.loc[te_idx, "split"] = "test"
        manifest["scaffold_group"] = groups
        manifest["seed"] = args.seed
        manifest.to_csv(
            out_dir / f"ebola_split_manifest_seed{args.seed}.csv",
            index=False)

        n_before = len(df_bdb_pool)
        df_bdb_pool["SMILES"] = df_bdb_pool["SMILES"].map(canonicalize_smiles)
        df_bdb_pool["Sequence"] = (df_bdb_pool["Sequence"]
                                   .astype(str).str.upper().str.strip())
        df_bdb_pool = df_bdb_pool.dropna(subset=["SMILES", "Sequence"])
        print(f"BindingDB: {n_before:,} → {len(df_bdb_pool):,} "
              f"after canonicalization")

        t0 = time.time()
        print(f"\nRetrieval type: {args.retrieval_type}")
        if args.retrieval_type == "protein":
            df_bdb = retrieve_bindingdb_by_protein_similarity(
                df_bdb_pool, df_ebola_tr,
                top_k=args.bindingdb_samples,
                kmer=args.protein_kmer,
                out_dir=out_dir,
                num_workers=args.retrieval_workers)
        elif args.retrieval_type == "compound":
            df_bdb = retrieve_bindingdb_by_compound_similarity(
                df_bdb_pool, df_ebola_tr,
                top_k=args.bindingdb_samples,
                out_dir=out_dir,
                num_workers=args.retrieval_workers)
        elif args.retrieval_type == "hybrid":
            df_bdb = retrieve_bindingdb_by_hybrid_similarity(
                df_bdb_pool, df_ebola_tr,
                top_k=args.bindingdb_samples,
                hybrid_alpha=args.hybrid_alpha,
                kmer=args.protein_kmer,
                out_dir=out_dir,
                num_workers=args.retrieval_workers)
        elif args.retrieval_type == "random":
            df_bdb = retrieve_bindingdb_random(
                df_bdb_pool, df_ebola_tr,
                top_k=args.bindingdb_samples,
                seed=args.seed,
                out_dir=out_dir)
        print(f"[TIMING] Retrieval: {time.time()-t0:.1f}s")

        df_bdb["source"]      = "bindingdb_retrieved"
        df_ebola_tr["source"] = "ebola_train"

        df_all = pd.concat([df_bdb, df_ebola_tr], ignore_index=True)
        df_all = df_all.drop_duplicates(
            subset=["SMILES", "Sequence", "pIC50"]).reset_index(drop=True)
        print(f"Stage 1 mixed: {len(df_bdb)} BDB + "
              f"{len(df_ebola_tr)} Ebola = {len(df_all)} total")

        # Internal split for early stopping
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.1,
                                     random_state=args.seed)
        tr_i, va_i = next(splitter.split(
            np.arange(len(df_all)),
            groups=df_all["SMILES"].to_numpy()))
        df_tr = df_all.iloc[tr_i].reset_index(drop=True)
        df_va = df_all.iloc[va_i].reset_index(drop=True)

        y_mean = float(df_tr["pIC50"].mean())
        y_std  = float(df_tr["pIC50"].std()) + 1e-8
        print(f"Stage1 pAffinity: mean={y_mean:.3f}, std={y_std:.3f}")
        print(f"Ebola  pAffinity: mean={df_ebola_tr['pIC50'].mean():.3f} "
              f"(gap = {abs(y_mean - df_ebola_tr['pIC50'].mean()):.3f})")
        np.save(out_dir / "y_stats.npy", np.array([y_mean, y_std]))

        print("Building datasets ...")
        tr_dl, va_dl, _ = make_loaders(
            df_tr, df_va, df_va, args.model, y_mean, y_std,
            args.stage1_batch, num_workers=args.retrieval_workers,
            seed=args.seed)

        if args.model == "graphdta":
            np.save(out_dir / "node_mean.npy", tr_dl.dataset.node_mean)
            np.save(out_dir / "node_std.npy",  tr_dl.dataset.node_std)

        model   = build_model(args.model, device)
        n_param = sum(p.numel() for p in model.parameters()
                      if p.requires_grad)
        print(f"Parameters: {n_param:,}")

        print(f"\nTraining Stage 1 ({args.stage1_epochs} epochs) ...")
        t0 = time.time()
        run_epochs(model, tr_dl, va_dl, device, args.model,
                   epochs=args.stage1_epochs, lr=args.stage1_lr,
                   wd=args.stage1_wd, patience=args.stage1_patience,
                   label=f"s1_{args.model}_seed{args.seed}",
                   y_mean=y_mean, y_std=y_std)
        print(f"[TIMING] Stage 1: {time.time()-t0:.1f}s")

        torch.save(model.state_dict(), out_dir / "model_weights.pt")
        print(f"✓ Stage 1 complete — {out_dir}/model_weights.pt")

    # ---------------------------------------------------------------- STAGE 2
    else:
        print("\n" + "="*70)
        print(f"STAGE 2 (SWAT): freeze backbone, fine-tune head on Ebola")
        print("="*70)

        if not args.pretrained_checkpoint:
            raise ValueError("--pretrained_checkpoint required for Stage 2")

        ckpt_dir  = Path(args.pretrained_checkpoint)
        ckpt_path = ckpt_dir / "model_weights.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"No checkpoint: {ckpt_path}")

        ystats = ckpt_dir / "y_stats.npy"
        if ystats.exists():
            bdb_y_mean, bdb_y_std = np.load(ystats).tolist()
            print(f"Stage 1 y stats: mean={bdb_y_mean:.3f}, "
                  f"std={bdb_y_std:.3f}")
        else:
            bdb_y_mean, bdb_y_std = 7.2, 1.8
            print("⚠ No y_stats — using defaults")

        df_ebola = load_ebola_data(args.ebola_zip)

        # Recreate exactly the same split as Stage 1
        tr_idx, va_idx, te_idx, groups = make_grouped_split(
            df_ebola, args.seed, args.split_mode)
        df_tr = df_ebola.iloc[tr_idx].reset_index(drop=True)
        df_va = df_ebola.iloc[va_idx].reset_index(drop=True)
        df_te = df_ebola.iloc[te_idx].reset_index(drop=True)
        print(f"Split: train={len(df_tr)}, "
              f"val={len(df_va)}, test={len(df_te)}")

        y_mean, y_std = bdb_y_mean, bdb_y_std
        ebola_norm = float((df_tr["pIC50"].mean() - y_mean) / y_std)
        print(f"Ebola mean in Stage1-normalised space: {ebola_norm:.4f}")
        print(f"(smaller absolute value = smaller domain gap = better)")

        node_mean = node_std = None
        if args.model == "graphdta":
            nm = ckpt_dir / "node_mean.npy"
            ns = ckpt_dir / "node_std.npy"
            if nm.exists():
                node_mean = np.load(nm)
                node_std  = np.load(ns)
                print("✓ Loaded node stats from Stage 1")

        print("Building datasets ...")
        pin = torch.cuda.is_available()
        if args.model == "graphdta" and node_mean is not None:
            kw = dict(num_workers=0, pin_memory=pin,
                      collate_fn=graph_collate)
            tr_ds = GraphDTADataset(df_tr, y_mean, y_std,
                                    node_mean=node_mean, node_std=node_std,
                                    num_workers=args.retrieval_workers)
            va_ds = GraphDTADataset(df_va, y_mean, y_std,
                                    node_mean=node_mean, node_std=node_std,
                                    num_workers=args.retrieval_workers)
            te_ds = GraphDTADataset(df_te, y_mean, y_std,
                                    node_mean=node_mean, node_std=node_std,
                                    num_workers=args.retrieval_workers)
            stage2_generator = torch.Generator()
            stage2_generator.manual_seed(args.seed)
            tr_dl = DataLoader(tr_ds, batch_size=args.stage2_batch,
                               shuffle=True, generator=stage2_generator, **kw)
            va_dl = DataLoader(va_ds, batch_size=args.stage2_batch,
                               shuffle=False, **kw)
            te_dl = DataLoader(te_ds, batch_size=args.stage2_batch,
                               shuffle=False, **kw)
        else:
            tr_dl, va_dl, te_dl = make_loaders(
                df_tr, df_va, df_te, args.model, y_mean, y_std,
                args.stage2_batch, num_workers=args.retrieval_workers,
                seed=args.seed)

        model = build_model(args.model, device)
        model.load_state_dict(
            torch.load(ckpt_path, map_location=device), strict=False)
        with torch.no_grad():
            model.calibration.bias.fill_(ebola_norm)
        print(f"✓ Loaded Stage 1 weights")
        print(f"  Calibration bias = {ebola_norm:.4f}")

        print(f"\n[SWAT Stage 2] Head retraining "
              f"({args.probe_epochs} epochs, lr={args.probe_lr}) ...")
        model.freeze_backbone()
        t0 = time.time()
        run_epochs(model, tr_dl, va_dl, device, args.model,
                   epochs=args.probe_epochs, lr=args.probe_lr,
                   wd=args.stage2_wd, patience=args.probe_patience,
                   label=f"s2_{args.model}_seed{args.seed}",
                   y_mean=y_mean, y_std=y_std)
        print(f"[TIMING] Stage 2: {time.time()-t0:.1f}s")

        metrics = eval_model(
            model, te_dl, device, args.model, y_mean, y_std)

        print("\n" + "="*70)
        print(f"Stage 2 Results — {args.model.upper()}  seed={args.seed}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        beat = metrics["RMSE"] < BASELINE_RMSE
        print(f"\n  Baseline : {BASELINE_RMSE}")
        print(f"  {'✓ BEATS BASELINE' if beat else '✗ Below baseline'} "
              f"({metrics['RMSE']:.4f})")
        print("="*70)

        ckpt_out = out_dir / f"{args.model}_ebola_seed{args.seed}.pt"
        torch.save(model.state_dict(), ckpt_out)
        print(f"✓ Saved: {ckpt_out}")

        # Save per-seed metrics for shell script aggregation
        row = {
            "model":          args.model,
            "seed":           args.seed,
            "split_mode":     args.split_mode,
            "retrieval_type": args.retrieval_type,
            "hybrid_alpha":   args.hybrid_alpha,
            "n_train":        len(df_tr),
            "n_val":          len(df_va),
            "n_test":         len(df_te),
            "beats_baseline": beat,
            **metrics,
        }
        pd.DataFrame([row]).to_csv(
            out_dir / f"metrics_{args.model}_seed{args.seed}.csv",
            index=False)
        print(f"✓ Metrics: "
              f"{out_dir}/metrics_{args.model}_seed{args.seed}.csv")


if __name__ == "__main__":
    main()
