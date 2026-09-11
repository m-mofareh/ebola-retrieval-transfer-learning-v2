#!/usr/bin/env python3
import argparse, zipfile, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import pearsonr

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False


def find_col(cols, candidates):
    for cand in candidates:
        for c in cols:
            if cand.lower() in c.lower():
                return c
    return None


def load_ebola(zip_path):
    with zipfile.ZipFile(zip_path, "r") as z:
        csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not csvs:
            raise ValueError("No CSV file found inside ebola.zip")
        with z.open(csvs[0]) as f:
            df = pd.read_csv(f)

    smiles_col = find_col(df.columns, ["smiles", "canonical_smiles", "compound_smiles"])
    seq_col = find_col(df.columns, ["sequence", "protein", "target_sequence", "target"])
    y_col = find_col(df.columns, ["pIC50", "affinity", "value", "label", "y", "IC50"])

    if smiles_col is None or seq_col is None or y_col is None:
        raise ValueError(f"Could not detect required columns. Columns are: {list(df.columns)}")

    df = df[[smiles_col, seq_col, y_col]].copy()
    df.columns = ["SMILES", "Sequence", "y"]
    df = df.dropna()
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["y"])
    df = df[df["y"] > 0]
    df["y"] = 9.0 - np.log10(df["y"].astype(float))

    # If the target column is IC50-like and values are large, convert IC50 nM to pIC50.
    if "ic50" in y_col.lower() and df["y"].median() > 20:
        df = df[df["y"] > 0]
        df["y"] = 9.0 - np.log10(df["y"].astype(float))

    df = df.drop_duplicates(["SMILES", "Sequence", "y"]).reset_index(drop=True)
    return df


def morgan_fp(smiles, n_bits=2048, radius=2):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    ConvertToNumpyArray(fp, arr)
    return arr.astype(np.float32)


AA = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AA)}


def protein_aac(seq):
    seq = str(seq).upper()
    vec = np.zeros(len(AA), dtype=np.float32)
    total = 0
    for ch in seq:
        if ch in AA_IDX:
            vec[AA_IDX[ch]] += 1
            total += 1
    if total > 0:
        vec /= total
    return vec


def protein_kmer3(seq):
    seq = str(seq).upper()
    dim = 512
    vec = np.zeros(dim, dtype=np.float32)
    total = 0
    for i in range(len(seq) - 2):
        k = seq[i:i+3]
        if all(c in AA_IDX for c in k):
            vec[hash(k) % dim] += 1
            total += 1
    if total > 0:
        vec /= total
    return vec


def build_features(df):
    print("Building Morgan fingerprints ...")
    Xc = np.vstack([morgan_fp(s) for s in df["SMILES"]])
    print("Building protein AAC + k-mer features ...")
    Xa = np.vstack([protein_aac(s) for s in df["Sequence"]])
    Xk = np.vstack([protein_kmer3(s) for s in df["Sequence"]])
    return np.hstack([Xc, Xa, Xk]).astype(np.float32), df["y"].values.astype(np.float32)


def metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    p = pearsonr(y_true, y_pred)[0]
    return rmse, mae, r2, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ebola_zip", default="ebola.zip")
    ap.add_argument("--seeds", default="0,1,2,3,42")
    ap.add_argument("--out_csv", default="ebola_only_ml_results.csv")
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    df = load_ebola(args.ebola_zip)
    print(f"Ebola samples after preprocessing: {len(df)}")
    print(f"Target stats: min={df.y.min():.3f}, max={df.y.max():.3f}, mean={df.y.mean():.3f}, std={df.y.std():.3f}")

    X, y = build_features(df)
    rows = []

    for seed in seeds:
        print(f"\n===== Seed {seed} =====")
        idx = np.arange(len(y))
        train_idx, temp_idx = train_test_split(idx, test_size=0.30, random_state=seed)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=seed)

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        models = {
            "RandomForest": RandomForestRegressor(n_estimators=500, random_state=seed, n_jobs=-1, max_features="sqrt"),
            "ExtraTrees": ExtraTreesRegressor(n_estimators=500, random_state=seed, n_jobs=-1, max_features="sqrt"),
            "SVR_RBF": make_pipeline(StandardScaler(with_mean=False), SVR(C=10.0, epsilon=0.1, gamma="scale")),
            "Ridge": make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=1.0)),
        }

        if HAS_XGB:
            models["XGBoost"] = XGBRegressor(
                n_estimators=800, max_depth=6, learning_rate=0.03,
                subsample=0.9, colsample_bytree=0.8,
                objective="reg:squarederror", random_state=seed, n_jobs=-1
            )

        for name, model in models.items():
            print(f"Training {name} ...")
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            rmse, mae, r2, p = metrics(y_test, pred)
            print(f"{name} | RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f} Pearson={p:.4f}")
            rows.append({
                "seed": seed, "model": name, "RMSE": rmse, "MAE": mae,
                "R2": r2, "Pearson": p,
                "train": len(train_idx), "val": len(val_idx), "test": len(test_idx)
            })

    res = pd.DataFrame(rows)
    res.to_csv(args.out_csv, index=False)

    summary = res.groupby("model")[["RMSE", "MAE", "R2", "Pearson"]].agg(["mean", "std"])
    print("\n===== Mean ± Std over seeds =====")
    print(summary)
    summary.to_csv(args.out_csv.replace(".csv", "_summary.csv"))


if __name__ == "__main__":
    main()
