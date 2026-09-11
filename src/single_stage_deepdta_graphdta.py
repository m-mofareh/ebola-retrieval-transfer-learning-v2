"""
Ebola-only DeepDTA / GraphDTA (no BindingDB retrieval)
Same architectures and scaffold split as the SWAT retrieval pipeline.
Used as an ablation baseline — shows what the models achieve without
transfer learning.

CHANGES vs. original:
  1. load_ebola_data() replaces load_ebola_ic50()
     - Reads ebola_dti_ready.csv (IC50+EC50+Ki+Kd as pAffinity)
     - All 590 deduplicated rows used (was IC50-only before)
  2. --seed exposed as CLI argument so shell script can sweep seeds
  3. Saves per-seed metrics to CSV for easy aggregation across seeds

Usage:
  # Single model, single seed
  python single_stage_deepdta_graphdta.py \
    --model deepdta --seed 42 --output_dir ebola_only_deepdta

  # Both models, single seed
  python single_stage_deepdta_graphdta.py \
    --model both --seed 42 --output_dir ebola_only_results

  # Multi-seed sweep via shell script (see run_single_stage.sh)
  bash run_single_stage.sh
"""

import argparse
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


# ============================================================================
# CONSTANTS
# ============================================================================

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
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
# SCAFFOLD SPLIT — identical to ML baselines and SWAT pipeline
# ============================================================================

def make_grouped_split(df, seed, mode="scaffold"):
    """
    Leakage-resistant 70/15/15 split by Bemis-Murcko scaffold.
    Identical logic and seeding to ebola_ml_baselines.py and
    deepdta_graphdta_protien4.py so all three pipelines share
    the same train/val/test rows at each seed.
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
# DATA LOADING — all assay types as pAffinity
# ============================================================================

def load_ebola_data(zip_path="ebola.zip", out_dir="ebola_data"):
    """
    Load Ebola bioactivity data (IC50 + EC50 + Ki + Kd) as pAffinity.
    Reads from ebola_dti_ready.csv (produced by build_ebola_dataset.py).
    Falls back to individual split files for backward compatibility.
    """
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out)

    # --- Primary: unified DTI-ready file ---
    dti_files = list(out.rglob("ebola_dti_ready.csv"))
    if dti_files:
        print(f"  Loading: {dti_files[0].name}")
        df = pd.read_csv(dti_files[0])
        df = df[["canonical_smiles", "sequence", "pAffinity"]].copy()
        df.columns = ["SMILES", "Sequence", "pIC50"]
    else:
        # --- Fallback: load all split files ---
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
                f"No Ebola data files found in {out}. "
                "Run build_ebola_dataset.py first.")
        df = pd.concat(all_dfs, ignore_index=True)

    # --- Clean and canonicalize ---
    df["SMILES"]   = df["SMILES"].map(canonicalize_smiles)
    df["Sequence"] = df["Sequence"].astype(str).str.upper().str.strip()
    df = df.dropna(subset=["SMILES", "Sequence", "pIC50"])
    df = df[np.isfinite(df["pIC50"])]

    # Median-aggregate duplicate (SMILES, Sequence) pairs
    df = (df.groupby(["SMILES", "Sequence"], as_index=False)["pIC50"]
            .median()
            .reset_index(drop=True))

    n_low = int((df["pIC50"] < 4.0).sum())
    print(f"  Ebola dataset: {len(df)} rows "
          f"(IC50+EC50+Ki+Kd pooled as pAffinity)")
    print(f"  pIC50 range : {df['pIC50'].min():.3f} — {df['pIC50'].max():.3f}")
    print(f"  pIC50 mean  : {df['pIC50'].mean():.3f}  "
          f"std: {df['pIC50'].std():.3f}")
    print(f"  Rare pIC50 < 4.0: {n_low} kept (no cutoff)")
    return df


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
            if x not in allowable:
                x = allowable[-1]
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
                one_of_k(_get_implicit_valence(atom), [0,1,2,3,4,5,6,7,8,9,10]) +
                [atom.GetIsAromatic()],
                dtype=np.float32
            )

        node_f = np.array([atom_features(a) for a in mol.GetAtoms()],
                          dtype=np.float32)
        edges = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edges += [[i, j], [j, i]]
        if not edges:
            edges = [[0, 0]]
        edge_index = np.array(edges, dtype=np.int64).T
        return node_f, edge_index
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
        return (
            torch.tensor(self.smiles_enc[i],  dtype=torch.long),
            torch.tensor(self.protein_enc[i], dtype=torch.long),
            torch.tensor(self.y[i],           dtype=torch.float32),
        )


class GraphDTADataset(Dataset):
    def __init__(self, df, y_mean=0., y_std=1.,
                 node_mean=None, node_std=None):
        y_raw       = df["pIC50"].values.astype(np.float32)
        smiles_list = df["SMILES"].tolist()
        seq_list    = df["Sequence"].tolist()
        graphs_flat = [smiles_to_graph(s) for s in smiles_list]

        self.graphs      = []
        self.protein_enc = []
        self.y           = []
        for i, g in enumerate(graphs_flat):
            if g is None:
                continue
            self.graphs.append(g)
            self.protein_enc.append(encode_protein(seq_list[i]))
            self.y.append((y_raw[i] - y_mean) / y_std)
        self.y = np.array(self.y, dtype=np.float32)

        if node_mean is None:
            all_nodes      = np.concatenate([nf for nf, _ in self.graphs], axis=0)
            self.node_mean = all_nodes.mean(0, keepdims=True)
            self.node_std  = all_nodes.std(0,  keepdims=True) + 1e-8
        else:
            self.node_mean = node_mean
            self.node_std  = node_std

        self.graphs = [
            ((nf - self.node_mean) / self.node_std, ei)
            for nf, ei in self.graphs
        ]
        dropped = len(smiles_list) - len(self.y)
        print(f"  GraphDTA: {len(self.y)} valid / {len(smiles_list)} input "
              f"({dropped} dropped)")

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        node_f, edge_index = self.graphs[i]
        return (
            torch.tensor(node_f,              dtype=torch.float32),
            torch.tensor(edge_index,          dtype=torch.long),
            torch.tensor(self.protein_enc[i], dtype=torch.long),
            torch.tensor(self.y[i],           dtype=torch.float32),
        )


def graph_collate(batch):
    node_fs, edge_indices, proteins, ys = zip(*batch)
    offset, batch_node_f, batch_edge_idx, batch_assign = 0, [], [], []
    for b, (nf, ei) in enumerate(zip(node_fs, edge_indices)):
        nf_t = nf.float() if isinstance(nf, torch.Tensor) else torch.tensor(nf, dtype=torch.float32)
        ei_t = ei.long()  if isinstance(ei, torch.Tensor) else torch.tensor(ei, dtype=torch.long)
        batch_node_f.append(nf_t)
        batch_edge_idx.append(ei_t + offset)
        batch_assign.extend([b] * len(nf_t))
        offset += len(nf_t)
    return (
        torch.cat(batch_node_f,   dim=0),
        torch.cat(batch_edge_idx, dim=1),
        torch.tensor(batch_assign, dtype=torch.long),
        torch.stack([p.clone().detach().to(torch.long)    for p in proteins]),
        torch.stack([y.clone().detach().to(torch.float32) for y in ys]),
    )


# ============================================================================
# MODELS
# ============================================================================

class DeepDTA(nn.Module):
    def __init__(self,
                 smiles_vocab_size  = len(SMILES_CHARS) + 1,
                 protein_vocab_size = len(PROTEIN_CHARS) + 1,
                 smiles_embed_dim   = 128,
                 protein_embed_dim  = 128,
                 num_filters        = 32,
                 smiles_kernel      = 8,
                 protein_kernel     = 12,
                 fc_dim             = 1024,
                 dropout            = 0.1):
        super().__init__()

        self.smiles_embed = nn.Embedding(smiles_vocab_size, smiles_embed_dim, padding_idx=0)
        self.smiles_norm  = nn.LayerNorm(smiles_embed_dim)
        self.smiles_conv  = nn.Sequential(
            nn.Conv1d(smiles_embed_dim, num_filters,     smiles_kernel),
            nn.BatchNorm1d(num_filters), nn.ReLU(),
            nn.Conv1d(num_filters,      num_filters * 2, smiles_kernel),
            nn.BatchNorm1d(num_filters * 2), nn.ReLU(),
            nn.Conv1d(num_filters * 2,  num_filters * 3, smiles_kernel),
            nn.BatchNorm1d(num_filters * 3), nn.ReLU(),
        )
        self.protein_embed = nn.Embedding(protein_vocab_size, protein_embed_dim, padding_idx=0)
        self.protein_norm  = nn.LayerNorm(protein_embed_dim)
        self.protein_conv  = nn.Sequential(
            nn.Conv1d(protein_embed_dim, num_filters,     protein_kernel),
            nn.BatchNorm1d(num_filters), nn.ReLU(),
            nn.Conv1d(num_filters,       num_filters * 2, protein_kernel),
            nn.BatchNorm1d(num_filters * 2), nn.ReLU(),
            nn.Conv1d(num_filters * 2,   num_filters * 3, protein_kernel),
            nn.BatchNorm1d(num_filters * 3), nn.ReLU(),
        )
        fused = num_filters * 3 * 2
        self.fc = nn.Sequential(
            nn.LayerNorm(fused),
            nn.Linear(fused, fc_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(fc_dim, fc_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(fc_dim // 2, 1),
        )
        self.calibration = nn.Linear(1, 1)
        nn.init.ones_(self.calibration.weight)
        nn.init.zeros_(self.calibration.bias)

    def forward(self, smiles, protein):
        d = self.smiles_norm(self.smiles_embed(smiles))
        d = F.adaptive_max_pool1d(self.smiles_conv(d.permute(0,2,1)), 1).squeeze(-1)
        p = self.protein_norm(self.protein_embed(protein))
        p = F.adaptive_max_pool1d(self.protein_conv(p.permute(0,2,1)), 1).squeeze(-1)
        return self.calibration(self.fc(torch.cat([d, p], dim=1)))


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
        x_out = self.linear((x + agg) / deg.clamp(min=1).unsqueeze(-1))
        return F.relu(self.bn(x_out))


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
                 node_feat_dim      = 78,
                 gcn_hidden         = 64,
                 gcn_out            = 128,
                 protein_vocab_size = len(PROTEIN_CHARS) + 1,
                 protein_embed_dim  = 128,
                 num_filters        = 32,
                 protein_kernel     = 12,
                 fc_dim             = 1024,
                 dropout            = 0.1):
        super().__init__()
        self.gcn1    = GCNLayer(node_feat_dim, gcn_hidden)
        self.gcn2    = GCNLayer(gcn_hidden,    gcn_hidden)
        self.gcn3    = GCNLayer(gcn_hidden,    gcn_out)
        self.drug_fc = nn.Linear(gcn_out * 2,  gcn_out)
        self.drug_bn = nn.BatchNorm1d(gcn_out)

        self.protein_embed = nn.Embedding(protein_vocab_size, protein_embed_dim, padding_idx=0)
        self.protein_conv  = nn.Sequential(
            nn.Conv1d(protein_embed_dim, num_filters,     protein_kernel),
            nn.BatchNorm1d(num_filters), nn.ReLU(),
            nn.Conv1d(num_filters,       num_filters * 2, protein_kernel),
            nn.BatchNorm1d(num_filters * 2), nn.ReLU(),
            nn.Conv1d(num_filters * 2,   num_filters * 3, protein_kernel),
            nn.BatchNorm1d(num_filters * 3), nn.ReLU(),
        )
        fused = gcn_out + num_filters * 3
        self.fc = nn.Sequential(
            nn.LayerNorm(fused),
            nn.Linear(fused, fc_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(fc_dim, fc_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(fc_dim // 2, 1),
        )
        self.calibration = nn.Linear(1, 1)
        nn.init.ones_(self.calibration.weight)
        nn.init.zeros_(self.calibration.bias)

    def forward(self, node_f, edge_index, batch, protein):
        x = self.gcn3(self.gcn2(self.gcn1(node_f, edge_index, batch),
                                 edge_index, batch), edge_index, batch)
        ng = batch.max().item() + 1
        sp = torch.zeros(ng, x.size(1), device=x.device)
        ct = torch.zeros(ng, 1,         device=x.device)
        sp.index_add_(0, batch, x)
        ct.index_add_(0, batch, torch.ones(batch.size(0), 1, device=x.device))
        d = torch.cat([sp / ct.clamp(min=1), graph_max_pool(x, batch, ng)], dim=1)
        d = F.relu(self.drug_bn(self.drug_fc(d)))
        p = F.adaptive_max_pool1d(
            self.protein_conv(self.protein_embed(protein).permute(0,2,1)), 1).squeeze(-1)
        return self.calibration(self.fc(torch.cat([d, p], dim=1)))


def build_model(model_type, device):
    m = DeepDTA() if model_type == "deepdta" else GraphDTA()
    return m.to(device)


# ============================================================================
# TRAINING LOOP
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
            ps.extend((pred.cpu().numpy().ravel() * y_std + y_mean).tolist())
            ys.extend((y.numpy().ravel() * y_std + y_mean).tolist())
    return calc_metrics(ys, ps)


def run_epochs(model, tr_dl, va_dl, device, model_type,
               epochs, lr, wd, patience, label, y_mean=0., y_std=1.):
    opt     = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=wd)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=lr / 20)
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


def make_loaders(df_tr, df_va, df_te, model_type, y_mean, y_std, batch_size):
    pin = torch.cuda.is_available()
    if model_type == "deepdta":
        kw = dict(num_workers=0, pin_memory=pin)
        return (
            DataLoader(DeepDTADataset(df_tr, y_mean, y_std),
                       batch_size=batch_size, shuffle=True,  **kw),
            DataLoader(DeepDTADataset(df_va, y_mean, y_std),
                       batch_size=batch_size, shuffle=False, **kw),
            DataLoader(DeepDTADataset(df_te, y_mean, y_std),
                       batch_size=batch_size, shuffle=False, **kw),
        )
    else:
        kw = dict(num_workers=0, pin_memory=pin, collate_fn=graph_collate)
        tr_ds = GraphDTADataset(df_tr, y_mean, y_std)
        va_ds = GraphDTADataset(df_va, y_mean, y_std,
                                node_mean=tr_ds.node_mean,
                                node_std=tr_ds.node_std)
        te_ds = GraphDTADataset(df_te, y_mean, y_std,
                                node_mean=tr_ds.node_mean,
                                node_std=tr_ds.node_std)
        return (
            DataLoader(tr_ds, batch_size=batch_size, shuffle=True,  **kw),
            DataLoader(va_ds, batch_size=batch_size, shuffle=False, **kw),
            DataLoader(te_ds, batch_size=batch_size, shuffle=False, **kw),
        )


# ============================================================================
# TRAIN ONE MODEL
# ============================================================================

def train_one_model(model_type, df_ebola, args, device, out_dir):
    print("\n" + "="*70)
    print(f"EBOLA-ONLY  {model_type.upper()}  seed={args.seed}")
    print("="*70)

    tr_idx, va_idx, te_idx, groups = make_grouped_split(
        df_ebola, args.seed, args.split_mode)
    df_tr = df_ebola.iloc[tr_idx].reset_index(drop=True)
    df_va = df_ebola.iloc[va_idx].reset_index(drop=True)
    df_te = df_ebola.iloc[te_idx].reset_index(drop=True)

    print(f"Split ({args.split_mode}): "
          f"train={len(df_tr)}, val={len(df_va)}, test={len(df_te)}")
    print(f"  y_tr: mean={df_tr['pIC50'].mean():.3f}  "
          f"std={df_tr['pIC50'].std():.3f}")
    print(f"  y_te: mean={df_te['pIC50'].mean():.3f}  "
          f"std={df_te['pIC50'].std():.3f}")

    # Save split manifest (one per seed/model)
    manifest = df_ebola.copy()
    manifest["split"] = "train"
    manifest.loc[va_idx, "split"] = "validation"
    manifest.loc[te_idx, "split"] = "test"
    manifest["scaffold_group"] = groups
    manifest["seed"] = args.seed
    manifest.to_csv(
        out_dir / f"split_manifest_{model_type}_seed{args.seed}.csv",
        index=False)

    y_mean = float(df_tr["pIC50"].mean())
    y_std  = float(df_tr["pIC50"].std()) + 1e-8

    print("Building datasets ...")
    tr_dl, va_dl, te_dl = make_loaders(
        df_tr, df_va, df_te, model_type, y_mean, y_std, args.batch_size)

    model   = build_model(model_type, device)
    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_param:,}")

    print(f"\nTraining ({args.epochs} epochs, lr={args.lr}) ...")
    t0 = time.time()
    run_epochs(model, tr_dl, va_dl, device, model_type,
               epochs=args.epochs, lr=args.lr, wd=args.wd,
               patience=args.patience, label=f"{model_type}_s{args.seed}",
               y_mean=y_mean, y_std=y_std)
    elapsed = time.time() - t0
    print(f"[TIMING] {elapsed:.1f}s ({elapsed/60:.2f} min)")

    metrics = eval_model(model, te_dl, device, model_type, y_mean, y_std)

    print("\n" + "="*70)
    print(f"Results  {model_type.upper()}  seed={args.seed}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    beat = metrics["RMSE"] < BASELINE_RMSE
    print(f"\n  Baseline RMSE: {BASELINE_RMSE}")
    print(f"  {'✓ BEATS BASELINE' if beat else '✗ Below baseline'} "
          f"({metrics['RMSE']:.4f})")
    print("="*70)

    # Save model weights
    ckpt = out_dir / f"{model_type}_seed{args.seed}.pt"
    torch.save(model.state_dict(), ckpt)
    print(f"✓ Saved: {ckpt}")

    # Save per-seed metrics row
    row = {
        "model":      model_type,
        "seed":       args.seed,
        "split_mode": args.split_mode,
        "n_train":    len(df_tr),
        "n_val":      len(df_va),
        "n_test":     len(df_te),
        "beats_baseline": beat,
        **metrics,
    }
    row_df = pd.DataFrame([row])
    metrics_csv = out_dir / f"metrics_{model_type}_seed{args.seed}.csv"
    row_df.to_csv(metrics_csv, index=False)
    print(f"✓ Metrics: {metrics_csv}")

    return row


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ebola-only DeepDTA/GraphDTA ablation (scaffold split, pAffinity)")

    parser.add_argument("--model", required=True,
                        choices=["deepdta", "graphdta", "both"])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ebola_zip",  default="ebola.zip")
    parser.add_argument("--split_mode", default="scaffold",
                        choices=["scaffold", "cold_drug"])

    # Seed — primary CLI argument for shell script sweeps
    parser.add_argument("--seed", type=int, default=3,
                        help="Random seed. Pass different seeds from shell "
                             "script to sweep (e.g. 42 1 2 3 0)")

    # Training hyperparameters
    parser.add_argument("--epochs",     type=int,   default=200)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--wd",         type=float, default=1e-4)
    parser.add_argument("--patience",   type=int,   default=40)
    parser.add_argument("--batch_size", type=int,   default=64)

    args   = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device     : {device}")
    print(f"Model      : {args.model}")
    print(f"Seed       : {args.seed}")
    print(f"Split mode : {args.split_mode}")

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    df_ebola    = load_ebola_data(args.ebola_zip)
    model_types = ["deepdta", "graphdta"] if args.model == "both" else [args.model]

    all_rows = []
    for mt in model_types:
        set_seed(args.seed)
        row = train_one_model(mt, df_ebola, args, out_dir=out_dir,
                               device=device)
        all_rows.append(row)

    # Aggregate summary if both models ran
    if len(all_rows) > 1:
        pd.DataFrame(all_rows).to_csv(
            out_dir / f"comparison_seed{args.seed}.csv", index=False)


if __name__ == "__main__":
    main()
