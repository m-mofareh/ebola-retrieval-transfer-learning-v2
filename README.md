# Retrieval-Guided Transfer Learning for Low-Resource Ebola Drug–Target Affinity Prediction

This repository contains the data, source code, experiment scripts, and selected results associated with the manuscript:

**Retrieval-Guided Transfer Learning for Low-Resource Ebola Drug–Target Affinity Prediction**

**Authors:** Mubarakah Alotaibi and Nada Al Taweraqi

The study investigates whether selecting biologically relevant source-domain interactions from BindingDB can improve transfer learning for drug–target affinity (DTA) prediction in a low-resource Ebola setting.

## Overview

The framework consists of two stages:

1. **Source-domain pretraining:** interactions are selected from BindingDB using random, protein-guided, compound-guided, or hybrid retrieval.
2. **Target-domain fine-tuning:** the pretrained model is fine-tuned using the Ebola bioactivity dataset.

Two deep-learning DTA architectures are evaluated:

- DeepDTA
- GraphDTA

The study also includes conventional machine-learning baselines and single-stage deep-learning baselines for comparison.

## Repository Structure

```text
ebola-retrieval-transfer-learning-v2/
│
├── data/
│   └── ebola_quant.zip
│
├── src/
│   ├── ebola_only_ml_baselines.py
│   ├── single_stage_deepdta_graphdta.py
│   └── deepdta_graphdta_swat_fixed.py
│
├── scripts/
│   ├── run_single_stage_quant.sh
│   └── run_swat_quant.sh
│
├── results/
│   ├── 50k/
│   └── 300k/
│
└── README.md
```

## Ebola Dataset

The final quantitative Ebola dataset contains **2,842 drug–target interaction records**.

Quantitative bioactivity measurements were collected from publicly available sources, including ChEMBL and PubChem. IC50 and EC50 measurements were standardized to a common pAffinity scale:

```text
pAffinity = -log10(Activity [M])
```

The target-domain dataset covers six Ebola protein targets:

- L protein (RNA-dependent RNA polymerase; RdRp)
- Nucleoprotein (NP)
- Envelope glycoprotein (GP)
- Truncated glycoprotein (GP-trunc)
- Secreted glycoprotein (sGP)
- VP35 polymerase cofactor

The processed quantitative dataset used in the experiments is provided under `data/`.

## Source Domain

BindingDB was used as the source domain for transfer learning.

Source interactions were selected using the following strategies:

- **Random retrieval**
- **Protein-guided retrieval**
- **Compound-guided retrieval**
- **Hybrid retrieval**

The principal source-data budgets evaluated in the study were **50k** and **300k** interactions.

Held-out Ebola compounds were excluded from the source-domain pool as part of the leakage-control procedure.

## Models

### Machine-Learning Baselines

Conventional machine-learning models are implemented in:

```text
src/ebola_only_ml_baselines.py
```

### Single-Stage Deep Learning

DeepDTA and GraphDTA trained directly on the Ebola target-domain data are implemented in:

```text
src/single_stage_deepdta_graphdta.py
```

### Retrieval-Guided Transfer Learning

The two-stage retrieval-guided DeepDTA and GraphDTA experiments are implemented in:

```text
src/deepdta_graphdta_swat_fixed.py
```

## Experimental Seeds

The principal experiments were repeated using five random seeds:

```text
0, 1, 2, 3, 42
```

Reported results in the manuscript are aggregated across these five runs.

## Software Environment

The experiments were conducted using:

- Python 3.12.3
- PyTorch 2.12.0
- NumPy 2.4.6
- CUDA 13.0
- cuDNN 9.2.0

Experiments were executed on an NVIDIA Tesla T4 GPU with 16 GB GPU memory.

MMseqs2 was additionally used for post-hoc sequence-similarity characterization of the retrieved source proteins.

## Running the Experiments

The shell scripts used to launch the main experiments are provided in the `scripts/` directory.

For the single-stage experiments:

```bash
bash scripts/run_single_stage_quant.sh
```

For the retrieval-guided transfer-learning experiments:

```bash
bash scripts/run_swat_quant.sh
```

Paths and environment-specific settings may need to be adjusted according to the local computing environment.

## Results

Selected experimental outputs are provided in the `results/` directory.

The main evaluation metrics reported in the manuscript include:

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- coefficient of determination (R²)
- Pearson correlation coefficient

The best overall configuration was obtained with **protein-guided GraphDTA using a 50k source-data budget**, achieving an average RMSE of **0.5498 ± 0.1073** across five seeds.

## Reproducibility

The repository is intended to provide the principal materials required to reproduce the experiments reported in the manuscript. Because some source-domain data originate from external public databases, users should also consult the corresponding database terms and source records when reconstructing the complete source-domain dataset.

## Data Sources

- BindingDB: https://www.bindingdb.org/
- ChEMBL: https://www.ebi.ac.uk/chembl/
- PubChem: https://pubchem.ncbi.nlm.nih.gov/

## Citation

If you use this repository, please cite the associated manuscript:

> Alotaibi, M.; Al Taweraqi, N. Retrieval-Guided Transfer Learning for Low-Resource Ebola Drug–Target Affinity Prediction.

Full publication details and DOI will be added after publication.

## License

Please refer to the licenses and terms of use of the original public databases for externally sourced bioactivity data.
