# DNABERT-2 Promoter Expression Prediction

This repository contains scripts for **training, extending, and interpreting DNABERT-2 models** for **core promoter expression prediction** from DNA sequences.

The project is organized around three main workflows:

1. **baseline sequence-only regression**
2. **extended regression with additional biological features**
3. **explainability and interpretation analysis**

The scripts are designed for promoter datasets stored in Excel format and use **6-mer tokenization** with a pretrained **DNABERT-2** model.

---

## Repository overview

### `core_promoter_base_model.py` — baseline DNABERT-2 regression
The first script trains a standard **DNABERT-2 regression model** using promoter sequences as input and expression values as targets.

It performs:
- data loading from Excel
- `log2` transformation of expression values
- 6-mer tokenization
- train/validation split
- model training and evaluation
- export of plots, prediction tables, checkpoints, and run information

Detailed description and usage:  
[`docs/core_promoter_base_model.md`](docs/core_promoter_base_model.md)

---

### `core_promoter_ecd_motifs_model.py` — DNABERT-2 + Ecd + motif features
The second script extends the baseline model by combining **DNABERT-2 sequence information** with additional biological features, including:

- **Ecd presence/absence**
- **motif-derived features**
- optional **gene-wise cross-validation**

It implements a custom hybrid model that merges transformer output with tabular features, and provides a more advanced training setup with additional metrics, TensorBoard logging, and cosine learning-rate scheduling.

Detailed description and usage:  
[`docs/core_promoter_ecd_motifs_model.md`](docs/core_promoter_ecd_motifs_model.md)

---

### `shap_analysis.py` — interpretability and explainability
The third script is a **post-training interpretation pipeline** for a fine-tuned DNABERT-2 model.

It supports:
- SHAP analysis at sequence, token, and k-mer level
- export of token contribution tables
- prediction comparison between real and random control sequences
- token frequency analysis
- optional attention visualization and LIME utilities

This script is intended for **model interpretation**, not training.

Detailed description and usage:  
[`docs/shap_analysis.md`](docs/shap_analysis.md)

---

## Typical workflow

A common workflow in this repository is:

1. train a baseline DNABERT-2 regression model  
2. train the extended Ecd/motif-aware model  
3. interpret the trained model using SHAP and attention-based analyses  

---

## Requirements

The scripts rely on Python packages such as:

- `torch`
- `transformers`
- `scikit-learn`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `openpyxl`

Some scripts also require:

- `scipy`
- `tensorboard`
- `shap`
- `lime`
- `safetensors`

Install the required packages as needed for the script you want to run.

---

## Data and model inputs

Depending on the script, the repository expects:

- a pretrained or fine-tuned **DNABERT-2** model directory
- an Excel dataset containing promoter sequences and expression values
- optionally a motif feature file in TSV format

Please see the corresponding documentation pages for the exact input format and required columns.

---

## Documentation

Detailed script-specific documentation is available in the `docs/` folder:

- [`docs/script1.md`](docs/script1.md)
- [`docs/script2.md`](docs/script2.md)
- [`docs/shap_analysis.md`](docs/shap_analysis.md)

---

## Notes

- Several scripts contain hard-coded paths that should be adapted before running on a new machine.
- Sequence processing is based on **6-mer tokenization**.
- Some scripts support GPU execution and may be computationally intensive, especially the SHAP analysis pipeline.

---

## Author

Christophe Jung
Gene Center Munich / LMU