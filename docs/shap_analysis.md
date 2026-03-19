This script performs model interpretation and sequence-level explanation analysis for a fine-tuned DNABERT-2 regression model. It is designed to help understand which sequence regions, k-mers, and token positions contribute most strongly to model predictions.

Compared with the first two scripts, this one is not about model fitting. Its role is to answer questions like:
-Which k-mers contribute most strongly to predictions?
-Which token positions receive the most attention?
-Do real promoter sequences show different attention patterns from random DNA?
-Which motifs or local contexts repeatedly appear among important regions?
-How does the model behave on a representative subset of real sequences?

So this is best described as an interpretation / explainability / post-training analysis script.

It works on a trained DNABERT-2 model and a promoter dataset, then applies a combination of:
-SHAP explanations
-attention extraction and visualization
-token frequency analysis
-prediction export for real and random control sequences
-optional LIME-based k-mer explanation

The main purpose of the script is not training, but interpreting an already fine-tuned model.

What this script does:
1. Loads a fine-tuned DNABERT-2 model
The script loads a DNABERT-2 sequence classification/regression model from a model directory:
model_dir = "/path/to/DNABERT-2-117M_base_model_20250923_144123"

So this script assumes that training has already been completed and that a usable fine-tuned model  exists.

2. Loads promoter sequences from an Excel dataset
The script reads an Excel file:
excel_path = "/path/to/core_promoter_training_data.xlsx"
It extracts at least these columns:
SequenceID
NORM
SequenceSample

It then filters out invalid or empty sequences

3. Randomly samples sequences for analysis
Because SHAP and attention analysis are computationally expensive, the script randomly selects a subset of sequences:
n_random = 2592

These sampled sequences are used for downstream interpretation.
The selected set is exported to a tab-separated file so the analyzed subset is recorded.

This is useful because SHAP on thousands of sequences is slow and expensive, and this script is clearly designed to analyze a representative subset, not necessarily the whole dataset.

4. Predicts model outputs for selected sequences and random controls
The script generates predictions for the selected promoter sequences and a small set of randomly generated control DNA sequences.

These predictions are exported to a text file so you can compare real promoter predictions with random-sequence baseline predictions

This gives a simple sanity check on whether the model treats real and random DNA differently.



This script contains several interpretability components:

A. SHAP analysis
The SHAP-related functions are a major part of the script.
SHAP is used to estimate how much each token or k-mer contributes to the model’s prediction.

The script runs SHAP in several ways:
1. Per-sequence SHAP summary plots
Function:
explain_with_shap(...)

For each selected sequence, the script converts the sequence into k-mer token IDs, builds a random background of synthetic DNA sequences, runs a SHAP explainer and saves a SHAP summary plot

This gives a local explanation for individual sequences.

2. Aggregated SHAP by unique k-mer
Function:
batch_explain_with_shap_and_summary(...)

This function goes beyond per-sequence explanations. It computes SHAP values across multiple sequences

aggregates contributions by unique token / k-mer, computes the mean SHAP value for each k-mer, plots a summary figure and exports a text file with:
k-mer
mean SHAP value
number of occurrences
individual SHAP values

This is useful for identifying which k-mers are globally most influential across the analyzed dataset.

3. Global token-level SHAP contribution export
Function:
export_token_shap_contributions(...)

This function aggregates SHAP values by token and writes a summary table:
token
mean SHAP contribution
count

This gives a compact global view of which tokens tend to increase or decrease predictions.

4. Per-token SHAP values for each sequence
Function:
export_per_token_shap_values(...)

This exports a detailed table of:
sequence index
token index
token identity
SHAP value(s)

This is useful if you want to perform custom downstream analysis outside Python, for example in Excel, R, or a later plotting script.

B. Attention analysis
The script also includes a substantial attention visualization pipeline.
It extracts transformer attention weights and visualizes them as heatmaps.

This computes:
average attention over random control sequences
average attention over promoter sequences
difference maps between promoter and random controls

This can help highlight attention patterns that are more specific to real promoter sequences.

Important note: in the current main() script, most of the attention-analysis calls are commented out, so the functionality exists, but is not fully active by default.

C. Token frequency analysis
Function:
export_most_frequent_tokens(...)

This exports the most frequent k-mer tokens

checking whether frequent tokens also appear among the most influential SHAP contributors

D. LIME analysis
The script also contains an experimental LIME-based explanation mode:
explain_with_lime_kmer_tabular(...)
batch_explain_with_lime(...)

This attempts to explain predictions at the k-mer presence/absence level using LIME.

Main workflow in main();

The main function performs the following steps:
load the fine-tuned model and tokenizer
load the Excel dataset
extract valid sequence/name/expression triplets
randomly sample n_random sequences for analysis
save the selected sequences to file
export the most frequent tokens 
generate a few random control DNA sequences
run model predictions for selected + random sequences
export predictions to a text file
run SHAP explanations on the selected sequences
export:
-SHAP summary plots
-aggregated k-mer SHAP summaries
-token contribution tables
-per-token SHAP tables

Several attention-analysis and LIME-analysis steps are present but commented out.
So in its current active state, this script is mostly a SHAP-centered interpretability workflow with auxiliary prediction and token statistics exports.

Main input files
1. Fine-tuned model directory
model_dir = "/path/to/fine_tuned_model_directory"
This should contain the model configuration, tokenizer files, and fine-tuned weights.

2. Excel dataset
excel_path = "/path/to/core_promoter_training_data.xlsx"

Expected columns include:
SequenceID
NORM
SequenceSample

Main parameters:
At the top of the script:

model_dir = "/path/to/model"
export_path = "/path/to/results/"
excel_path = "/path/to/data.xlsx"
SeqLength = 130
n_random = 2592

model_dir — directory containing the fine-tuned model and tokenizer
export_path — output directory for plots and text files
excel_path — dataset file used to select sequences for interpretation
SeqLength — fixed padded token length
n_random — number of randomly sampled real sequences used for analysis


Usage:
Install dependencies
pip install torch transformers shap lime safetensors pandas numpy matplotlib seaborn openpyxl

Depending on your setup, you may also need:

pip install scipy
Configure paths

Edit these variables:

model_dir = "/path/to/fine_tuned_model"
export_path = "/path/to/results/"
excel_path = "/path/to/core_promoter_training_data.xlsx"

Also review any hard-coded paths later in the script, especially those beginning with /home/... or /home/be-em/....

Run
python shap_analysis.py
