# -*- coding: utf-8 -*-
"""
Load a trained promoter model.
Predict expression for new promoter sequences.
Save plots, tables, and score values.
"""

#  Notes
# - Main settings are near the top of this file.
# - Change MODEL_DIR, EXCEL_INPUT, RESULTS_DIR, and motif_feature_file before running.
# - Run this file after a model has already been trained.


import os, time
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from transformers import AutoTokenizer, BertConfig, AutoModelForSequenceClassification
import torch.nn as nn

# User settings
MODEL_DIR = r"/.../Core_Promoter_2015/results/" \
"DNABERT-2-117M_model_20260512_172646"  # Set this to the folder that holds the trained model files.
# Optional example; edit and uncomment only if you need it.
# Optional example; edit and uncomment only if you need it.


# Optional example; edit and uncomment only if you need it.
# Optional example; edit and uncomment only if you need it.
EXCEL_INPUT = r"/.../Fly_enhancer_screens_all_CPs_normalized_tagcount_merged_core_prom.xlsx"
# Optional example; edit and uncomment only if you need it.
  # Set this to the Excel file with new data.
RESULTS_DIR = r"/.../results"
# Motif score values used by the model.
motif_feature_file = r"/.../core_promoter_training_data_250604_truncated.tsv"

MAX_LEN = 130  # Longest sequence length used by the model.
BATCH_SIZE = 28
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model class
class ModifiedModel(nn.Module):
    def __init__(self, base_model, hidden_size, num_motif_score_columns):
        super(ModifiedModel, self).__init__()
        self.base_model = base_model
        self.base_projection = nn.Linear(1, hidden_size)
        self.batch_norm1_base = nn.BatchNorm1d(hidden_size)
        self.ecd_embedding = nn.Embedding(2, hidden_size // 2)
        self.all_motifs_projection = nn.Linear(num_motif_score_columns, hidden_size // 2)
        self.additional_layer1 = nn.Linear(hidden_size * 2, hidden_size)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.additional_layer2 = nn.Linear(hidden_size, hidden_size // 2)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size // 2)
        self.dropout = nn.Dropout(p=0.00001)
        self.classifier = nn.Linear(hidden_size // 2, 1)
        # Set starting values for model weights.
        nn.init.xavier_uniform_(self.base_projection.weight)
        nn.init.xavier_uniform_(self.ecd_embedding.weight)
        nn.init.xavier_uniform_(self.all_motifs_projection.weight)
        nn.init.xavier_uniform_(self.additional_layer1.weight)
        nn.init.xavier_uniform_(self.additional_layer2.weight)
        nn.init.xavier_uniform_(self.classifier.weight)
    def forward(self, input_ids, ecd_feature, all_motif_scores):
        base_output = self.base_model(input_ids).logits
        base_output = self.base_projection(base_output)
        base_output = self.batch_norm1_base(base_output)
        ecd_embedded = self.ecd_embedding(ecd_feature).squeeze(1)
        all_motifs_projected = self.all_motifs_projection(all_motif_scores)
        combined_input = torch.cat((base_output, ecd_embedded, all_motifs_projected), dim=1)
        output = torch.relu(self.additional_layer1(combined_input))
        output = self.batch_norm1(output)
        output = torch.relu(self.additional_layer2(output))
        output = self.batch_norm2(output)
        output = self.dropout(output)
        output = self.classifier(output)
        return {'logits': output}

# Load the model and tokenizer
def load_model_and_tokenizer(model_dir, motif_feature_columns):
    # Set the path to the original DNABERT base model directory
    BASE_DNABERT_DIR = r"/home/be-em/data/Core_Promoter_2015/DNABERT-2-117M_model"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    config = BertConfig.from_pretrained(BASE_DNABERT_DIR)
    config.hidden_dropout_prob = 0.00001
    config.attention_probs_dropout_prob = 0.00001
    config.num_labels = 1
    # Load base DNABERT model only (not fine-tuned weights)
    base_model = AutoModelForSequenceClassification.from_pretrained(BASE_DNABERT_DIR, config=config, trust_remote_code=True)
    hidden_size = base_model.config.hidden_size
    num_motif_score_columns = len(motif_feature_columns)
    # Instantiate custom model
    model = ModifiedModel(base_model, hidden_size, num_motif_score_columns)
    # Load fine-tuned weights for custom model
    state_dict = torch.load(os.path.join(model_dir, "pytorch_model.bin"), map_location=DEVICE, weights_only=True)
    print(f"Loaded state_dict keys: {list(state_dict.keys())[:5]} ... total: {len(state_dict)}")
    print(f"Model expects all_motifs_projection.weight shape: {model.all_motifs_projection.weight.shape}")
    model.load_state_dict(state_dict)
    print(f"Model loaded with weights from: {model_dir}")
    model.to(DEVICE)
    model.eval()
    return model, tokenizer

# Prepare the data
def prepare_input_data(excel_path):
    df = pd.read_excel(excel_path)
    # Rename input columns to the names this script expects.
    df = df.rename(columns={'sequence': 'Sequence'})
    
    # Optional example; edit and uncomment only if you need it.
    df = df.rename(columns={'no_enh_Rep1': 'Expression'})
    # Optional example; edit and uncomment only if you need it.

    # Optional example; edit and uncomment only if you need it.
    df = df.rename(columns={'oligo_id': 'SequenceID'})
    # Optional example; edit and uncomment only if you need it.
    
    if 'Sequence' not in df.columns or 'Expression' not in df.columns:
        raise ValueError("Input Excel must have 'Sequence' and 'Expression' columns.")
    if len(df) == 0:
        raise ValueError("Input Excel contains no valid rows after filtering. Check your input file.")  
    # Optional example; edit and uncomment only if you need it.

    # Sequence cleaning (as in base model)
    df['Sequence'] = df['Sequence'].apply(lambda seq: seq[31:] if isinstance(seq, str) else seq)
    # Optional example; edit and uncomment only if you need it.

    # Optional sequence trimming step.
    df['Sequence'] = df['Sequence'].astype(str).str[0:]

    # Keep only rows that pass this rule.
    norm_threshold = -20.0  # Set your desired threshold value here
    df = df[df['Expression'].astype(float)> norm_threshold].copy()
    # Convert expression values to log2 scale.
    df['Expression'] = np.log2(df['Expression'].astype(float))
    df['Expression'] = df['Expression']+0.45 # Scale expression values to match training data
    # Make sure the sequence is text.
    if 'Ecd' not in df.columns:
        df['Ecd'] =1
    else:
        # Convert 'Yes'/'No' to 1/0, or any string to 0, else keep as int
        df['Ecd'] = df['Ecd'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else (0 if str(x).strip().lower() == 'no' else int(x) if str(x).isdigit() else 0))
    # Motif score values used by the model.
    motif_df_full = pd.read_csv(motif_feature_file, sep='\t', dtype={"motif_id": str, "sequence_name": str, "strand": str, "score": float, "p-value": float})
    motif_df_full = motif_df_full[motif_df_full['strand'] != '-']
    # Find the motif columns used by the model.
    motif_score_columns_main = ['Block1', 'Block7']
    motif_ids = sorted(set(motif_df_full['motif_id'].unique()) - set(motif_score_columns_main))
    motif_ids_score = [m + '_score' if not m.endswith('_score') else m for m in motif_ids]
    motif_score_columns = ['Block1', 'Block7'] + motif_ids_score
    # Keep the selected motif score columns.
    motif_score_columns = motif_score_columns[ :2]
    
    # Keep motif rows that pass the p-value cutoff.
    pvalue_threshold = 0.0001  # Set your desired threshold here
    motif_df = motif_df_full[motif_df_full['p-value'] < pvalue_threshold].copy()
    print(f"Filtered motif_df by p-value < {pvalue_threshold}, remaining motifs: {len(motif_df)}")

    motif_scores_list = []
    for idx, row in df.iterrows():
        sample_id = row['SampleID'] if 'SampleID' in row else row['SequenceID']
        sample_id = str(sample_id)
        motif_rows = motif_df[motif_df['sequence_name'].astype(str) == sample_id]
        motif_scores = {}
        for motif_id, motif_id_score in zip(motif_ids, motif_ids_score):
            scores = motif_rows[motif_rows['motif_id'] == motif_id]['score']
            if not scores.empty:
                motif_scores[motif_id_score] = scores.astype(float).max()
            else:
                motif_scores[motif_id_score] = 0.0
        motif_scores_list.append(motif_scores)
    # Make a table and save it to Excel.
    motif_scores_df = pd.DataFrame(motif_scores_list, index=df.index)
    # Set Block1 and Block7 values used by this model.
    for col in ['Block1', 'Block7']:
       motif_scores_df[col] = 15
    # Optional example; edit and uncomment only if you need it.
    # Keep SequenceID so rows can be checked later.
    if 'SequenceID' in df.columns:
        motif_scores_df['SequenceID'] = df['SequenceID']
    # Select only the columns used for prediction (Block1 and Block7)
    motif_scores_for_model = motif_scores_df[[col for col in motif_score_columns if col in motif_scores_df.columns]].copy()
    # Motif score values used by the model.
    from sklearn.preprocessing import StandardScaler
    # Optional example; edit and uncomment only if you need it.

    all_motif_scores_scaled= motif_scores_for_model.values
    df['all_motif_scores_scaled'] = list(all_motif_scores_scaled)
    # Export df to Excel for inspection (optional, can be removed if not needed)
    # Optional example; edit and uncomment only if you need it.
    # Save motif score table with matching output names.
    # Main run.
    global motif_scores_df_for_export
    motif_scores_df_for_export = motif_scores_df.copy()
    return df, motif_score_columns

def seq_to_kmers(seq, k=6):
    seq = str(seq).upper().replace(" ", "").replace("\n", "")
    return [seq[i:i+k] for i in range(len(seq)-k+1)]

# Prediction
def predict(model, tokenizer, df, motif_feature_columns, max_len=130):
    all_preds = []
    model.eval()
    with torch.no_grad():
        for idx, row in df.iterrows():
            seq = row['Sequence']
            kmers = seq_to_kmers(seq, 6)
            n_kmers_allowed = max_len - 2
            if len(kmers) > n_kmers_allowed:
                kmers = kmers[:n_kmers_allowed]
            tokens = ['[CLS]'] + kmers + ['[SEP]']
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            if len(input_ids) < max_len:
                input_ids += [tokenizer.pad_token_id] * (max_len - len(input_ids))
            input_ids = torch.tensor(input_ids).unsqueeze(0).to(DEVICE)
            # Ecd value used by the model.
            ecd_feature = torch.tensor([row['Ecd']], dtype=torch.long).unsqueeze(0).to(DEVICE)
            # Motif score values used by the model.
            motif_scores = np.array(row['all_motif_scores_scaled'])
            motif_scores = torch.tensor(motif_scores, dtype=torch.float).unsqueeze(0).to(DEVICE)
            outputs = model(input_ids, ecd_feature, motif_scores)
            pred = outputs['logits'].view(-1).cpu().numpy()[0]
            all_preds.append(pred)
    return np.array(all_preds)

# Check prediction quality
def evaluate(y_true, y_pred):
    pearson = pearsonr(y_true, y_pred)
    spearman = spearmanr(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        'pearson_r': pearson[0], 'pearson_p': pearson[1],
        'spearman_r': spearman.correlation, 'spearman_p': spearman.pvalue,
        'mse': mse, 'mae': mae, 'r2': r2
    }

# Make plots
def plot_results(y_true, y_pred, out_prefix):
    plt.figure(figsize=(7,6))
    plt.scatter(y_true, y_pred, alpha=0.3, edgecolors='none')  # Make points partly transparent.
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.xlabel('Actual Expression')
    plt.ylabel('Predicted Expression')
    plt.title('Predicted vs Actual')
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_pred_vs_actual.png")
    plt.close()
    # Prediction errors.
    residuals = y_pred - y_true
    plt.figure(figsize=(7,6))
    sns.histplot(residuals, kde=True, alpha=0.3)  # Make points partly transparent.
    plt.xlabel('Residual (Predicted - Actual)')
    plt.title('Residuals Distribution')
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_residuals.png")
    plt.close()

# Save files
def export_results(df, y_pred, eval_metrics, out_prefix):
    df_out = df.copy()
    df_out['Predicted'] = y_pred
    df_out['Residual'] = y_pred - df_out['Expression']
    df_out.to_excel(f"{out_prefix}_results.xlsx", index=False)
    # Save score values.
    with open(f"{out_prefix}_metrics.txt", 'w') as f:
        for k, v in eval_metrics.items():
            f.write(f"{k}: {v}\n")

# Main run
def main():
    timestamp = time.time()
    out_prefix = os.path.join(RESULTS_DIR, f"predict_eval_v3_4_{timestamp}")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # Read and prepare the input data.
    df, motif_feature_columns = prepare_input_data(EXCEL_INPUT)
    # Load model
    print("Loading model and tokenizer...")
    print(motif_feature_columns)
    model, tokenizer = load_model_and_tokenizer(MODEL_DIR, motif_feature_columns)
    # Predict
    y_true = df['Expression'].values
    y_pred = predict(model, tokenizer, df, motif_feature_columns, MAX_LEN)
    print(f"First 10 predictions: {y_pred[:10]}")
    # Remove rows with missing values before scoring.
    mask = ~((np.isnan(y_true)) | (np.isnan(y_pred)))
    n_nans = (~mask).sum()
    if n_nans > 0:
        print(f"Warning: {n_nans} rows with NaN in y_true or y_pred will be excluded from evaluation.")
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    # Evaluate
    eval_metrics = evaluate(y_true_clean, y_pred_clean)
    print("Evaluation metrics:")
    for k, v in eval_metrics.items():
        print(f"{k}: {v}")
    # Plot
    plot_results(y_true_clean, y_pred_clean, out_prefix)
    # Save files
    export_results(df.iloc[mask], y_pred_clean, eval_metrics, out_prefix)
    # Save motif score table with matching output names.
    motif_scores_export_path = f"{out_prefix}_motif_scores.xlsx"
    if 'motif_scores_df_for_export' in globals():
        motif_scores_df_for_export.to_excel(motif_scores_export_path, index=False)
        print(f"Motif scores exported to {motif_scores_export_path}")
    print(f"Results exported to {out_prefix}_results.xlsx and plots saved.")
    training_time = (time.time() - timestamp)/60
    print(f"Predictions: {training_time:.2f} minutes")

if __name__ == "__main__":
    main()
