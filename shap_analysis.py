# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 14:25:31 2024

@author: Christophe Jung

DNABERT-2-117M_model contains the original DNABERT-2 model code and pre-trained weights (with custom code like bert_layers.py).
DNABERT-2-117M_model_model_20250627_121629 contains the fine-tuned weights (model.safetensors) and a config.json from your training.
"""

import torch, shap, os, gc, time
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertConfig
from safetensors.torch import safe_open
from datetime import datetime
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import random
import multiprocessing
import subprocess
import sys


model_dir = r"/home/.../DNABERT-2-117M_base_model_20250923_144123"
# Export path for results
export_path = "/home/.../results/"
# Path to the training dataset Excel file
excel_path = r"/home/.../core_promoter_training_data.xlsx"
# Seqences length
SeqLength= 130
# Number of random sequences to use for both SHAP and attention
n_random = 2592 # processing speed ca 50 sequences per hour on a single GPU

def load_pth_model(model_dir, config, device):
    """Load model weights from a .pth file (PyTorch state_dict)"""
    # Load model architecture from config
    model = AutoModelForSequenceClassification.from_config(config)
    # Load state_dict from .pth file
    pth_path = os.path.join(model_dir, "pytorch_model.bin")
    print(f"Loading model weights from: {pth_path}")
    state_dict = torch.load(pth_path, map_location=device)
    model.load_state_dict(state_dict)
    return model

# Attention visualization
def extract_attention_weights(model, input_ids):
    """Extract attention weights from the model"""
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)
        attention_weights = outputs.attentions
        print('attention_weights shape:', outputs)
    return attention_weights

def plot_attention_grid(attention_weights, sequence, num_layers=9, num_heads=9, seq_nr=None):
    """Plot a grid of attention maps for each layer and head."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nr_str = f"_seq{seq_nr}" if seq_nr is not None else ""
    fig, axes = plt.subplots(num_layers, num_heads, figsize=(60, 45))
    seq_len = len(sequence)
    step = 10
    # Use numeric positions for ticks
    xticks = list(range(1, seq_len + 1))
    tick_locs = np.arange(0, seq_len, step)
    tick_labels = [str(i + 1) for i in tick_locs]
    for layer_num in range(num_layers):
        for head_num in range(num_heads):
            ax = axes[layer_num, head_num]
            attention_map = attention_weights[layer_num][0, head_num].cpu().numpy()
            sns.heatmap(attention_map, ax=ax, cmap='viridis',
                        xticklabels=False, yticklabels=False)
            ax.set_title(f'Layer {layer_num + 1}, Head {head_num + 1}')
            ax.set_xlabel('Token Position')
            ax.set_ylabel('Token Position')
            # Set numeric ticks every 'step' positions
            ax.set_xticks(tick_locs)
            ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
            ax.set_yticks(tick_locs)
            ax.set_yticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(export_path, f"attention_grid_summary{nr_str}_{timestamp}.png"))
    plt.close()

def plot_average_attention(attention_weights, sequence, num_layers=9, num_heads=9, top_n=25, seq_nr=None):
    """Plot the average attention map across all layers and heads, and print/save most-attended positions."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nr_str = f"_seq{seq_nr}" if seq_nr is not None else ""
    avg_attention = np.zeros_like(attention_weights[0][0, 0].cpu().numpy())
    for layer_num in range(num_layers):
        for head_num in range(num_heads):
            attention_map = attention_weights[layer_num][0, head_num].cpu().numpy()
            avg_attention += attention_map
    avg_attention /= (num_layers * num_heads)
    col_sums = avg_attention.sum(axis=0)
    top_indices = np.argsort(col_sums)[-top_n:][::-1]
    print(f"Top {top_n} most-attended positions (vertical stripes):")
    motif_lines = []
    for idx in top_indices:
        context = sequence[max(0, idx-5):idx+6]
        line = f"  Position {idx}: {context} (sum attention={col_sums[idx]:.3f})"
        print(line)
        motif_lines.append(line)
        motifs_filename = os.path.join(export_path, f"attention_top{top_n}_motifs{nr_str}_{timestamp}.txt")
    with open(motifs_filename, 'w') as f:
        f.write(f"Top {top_n} most-attended positions (vertical stripes):\n")
        for line in motif_lines:
            f.write(line + '\n')
    print(f"Saved top {top_n} motif contexts to: {motifs_filename}")
    plt.figure(figsize=(10, 8))
    seq_len = len(sequence)
    step = 10
    xticks = list(range(1, seq_len + 1))
    tick_locs = np.arange(0, seq_len, step)
    tick_labels = [str(i + 1) for i in tick_locs]
    ax = sns.heatmap(avg_attention, cmap='viridis', xticklabels=False, yticklabels=False)
    for idx in top_indices:
        ax.axvline(idx, color='red', linestyle='--', linewidth=1)
    plt.title(f'Average Attention Map (Across {num_layers} Layers and {num_heads} Heads)')
    plt.xlabel('Token Position')
    plt.ylabel('Token Position')
    plt.xticks(tick_locs, tick_labels, rotation=45, ha='right', fontsize=8)
    plt.yticks(tick_locs, tick_labels, rotation=45, ha='right', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(export_path, f"average_attention_summary{nr_str}_{timestamp}.png"))
    plt.close()


def visualize_attention(model, tokenizer, sequence, device, num_layers=9, num_heads=9, seq_nr=None):
    """Visualize attention for a given sequence"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sequence = sequence.upper()
    input_ids = get_kmer_input_ids(sequence, tokenizer, max_len=SeqLength, k=6).to(device)
    attention_weights = extract_attention_weights(model, input_ids)
    
    # Plot a grid of all attention maps
    plot_attention_grid(attention_weights, sequence, num_layers, num_heads, seq_nr=seq_nr)

    # Plot the average attention map across all layers and heads
    plot_average_attention(attention_weights, sequence, num_layers, num_heads, top_n=25, seq_nr=seq_nr)


def batch_visualize_attention(model, tokenizer, sequences, device, num_layers=9, num_heads=9):
    """Run attention visualization for a batch of sequences."""
    for i, seq in enumerate(sequences):
        print(f"\nVisualizing attention for sequence {i+1}/{len(sequences)}...")
        visualize_attention(model, tokenizer, seq, device, num_layers=num_layers, num_heads=num_heads, seq_nr=i+1)

def batch_visualize_attention_and_summary(model, tokenizer, sequences, device, num_layers=9, num_heads=9):
    """Run attention visualization for a batch of sequences and save a summary plot for all."""
    # This function will just call batch_visualize_attention, as attention plots are per-sequence.
    # Optionally, you could aggregate or summarize, but for now, just run all.
    batch_visualize_attention(model, tokenizer, sequences, device, num_layers=num_layers, num_heads=num_heads)
    print("All attention visualizations complete.")


def compute_mean_attention_map(model, tokenizer, sequence, device, num_layers=9, num_heads=9):
    """Compute the mean attention map (averaged over layers and heads) for a single sequence."""
    sequence = sequence.upper()
    input_ids = get_kmer_input_ids(sequence, tokenizer, max_len=SeqLength, k=6).to(device)
    attention_weights = extract_attention_weights(model, input_ids)
    # attention_weights: list of tensors (num_layers), each (batch, num_heads, seq_len, seq_len)
    # Stack and average over layers and heads
    attn_stack = torch.stack([aw[0] for aw in attention_weights])  # (num_layers, num_heads, seq_len, seq_len)
    mean_attn = attn_stack.mean(dim=(0,1))  # (seq_len, seq_len)
    return mean_attn.cpu().numpy()

def compute_global_average_attention_map(model, tokenizer, sequences, device, num_layers=9, num_heads=9):
    """Compute the global average attention map across all sequences."""
    mean_maps = []
    for i, seq in enumerate(sequences):
        print(f"Computing mean attention map for sequence {i+1}/{len(sequences)}...")
        mean_map = compute_mean_attention_map(model, tokenizer, seq, device, num_layers, num_heads)
        mean_maps.append(mean_map)
    global_mean_map = np.mean(np.stack(mean_maps), axis=0)
    return global_mean_map

def plot_global_average_attention_heatmap(global_mean_map, out_path=None):
    """Plot the global average attention map as a heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(global_mean_map, cmap='viridis')
    plt.title('Global Average Attention Map (All Sequences)')
    plt.xlabel('Token Position')
    plt.ylabel('Token Position')
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path)
        print(f"Global average attention map saved to: {out_path}")
    else:
        plt.show()
    plt.close()

def plot_difference_attention_map(diff_map, sequence, out_path=None):
    plt.figure(figsize=(10, 8))
    sns.heatmap(diff_map, cmap='bwr', center=0)
    plt.title('Promoter - Random Control: Difference Attention Map')
    plt.xlabel('Token Position')
    plt.ylabel('Token Position')
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path)
        print(f"Difference attention map saved to: {out_path}")
    else:
        plt.show()
    plt.close()

def subtract_random_control_attention_maps(model, tokenizer, promoter_sequences, random_controls, device, num_layers=9, num_heads=9, out_dir="/home/.../diff_attention_maps"):
    os.makedirs(out_dir, exist_ok=True)
    # Compute average attention map for random controls
    print("Computing average attention map for random controls...")
    random_mean_map = compute_global_average_attention_map(model, tokenizer, random_controls, device, num_layers, num_heads)
    # For each promoter, compute its mean map and subtract random mean
    for i, seq in enumerate(promoter_sequences):
        print(f"Computing and saving difference map for promoter {i+1}/{len(promoter_sequences)}...")
        promoter_mean_map = compute_mean_attention_map(model, tokenizer, seq, device, num_layers, num_heads)
        diff_map = promoter_mean_map - random_mean_map
        out_path = os.path.join(out_dir, f"diff_attention_map_promoter_{i+1}.png")
        plot_difference_attention_map(diff_map, seq, out_path=out_path)



def explain_with_shap(model, tokenizer, sequence, device, class_names=None, max_length=SeqLength, n_background=6, seq_nr=None):
    """Generate SHAP explanations for a single sequence and save the summary plot, using a random background."""
    import shap
    import matplotlib.pyplot as plt
    import numpy as np
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nr_str = f"_seq{seq_nr}" if seq_nr is not None else ""
    model.eval()
    input_ids = get_kmer_input_ids(sequence, tokenizer, max_len=max_length, k=6).to(device)
    seq_len = input_ids.shape[1]
    bases = ['A', 'C', 'G', 'T']
    random_seqs = [''.join(random.choices(bases, k=seq_len-2+6)) for _ in range(n_background)]
    background_input_ids = torch.cat([
        get_kmer_input_ids(s, tokenizer, max_len=max_length, k=6) for s in random_seqs], dim=0).cpu().numpy()
    def predict_fn(input_ids_numpy):
        input_ids_tensor = torch.tensor(input_ids_numpy, dtype=torch.long).to(device)
        with torch.no_grad():
            outputs = model(input_ids_tensor)
            logits = outputs.logits.cpu().numpy()
        return logits
    explainer = shap.Explainer(predict_fn, background_input_ids)
    shap_values = explainer(input_ids.cpu().numpy())
    feature_names = np.array(tokenizer.convert_ids_to_tokens(input_ids[0]))
    shap.summary_plot(shap_values, feature_names=feature_names, show=False)
    shap_plot_filename = os.path.join(export_path, f"shap_summary{nr_str}_{timestamp}.png")
    plt.savefig(shap_plot_filename)
    plt.close()
    print(f"SHAP summary plot saved to: {shap_plot_filename}")

def batch_explain_with_shap(model, tokenizer, sequences, device, max_length=SeqLength, n_background=10):
    """Run SHAP explanations for a batch of sequences from a list."""
    for i, seq in enumerate(sequences):
        print(f"\nProcessing sequence {i+1}/{len(sequences)}...")
        explain_with_shap(model, tokenizer, seq, device, max_length=max_length, n_background=n_background, seq_nr=i+1)



def batch_explain_with_shap_and_summary(model, tokenizer, sequences, device, max_length=SeqLength, n_background=10):
    """Aggregate SHAP values by unique k-mer across all sequences and plot a custom summary with individual points."""
    import shap
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import defaultdict
    from datetime import datetime
    seq_len = len(seq_to_kmers(sequences[0], 6)) + 2 if sequences else max_length
    bases = ['A', 'C', 'G', 'T']
    random_seqs = [''.join(random.choices(bases, k=seq_len-2+6)) for _ in range(n_background)]
    background_input_ids = torch.cat([
        get_kmer_input_ids(s, tokenizer, max_len=max_length, k=6) for s in random_seqs
    ], dim=0).cpu().numpy()
    def predict_fn(input_ids_numpy):
        input_ids_tensor = torch.tensor(input_ids_numpy, dtype=torch.long).to(device)
        with torch.no_grad():
            outputs = model(input_ids_tensor)
            logits = outputs.logits.cpu().numpy()
        return logits
    explainer = shap.Explainer(predict_fn, background_input_ids)
    all_input_ids = []
    all_kmers = []
    for seq in sequences:
        input_ids = get_kmer_input_ids(seq, tokenizer, max_len=max_length, k=6).cpu().numpy()[0]
        all_input_ids.append(input_ids)
        kmers = seq_to_kmers(seq, 6)
        all_kmers.append(['[CLS]'] + kmers + ['[SEP]'])
    all_input_ids = np.stack(all_input_ids)
    shap_values = explainer(all_input_ids)
    # Aggregate SHAP values by unique k-mer
    kmer_shap = defaultdict(list)
    for seq_idx, (input_ids, kmer_list) in enumerate(zip(all_input_ids, all_kmers)):
        tokens = tokenizer.convert_ids_to_tokens(list(input_ids))
        for i, token in enumerate(tokens):
            if token == '[UNK]' and 1 <= i < len(kmer_list) - 1:
                tokens[i] = kmer_list[i]
        if hasattr(shap_values, 'values'):
            values = shap_values.values[seq_idx]
        else:
            values = shap_values[seq_idx]
        for token, shap_val in zip(tokens, values):
            kmer_shap[token].append(np.mean(shap_val) if isinstance(shap_val, np.ndarray) else shap_val)
    # Prepare data for plotting
    unique_kmers = list(kmer_shap.keys())
    mean_shap = [np.mean(kmer_shap[k]) for k in unique_kmers]
    count = [len(kmer_shap[k]) for k in unique_kmers]
    # Sort by absolute mean SHAP value
    sorted_idx = np.argsort(-np.abs(mean_shap))
    unique_kmers = [unique_kmers[i] for i in sorted_idx]
    mean_shap = [mean_shap[i] for i in sorted_idx]
    count = [count[i] for i in sorted_idx]
    # Plot mean and individual points
    plt.figure(figsize=(12, max(6, len(unique_kmers)//4)))
    y_pos = np.arange(len(unique_kmers))
    # Plot individual points (jittered)
    for i, kmer in enumerate(unique_kmers):
        y = np.full(len(kmer_shap[kmer]), y_pos[i])
        plt.scatter(kmer_shap[kmer], y, color='gray', alpha=0.5, s=20, label='_nolegend_')
    # Plot mean as blue bars
    bars = plt.barh(y_pos, mean_shap, color='skyblue', alpha=0.8, label='Mean SHAP')
    plt.yticks(y_pos, unique_kmers)
    plt.xlabel('Mean SHAP Value')
    plt.ylabel('k-mer')
    plt.title('Aggregated SHAP values by unique k-mer (dots: individual values)')
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    shap_plot_filename = os.path.join(export_path, f"shap_summary_kmer_{timestamp}.png")
    plt.savefig(shap_plot_filename)
    plt.close()
    print(f"Custom k-mer SHAP summary plot with individual points saved to: {shap_plot_filename}")
    # Export mean and individual SHAP values for each unique k-mer
    export_kmer_path = os.path.join(export_path, f"shap_kmer_values_{timestamp}.txt")
    with open(export_kmer_path, 'w') as f:
        f.write("kmer\tmean_shap\tcount\tindividual_shap_values\n")
        for kmer, mean, cnt in zip(unique_kmers, mean_shap, count):
            indiv_str = ','.join(f"{v:.6f}" for v in kmer_shap[kmer])
            f.write(f"{kmer}\t{mean:.6f}\t{cnt}\t{indiv_str}\n")
    print(f"Exported k-mer mean and individual SHAP values to: {export_path}")


def export_token_shap_contributions(model, tokenizer, sequences, device, max_length=SeqLength, n_background=10, out_path="/home/be-em/data/Core_Promoter_2015/results/token_shap_contributions.txt"):
    """Aggregate SHAP values per token across sequences and export their mean contribution."""
    import shap
    from collections import defaultdict, Counter
    import numpy as np
    seq_len = len(seq_to_kmers(sequences[0], 6)) + 2 if sequences else max_length
    bases = ['A', 'C', 'G', 'T']
    random_seqs = [''.join(random.choices(bases, k=seq_len-2+6)) for _ in range(n_background)]
    background_input_ids = torch.cat([
        get_kmer_input_ids(s, tokenizer, max_len=max_length, k=6) for s in random_seqs
    ], dim=0).cpu().numpy()
    def predict_fn(input_ids_numpy):
        input_ids_tensor = torch.tensor(input_ids_numpy, dtype=torch.long).to(device)
        with torch.no_grad():
            outputs = model(input_ids_tensor)
            logits = outputs.logits.cpu().numpy()
        return logits
    explainer = shap.Explainer(predict_fn, background_input_ids)
    all_input_ids = []
    for seq in sequences:
        input_ids = get_kmer_input_ids(seq, tokenizer, max_len=max_length, k=6).cpu().numpy()[0]
        all_input_ids.append(input_ids)
    all_input_ids = np.stack(all_input_ids)
    shap_values = explainer(all_input_ids)
    
    # Aggregate SHAP values by token
    token_shap = defaultdict(list)
    token_counts = Counter()
    for i, input_ids in enumerate(all_input_ids):
        tokens = tokenizer.convert_ids_to_tokens(list(input_ids))
        if hasattr(shap_values, 'values'):
            values = shap_values.values[i]
        else:
            values = shap_values[i]
        for token, val in zip(tokens, values):
            token_shap[token].append(val)
            token_counts[token] += 1
    if out_path is None:
        out_path = os.path.join(export_path, "token_shap_contributions.txt")
    with open(out_path, 'w') as f:
        f.write("Token\tMean_SHAP\tCount\n")
        for token, vals in sorted(token_shap.items(), key=lambda x: -abs(np.mean(x[1]))):
            f.write(f"{token}\t{np.mean(vals):.6f}\t{token_counts[token]}\n")
    print(f"Token SHAP contributions exported to: {out_path}")

def batch_predict_kmer(model, tokenizer, sequences, device, max_len=SeqLength, k=6):
    """Batch prediction using k-mer batching, matching base model validation."""
    model.eval()
    all_preds = []
    valid_indices = []  # To keep track of indices with valid predictions
    with torch.no_grad():
        batch_input_ids = []
        for seq in sequences:
            input_ids = get_kmer_input_ids(seq, tokenizer, max_len=max_len, k=6)
            batch_input_ids.append(input_ids[0])
        input_ids_tensor = torch.tensor(np.stack(batch_input_ids)).to(device)
        outputs = model(input_ids_tensor)
        logits = outputs.logits.squeeze()
        all_preds = logits.cpu().numpy()
        valid_indices = list(range(len(sequences)))  # All indices are valid in this batch
    return all_preds, valid_indices

def export_most_frequent_tokens(tokenizer, sequences, top_n=500, out_path="/home/be-em/data/Core_Promoter_2015/results/most_frequent_tokens.txt"):
    """Export the most frequent tokens in the given sequences to a txt file."""
    from collections import Counter
    all_tokens = []
    for seq in sequences:
        kmers = seq_to_kmers(seq, 6)
        n_kmers_allowed = SeqLength - 2
        if len(kmers) > n_kmers_allowed:
            kmers = kmers[:n_kmers_allowed]
        tokens = ['[CLS]'] + kmers + ['[SEP]']
        all_tokens.extend(tokens)
    token_counts = Counter(all_tokens)
    most_common = token_counts.most_common(top_n)
    if out_path is None:
        out_path = os.path.join(export_path, "most_frequent_tokens.txt")
    with open(out_path, 'w') as f:
        f.write("Token\tCount\n")
        for token, count in most_common:
            f.write(f"{token}\t{count}\n")
    print(f"Most frequent tokens exported to: {out_path}")


def export_per_token_shap_values(model, tokenizer, sequences, device, max_length=SeqLength, n_background=10, out_path="/home/be-em/data/Core_Promoter_2015/results/per_token_shap_values.txt"):
    """Export SHAP values for each token in each sequence as a table. Handles both scalar and multi-dimensional SHAP values."""
    import shap
    import numpy as np
    seq_len = len(seq_to_kmers(sequences[0], 6)) + 2 if sequences else max_length
    bases = ['A', 'C', 'G', 'T']
    random_seqs = [''.join(random.choices(bases, k=seq_len-2+6)) for _ in range(n_background)]
    background_input_ids = torch.cat([
        get_kmer_input_ids(s, tokenizer, max_len=max_length, k=6) for s in random_seqs
    ], dim=0).cpu().numpy()
    def predict_fn(input_ids_numpy):
        input_ids_tensor = torch.tensor(input_ids_numpy, dtype=torch.long).to(device)
        with torch.no_grad():
            outputs = model(input_ids_tensor)
            logits = outputs.logits.cpu().numpy()
        return logits
    explainer = shap.Explainer(predict_fn, background_input_ids)
    all_input_ids = []
    for seq in sequences:
        input_ids = get_kmer_input_ids(seq, tokenizer, max_len=max_length, k=6).cpu().numpy()[0]
        all_input_ids.append(input_ids)
    all_input_ids = np.stack(all_input_ids)
    shap_values = explainer(all_input_ids)
    with open(out_path, 'w') as f:
        out_path = os.path.join(export_path, "per_token_shap_values.txt")
        f.write("SequenceIndex\tTokenIndex\tToken\tSHAP_Value(s)\n")
        for seq_idx, input_ids in enumerate(all_input_ids):
            tokens = tokenizer.convert_ids_to_tokens(list(input_ids))
            kmers = seq_to_kmers(sequences[seq_idx], 6)
            # Replace [UNK] tokens with the actual k-mer for this sequence
            for i, token in enumerate(tokens):
                if token == '[UNK]' and 1 <= i < len(kmers) + 1:
                    tokens[i] = kmers[i - 1]
            if hasattr(shap_values, 'values'):
                values = shap_values.values[seq_idx]
            else:
                values = shap_values[seq_idx]
            for token_idx, (token, shap_val) in enumerate(zip(tokens, values)):
                if isinstance(shap_val, np.ndarray) and shap_val.size > 1:
                    shap_str = ','.join(f"{v:.6f}" for v in shap_val.flatten())
                else:
                    shap_str = f"{float(shap_val):.6f}"
                f.write(f"{seq_idx}\t{token_idx}\t{token}\t{shap_str}\n")
    print(f"Per-token SHAP values exported to: {out_path}")



# LIME explanation
def explain_with_lime_kmer_tabular(model, tokenizer, sequence, kmer_list, device, seq_nr=None):
    """LIME explanation at the k-mer level using LIME's tabular explainer."""
    from lime.lime_tabular import LimeTabularExplainer
    import numpy as np
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nr_str = f"_seq{seq_nr}" if seq_nr is not None else ""
    # Convert sequence to k-mer presence vector
    def sequence_to_vector(seq, kmer_list):
        kmers = set(seq_to_kmers(seq, 6))
        return np.array([1 if kmer in kmers else 0 for kmer in kmer_list])
    X = np.array([sequence_to_vector(sequence, kmer_list)])
    def lime_predict(X_bin):
        seqs = []
        for row in X_bin:
            present_kmers = [kmer for kmer, present in zip(kmer_list, row) if present]
            # Reconstruct a sequence by masking absent k-mers in the original sequence
            orig_kmers = seq_to_kmers(sequence, 6)
            masked_kmers = [k if k in present_kmers else 'N'*6 for k in orig_kmers]
            # Rebuild sequence from masked k-mers (overlap by 5 bases)
            if masked_kmers:
                seq = masked_kmers[0] + ''.join([k[-1] for k in masked_kmers[1:]])
            else:
                seq = ''
            seqs.append(seq)
        preds = predict(model, tokenizer, seqs, device=device, max_len=SeqLength, k=6)
        return preds
    explainer = LimeTabularExplainer(X, feature_names=kmer_list, mode='regression')
    exp = explainer.explain_instance(X[0], lime_predict, num_features=10)
    exp.save_to_file(f"/home/be-em/data/Core_Promoter_2015/lime_kmer_explanation{nr_str}_{timestamp}.html")
    print(f"LIME k-mer explanation saved to: /home/be-em/data/Core_Promoter_2015/lime_kmer_explanation{nr_str}_{timestamp}.html")
    # Check for UNK tokens in the prediction
    test_seq = seq_to_kmers(sequence, 6)
    test_ids = tokenizer.convert_tokens_to_ids(['[CLS]'] + test_seq + ['[SEP]'])
    test_tokens = tokenizer.convert_ids_to_tokens(test_ids)
    unk_count = test_tokens.count('[UNK]')
    print(f"Number of [UNK] tokens in the original sequence: {unk_count}")

def batch_explain_with_lime(model, tokenizer, sequences, device):
    """Run LIME k-mer explanations for a batch of sequences."""
    # Build k-mer vocabulary from all sequences
    all_kmers = set()
    for seq in sequences:
        all_kmers.update(seq_to_kmers(seq, 6))
    kmer_list = sorted(list(all_kmers))
    for i, seq in enumerate(sequences):
        print(f"\nRunning LIME k-mer explanation for sequence {i+1}/{len(sequences)}...")
        explain_with_lime_kmer_tabular(model, tokenizer, seq, kmer_list, device, seq_nr=i+1)

def get_top_n_attended_positions(mean_attention, sequence, top_n=51):
    """Return the top N most-attended positions and their motif contexts for a single sequence."""
    col_sums = mean_attention.sum(axis=0)
    top_indices = np.argsort(col_sums)[-top_n:][::-1]
    motifs = [(idx, sequence[max(0, idx-5):idx+6]) for idx in top_indices]
    return top_indices, motifs

def aggregate_top_attended_positions(model, tokenizer, sequences, device, top_n=51, num_layers=9, num_heads=9, out_path=None):
    """Aggregate top N most-attended positions and motif contexts across all sequences."""
    from collections import Counter, defaultdict
    position_counter = Counter()
    motif_counter = Counter()
    motif_contexts = defaultdict(list)
    for i, seq in enumerate(sequences):
        print(f"Finding top attended positions for sequence {i+1}/{len(sequences)}...")
        mean_attn = compute_mean_attention_map(model, tokenizer, seq, device, num_layers, num_heads)
        top_indices, motifs = get_top_n_attended_positions(mean_attn, seq, top_n=top_n)
        for idx, motif in motifs:
            position_counter[idx] += 1
            motif_counter[motif] += 1
            motif_contexts[idx].append(motif)
    # Save summary
    if out_path:
        with open(out_path, 'w') as f:
            f.write("Position\tCount\tExample_Motif\n")
            for idx, count in position_counter.most_common():
                example_motif = motif_contexts[idx][0] if motif_contexts[idx] else ''
                f.write(f"{idx}\t{count}\t{example_motif}\n")
        print(f"Aggregated top attended positions saved to: {out_path}")
    return position_counter, motif_counter, motif_contexts

def extract_block_name(seq_name):
    """Extract the substring between the first and last occurrence of 'Block' in the sequence name, removing 'Block' and anything before/after."""
    import re
    # Find all 'Block' occurrences
    matches = list(re.finditer(r'Block', seq_name))
    if not matches:
        return seq_name  # fallback: return original if no 'Block'
    first = matches[0].end()
    last = matches[-1].start()
    # Find the next space or end after first 'Block'
    after_first = seq_name[first:]
    # Find the first non-alphanumeric after 'Block'
    m = re.search(r'[^\w.]+', after_first)
    end_idx = m.start() if m else len(after_first)
    # Extract between first and last 'Block', or just after first if only one
    if len(matches) == 1:
        return after_first[:end_idx].strip()
    else:
        return seq_name[first:last].strip()



# Print tokens for first 5 sequences for verification (after loading sequences)
def print_first5_tokens(sequences, tokenizer, max_len=SeqLength, k=6):
    print("\n--- Token examples for first 5 sequences ---")
    for i in range(min(5, len(sequences))):
        kmers = seq_to_kmers(sequences[i], k)
        n_kmers_allowed = max_len - 2
        if len(kmers) > n_kmers_allowed:
            kmers = kmers[:n_kmers_allowed]
        tokens = ['[CLS]'] + kmers + ['[SEP]']
        print(f"Sequence {i+1} tokens: {tokens}")
    print("--- End token examples ---\n")


def seq_to_kmers(seq, k=6):
    seq = seq.upper().replace(" ", "").replace("\n", "")
    return [seq[i:i+k] for i in range(len(seq)-k+1)]

def get_kmer_input_ids(seq, tokenizer, max_len=SeqLength, k=6):
    kmers = seq_to_kmers(seq, k)
    n_kmers_allowed = max_len - 2
    if len(kmers) > n_kmers_allowed:
        kmers = kmers[:n_kmers_allowed]
    tokens = ['[CLS]'] + kmers + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    if len(input_ids) < max_len:
        input_ids += [tokenizer.pad_token_id] * (max_len - len(input_ids))
    return torch.tensor([input_ids])

def predict(model, tokenizer, sequences, device='cpu', max_len=SeqLength, k=6):
    model.eval()
    all_preds = []
    with torch.no_grad():
        batch_input_ids = []
        for seq in sequences:
            input_ids = get_kmer_input_ids(seq, tokenizer, max_len=max_len, k=k)
            batch_input_ids.append(input_ids[0])
        input_ids_tensor = torch.stack(batch_input_ids).to(device)
        outputs = model(input_ids_tensor)
        logits = outputs.logits.squeeze()
        all_preds = logits.cpu().numpy()
    return all_preds

def try2(target_func, *args, timeout=600):
    def wrapper(queue, *args):
        try:
            result = target_func(*args)
            queue.put(result)
        except Exception as e:
            queue.put(e)
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=wrapper, args=(queue, *args))
    p.start()
    p.join(timeout)
    if p.exitcode == 0:
        result = queue.get()
        if isinstance(result, Exception):
            print(f"Error: {result}")
            return None
        return result
    else:
        print(f"Subprocess crashed or timed out (exitcode={p.exitcode})")
        return None

def get_free_gpu_memory():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        free_mem = int(result.stdout.strip().split('\n')[0])  # in MiB
        return free_mem
    except Exception as e:
        print(f"Could not query GPU memory: {e}")
        return None
def main():
    start_time_total = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = BertConfig.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, config=config, trust_remote_code=True, ignore_mismatched_sizes=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model.to(device)
    
    # Load all relevant columns from the Excel file
    df = pd.read_excel(excel_path, engine='openpyxl')
    # Extract names (col B, index 1), expression (col D, index 3), sequences (col E, index 4)
    names = df['SequenceID'].astype(str).tolist()
    expressions = df['NORM'].tolist()
    sequences = df['SequenceSample'].astype(str).tolist()
    # Filter out rows with missing sequence
    filtered = [(n, e, s) for n, e, s in zip(names, expressions, sequences) if isinstance(s, str) and s.strip()]
    print(f"Loaded {len(filtered)} valid (name, expr, seq) triplets from {excel_path}")
    # Save all valid sequences before sampling
    all_sequences = [s for n, e, s in filtered]
    # Randomly select n_random triplets. random.seed(42)  COULD BE there for reproducibility
    #random.seed(42)
    sampled = random.sample(filtered, n_random)
    print(f"Randomly selected {len(sampled)} triplets for analysis.")
    # Save selected triplets to a file (tab-separated)
    selected_path = "/home/be-em/data/Core_Promoter_2015/results/selected_sequences_for_shap.tsv"
    with open(selected_path, 'w') as f:
        f.write("Name\tExpression\tSequence\n")
        for name, expr, seq in sampled:
            f.write(f"{name}\t{expr}\t{seq}\n")

    # Unpack for downstream analysis
    selected_names = [n for n, e, s in sampled]
    selected_expressions = [e for n, e, s in sampled]
    selected_sequences = [s for n, e, s in sampled]
    # Only to check if 6-mers tokeniuzation worked correctly:
    #print_first5_tokens(selected_sequences, tokenizer, max_len=SeqLength, k=6)

    # Export most frequent tokens in the full dataset (all valid sequences)
    export_most_frequent_tokens(tokenizer, all_sequences, top_n=500, out_path="/home/.../most_frequent_tokens_full_dataset.txt")
    # Export most frequent tokens in the selected sequences (for comparison, optional)
    export_most_frequent_tokens(tokenizer, selected_sequences, top_n=500)
    
    # Add 3 random control sequences of the same length as the real sequences
    seq_len = len(selected_sequences[0]) if selected_sequences else SeqLength
    bases = ['A', 'C', 'G', 'T']
    random_controls = [''.join(random.choices(bases, k=seq_len)) for _ in range(3)]
    control_names = [f"random_control_{i+1}" for i in range(3)]
    # Optionally print or save the random controls
    print("\nRandom control sequences:")
    for name, seq in zip(control_names, random_controls):
        print(f"{name}: {seq}")
    # Add to selected_sequences for downstream analysis
    selected_sequences_with_controls = selected_sequences + random_controls

    # Export model expression predictions for selected + random sequences
    print("\nExporting model expression predictions for selected and random sequences...")
    predictions, valid_indices = batch_predict_kmer(model, tokenizer, selected_sequences_with_controls, device, max_len=SeqLength, k=6)
    pred_out_path = os.path.join(export_path, "expression_predictions_selected_and_random.txt")
    with open(pred_out_path, 'w') as f:
        f.write("Index\tName\tSequence\tModel_Prediction\tExpression\n")
        for i, idx in enumerate(valid_indices):
            seq = selected_sequences_with_controls[idx]
            pred = predictions[i]
            if idx < len(selected_sequences):
                name = selected_names[idx]
                expr = selected_expressions[idx]
            else:
                name = control_names[idx - len(selected_sequences)]
                expr = "NA"
            f.write(f"{idx+1}\t{name}\t{seq}\t{pred}\t{expr}\n")
    print(f"Model predictions exported to: {pred_out_path}")
    
    # SHAP analysis
    batch_explain_with_shap(model, tokenizer, selected_sequences, device, max_length=SeqLength, n_background=2)
    batch_explain_with_shap_and_summary(model, tokenizer, selected_sequences, device, max_length=SeqLength, n_background=2)
    export_token_shap_contributions(model, tokenizer, selected_sequences, device, max_length=SeqLength, n_background=2, out_path="/home/.../token_shap_contributions.txt")
    export_per_token_shap_values(model, tokenizer, selected_sequences, device, max_length=SeqLength, n_background=2, out_path="/home/.../per_token_shap_values.txt")

    # Attention Visualization
    #print("Visualizing attention...")
    # Run attention visualization for selected sequences + controls
    ###batch_visualize_attention(model, tokenizer, selected_sequences_with_controls, device, num_layers=9, num_heads=9)
    # Global Attention Analysis: Compute and plot global average attention map
    #print("\nComputing and plotting global average attention map for all selected sequences...")
    #global_mean_map = compute_global_average_attention_map(model, tokenizer, selected_sequences_with_controls, device)
    ###plot_global_average_attention_heatmap(global_mean_map, out_path="/home/be-em/data/Core_Promoter_2015/global_average_attention_map.png")

    # Aggregate and save most-attended positions across dataset
    #print("\nAggregating most-attended positions across all selected sequences...")
    #aggregate_top_attended_positions(
    #    model, tokenizer, selected_sequences, device, top_n=25, num_layers=9, num_heads=9,
    #    out_path="/home/be-em/data/Core_Promoter_2015/most_attended_positions_across_dataset.txt"
    #)
       
    # SHAP analysis for selected sequences + controls
    #batch_explain_with_shap_and_summary(model, tokenizer, selected_sequences, device, max_length=SeqLength, n_background=3)
    #export_token_shap_contributions(model, tokenizer, selected_sequences, device, max_length=SeqLength, n_background=3, out_path="/home/be-em/data/Core_Promoter_2015/token_shap_contributions.txt")    
    #export_per_token_shap_values(model, tokenizer, selected_sequences, device, max_length=SeqLength, n_background=3, out_path="/home/be-em/data/Core_Promoter_2015/per_token_shap_values.txt")


    # Call the subtraction at the end of the main script
    #subtract_random_control_attention_maps(
    #    model, tokenizer, selected_sequences, random_controls, device, num_layers=9, num_heads=9,
    #    out_dir="/home/be-em/data/Core_Promoter_2015/diff_attention_maps"
    #)
    # Lime visualization for selected sequences
    # ---------------------------------------------------------------------------
    # does not work with safetensors model: Running LIME explanation for sequence 1/4...
    # Killed. Probably memory overflow. Same issue with GPU memory.
    #print("\nRunning LIME explanations for selected sequences...")  
    #batch_explain_with_lime(model, tokenizer, selected_sequences, device)
    
    print("All  runs complete.")
    end_time_total = time.time()
    total_run_time = (end_time_total - start_time_total) / 60  # Convert to minutes
    print(f"Total training and evaluation time for all genes: {total_run_time:.2f} minutes")


if __name__ == "__main__":
    main()