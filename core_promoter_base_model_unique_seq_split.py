# -*- coding: utf-8 -*-
"""
Train a base DNABERT-2 promoter model.
Use DNA sequence only as model input.
Save model files, plots, tables, and run notes.

IMPORTANT:
In this version, the training data is split by unique DNA sequence.
"""

# Beginner notes
# - Main settings are near the top of this file.
# - Change model_name and data_filepath before running.
# - Run this file to train the sequence-only model.
# - Lines that start with # are comments.
# - Comments explain the script but do not change the result.


# Model settings
batch_size= 28
lr=1e-5
weight_decay=0.0004
epochs=25

# Input and output paths
model_name = r"/home/be-em/data/Core_Promoter_2015/DNABERT-2-117M_model"    # Folder with the DNABERT-2 model.
# Optional example; edit and uncomment only if you need it.
# Optional example; edit and uncomment only if you need it.
data_filepath = r"/home/be-em/data/Core_Promoter_2015/data/core_promoter_training_data_250604.xlsx"

# Install DNABERT-2 so the tokenizer can load.
# Optional terminal command; run it only if you need it.
# Optional terminal command; run it only if you need it.
import torch, time, os, gc
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertConfig
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer 
# Show full long text values in pandas output.
pd.set_option('display.max_colwidth', None)

def load_model_and_tokenizer(model_name):
    """
    Load the saved model files and the tokenizer.
    Return both objects for later use.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = BertConfig.from_pretrained(model_name)
    
    # Chance to skip part of the model during training.
    config.hidden_dropout_prob = 0.00001
    config.attention_probs_dropout_prob = 0.00001
    config.num_labels = 1  # The model predicts one number.

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        config=config,
        trust_remote_code=True
    )
    
    return model, tokenizer


def prepare_data(filepath, tokenizer, batch_size, max_len=130):
    """
    Read the input data file.
    Clean the needed columns.
    Build data loaders for training and testing.
    """
    df = pd.read_excel(filepath)
    df = df[df['SequenceID'].str.contains('Block 1.15') & df['SequenceID'].str.contains('Block 7.15') &
            df['Ecd'].str.contains('No') & df['SequenceID'].str.endswith('Block 7.15')]
    # Rename input columns to the names this script expects.
    df['SequenceSample'] = df['SequenceSample'].apply(lambda seq: seq[10:] if isinstance(seq, str) and len(seq)==153 else seq)
    df['SequenceSample'] = df['SequenceSample'].apply(lambda seq: seq[12:] if isinstance(seq, str) and len(seq)==157 else seq)
    # Optional sequence trimming step.
    # Optional example; edit and uncomment only if you need it.
    
    # Keep only rows that pass this rule.
    # Optional example; edit and uncomment only if you need it.
    # Optional example; edit and uncomment only if you need it.
    # Convert expression values to log2 scale.
    df['NORM'] = np.log2(np.clip(df['NORM'].astype(float), 5e-3, None))
    print(r'Expression: ', df['NORM'])
    # Optional check: shuffle expression values.
    # Optional example; edit and uncomment only if you need it.
    
    # Optional: use log10 scale instead.
    # Optional example; edit and uncomment only if you need it.
    # Optional: scale values between 0 and 1.
    # Optional example; edit and uncomment only if you need it.
    # Fit the scaler on the expression values.
    # Optional example; edit and uncomment only if you need it.
    # Normalize the training and validation data using the fitted scaler
    # Optional example; edit and uncomment only if you need it.
    
    # Make sequence text uppercase.
    df =df.apply(lambda col: col.str.upper() if col.dtype == "object" else col)

    print('Data preparation complete. Sample data:')
    print(df['SequenceSample'] .head(1))  # Print example sequences for checking.
    # Dataset class used by PyTorch.
    def seq_to_kmers(seq, k=6):
        seq = seq.upper().replace(" ", "").replace("\n", "")
        return [seq[i:i+k] for i in range(len(seq)-k+1)]
    
    class GeneExpressionDataset(Dataset):
        def __init__(self, sequences, expressions, tokenizer, max_len, k=6):
            self.sequences = sequences
            self.expressions = expressions
            self.tokenizer = tokenizer
            self.max_len = max_len
            self.k = k
            # Print examples of model input tokens.
            print("\n--- Token examples for first 5 sequences ---")
            for i in range(min(2, len(self.sequences))):
                kmers = seq_to_kmers(self.sequences[i], self.k)
                n_kmers_allowed = self.max_len - 2
                if len(kmers) > n_kmers_allowed:
                    kmers = kmers[:n_kmers_allowed]
                tokens = ['[CLS]'] + kmers + ['[SEP]']
                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                print(f"Sequence {i+1} tokens: {tokens}")
            print("--- End token examples ---\n")

        def __len__(self):
            return len(self.sequences)

        def __getitem__(self, item):
            sequence = self.sequences[item]
            # Make sure the sequence is text.
            if not isinstance(sequence, str):
                sequence = str(sequence)
            expression = self.expressions[item]
            # Split the DNA sequence into short words.
            kmers = seq_to_kmers(sequence, self.k)
            n_kmers_allowed = self.max_len - 2
            if len(kmers) > n_kmers_allowed:
                kmers = kmers[:n_kmers_allowed]
            tokens = ['[CLS]'] + kmers + ['[SEP]']
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            # Add pad tokens so all inputs have the same length.
            if len(input_ids) < self.max_len:
                input_ids += [self.tokenizer.pad_token_id] * (self.max_len - len(input_ids))
            input_ids = torch.tensor(input_ids)
            return input_ids, torch.tensor(expression, dtype=torch.float)

    # Split data into training and test sets.
# Clean DNA text before the split.
    df["SequenceSample"] = (
        df["SequenceSample"]
        .astype(str)
        .str.upper()
        .str.replace(r"\s+", "", regex=True)
    )

    # Split by unique DNA sequence.
    # This keeps all replicate measurements, but ensures that the same sequence
    # Can never appear in both training and validation sets.
    unique_sequences = df["SequenceSample"].drop_duplicates()

    train_sequences, val_sequences = train_test_split(
        unique_sequences,
        test_size=0.1,
        random_state=42,
        shuffle=True
    )

    train_df = df[df["SequenceSample"].isin(train_sequences)].copy()
    val_df = df[df["SequenceSample"].isin(val_sequences)].copy()

    # Check that the same sequence is not in both splits.
    train_seq_set = set(train_df["SequenceSample"])
    val_seq_set = set(val_df["SequenceSample"])
    overlap = train_seq_set & val_seq_set

    print("Training rows:", len(train_df))
    print("Validation rows:", len(val_df))
    print("Training unique sequences:", train_df["SequenceSample"].nunique())
    print("Validation unique sequences:", val_df["SequenceSample"].nunique())
    print("Overlapping sequences:", len(overlap))

    assert len(overlap) == 0, (
        f"Sequence leakage detected: {len(overlap)} overlapping sequences"
    )

    X_train = train_df["SequenceSample"].values
    y_train = train_df["NORM"].values

    X_val = val_df["SequenceSample"].values
    y_val = val_df["NORM"].values

    train_dataset = GeneExpressionDataset(X_train, y_train, tokenizer, max_len, k=6)
    val_dataset = GeneExpressionDataset(X_val, y_val, tokenizer, max_len, k=6)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size)

    return train_loader, val_loader, X_val


def train_model(model, model_name, train_loader, optimizer, loss_fn, device, epochs):
    """
    Train the model for the number of rounds set above.
    Return the loss values from training.
    """
    model.train()
    train_losses = []
    for epoch in range(epochs):
        total_loss = 0
        for i, batch in enumerate(train_loader):
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            optimizer.zero_grad() # Clear old gradients before the next update.
            outputs = model(input_ids)    
            logits = outputs.logits
            # Optional example; edit and uncomment only if you need it.
            logits = logits.squeeze()      
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            loss.backward()  # Compute gradients from the loss.
            # Optional check: print gradient values for each layer.
            # Optional example; edit and uncomment only if you need it.
            # Optional example; edit and uncomment only if you need it.
            # Optional example; edit and uncomment only if you need it.
            optimizer.step() # Update model weights.

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)        
    return train_losses


def evaluate_model(model, val_loader, device):
    """
    Run the model on the test data.
    Return predictions and score values.
    """
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            outputs = model(input_ids)
            preds = outputs.logits.squeeze()

            # Optional example; edit and uncomment only if you need it.
            # Optional example; edit and uncomment only if you need it.
            all_preds.extend(np.atleast_1d(preds.cpu().numpy()))
            all_labels.extend(np.atleast_1d(labels.cpu().numpy()))

    mse = mean_squared_error(all_labels, all_preds) 
    r2 = r2_score(all_labels, all_preds)

    return all_preds, all_labels, mse, r2


def plot_evaluation_results(all_preds, all_labels, model_name, output_dir=None):
    """
    Make plots that compare real values and predicted values.
    Save the plots to the output folder.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot predicted values against real values.
    plt.figure(figsize=(8, 6))
    plt.scatter(all_labels, all_preds, alpha=0.7)
    plt.plot([min(all_labels), max(all_labels)], [min(all_labels), max(all_labels)], color='red', lw=2)
    plt.xlabel('log2(X)')
    plt.ylabel('log2(Y)')
    plt.title(f'Predicted vs Actual Values \n{model_name}', fontsize=14)
    plt.xlabel('Actual', fontsize=12)
    plt.ylabel('Predicted', fontsize=12)
    plt.tight_layout()
    plt.grid(True)
    if output_dir:
        plt.savefig(os.path.join(output_dir, f"{model_name}_pred_vs_actual{timestamp}.png"))
    else:
        plt.savefig(f"{model_name}_pred_vs_actual{timestamp}.png")
    plt.show()
    plt.close()

    # Plot prediction errors.
    residuals = np.array(all_preds) - np.array(all_labels)
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True)
    plt.title(f'Residuals Distribution\n{model_name}', fontsize=14)
    plt.xlabel('Residuals', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, f"{model_name}_residuals_{timestamp}.png"))
    else:
        plt.savefig(f"{model_name}_residuals_{timestamp}.png")
    plt.grid(True)
    plt.show()
    plt.close()


def export_results_to_excel(all_preds, all_labels, model_name, final_val_mse=None, sequences=None, names=None, output_dir=None):
    """Save real values and predicted values to an Excel file."""
    results_df = pd.DataFrame({
        'Actual': all_labels,
        'Predicted': all_preds,
        'Residuals': np.array(all_preds) - np.array(all_labels)
    })
    if final_val_mse is not None:
        results_df['Final_Validation_MSE'] = [final_val_mse] + [None]*(len(results_df)-1)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{model_name}_evaluation_results_{timestamp}.xlsx"
    if output_dir:
        file_path = os.path.join(output_dir, file_name)
    else:
        file_path = file_name
    results_df.to_excel(file_path, index=False)
    print(f"Results exported to {file_path}")


def export_outliers_and_all_results(val_names, val_sequences, all_labels, all_preds, model_name, timestamp):
    import numpy as np
    residuals = np.array(all_preds) - np.array(all_labels)
    std_res = np.std(residuals)
    abs_residuals = np.abs(residuals)
    outlier_mask = abs_residuals > 2 * std_res
    outlier_lines = ["Index\tName\tSequence\tActual\tPredicted\tResidual"]
    for i, is_outlier in enumerate(outlier_mask):
        if is_outlier:
            name = val_names[i] if i < len(val_names) else 'N/A'
            seq = val_sequences[i] if i < len(val_sequences) else 'N/A'
            actual = all_labels[i]
            pred = all_preds[i]
            resid = residuals[i]
            outlier_lines.append(f"{i}\t{name}\t{seq}\t{actual}\t{pred}\t{resid}")
    n_best = max(1, int(0.2 * len(all_labels)))
    best_indices = np.argsort(abs_residuals)[:n_best]
    outlier_lines.append("\n# 20% Best Predicted Sequences (lowest abs residuals)")
    outlier_lines.append("Index\tName\tSequence\tActual\tPredicted\tResidual")
    for i in best_indices:
        name = val_names[i] if i < len(val_names) else 'N/A'
        seq = val_sequences[i] if i < len(val_sequences) else 'N/A'
        actual = all_labels[i]
        pred = all_preds[i]
        resid = residuals[i]
        outlier_lines.append(f"{i}\t{name}\t{seq}\t{actual}\t{pred}\t{resid}")
    # Create the output folder if it is missing.
    results_dir = "/home/be-em/data/Core_Promoter_2015/results"
    os.makedirs(results_dir, exist_ok=True)
    outlier_txt_path = os.path.join(results_dir, f"{os.path.basename(model_name)}_outliers_{timestamp}.txt")
    with open(outlier_txt_path, 'w') as f:
        f.write("\n".join(outlier_lines))
    print(f"Outlier info written to {outlier_txt_path}")
    all_lines = ["Index\tName\tSequence\tActual\tPredicted\tResidual"]
    for i in range(len(all_labels)):
        name = val_names[i] if i < len(val_names) else 'N/A'
        seq = val_sequences[i] if i < len(val_sequences) else 'N/A'
        actual = all_labels[i]
        pred = all_preds[i]
        resid = residuals[i]
        all_lines.append(f"{i}\t{name}\t{seq}\t{actual}\t{pred}\t{resid}")
    all_txt_path = os.path.join(results_dir, f"{os.path.basename(model_name)}_all_results_{timestamp}.txt")
    with open(all_txt_path, 'w') as f:
        f.write("\n".join(all_lines))
    print(f"All results info written to {all_txt_path}")


def save_model_with_timestamp(model, model_name):
    import torch
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"/home/be-em/data/Core_Promoter_2015/results/{model_name}_model_{timestamp}.pth"
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved to {model_filename}")



def predict(model, tokenizer, sequences, device, max_len=130, k=6, save_attention_path=None):
    """
    Predict values for new DNA sequences.
    Optionally save attention tables.
    """
    model.eval()
    predictions = []
    all_attentions = []
    with torch.no_grad():
        if isinstance(sequences, str):
            sequences = [sequences]  # Accept one sequence or a list of sequences.

        input_ids_list = []
        for seq in sequences:
            # Use the same DNA word split used during training.
            seq = seq.upper().replace(" ", "").replace("\n", "")
            kmers = [seq[i:i+k] for i in range(len(seq)-k+1)]
            n_kmers_allowed = max_len - 2
            if len(kmers) > n_kmers_allowed:
                kmers = kmers[:n_kmers_allowed]
            tokens = ['[CLS]'] + kmers + ['[SEP]']
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            if len(input_ids) < max_len:
                input_ids += [tokenizer.pad_token_id] * (max_len - len(input_ids))
            input_ids_list.append(input_ids)
        input_ids_tensor = torch.tensor(input_ids_list).to(device)

        outputs = model(input_ids_tensor, output_attentions=True)
        logits = outputs.logits.squeeze()
        predictions = logits.cpu().numpy()
        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            # Save attention values as NumPy arrays.
            all_attentions = [att.cpu().numpy() for att in outputs.attentions]
            if save_attention_path is not None:
                np.save(save_attention_path, all_attentions)
                print(f"Attention weights saved to {save_attention_path}")
    return predictions, all_attentions


def export_runinfo(model, model_name, data_filepath, batch_size, lr, weight_decay, epochs, device, all_labels, all_preds, mse, timestamp, validation_gene_name=None, max_len=None, script_name=None, model_description=None):
    """Save settings, scores, and software versions to a text file."""
    from sklearn.metrics import r2_score, mean_absolute_error
    params_txt_path = f"/home/be-em/data/Core_Promoter_2015/results/{model_name}_runinfo_{timestamp}.txt"
    with open(params_txt_path, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Training data file (full path): {data_filepath}\n")
        f.write(f"Training data file (basename): {os.path.basename(data_filepath)}\n")
        f.write(f"Python script used: {script_name}\n")
        f.write(f"Max sequence length (max_len): {max_len}\n")
        f.write(f"Validation strategy: Gene-wise holdout\n") 
        if validation_gene_name:
            f.write(f"Validation gene: {validation_gene_name}\n")
        f.write(f"Validation MSE: {mse:.4f}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Learning rate: {lr}\n")
        f.write(f"Weight decay: {weight_decay}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Loss function: torch.nn.MSELoss\n")
        f.write(f"Scheduler: Cosine annealing with warmup\n")
        f.write(f"Device: {device}\n")
        f.write(f"Validation set size: {len(all_labels)}\n")
        # Optional example; edit and uncomment only if you need it.
        # Optional example; edit and uncomment only if you need it.
        r2 = r2_score(all_labels, all_preds)
        mae = mean_absolute_error(all_labels, all_preds)
        norm_mae = mae / (np.max(all_labels) - np.min(all_labels)) if (np.max(all_labels) - np.min(all_labels)) > 0 else float('nan')
        f.write(f"Validation MAE: {mae:.4f}\n")
        f.write(f"Validation R2: {r2:.4f}\n")
        f.write(f"Normalized MAE: {norm_mae:.4f}\n")
        # Save seed values so the run is easier to repeat.
        import sys
        import transformers
        import random
        f.write(f"Random seed (numpy): {getattr(np.random, 'seed', 'N/A')}\n")
        f.write(f"Random seed (torch): {getattr(torch, 'initial_seed', lambda: 'N/A')()}\n")
        f.write(f"Random seed (random): {getattr(random, 'seed', 'N/A')}\n")
        f.write(f"train_test_split random_state:\n")
        # Save software versions used in this run.
        f.write(f"Python version: {sys.version}\n")
        f.write(f"PyTorch version: {torch.__version__}\n")
        f.write(f"Transformers version: {transformers.__version__}\n")
        f.write(f"Pandas version: {pd.__version__}\n")
        f.write(f"Numpy version: {np.__version__}\n")
        # Save a short model file check value when available.
        if hasattr(model, 'config') and hasattr(model.config, 'to_json_string'):
            import hashlib
            config_str = model.config.to_json_string()
            config_hash = hashlib.md5(config_str.encode('utf-8')).hexdigest()
            f.write(f"Model config hash: {config_hash}\n")
        # Write a short model summary to the run file.
        if model_description:
            f.write("\nModel Description:\n")
            f.write(model_description + "\n")
    print(f"Run info written to {params_txt_path}")


def count_model_parameters(model):
    """Count how many model weights can be learned during training."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ----------------------------------------
# Main run
# ----------------------------------------
def main():
    # Use a GPU if one is available.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Set GPU memory options.
    # Optional example; edit and uncomment only if you need it.
    # Optional terminal command; run it only if you need it.
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
    torch.cuda.empty_cache()
    gc.collect()

    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)
    model.to(device)
    print(f"Model loaded: {model_name}")
    print(f"Tokenizer loaded: {model_name}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}") 
    print("Total trainable parameters:", count_model_parameters(model))
    train_loader, val_loader, X_val = prepare_data(data_filepath, tokenizer, batch_size=batch_size)

    # Create the training tools.
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay) 
    loss_fn = torch.nn.MSELoss()

    # Train the model
    print("######################################")
    print("Training model...")
    start_time = time.time()
    train_losses = []
    val_mses = []
    val_r2s = []

    for epoch in range(epochs):
        train_loss = train_model(
            model,
            model_name,
            train_loader,
            optimizer,
            loss_fn,
            device,
            epochs=1
        )[0]

        train_losses.append(train_loss)

        # Check the model on the test data after each round.
        all_preds, all_labels, mse, r2 = evaluate_model(model, val_loader, device)

        val_mses.append(mse)
        val_r2s.append(r2)

        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Training Loss: {train_loss:.4f}, "
            f"Validation MSE: {mse:.4f}, "
            f"Validation R2: {r2:.4f}"
        )
    training_time = (time.time() - start_time)/60
    print(f"Training time: {training_time:.2f} minutes")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_run = os.path.join("/home/be-em/data/Core_Promoter_2015/results", f"{os.path.basename(model_name)}_{timestamp}")
    os.makedirs(output_dir_run, exist_ok=True)
    # Check loss on the test data.
    plt.figure(figsize=(8, 6))
    plt.plot(range(epochs), train_losses, marker='o', linestyle='-', color='b', label='Training Loss')
    plt.plot(range(epochs), val_mses, marker='s', linestyle='--', color='r', label='Validation MSE')
    plt.title('Training Loss and Validation MSE over Epochs', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss / MSE', fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_run, f"{os.path.basename(model_name)}_training_loss_val_mse_{timestamp}.png"))
    plt.close()
    # Save the model and tokenizer.
    torch.save(model.state_dict(), os.path.join(output_dir_run, "pytorch_model.bin"))
    tokenizer.save_pretrained(output_dir_run)
    print(f"Model and tokenizer saved to {output_dir_run}")

    # Evaluate the model
    print("Evaluating model...")
    start_time = time.time()
    all_preds, all_labels, mse, r2 = evaluate_model(model, val_loader, device)
    eval_time = (time.time() - start_time)/60

    # Check that predictions and labels have matching shapes.
    print(f"Predictions shape: {np.array(all_preds).shape}")
    print(f"Labels shape: {np.array(all_labels).shape}")
    print(f"Evaluation time: {eval_time:.2f} minutes")
    print(f"Validation MSE: {mse:.4f}")
    print(f"Validation R2: {r2:.4f}")

    # Plot evaluation results
        # Prediction errors.
    plot_evaluation_results(all_preds, all_labels, os.path.basename(model_name), output_dir=output_dir_run)
    # Export results to Excel
    export_results_to_excel(all_preds, all_labels, model_name, final_val_mse=val_mses[-1], sequences=X_val, names=[str(i) for i in range(len(X_val))], output_dir=output_dir_run)

    # Example for predicting new sequences.
    new_sequences = ['GGTCTCAGGATTTTAAATAGATTTAGCTAGAAAATAGCTGACAGACACATATCGATATATCGCTGCGATAGCCACAGCTGTTCACGCCCGCAGTTTAAGCGtaGatcaccgaagctaCGGCCACCAAAAAATAAACATTGGATCTGTGAGACC', 'GGTCTCAGGATGAGAGAACCAGTGCGCTCTTATCACGTGAGAACGCTTTTGGGCATTCAGTTTGGCTTTTGCGGCGCTGACCGCTGGCGcttagtgCGAATCCATAGgcgctttcaccaatcgcAACGTAGGCCAGAACGGATCTGTGAGACC', 'GGTCTCAGGATGTGTGGCCCCTGTTAGCTTTCTGTTAAATTTAAATTTCTGTAAAGTGCCcgacgcctctctctctctctctctcATCAGAtcagttgTTGTCTGGATAtcgacgcgagcggtcggGATCGCGCATTAGTGTCATCTGTGAGACC']
    measured = [0.56960111, -4.839324055, 1.885391158]
    predictions, attentions = predict(model, tokenizer, new_sequences, device, save_attention_path="attention_weights_example.npy")
    print("Predictions on new sequences:", predictions)
    print("Measured on new sequences:", measured)
    print("Attention weights shape (per layer):", [a.shape for a in attentions] if attentions else None)

    # After testing, save names, sequences, labels, and predictions.
    # These are the variable names used below.
    # After testing, save names, sequences, labels, and predictions.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Use validation sequences and the final validation loss.
    val_sequences = X_val
    val_names = [str(i) for i in range(len(X_val))]  # Use row numbers when no names are available.
    export_outliers_and_all_results(val_names, val_sequences, all_labels, all_preds, model_name, timestamp)
    export_runinfo(
        model,
        os.path.basename(model_name),
        data_filepath,
        batch_size,
        lr,
        weight_decay,
        epochs,
        device,
        all_labels,
        all_preds,
        val_mses[-1],
        timestamp,
        validation_gene_name=None,
        max_len=None,
        script_name=os.path.basename(__file__),
        model_description=None
    )
    # Save the model after the run notes are written.
    save_model_with_timestamp(model, os.path.basename(model_name))

if __name__ == "__main__":
    main()