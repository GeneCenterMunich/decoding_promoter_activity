# -*- coding: utf-8 -*-
"""
Train a DNABERT-2 model on promoter DNA sequences.
Also train two simple comparison models.
Save model files, tables, plots, and run notes.

IMPORTANT:
The training data is split by unique DNA sequence.

"""


import os
import gc
import time
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertConfig
from transformers import get_linear_schedule_with_warmup

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge


# ----------------------------------------
# Model settings
# ----------------------------------------

batch_size = 28
lr = 1e-5
weight_decay = 0.0006
epochs = 1
max_len = 130
seed = 42

# CNN comparison model settings
cnn_epochs = 25
cnn_lr = 1e-3
cnn_weight_decay = 1e-4
cnn_dropout = 0.2

# Input and output paths
model_name = r".../DNABERT-2-117M_model"
data_filepath = r".../core_promoter_training_data_250604.xlsx"
results_root = r".../results"

# Show full long text values in pandas output.
pd.set_option("display.max_colwidth", None)


# ----------------------------------------
# Repeatable runs
# ----------------------------------------


def set_global_seed(seed_value=42):
    """Set random seeds so repeat runs are easier to compare."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    # This can make runs slower, but results are easier to repeat.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------------------
# Load the model
# ----------------------------------------


def load_model_and_tokenizer(model_name):
    """
    Load the saved model files and the tokenizer.
    Return both objects for later use.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = BertConfig.from_pretrained(model_name)

    # The model predicts one number.
    config.num_labels = 1

    # These values match the earlier script.
    # Try larger dropout if the model memorizes the data.
    config.hidden_dropout_prob = 0.00001
    config.attention_probs_dropout_prob = 0.00001

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        trust_remote_code=True
    )

    return model, tokenizer


# ----------------------------------------
# Prepare the data
# ----------------------------------------


def clean_sequence(seq):
    """Return a DNA sequence in capital letters with spaces removed."""
    return str(seq).upper().replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")


class GeneExpressionDataset(Dataset):
    """Store promoter sequences and expression values for PyTorch."""

    def __init__(self, sequences, expressions, tokenizer, max_len):
        self.sequences = list(sequences)
        self.expressions = list(expressions)
        self.tokenizer = tokenizer
        self.max_len = max_len

        print("\n--- Token examples for first 2 sequences ---")
        for i in range(min(2, len(self.sequences))):
            seq = clean_sequence(self.sequences[i])
            encoded = self.tokenizer(
                seq,
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt"
            )
            tokens = self.tokenizer.convert_ids_to_tokens(encoded["input_ids"].squeeze(0)[:40])
            print(f"Sequence {i + 1} first 40 tokens: {tokens}")
        print("--- End token examples ---\n")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, item):
        sequence = clean_sequence(self.sequences[item])
        expression = float(self.expressions[item])

        encoding = self.tokenizer(
            sequence,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(expression, dtype=torch.float32)
        }


def prepare_data(filepath, tokenizer, batch_size, max_len=130):
    """
    Read the input data file.
    Clean the needed columns.
    Build data loaders for training and testing.
    """
    df = pd.read_excel(filepath)

    df = df[
        df["SequenceID"].str.contains("Block 1.15", na=False)
        & df["SequenceID"].str.contains("Block 7.15", na=False)
        & df["Ecd"].str.contains("No", na=False)
        & df["SequenceID"].str.endswith("Block 7.15", na=False)
    ].copy()

    # Remove extra sequence bases from known input lengths.
    df["SequenceSample"] = df["SequenceSample"].apply(
        lambda seq: seq[10:] if isinstance(seq, str) and len(seq) == 153 else seq
    )
    df["SequenceSample"] = df["SequenceSample"].apply(
        lambda seq: seq[12:] if isinstance(seq, str) and len(seq) == 157 else seq
    )

    # Convert expression values to log2 scale and avoid log of zero.
    df["NORM"] = np.log2(np.clip(df["NORM"].astype(float), 5e-3, None))
    print("Expression values after log2 transformation:")
    print(df["NORM"].describe())

    # Clean DNA text before the split.
    df["SequenceSample"] = df["SequenceSample"].apply(clean_sequence)

    print("Data preparation complete. Sample sequence:")
    print(df["SequenceSample"].head(1))

    # Split by unique DNA sequence.
    # Replicate rows stay together in the same split.
    unique_sequences = df["SequenceSample"].drop_duplicates()

    train_sequences, val_sequences = train_test_split(
        unique_sequences,
        test_size=0.1,
        random_state=seed,
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

    assert len(overlap) == 0, f"Sequence leakage detected: {len(overlap)} overlapping sequences"

    X_train = train_df["SequenceSample"].values
    y_train = train_df["NORM"].astype(float).values
    X_val = val_df["SequenceSample"].values
    y_val = val_df["NORM"].astype(float).values

    train_dataset = GeneExpressionDataset(X_train, y_train, tokenizer, max_len=max_len)
    val_dataset = GeneExpressionDataset(X_val, y_val, tokenizer, max_len=max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, X_train, y_train, X_val, y_val


# ----------------------------------------
# Train and test DNABERT-2
# ----------------------------------------


def train_one_epoch(model, train_loader, optimizer, loss_fn, device, scheduler=None):
    """
    Train the model one time through the training data.
    Return the average loss for that round.
    """
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.view(-1)
        labels = labels.view(-1)

        loss = loss_fn(logits, labels)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

    avg_train_loss = total_loss / len(train_loader)
    return avg_train_loss


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
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.view(-1)

            all_preds.extend(np.atleast_1d(preds.cpu().numpy()))
            all_labels.extend(np.atleast_1d(labels.cpu().numpy()))

    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)

    return all_preds, all_labels, mse, mae, r2


# ----------------------------------------
# Comparison model 1: short DNA word counts
# ----------------------------------------


def sequence_to_spaced_kmers(seq, k):
    """
    Split a DNA sequence into short overlapping words.
    Separate the words with spaces.
    """
    seq = clean_sequence(seq)
    if len(seq) < k:
        return ""
    return " ".join(seq[i:i + k] for i in range(len(seq) - k + 1))


def run_kmer_ridge_baseline(X_train, y_train, X_val, y_val, k=6):
    """
    Train a simple count-based comparison model.
    Return its test scores.
    """
    print("######################################")
    print(f"Running {k}-mer Ridge baseline...")

    X_train_kmers = [sequence_to_spaced_kmers(seq, k) for seq in X_train]
    X_val_kmers = [sequence_to_spaced_kmers(seq, k) for seq in X_val]

    model = Pipeline([
        ("vectorizer", CountVectorizer(token_pattern=r"(?u)\b\w+\b")),
        ("regressor", Ridge(alpha=1.0))
    ])

    model.fit(X_train_kmers, y_train)
    preds = model.predict(X_val_kmers)

    mse = mean_squared_error(y_val, preds)
    mae = mean_absolute_error(y_val, preds)
    r2 = r2_score(y_val, preds)

    print(
        f"{k}-mer Ridge baseline, "
        f"Validation MSE: {mse:.4f}, "
        f"MAE: {mae:.4f}, "
        f"R2: {r2:.4f}"
    )

    return {
        "Model": f"{k}-mer Ridge",
        "Validation_MSE": mse,
        "Validation_MAE": mae,
        "Validation_R2": r2
    }


# ----------------------------------------
# Comparison model 2: small DNA model
# ----------------------------------------


class OneHotSequenceDataset(Dataset):
    """Store DNA sequences as four rows: A, C, G, and T."""

    def __init__(self, sequences, expressions, max_len):
        self.sequences = list(sequences)
        self.expressions = list(expressions)
        self.max_len = max_len
        self.base_to_idx = {
            "A": 0,
            "C": 1,
            "G": 2,
            "T": 3
        }

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = clean_sequence(self.sequences[idx])
        x = torch.zeros(4, self.max_len, dtype=torch.float32)

        for pos, base in enumerate(seq[:self.max_len]):
            if base in self.base_to_idx:
                x[self.base_to_idx[base], pos] = 1.0

        y = torch.tensor(float(self.expressions[idx]), dtype=torch.float32)
        return x, y


class SimplePromoterCNN(torch.nn.Module):
    """Small comparison model that learns from one-hot DNA input."""

    def __init__(self, dropout=0.2):
        super().__init__()

        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(4, 64, kernel_size=8, padding=4),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2),

            torch.nn.Conv1d(64, 128, kernel_size=8, padding=4),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2),

            torch.nn.Conv1d(128, 128, kernel_size=8, padding=4),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.AdaptiveMaxPool1d(1)
        )

        self.regressor = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.regressor(x)
        return x.view(-1)


def evaluate_cnn_model(model, val_loader, device):
    """
    Run the CNN comparison model on the test data.
    Return predictions and score values.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)

            preds = model(x)

            all_preds.extend(np.atleast_1d(preds.cpu().numpy()))
            all_labels.extend(np.atleast_1d(y.cpu().numpy()))

    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)

    return all_preds, all_labels, mse, mae, r2


def run_cnn_baseline(X_train, y_train, X_val, y_val, device, output_dir, timestamp,
                     max_len=130, batch_size=28, epochs=25, lr=1e-3, weight_decay=1e-4,
                     dropout=0.2):
    """
    Train the CNN comparison model.
    Save its scores, plots, and predictions.
    """
    print("######################################")
    print("Running simple one-hot CNN baseline...")

    train_dataset = OneHotSequenceDataset(X_train, y_train, max_len=max_len)
    val_dataset = OneHotSequenceDataset(X_val, y_val, max_len=max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = SimplePromoterCNN(dropout=dropout).to(device)
    _, _, initial_mse, initial_mae, initial_r2 = evaluate_cnn_model(model, val_loader, device)

    print(
        f"Simple CNN before training, "
        f"Validation MSE: {initial_mse:.4f}, "
        f"MAE: {initial_mae:.4f}, "
        f"R2: {initial_r2:.4f}"
    )
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()

    train_losses = []
    val_mses = []
    val_maes = []
    val_r2s = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            preds = model(x)
            loss = loss_fn(preds, y.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        _, _, mse, mae, r2 = evaluate_cnn_model(model, val_loader, device)

        
        val_mses.append(mse)
        val_maes.append(mae)
        val_r2s.append(r2)

        print(
            f"Simple CNN epoch {epoch + 1}/{epochs}, "
            f"Training Loss: {avg_train_loss:.4f}, "
            f"Validation MSE: {mse:.4f}, "
            f"Validation MAE: {mae:.4f}, "
            f"Validation R2: {r2:.4f}"
        )

    # Check the final trained model.
    all_preds, all_labels, mse, mae, r2 = evaluate_cnn_model(model, val_loader, device)

    # Save CNN scores from each training round.
    cnn_metrics_df = pd.DataFrame({
        "Epoch": np.arange(1, len(train_losses) + 1),
        "Training_Loss": train_losses,
        "Validation_MSE": val_mses,
        "Validation_MAE": val_maes,
        "Validation_R2": val_r2s
    })

    cnn_metrics_path = os.path.join(output_dir, f"simple_cnn_training_metrics_{timestamp}.xlsx")
    cnn_metrics_df.to_excel(cnn_metrics_path, index=False)
    print(f"Simple CNN training metrics exported to {cnn_metrics_path}")

    # Save final CNN test predictions.
    cnn_results_df = pd.DataFrame({
        "Sequence": X_val,
        "Actual": all_labels,
        "Predicted": all_preds,
        "Residuals": np.array(all_preds) - np.array(all_labels)
    })

    cnn_results_path = os.path.join(output_dir, f"simple_cnn_evaluation_results_{timestamp}.xlsx")
    cnn_results_df.to_excel(cnn_results_path, index=False)
    print(f"Simple CNN evaluation results exported to {cnn_results_path}")

    # Save the learned CNN weights.
    cnn_model_path = os.path.join(output_dir, f"simple_cnn_model_{timestamp}.pth")
    torch.save(model.state_dict(), cnn_model_path)
    print(f"Simple CNN model saved to {cnn_model_path}")

    # Save the CNN R2 plot.
    epoch_axis = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epoch_axis, train_losses, marker="o", linestyle="-", label="Training Loss")
    plt.plot(epoch_axis, val_mses, marker="s", linestyle="--", label="Validation MSE")
    plt.plot(epoch_axis, val_maes, marker="^", linestyle=":", label="Validation MAE")
    plt.title("Simple CNN Training and Validation Metrics", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss / Error", fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"simple_cnn_training_loss_val_metrics_{timestamp}.png"))
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(epoch_axis, val_r2s, marker="o", linestyle="-", label="Validation R2")
    plt.title("Simple CNN Validation R2 over Epochs", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("R2", fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"simple_cnn_validation_r2_{timestamp}.png"))
    plt.close()

    print(
        f"Simple one-hot CNN baseline, "
        f"Validation MSE: {mse:.4f}, "
        f"MAE: {mae:.4f}, "
        f"R2: {r2:.4f}"
    )

    return {
        "Model": "Simple one-hot CNN",
        "Validation_MSE": mse,
        "Validation_MAE": mae,
        "Validation_R2": r2
    }


# ----------------------------------------
# Make plots and save files
# ----------------------------------------


def plot_training_metrics(train_losses, val_mses, val_maes, val_r2s, model_name, timestamp, output_dir):
    """Plot training and test scores for each training round."""
    safe_model_name = os.path.basename(model_name)
    epoch_axis = range(1, len(train_losses) + 1)

    # Plot loss and error scores.
    plt.figure(figsize=(8, 6))
    plt.plot(epoch_axis, train_losses, marker="o", linestyle="-", label="Training Loss")
    plt.plot(epoch_axis, val_mses, marker="s", linestyle="--", label="Validation MSE")
    plt.plot(epoch_axis, val_maes, marker="^", linestyle=":", label="Validation MAE")
    plt.title("Training Loss and Validation Error over Epochs", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss / Error", fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{safe_model_name}_training_loss_val_metrics_{timestamp}.png"))
    plt.close()

    # Plot R2 values.
    plt.figure(figsize=(8, 6))
    plt.plot(epoch_axis, val_r2s, marker="o", linestyle="-", label="Validation R2")
    plt.title("Validation R2 over Epochs", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("R2", fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{safe_model_name}_validation_r2_{timestamp}.png"))
    plt.close()


def plot_evaluation_results(all_preds, all_labels, model_name, output_dir=None):
    """
    Make plots that compare real values and predicted values.
    Save the plots to the output folder.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = os.path.basename(model_name)

    # Plot predicted values against real values.
    plt.figure(figsize=(8, 6))
    plt.scatter(all_labels, all_preds, alpha=0.7)
    plt.plot(
        [min(all_labels), max(all_labels)],
        [min(all_labels), max(all_labels)],
        color="red",
        lw=2
    )
    plt.title(f"Predicted vs Actual Values\n{safe_model_name}", fontsize=14)
    plt.xlabel("Actual", fontsize=12)
    plt.ylabel("Predicted", fontsize=12)
    plt.tight_layout()
    plt.grid(True)

    if output_dir:
        plt.savefig(os.path.join(output_dir, f"{safe_model_name}_pred_vs_actual_{timestamp}.png"))
    else:
        plt.savefig(f"{safe_model_name}_pred_vs_actual_{timestamp}.png")

    plt.show()
    plt.close()

    # Plot prediction errors.
    residuals = np.array(all_preds) - np.array(all_labels)
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True)
    plt.title(f"Residuals Distribution\n{safe_model_name}", fontsize=14)
    plt.xlabel("Residuals", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.tight_layout()
    plt.grid(True)

    if output_dir:
        plt.savefig(os.path.join(output_dir, f"{safe_model_name}_residuals_{timestamp}.png"))
    else:
        plt.savefig(f"{safe_model_name}_residuals_{timestamp}.png")

    plt.show()
    plt.close()


def export_training_metrics(train_losses, val_mses, val_maes, val_r2s, model_name, timestamp, output_dir):
    """Save score values from each training round to Excel."""
    safe_model_name = os.path.basename(model_name)

    metrics_df = pd.DataFrame({
        "Epoch": np.arange(1, len(train_losses) + 1),
        "Training_Loss": train_losses,
        "Validation_MSE": val_mses,
        "Validation_MAE": val_maes,
        "Validation_R2": val_r2s
    })

    file_path = os.path.join(output_dir, f"{safe_model_name}_training_metrics_{timestamp}.xlsx")
    metrics_df.to_excel(file_path, index=False)
    print(f"Training metrics exported to {file_path}")

    return metrics_df


def export_results_to_excel(all_preds, all_labels, model_name, final_val_mse=None, final_val_mae=None,
                            final_val_r2=None, sequences=None, names=None, output_dir=None):
    """Save real values and predicted values to an Excel file."""
    results_df = pd.DataFrame({
        "Actual": all_labels,
        "Predicted": all_preds,
        "Residuals": np.array(all_preds) - np.array(all_labels)
    })

    if names is not None:
        results_df.insert(0, "Name", names)

    if sequences is not None:
        results_df.insert(1 if names is not None else 0, "Sequence", sequences)

    if final_val_mse is not None:
        results_df["Final_Validation_MSE"] = [final_val_mse] + [None] * (len(results_df) - 1)

    if final_val_mae is not None:
        results_df["Final_Validation_MAE"] = [final_val_mae] + [None] * (len(results_df) - 1)

    if final_val_r2 is not None:
        results_df["Final_Validation_R2"] = [final_val_r2] + [None] * (len(results_df) - 1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = os.path.basename(model_name)
    file_name = f"{safe_model_name}_evaluation_results_{timestamp}.xlsx"

    if output_dir:
        file_path = os.path.join(output_dir, file_name)
    else:
        file_path = file_name

    results_df.to_excel(file_path, index=False)
    print(f"Results exported to {file_path}")


def export_baseline_comparison(baseline_results, dnabert_metrics, output_dir, timestamp):
    """Save a table that compares all models."""
    rows = [{
        "Model": "DNABERT-2 fine-tuned",
        "Validation_MSE": dnabert_metrics["Validation_MSE"],
        "Validation_MAE": dnabert_metrics["Validation_MAE"],
        "Validation_R2": dnabert_metrics["Validation_R2"]
    }]

    rows.extend(baseline_results)
    comparison_df = pd.DataFrame(rows)

    file_path = os.path.join(output_dir, f"model_baseline_comparison_{timestamp}.xlsx")
    comparison_df.to_excel(file_path, index=False)
    print(f"Baseline comparison exported to {file_path}")

    return comparison_df


def export_outliers_and_all_results(val_names, val_sequences, all_labels, all_preds, model_name, timestamp, output_dir):
    """Save all predictions and the largest prediction errors."""
    residuals = np.array(all_preds) - np.array(all_labels)
    std_res = np.std(residuals)
    abs_residuals = np.abs(residuals)
    outlier_mask = abs_residuals > 2 * std_res

    outlier_lines = ["Index\tName\tSequence\tActual\tPredicted\tResidual"]

    for i, is_outlier in enumerate(outlier_mask):
        if is_outlier:
            name = val_names[i] if i < len(val_names) else "N/A"
            seq = val_sequences[i] if i < len(val_sequences) else "N/A"
            actual = all_labels[i]
            pred = all_preds[i]
            resid = residuals[i]
            outlier_lines.append(f"{i}\t{name}\t{seq}\t{actual}\t{pred}\t{resid}")

    n_best = max(1, int(0.2 * len(all_labels)))
    best_indices = np.argsort(abs_residuals)[:n_best]

    outlier_lines.append("\n# 20% Best Predicted Sequences (lowest absolute residuals)")
    outlier_lines.append("Index\tName\tSequence\tActual\tPredicted\tResidual")

    for i in best_indices:
        name = val_names[i] if i < len(val_names) else "N/A"
        seq = val_sequences[i] if i < len(val_sequences) else "N/A"
        actual = all_labels[i]
        pred = all_preds[i]
        resid = residuals[i]
        outlier_lines.append(f"{i}\t{name}\t{seq}\t{actual}\t{pred}\t{resid}")

    safe_model_name = os.path.basename(model_name)
    outlier_txt_path = os.path.join(output_dir, f"{safe_model_name}_outliers_{timestamp}.txt")
    with open(outlier_txt_path, "w") as f:
        f.write("\n".join(outlier_lines))
    print(f"Outlier info written to {outlier_txt_path}")

    all_lines = ["Index\tName\tSequence\tActual\tPredicted\tResidual"]

    for i in range(len(all_labels)):
        name = val_names[i] if i < len(val_names) else "N/A"
        seq = val_sequences[i] if i < len(val_sequences) else "N/A"
        actual = all_labels[i]
        pred = all_preds[i]
        resid = residuals[i]
        all_lines.append(f"{i}\t{name}\t{seq}\t{actual}\t{pred}\t{resid}")

    all_txt_path = os.path.join(output_dir, f"{safe_model_name}_all_results_{timestamp}.txt")
    with open(all_txt_path, "w") as f:
        f.write("\n".join(all_lines))
    print(f"All results info written to {all_txt_path}")


def export_runinfo(model, model_name, data_filepath, batch_size, lr, weight_decay, epochs, device,
                   all_labels, all_preds, mse, mae, r2, timestamp, output_dir,
                   validation_gene_name=None, max_len=None, script_name=None, model_description=None,
                   val_mses=None, val_maes=None, val_r2s=None, baseline_results=None):
    """Save settings, scores, and software versions to a text file."""
    safe_model_name = os.path.basename(model_name)
    params_txt_path = os.path.join(output_dir, f"{safe_model_name}_runinfo_{timestamp}.txt")

    with open(params_txt_path, "w") as f:
        f.write(f"Model: {safe_model_name}\n")
        f.write(f"Training data file (full path): {data_filepath}\n")
        f.write(f"Training data file (basename): {os.path.basename(data_filepath)}\n")
        f.write(f"Python script used: {script_name}\n")
        f.write(f"Max sequence length (max_len): {max_len}\n")
        f.write("Validation strategy: sequence-wise holdout; exact duplicate sequences disjoint\n")

        if validation_gene_name:
            f.write(f"Validation gene: {validation_gene_name}\n")

        f.write(f"Final Validation MSE: {mse:.4f}\n")
        f.write(f"Final Validation MAE: {mae:.4f}\n")
        f.write(f"Final Validation R2: {r2:.4f}\n")

        if val_mses is not None and len(val_mses) > 0:
            f.write(f"Best Validation MSE: {np.min(val_mses):.4f}\n")
            f.write(f"Final Epoch Validation MSE: {val_mses[-1]:.4f}\n")

        if val_maes is not None and len(val_maes) > 0:
            f.write(f"Best Validation MAE: {np.min(val_maes):.4f}\n")
            f.write(f"Final Epoch Validation MAE: {val_maes[-1]:.4f}\n")

        if val_r2s is not None and len(val_r2s) > 0:
            f.write(f"Best Validation R2: {np.max(val_r2s):.4f}\n")
            f.write(f"Final Epoch Validation R2: {val_r2s[-1]:.4f}\n")

        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Learning rate: {lr}\n")
        f.write(f"Weight decay: {weight_decay}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write("Loss function: torch.nn.MSELoss\n")
        f.write("Optimizer: AdamW\n")
        f.write("Scheduler: linear warmup/decay\n")
        f.write(f"Device: {device}\n")
        f.write(f"Validation set size: {len(all_labels)}\n")
        f.write(f"Random seed: {seed}\n")
        f.write(f"CNN baseline epochs: {cnn_epochs}\n")
        f.write(f"CNN baseline learning rate: {cnn_lr}\n")
        f.write(f"CNN baseline weight decay: {cnn_weight_decay}\n")
        f.write(f"CNN baseline dropout: {cnn_dropout}\n")

        import sys
        import transformers

        f.write(f"Python version: {sys.version}\n")
        f.write(f"PyTorch version: {torch.__version__}\n")
        f.write(f"Transformers version: {transformers.__version__}\n")
        f.write(f"Pandas version: {pd.__version__}\n")
        f.write(f"Numpy version: {np.__version__}\n")

        if hasattr(model, "config") and hasattr(model.config, "to_json_string"):
            import hashlib
            config_str = model.config.to_json_string()
            config_hash = hashlib.md5(config_str.encode("utf-8")).hexdigest()
            f.write(f"Model config hash: {config_hash}\n")

        if baseline_results is not None:
            f.write("\nBaseline Results:\n")
            for result in baseline_results:
                f.write(
                    f"{result['Model']}: "
                    f"MSE={result['Validation_MSE']:.4f}, "
                    f"MAE={result['Validation_MAE']:.4f}, "
                    f"R2={result['Validation_R2']:.4f}\n"
                )

        if model_description:
            f.write("\nModel Description:\n")
            f.write(model_description + "\n")

    print(f"Run info written to {params_txt_path}")


# ----------------------------------------
# Predict new sequences
# ----------------------------------------


def predict(model, tokenizer, sequences, device, max_len=130, save_attention_path=None):
    """
    Predict values for new DNA sequences.
    Optionally save attention tables.
    """
    model.eval()
    all_attentions = []

    if isinstance(sequences, str):
        sequences = [sequences]

    cleaned_sequences = [clean_sequence(seq) for seq in sequences]

    encoding = tokenizer(
        cleaned_sequences,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        logits = outputs.logits.view(-1)
        predictions = logits.cpu().numpy()

        if hasattr(outputs, "attentions") and outputs.attentions is not None:
            all_attentions = [att.cpu().numpy() for att in outputs.attentions]
            if save_attention_path is not None:
                np.save(save_attention_path, np.array(all_attentions, dtype=object), allow_pickle=True)
                print(f"Attention weights saved to {save_attention_path}")

    return predictions, all_attentions


# ----------------------------------------
# Small helper functions
# ----------------------------------------


def save_model_with_timestamp(model, model_name, output_dir):
    """Save the model with the current date and time in the folder name."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = os.path.basename(model_name)
    model_filename = os.path.join(output_dir, f"{safe_model_name}_model_{timestamp}.pth")
    torch.save(model.state_dict(), model_filename)
    print(f"Model state dict saved to {model_filename}")


def count_model_parameters(model):
    """Count how many model weights can be learned during training."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ----------------------------------------
# Main run
# ----------------------------------------


def main():
    set_global_seed(seed)

    # Set GPU memory behavior before the model starts.
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gc.collect()

    # Load the model and tokenizer.
    model, tokenizer = load_model_and_tokenizer(model_name)
    model.to(device)

    print(f"Model loaded: {model_name}")
    print(f"Tokenizer loaded: {model_name}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print("Total trainable parameters:", count_model_parameters(model))

    # Read and prepare the input data.
    train_loader, val_loader, X_train, y_train, X_val, y_val = prepare_data(
        data_filepath,
        tokenizer,
        batch_size=batch_size,
        max_len=max_len
    )

    # Create the training tools.
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()

    num_training_steps = epochs * len(train_loader)
    num_warmup_steps = int(0.1 * num_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # Train the model.
    print("######################################")
    print("Training model...")

    start_time = time.time()
    train_losses = []
    val_mses = []
    val_maes = []
    val_r2s = []

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            scheduler=scheduler
        )

        train_losses.append(train_loss)

        # Check the model on the test data after each round.
        all_preds, all_labels, mse, mae, r2 = evaluate_model(model, val_loader, device)
        val_mses.append(mse)
        val_maes.append(mae)
        val_r2s.append(r2)

        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Training Loss: {train_loss:.4f}, "
            f"Validation MSE: {mse:.4f}, "
            f"Validation MAE: {mae:.4f}, "
            f"Validation R2: {r2:.4f}"
        )

    training_time = (time.time() - start_time) / 60
    print(f"Training time: {training_time:.2f} minutes")

    # Make a new output folder with the current date and time.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_run = os.path.join(results_root, f"{os.path.basename(model_name)}_{timestamp}")
    os.makedirs(output_dir_run, exist_ok=True)

    # Save scores from each training round.
    export_training_metrics(
        train_losses,
        val_mses,
        val_maes,
        val_r2s,
        model_name,
        timestamp,
        output_dir_run
    )

    plot_training_metrics(
        train_losses,
        val_mses,
        val_maes,
        val_r2s,
        model_name,
        timestamp,
        output_dir_run
    )

    # Save the model and tokenizer.
    model.save_pretrained(output_dir_run)
    tokenizer.save_pretrained(output_dir_run)
    print(f"Model and tokenizer saved to {output_dir_run}")

    # Also save the raw learned weights.
    torch.save(model.state_dict(), os.path.join(output_dir_run, "pytorch_model.bin"))

    # Check the final DNABERT-2 model.
    print("######################################")
    print("Evaluating model...")

    start_time = time.time()
    all_preds, all_labels, mse, mae, r2 = evaluate_model(model, val_loader, device)
    eval_time = (time.time() - start_time) / 60

    print(f"Predictions shape: {np.array(all_preds).shape}")
    print(f"Labels shape: {np.array(all_labels).shape}")
    print(f"Evaluation time: {eval_time:.2f} minutes")
    print(f"Validation MSE: {mse:.4f}")
    print(f"Validation MAE: {mae:.4f}")
    print(f"Validation R2: {r2:.4f}")

    dnabert_metrics = {
        "Validation_MSE": mse,
        "Validation_MAE": mae,
        "Validation_R2": r2
    }

    # Check the final trained model.
    plot_evaluation_results(all_preds, all_labels, os.path.basename(model_name), output_dir=output_dir_run)

    # Save final test predictions.
    export_results_to_excel(
        all_preds,
        all_labels,
        model_name,
        final_val_mse=mse,
        final_val_mae=mae,
        final_val_r2=r2,
        sequences=X_val,
        names=[str(i) for i in range(len(X_val))],
        output_dir=output_dir_run
    )

    # Run the comparison models on the same split.
    print("######################################")
    print("Running reviewer-response baseline models...")

    baseline_results = []

    baseline_results.append(
        run_kmer_ridge_baseline(
            X_train,
            y_train,
            X_val,
            y_val,
            k=6
        )
    )

    baseline_results.append(
        run_cnn_baseline(
            X_train,
            y_train,
            X_val,
            y_val,
            device=device,
            output_dir=output_dir_run,
            timestamp=timestamp,
            max_len=max_len,
            batch_size=batch_size,
            epochs=cnn_epochs,
            lr=cnn_lr,
            weight_decay=cnn_weight_decay,
            dropout=cnn_dropout
        )
    )

    comparison_df = export_baseline_comparison(
        baseline_results,
        dnabert_metrics,
        output_dir_run,
        timestamp
    )

    print("Model comparison:")
    print(comparison_df)

    # Example for predicting new sequences.
    new_sequences = [
        "GGTCTCAGGATTTTAAATAGATTTAGCTAGAAAATAGCTGACAGACACATATCGATATATCGCTGCGATAGCCACAGCTGTTCACGCCCGCAGTTTAAGCGtaGatcaccgaagctaCGGCCACCAAAAAATAAACATTGGATCTGTGAGACC",
        "GGTCTCAGGATGAGAGAACCAGTGCGCTCTTATCACGTGAGAACGCTTTTGGGCATTCAGTTTGGCTTTTGCGGCGCTGACCGCTGGCGcttagtgCGAATCCATAGgcgctttcaccaatcgcAACGTAGGCCAGAACGGATCTGTGAGACC",
        "GGTCTCAGGATGTGTGGCCCCTGTTAGCTTTCTGTTAAATTTAAATTTCTGTAAAGTGCCcgacgcctctctctctctctctctcATCAGAtcagttgTTGTCTGGATAtcgacgcgagcggtcggGATCGCGCATTAGTGTCATCTGTGAGACC"
    ]

    measured = [0.56960111, -4.839324055, 1.885391158]
    attention_path = os.path.join(output_dir_run, "attention_weights_example.npy")

    predictions, attentions = predict(
        model,
        tokenizer,
        new_sequences,
        device,
        max_len=max_len,
        save_attention_path=attention_path
    )

    print("Predictions on new sequences:", predictions)
    print("Measured on new sequences:", measured)
    print("Attention weights shape per layer:", [a.shape for a in attentions] if attentions else None)

    # Save all predictions and large errors.
    val_sequences = X_val
    val_names = [str(i) for i in range(len(X_val))]

    export_outliers_and_all_results(
        val_names,
        val_sequences,
        all_labels,
        all_preds,
        model_name,
        timestamp,
        output_dir_run
    )

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
        mse,
        mae,
        r2,
        timestamp,
        output_dir_run,
        validation_gene_name=None,
        max_len=max_len,
        script_name=os.path.basename(__file__),
        model_description="DNABERT-2 fine-tuned for promoter-expression regression. Baselines: 6-mer Ridge regression and simple one-hot CNN trained from scratch on the identical sequence-wise holdout split.",
        val_mses=val_mses,
        val_maes=val_maes,
        val_r2s=val_r2s,
        baseline_results=baseline_results
    )

    save_model_with_timestamp(model, os.path.basename(model_name), output_dir_run)


if __name__ == "__main__":
    main()
