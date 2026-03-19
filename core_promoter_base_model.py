# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 10:35:24 2024

@author: Christophe Jung
"""

# Model hyper-parameters
batch_size= 16
lr=1e-5
weight_decay=0.0006
epochs=25

# Filepath to the dataset and model directory
model_name = r"/home/.../DNABERT-2-117M_model"    # DNABERT2 model
data_filepath = r"/home/.../core_promoter_expressions.xlsx" # Expression dataset with sequences and expression values

import torch, time, os, gc
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertConfig
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer 
# Prevent truncation of long strings in columns
pd.set_option('display.max_colwidth', None)

def load_model_and_tokenizer(model_name):
    """Load the pre-trained model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = BertConfig.from_pretrained(model_name)
    
    # Set dropout rate in the configuration
    config.hidden_dropout_prob = 0.00001
    config.attention_probs_dropout_prob = 0.00001
    config.num_labels = 1  # Regression task with a single continuous value

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        config=config,
        trust_remote_code=True
    )
    
    return model, tokenizer


def prepare_data(filepath, tokenizer, batch_size, max_len=130):
    """Prepare the dataset and dataloaders"""
    df = pd.read_excel(filepath)
    # log2 conversion
    df['NORM'] = np.log2(np.clip(df['NORM'].astype(float), 5e-3, None))
    print(r'Expression: ', df['NORM'])
    
    # Convert sequences to capital letters
    df =df.apply(lambda col: col.str.upper() if col.dtype == "object" else col)

    print('Data preparation complete. Sample data:')
    print(df['SequenceSample'] .head(5))  # Display first 5 sequences for verification  
    # Dataset class
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
            # Print all tokens for the first 5 sequences for inspection
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
            # Conversion to string
            if not isinstance(sequence, str):
                sequence = str(sequence)
            expression = self.expressions[item]
            # Convert to k-mer list
            kmers = seq_to_kmers(sequence, self.k)
            n_kmers_allowed = self.max_len - 2
            if len(kmers) > n_kmers_allowed:
                kmers = kmers[:n_kmers_allowed]
            tokens = ['[CLS]'] + kmers + ['[SEP]']
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            # Pad if necessary
            if len(input_ids) < self.max_len:
                input_ids += [self.tokenizer.pad_token_id] * (self.max_len - len(input_ids))
            input_ids = torch.tensor(input_ids)
            return input_ids, torch.tensor(expression, dtype=torch.float)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(df['SequenceSample'].values, df['NORM'].values, test_size=0.1, random_state=42)
    train_dataset = GeneExpressionDataset(X_train, y_train, tokenizer, max_len, k=6)
    val_dataset = GeneExpressionDataset(X_val, y_val, tokenizer, max_len, k=6)  
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size)
    return train_loader, val_loader, X_val


def train_model(model, model_name, train_loader, optimizer, loss_fn, device, epochs):
    """Training loop"""
    model.train()
    train_losses = []
    for epoch in range(epochs):
        total_loss = 0
        for i, batch in enumerate(train_loader):
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            optimizer.zero_grad() # Zero out previous gradients
            outputs = model(input_ids)    
            logits = outputs.logits        
            logits = logits.squeeze()      
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            loss.backward()  # Backpropagate to compute gradients                         
            optimizer.step() # Update model parameters using AdamW

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)        
    return train_losses


def evaluate_model(model, val_loader, device):
    """Model evaluation on the validation dataset"""
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
            all_preds.extend(np.atleast_1d(preds.cpu().numpy()))
            all_labels.extend(np.atleast_1d(labels.cpu().numpy()))

    mse = mean_squared_error(all_labels, all_preds) 
    return all_preds, all_labels, mse


def plot_evaluation_results(all_preds, all_labels, model_name, output_dir=None):
    """Generate and save evaluation plots"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Scatter plot of predicted vs actual values 
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

    # 2. Histogram of residuals (predictions - actual)
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
    """Export predicted and actual values to an Excel file, with optional final validation MSE"""
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
    # Ensure results directory exists
    results_dir = "/home/.../results"
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
    model_filename = f"/home/.../results/{model_name}_model_{timestamp}.pth"
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved to {model_filename}")



def predict(model, tokenizer, sequences, device, max_len=130, k=6, save_attention_path=None):
    """Make predictions on new sequences and optionally save attention weights."""
    model.eval()
    predictions = []
    all_attentions = []
    with torch.no_grad():
        if isinstance(sequences, str):
            sequences = [sequences]  # Convert to a list if a single sequence is passed

        input_ids_list = []
        for seq in sequences:
            # Manual k-mer splitting and tokenization (same as in training)
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
            # Convert attention tensors to numpy arrays for saving
            all_attentions = [att.cpu().numpy() for att in outputs.attentions]
            if save_attention_path is not None:
                np.save(save_attention_path, all_attentions)
                print(f"Attention weights saved to {save_attention_path}")
    return predictions, all_attentions


def export_runinfo(model, model_name, data_filepath, batch_size, lr, weight_decay, epochs, device, all_labels, all_preds, mse, timestamp, validation_gene_name=None, max_len=None, script_name=None, model_description=None):
    """
    Export model parameters and evaluation metrics to a text file.

        model: The trained model (needed for config hash).
        model_name (str): Name of the model.
        data_filepath (str): Path to the dataset file.
        batch_size (int): Batch size used for training.
        lr (float): Learning rate used for training.
        weight_decay (float): Weight decay used for training.
        epochs (int): Number of training epochs.
        device: Device used for training.
        all_labels (list): Actual values.
        all_preds (list): Predicted values.
        mse (float): Mean squared error of predictions.
        timestamp (str): Timestamp for filenames.
        validation_gene_name (str, optional): Name of the gene used for validation in this run.
        max_len (int, optional): Max sequence length used for training.
        script_name (str, optional): Name of the Python script used for training.
        model_description (str, optional): Description of the model architecture and features.
    """
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
        r2 = r2_score(all_labels, all_preds)
        mae = mean_absolute_error(all_labels, all_preds)
        norm_mae = mae / (np.max(all_labels) - np.min(all_labels)) if (np.max(all_labels) - np.min(all_labels)) > 0 else float('nan')
        f.write(f"Validation MAE: {mae:.4f}\n")
        f.write(f"Validation R2: {r2:.4f}\n")
        f.write(f"Normalized MAE: {norm_mae:.4f}\n")
        # Export random seeds and reproducibility info
        import sys
        import transformers
        import random
        f.write(f"Random seed (numpy): {getattr(np.random, 'seed', 'N/A')}\n")
        f.write(f"Random seed (torch): {getattr(torch, 'initial_seed', lambda: 'N/A')()}\n")
        f.write(f"Random seed (random): {getattr(random, 'seed', 'N/A')}\n")
        f.write(f"train_test_split random_state:\n")
        # Export package versions
        f.write(f"Python version: {sys.version}\n")
        f.write(f"PyTorch version: {torch.__version__}\n")
        f.write(f"Transformers version: {transformers.__version__}\n")
        f.write(f"Pandas version: {pd.__version__}\n")
        f.write(f"Numpy version: {np.__version__}\n")
        # Model/config hash (if available)
        if hasattr(model, 'config') and hasattr(model.config, 'to_json_string'):
            import hashlib
            config_str = model.config.to_json_string()
            config_hash = hashlib.md5(config_str.encode('utf-8')).hexdigest()
            f.write(f"Model config hash: {config_hash}\n")
        # Add model description
        if model_description:
            f.write("\nModel Description:\n")
            f.write(model_description + "\n")
    print(f"Run info written to {params_txt_path}")


def count_model_parameters(model):
    """
    Returns the total number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


###################################################################################################
#                   MAIN
###################################################################################################
def main():
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # CUDA initialisation settings
    # if memory issues run to avoid fragmentation in linux:
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

    # Define optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay) 
    loss_fn = torch.nn.MSELoss()

    # Train the model
    print("######################################")
    print("Training model...")
    start_time = time.time()
    train_losses = []
    val_mses = []
    for epoch in range(epochs):
        train_loss = train_model(model, model_name, train_loader, optimizer, loss_fn, device, epochs=1)[0]
        train_losses.append(train_loss)
        # Evaluate on validation set after each epoch
        all_preds, all_labels, mse = evaluate_model(model, val_loader, device)
        val_mses.append(mse)
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss:.4f}, Validation MSE: {mse:.4f}")
    training_time = (time.time() - start_time)/60
    print(f"Training time: {training_time:.2f} minutes")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_run = os.path.join("/home/.../results", f"{os.path.basename(model_name)}_{timestamp}")
    os.makedirs(output_dir_run, exist_ok=True)
    
    # Save training/validation loss plot in output_dir_run
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
    
    # Save model and tokenizer 
    torch.save(model.state_dict(), os.path.join(output_dir_run, "pytorch_model.bin"))
    tokenizer.save_pretrained(output_dir_run)
    print(f"Model and tokenizer saved to {output_dir_run}")

    # Evaluate the model
    print("Evaluating model...")
    start_time = time.time()
    all_preds, all_labels, mse = evaluate_model(model, val_loader, device)
    eval_time = (time.time() - start_time)/60

    # In evaluate_model, check the shapes of all_preds and all_labels
    print(f"Predictions shape: {np.array(all_preds).shape}")
    print(f"Labels shape: {np.array(all_labels).shape}")
    print(f"Evaluation time: {eval_time:.2f} minutes")
    print(f"Validation MSE: {mse:.4f}")

    # Plot evaluation results
    # Save predicted vs actual and residuals plots in output_dir_run
    plot_evaluation_results(all_preds, all_labels, os.path.basename(model_name), output_dir=output_dir_run)
    # Export results to Excel
    export_results_to_excel(all_preds, all_labels, model_name, final_val_mse=val_mses[-1], sequences=X_val, names=[str(i) for i in range(len(X_val))], output_dir=output_dir_run)

    # Example prediction on new sequences
    new_sequences = ['GGTCTCAGGATTTTAAATAGATTTAGCTAGAAAATAGCTGACAGACACATATCGATATATCGCTGCGATAGCCACAGCTGTTCACGCCCGCAGTTTAAGCGtaGatcaccgaagctaCGGCCACCAAAAAATAAACATTGGATCTGTGAGACC', 'GGTCTCAGGATGAGAGAACCAGTGCGCTCTTATCACGTGAGAACGCTTTTGGGCATTCAGTTTGGCTTTTGCGGCGCTGACCGCTGGCGcttagtgCGAATCCATAGgcgctttcaccaatcgcAACGTAGGCCAGAACGGATCTGTGAGACC', 'GGTCTCAGGATGTGTGGCCCCTGTTAGCTTTCTGTTAAATTTAAATTTCTGTAAAGTGCCcgacgcctctctctctctctctctcATCAGAtcagttgTTGTCTGGATAtcgacgcgagcggtcggGATCGCGCATTAGTGTCATCTGTGAGACC']
    measured = [0.56960111, -4.839324055, 1.885391158]
    predictions, attentions = predict(model, tokenizer, new_sequences, device, save_attention_path="attention_weights_example.npy")
    print("Predictions on new sequences:", predictions)
    print("Measured on new sequences:", measured)
    print("Attention weights shape (per layer):", [a.shape for a in attentions] if attentions else None)

    # After validation/prediction, collect names, sequences, labels, preds
    # Example variable names: val_names, val_sequences, all_labels, all_preds
    # Add export logic after validation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Use X_val for sequences, index for names, val_mses[-1] for val_loss
    val_sequences = X_val
    val_names = [str(i) for i in range(len(X_val))]  # Use index as name if no explicit names
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
    # Save model after exporting runinfo (use correct signature)
    save_model_with_timestamp(model, os.path.basename(model_name))

if __name__ == "__main__":
    main()