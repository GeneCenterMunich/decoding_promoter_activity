# -*- coding: utf-8 -*-
"""
Train a DNABERT-2 model with promoter sequence data.
Also use Ecd status and motif score columns.
Save model files, plots, tables, and run notes.

IMPORTANT:
In this version, the training data is split by unique DNA sequence.
"""



# ----------------------------------------
# Model settings
batch_size= 28 # Number of rows used in one training step.
lr=1e-5  # Step size used by the optimizer.
weight_decay=0.0004 # Small setting that helps the model avoid memorizing the data.
dropout=0.001  # Chance to skip part of the model during training.
epochs=25 # Number of full passes through the training data.
# Share of data used for the test split.
# Not used when one gene is held out.
test_size = 0.101
# Input and output paths
model_name = r".../DNABERT-2-117M_model"# Folder with the DNABERT-2 base model.

# Optional example; edit and uncomment only if you need it.
data_filepath = r".../core_promoter_training_data_250604.xlsx"


# Optional example; edit and uncomment only if you need it.
# Optional example; edit and uncomment only if you need it.
# Optional example; edit and uncomment only if you need it.

# Folder for model files.
# The folder must contain model and tokenizer files.
max_len = 130  # Maximum number of tokens used for each sequence.
# Use True for one random split; use False for gene hold-out runs.
RandomSplit = True

# ----------------------------------------


import torch, time, os, gc
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertConfig
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error 
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.model_selection import GroupShuffleSplit


class ModifiedModel(torch.nn.Module):
    """
    Model that joins three inputs.
    Input 1: DNABERT-2 sequence output.
    Input 2: Ecd yes/no value.
    Input 3: motif score values.
    Output: one predicted expression value.
    """
    def __init__(self, base_model, hidden_size, num_motif_score_columns): # Add num_motif_score_columns
        super(ModifiedModel, self).__init__()
        self.base_model = base_model
        
        # This layer changes the base model output size.
        self.base_projection = torch.nn.Linear(1, hidden_size) 
        self.batch_norm1_base = torch.nn.BatchNorm1d(hidden_size) # This layer changes the base model output size.

        self.ecd_embedding = torch.nn.Embedding(2, hidden_size // 2) # This stores the Ecd yes/no value.

        # This layer reads all motif score columns.
        self.all_motifs_projection = torch.nn.Linear(num_motif_score_columns, hidden_size // 2)

        # This layer reads sequence, Ecd, and motif features together.
        self.additional_layer1 = torch.nn.Linear(hidden_size * 2, hidden_size)
        self.batch_norm1 = torch.nn.BatchNorm1d(hidden_size) # This layer helps keep values stable during training.
        self.additional_layer2 = torch.nn.Linear(hidden_size, hidden_size // 2)
        self.batch_norm2 = torch.nn.BatchNorm1d(hidden_size // 2) # This layer helps keep values stable during training.
        self.dropout = torch.nn.Dropout(p=dropout) # Use the skip chance set above.
        self.classifier = torch.nn.Linear(hidden_size // 2, 1) # Final layer that predicts one number.

        # Set starting values for new model layers.
        torch.nn.init.xavier_uniform_(self.base_projection.weight)
        torch.nn.init.xavier_uniform_(self.ecd_embedding.weight)
        torch.nn.init.xavier_uniform_(self.all_motifs_projection.weight)
        torch.nn.init.xavier_uniform_(self.additional_layer1.weight)
        torch.nn.init.xavier_uniform_(self.additional_layer2.weight)
        torch.nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, input_ids, ecd_feature, all_motif_scores): 
        # Get the sequence model output.
        base_output = self.base_model(input_ids).logits
        # Change the sequence output to the needed size.
        base_output = self.base_projection(base_output)
        base_output = self.batch_norm1_base(base_output) # This layer helps keep values stable during training.

        # Turn the Ecd value into numbers for the model.
        ecd_embedded = self.ecd_embedding(ecd_feature)
        # Remove the extra size-one dimension.
        ecd_embedded = ecd_embedded.squeeze(1)
        # Turn motif scores into numbers for the model.
        all_motifs_projected = self.all_motifs_projection(all_motif_scores)

        # Join sequence, Ecd, and motif values.
        combined_input = torch.cat((base_output, ecd_embedded, all_motifs_projected), dim=1) # Expected tensor shape.

        # Use the final layers to make one prediction.
        output = torch.relu(self.additional_layer1(combined_input))
        output = self.batch_norm1(output)
        output = torch.relu(self.additional_layer2(output))
        output = self.batch_norm2(output)
        output = self.dropout(output)
        output = self.classifier(output)

        return SequenceClassifierOutput(logits=output)
 
    
def load_model_and_tokenizer(model_name):
    """
    Load the saved model files and the tokenizer.
    Return both objects for later use.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = BertConfig.from_pretrained(model_name)
    # Chance to skip part of the model during training.
    config.hidden_dropout_prob = dropout
    config.attention_probs_dropout_prob = dropout  # Set dropout for attention layers.
    config.num_labels = 1  # The model predicts one number.
    # Important note:
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        config=config,
        trust_remote_code=True
    )
    return model, tokenizer


def seq_to_kmers(seq, k=6):
    seq = seq.upper().replace(" ", "").replace("\n", "")
    return [seq[i:i+k] for i in range(len(seq)-k+1)]


def prepare_data(filepath, tokenizer, batch_size, max_len=130, log_transform_targets=False, validation_gene_name=None):
    """
    Read the input data file.
    Clean the needed columns.
    Build data loaders for training and testing.
    """
    df = pd.read_excel(filepath)
    df['NORM'] = np.log2(np.clip(df['NORM'].astype(float), 5e-3, None))
    # Add condition if necessary

    # Change Ecd text into 0 or 1.
    df['Ecd'] = df['Ecd'].apply(lambda x: 1 if x == 'Yes' else 0)   

    # List the motif score columns used by the model.
    motif_score_columns = [
        'Block1', 'Block7', 'INR_score', 'MTEDPE_score', 'CGpal_score',
        'GAGA_score', 'GAGArev_score', 'TATA-box_score', 'ATGAA_score',
        'CA-INR_score', 'INR2_score', 'Ohler6_score', 'DRE_score',
        'Ohler7_score', 'E-box1_score', 'TTGTT_score', 'TTGTTrev_score',
        'R-INR_score', 'RDPE_score'
    ]
    for col in motif_score_columns:
        df[col] = df[col].fillna(0).replace('', 0).astype(float)
        df[col] = df[col].apply(lambda x: 0 if x < 0 else x)
    # Keep the selected motif score columns.
    motif_score_columns = motif_score_columns[ :] # + all other motif scores if needed

    from sklearn.preprocessing import StandardScaler
    scaler_motifs = StandardScaler()
    print("motif_scor_col:",motif_score_columns)
    # Optional example; edit and uncomment only if you need it.
    # Use motif scores without scaling.
    all_motif_scores_scaled = df[motif_score_columns].values

    # Make sequence text uppercase.
    df = df.apply(lambda col: col.str.upper() if col.dtype == "object" else col)

    # Clean sequence text and remove known extra bases.
    df['SequenceSample'] = df['SequenceSample'].astype(str)

    
    df['SequenceSample'] = df['SequenceSample'].apply(lambda seq: seq[10:] if isinstance(seq, str) and len(seq)==153 else seq)
    df['SequenceSample'] = df['SequenceSample'].apply(lambda seq: seq[12:] if isinstance(seq, str) and len(seq)==157 else seq)
    # Optional sequence trimming step.
    # Optional example; edit and uncomment only if you need it.


    class GeneExpressionDataset(Dataset):
        def __init__(self, sequences, expressions, tokenizer, max_len, ecd_features, all_motif_scores_scaled, k=6):
            self.sequences = sequences
            self.expressions = expressions
            self.tokenizer = tokenizer
            self.max_len = max_len
            self.ecd_features = ecd_features
            self.all_motif_scores_scaled = all_motif_scores_scaled
            self.k = k
            '''
            print("\n--- Token examples for first sequences ---")
            for i in range(min(3, len(self.sequences))):
                kmers = seq_to_kmers(self.sequences[i], self.k)
                n_kmers_allowed = self.max_len - 2
                if len(kmers)   > n_kmers_allowed:
                    kmers = kmers[:n_kmers_allowed]
                tokens = ['[CLS]'] + kmers + ['[SEP]']
                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                print(f"Sequence {i+1} tokens: {tokens}")
            print("--- End token examples ---\n")
            '''
        def __len__(self):
            return len(self.sequences)

        def __getitem__(self, item):
            sequence = self.sequences[item]
            expression = self.expressions[item]
            ecd_feature = self.ecd_features[item]
            all_motif_scores = self.all_motif_scores_scaled[item]
            kmers = seq_to_kmers(sequence, self.k)
            n_kmers_allowed = self.max_len - 2
            if len(kmers) > n_kmers_allowed:
                kmers = kmers[:n_kmers_allowed]
            tokens = ['[CLS]'] + kmers + ['[SEP]']
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            if len(input_ids) < self.max_len:
                input_ids += [self.tokenizer.pad_token_id] * (self.max_len - len(input_ids))
            input_ids = torch.tensor(input_ids)
            return input_ids, torch.tensor(expression, dtype=torch.float), torch.tensor(ecd_feature, dtype=torch.long), torch.tensor(all_motif_scores, dtype=torch.float)

    df['Gene'] = df['Gene'].str.upper().str.replace(' ', '')
    if validation_gene_name:
        val_df = df[df['Gene'] == validation_gene_name]
        train_df = df[df['Gene'] != validation_gene_name]
        all_motif_scores_train = all_motif_scores_scaled[train_df.index]
        all_motif_scores_val = all_motif_scores_scaled[val_df.index] 
        X_train, y_train, ecd_train = train_df['SequenceSample'].values, train_df['NORM'].values, train_df['Ecd'].values
        X_val, y_val, ecd_val = val_df['SequenceSample'].values, val_df['NORM'].values, val_df['Ecd'].values
    else:
        groups = df['SequenceSample'].values   # This keeps equal sequences in the same split.

        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=101
        )

        train_idx, val_idx = next(gss.split(df, groups=groups))

        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()

        X_train = train_df['SequenceSample'].values
        X_val = val_df['SequenceSample'].values

        y_train = train_df['NORM'].values
        y_val = val_df['NORM'].values

        ecd_train = train_df['Ecd'].values
        ecd_val = val_df['Ecd'].values

        all_motif_scores_train = all_motif_scores_scaled[train_idx]
        all_motif_scores_val = all_motif_scores_scaled[val_idx]

        print("dim train_df:", len(train_df))
        print("dim val_df:", len(val_df))
        print("unique train sequences:", train_df['SequenceSample'].nunique())
        print("unique val sequences:", val_df['SequenceSample'].nunique())
        # Optional example; edit and uncomment only if you need it.
        print("dim val_df (total):",len(val_df))

    train_dataset = GeneExpressionDataset(X_train, y_train, tokenizer, max_len, ecd_train, all_motif_scores_train)
    val_dataset = GeneExpressionDataset(X_val, y_val, tokenizer, max_len, ecd_val, all_motif_scores_val)
    num_motif_score_columns = len(motif_score_columns)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size)   
    print("############################################################################################################")
    return train_loader, val_loader, (X_val, y_val, val_df), num_motif_score_columns
 

def train_model(model, model_name, train_loader, val_loader, optimizer, loss_fn, device, epochs, logger=None, scheduler=None):
    """
    Train the model for the number of rounds set above.
    Return the loss values from training.
    """
    train_losses = []
    val_losses = []
    log_interval = 10  # Write progress every 10 steps.
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, batch in enumerate(train_loader):         
            input_ids, labels, ecd_feature, all_motif_scores = batch 
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            ecd_feature = ecd_feature.to(device)
            all_motif_scores = all_motif_scores.to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, ecd_feature, all_motif_scores) 
            # Use the prediction values in the right shape.
            loss = loss_fn(outputs.logits.view(-1), labels)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # Save training loss for TensorBoard.
            if scheduler is not None:
                scheduler.step()
            # Step size used by the optimizer.
            if logger is not None:
                current_lr = optimizer.param_groups[0]['lr']
                global_step = epoch * len(train_loader) + i
                logger.add_scalar('LearningRate', current_lr, global_step)
            # Save gradient size values for TensorBoard.
            if logger is not None:
                current_lr = optimizer.param_groups[0]['lr']
                global_step = epoch * len(train_loader) + i

                if global_step % log_interval == 0:  # Write progress only at the chosen step interval.
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            logger.add_scalar(f'Gradients/{name}', param.grad.norm().item(), global_step)
            # Limit very large gradients.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        # Check loss on the test data.
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids, labels, ecd_feature, all_motif_scores = batch
                # Move data to CPU or GPU.
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                ecd_feature = ecd_feature.to(device)
                all_motif_scores = all_motif_scores.to(device) 
                
                # Call the model with all needed inputs.
                outputs = model(input_ids, ecd_feature, all_motif_scores)
                
                # Use model prediction values for the loss.
                logits = outputs.logits.view(-1) 
                loss = loss_fn(logits, labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        if logger is not None:
            logger.add_scalar('Loss/train', avg_train_loss, epoch)
            logger.add_scalar('Loss/val', avg_val_loss, epoch)
        model.train()
    return train_losses, val_losses


def evaluate_model(model, device, val_loader, loss_fn, log_transform_targets=False, y_val_orig=None):
    """
    Run the model on the test data.
    Return predictions and score values.
    """
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            # Read sequence, label, Ecd, and motif values from the batch.
            input_ids, labels, ecd_feature, all_motif_scores = batch 
            
            # Move data to CPU or GPU.
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            ecd_feature = ecd_feature.to(device)
            all_motif_scores = all_motif_scores.to(device) 
            
            # Call the model with all needed inputs.
            outputs = model(input_ids, ecd_feature, all_motif_scores)
            
            # Use model prediction values for the loss.
            logits = outputs.logits.view(-1) 
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            all_preds.extend(logits.cpu().numpy()) # Store prediction numbers, not the full model output.
            all_labels.extend(labels.cpu().numpy())
    
    mse = total_loss / len(val_loader)
    return all_preds, all_labels, mse


def plot_evaluation_results(all_preds, all_labels, model_name):
    """
    Make plots that compare real values and predicted values.
    Save the plots to the output folder.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot predicted values against real values.
    plt.figure(figsize=(8, 6))
    plt.scatter(all_labels, all_preds, alpha=0.3, edgecolors='none')  # Make points partly transparent.
    plt.plot([min(all_labels), max(all_labels)], [min(all_labels), max(all_labels)], color='red', lw=2)
    plt.xlabel('log2(X)')
    plt.ylabel('log2(Y)')
    plt.title(f'Predicted vs Actual Values \n{model_name}', fontsize=14)
    plt.xlabel('Actual', fontsize=12)
    plt.ylabel('Predicted', fontsize=12)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(f"/home/be-em/data/Core_Promoter_2015/results/{model_name}_pred_vs_actual{timestamp}.png")
    plt.show()
    plt.close()

    # Plot prediction errors.
    residuals = np.array(all_preds) - np.array(all_labels)
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, alpha=0.3)  # Make points partly transparent.
    plt.title(f'Residuals Distribution\n{model_name}', fontsize=14)
    plt.xlabel('Residuals', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"/home/be-em/data/Core_Promoter_2015/results/{model_name}_residuals_{timestamp}.png")
    plt.grid(True)
    plt.show()
    plt.close()


def export_results_to_excel(all_preds, all_labels, model_name, sequences=None, names=None):
    """Save real values and predicted values to an Excel file."""
    results_dict = {
        'Actual': all_labels,
        'Predicted': all_preds,
        'Residuals': np.array(all_preds) - np.array(all_labels)
    }
    """
    # Optional: Include sequences and names if provided - nor working yet....
    if names is not None:
        results_dict['Name'] = names
    if sequences is not None:
        results_dict['Sequence'] = sequences
    """
    # Make a table and save it to Excel.
    results_df = pd.DataFrame(results_dict)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"/home/be-em/data/Core_Promoter_2015/results/{model_name}_evaluation_results_{timestamp}.xlsx"
    results_df.to_excel(file_name, index=False)
    print(f"Results exported to {file_name}")


def save_model_with_timestamp(model, model_name):
    """Save the model with the current date and time in the folder name."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Use the current date and time in the file name.
    model_filename = f"{model_name}_model_{timestamp}.pth"  # Add the model name and time to the folder name.
    torch.save(model.state_dict(), model_filename)  # Save the learned model weights.
    print(f"Model saved to {model_filename}")


def predict(model, tokenizer, sequences, device, ecd_features_for_prediction, all_motif_scores_for_prediction):
    """
    Predict values for new DNA sequences.
    Optionally save attention tables.
    """
    model.eval()
    predictions = []
    with torch.no_grad():
        if isinstance(sequences, str):
            sequences = [sequences]

        encoding = tokenizer(sequences, truncation=True, padding='max_length', max_length=130, return_tensors="pt")
        input_ids = encoding['input_ids'].to(device)

        # Move Ecd and motif data to the same device as the model.
        ecd_features_for_prediction = ecd_features_for_prediction.to(device)
        all_motif_scores_for_prediction = all_motif_scores_for_prediction.to(device)

        outputs = model(input_ids, ecd_features_for_prediction, all_motif_scores_for_prediction) # Call the model with sequence, Ecd, and motif inputs.
        logits = outputs.logits.view(-1)
        predictions = logits.cpu().numpy()
    return predictions



def export_outliers_and_all_results(val_names, val_sequences, all_labels, all_preds, model_name, timestamp):
    """Save all predictions and the largest prediction errors."""
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
    # Save the rows with the smallest errors.
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
    outlier_txt_path = f"/home/be-em/data/Core_Promoter_2015/results/{model_name}_outliers_{timestamp}.txt"
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
    all_txt_path = f"/home/be-em/data/Core_Promoter_2015/results/{model_name}_all_results_{timestamp}.txt"
    with open(all_txt_path, 'w') as f:
        f.write("\n".join(all_lines))
    print(f"All results info written to {all_txt_path}")


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
        f.write(f"Dropout (hidden): {dropout}\n")
        f.write(f"Dropout (attention): {dropout}\n")
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


def plot_training_validation_loss(train_losses, val_losses, epochs, model_name, output_dir):
    """Plot training loss and test loss over time."""
    plt.figure(figsize=(8, 6))
    plt.plot(range(epochs), train_losses, marker='o', linestyle='-', color='b', label='Train Loss')
    plt.plot(range(epochs), val_losses, marker='s', linestyle='-', color='orange', label='Val Loss')
    plt.title('Training & Validation Loss over Epochs', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"/home/be-em/data/Core_Promoter_2015/results/{model_name}_train_val_loss.png")
    plt.savefig(output_path)
    plt.show()
    plt.close()


def main():
    start_time_total = time.time()
    # Optional example; edit and uncomment only if you need it.

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
    base_dnabert_model, tokenizer = load_model_and_tokenizer(model_name)
    # Existing code continues here.
    log_transform_targets = False  # Set this to True to use log10 values.

    # Read all data once to find gene names.
    df_full = pd.read_excel(data_filepath) 

    # Get one copy of each gene name.
    unique_genes = df_full['Gene'].str.upper().str.replace(' ', '').unique()
    print(f"Discovered {len(unique_genes)} unique genes for cross-validation: {unique_genes}")
    
    short_model_identifier = os.path.basename(model_name) # Get the model folder name from the full path.
    training_file_basename = os.path.basename(data_filepath)
    script_name = os.path.basename(__file__) if '__file__' in globals() else 'unknown'
    model_description = (
        "ModifiedModel: DNABERT-based regression model with additional features. "
        "Incorporates Ecd_Present (binary embedding) and motif scores (linear projection). "
        "Architecture: base DNABERT model, base_projection (1->hidden_size), BatchNorm, "
        "ecd_embedding (2->hidden_size//2), all_motifs_projection (motif_count->hidden_size//2), "
        "concatenation, additional_layer1 (hidden_size*2->hidden_size), BatchNorm, "
        "additional_layer2 (hidden_size->hidden_size//2), BatchNorm, dropout, classifier (hidden_size//2->1). "
        "Motif scores used: Block1, Block7. Sequences tokenized as 6-mers, padded to max_len. "
        "Training: MSE loss, AdamW optimizer, cosine scheduler. Validation: random split or gene-wise holdout."
    )

    if RandomSplit:
        # Run one training job with a random split.
        gene_name = 'random_split'
        print(f"\n--- Starting run for random split ---")
        train_loader, val_loader, val_data_tuple, num_motif_features = prepare_data(
            data_filepath, tokenizer, batch_size=batch_size,
            log_transform_targets=log_transform_targets,
            validation_gene_name=None
        )
        X_val, y_val, val_df = val_data_tuple
        model = ModifiedModel(base_dnabert_model, base_dnabert_model.config.hidden_size, num_motif_features)
        model.to(device)
        print(f"Model refined to incorporate Ecd_Present feature and {num_motif_features} motif features")
        print(f"Number of training samples: {len(train_loader.dataset)}")
        print(f"Number of validation samples: {len(val_loader.dataset)}")
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()
        total_steps = len(train_loader) * epochs
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)
        log_dir_run = os.path.join("runs", short_model_identifier, gene_name, datetime.now().strftime("%Y%m%d_%H%M%S"))
        logger_run = SummaryWriter(log_dir_run)
        print(f"Training model for random split...")
        start_time = time.time()
        train_losses, val_losses = train_model(model, model_name, train_loader, val_loader, optimizer, loss_fn, device, epochs=epochs, logger=logger_run, scheduler=scheduler)
        training_time = (time.time() - start_time)/60
        print(f"Training time for random split: {training_time:.2f} minutes")
        logger_run.close()
        timestamp_run = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir_run = os.path.join("/home/be-em/data/Core_Promoter_2015/results", f"{short_model_identifier}_{gene_name}_{training_file_basename}_{timestamp_run}")
        os.makedirs(output_dir_run, exist_ok=True)
        plot_training_validation_loss(train_losses, val_losses, epochs, f"{short_model_identifier}_{gene_name}", output_dir_run)
        torch.save(model.state_dict(), os.path.join(output_dir_run, "pytorch_model.bin"))
        tokenizer.save_pretrained(output_dir_run)
        print(f"Evaluating model for random split...")
        start_time = time.time()
        all_preds, all_labels, mse = evaluate_model(model, device, val_loader, loss_fn, log_transform_targets=log_transform_targets, y_val_orig=y_val if log_transform_targets else None)
        eval_time = (time.time() - start_time)/60
        print(f"Evaluation time for random split: {eval_time:.2f} minutes")
        print(f"Validation MSE for random split: {mse:.4f}")
        plot_evaluation_results(all_preds, all_labels, f"{short_model_identifier}_{gene_name}")
        val_sequences_current_run = val_df['SequenceSample'].values
        val_names_current_run = val_df['SequenceID'].values
        export_results_to_excel(all_preds, all_labels, f"{short_model_identifier}_{gene_name}",
                                sequences=val_sequences_current_run, names=val_names_current_run)
        export_outliers_and_all_results(val_names_current_run, val_sequences_current_run,
                                        all_labels, all_preds, f"{short_model_identifier}_{gene_name}", timestamp_run)
        export_runinfo(model, f"{short_model_identifier}_{gene_name}", data_filepath, batch_size, lr, weight_decay, epochs, device,
                    all_labels, all_preds, mse, timestamp_run, validation_gene_name=gene_name, max_len=max_len, script_name=script_name, model_description=model_description)
        print(f"--- Finished run for random split ---\n")
    else:
        for gene_name in unique_genes[:12]:
            print(f"\n--- Starting cross-validation run for held-out gene: {gene_name} ---")
            train_loader, val_loader, val_data_tuple, num_motif_features = prepare_data(
                data_filepath, tokenizer, batch_size=batch_size,
                log_transform_targets=log_transform_targets,
                validation_gene_name=gene_name
            )
            X_val, y_val, val_df = val_data_tuple
            model = ModifiedModel(base_dnabert_model, base_dnabert_model.config.hidden_size, num_motif_features)
            model.to(device)
            print(f"Model refined to incorporate Ecd_Present feature and {num_motif_features} motif features")
            print(f"Number of training samples: {len(train_loader.dataset)}")
            print(f"Number of validation samples: {len(val_loader.dataset)}")
            optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            loss_fn = nn.MSELoss()
            total_steps = len(train_loader) * epochs
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)
            log_dir_run = os.path.join("runs", short_model_identifier, gene_name, datetime.now().strftime("%Y%m%d_%H%M%S"))
            logger_run = SummaryWriter(log_dir_run)
            print(f"Training model for gene {gene_name}...")
            start_time = time.time()
            train_losses, val_losses = train_model(model, model_name, train_loader, val_loader, optimizer, loss_fn, device, epochs=epochs, logger=logger_run, scheduler=scheduler)
            training_time = (time.time() - start_time)/60
            print(f"Training time for gene {gene_name}: {training_time:.2f} minutes")
            logger_run.close()
            timestamp_run = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir_run = os.path.join("/home/be-em/data/Core_Promoter_2015/results", f"{short_model_identifier}_{gene_name}_{training_file_basename}_{timestamp_run}")
            os.makedirs(output_dir_run, exist_ok=True)
            plot_training_validation_loss(train_losses, val_losses, epochs, f"{short_model_identifier}_{gene_name}", output_dir_run)
            torch.save(model.state_dict(), os.path.join(output_dir_run, "pytorch_model.bin"))
            tokenizer.save_pretrained(output_dir_run)
            print(f"Evaluating model for gene {gene_name}...")
            start_time = time.time()
            all_preds, all_labels, mse = evaluate_model(model, device, val_loader, loss_fn, log_transform_targets=log_transform_targets, y_val_orig=y_val if log_transform_targets else None)
            eval_time = (time.time() - start_time)/60
            print(f"Evaluation time for gene {gene_name}: {eval_time:.2f} minutes")
            print(f"Validation MSE for gene {gene_name}: {mse:.4f}")
            plot_evaluation_results(all_preds, all_labels, f"{short_model_identifier}_{gene_name}")
            val_sequences_current_run = val_df['SequenceSample'].values
            val_names_current_run = val_df['SequenceID'].values
            export_results_to_excel(all_preds, all_labels, f"{short_model_identifier}_{gene_name}",
                                    sequences=val_sequences_current_run, names=val_names_current_run)
            export_outliers_and_all_results(val_names_current_run, val_sequences_current_run,
                                            all_labels, all_preds, f"{short_model_identifier}_{gene_name}", timestamp_run)
            export_runinfo(model, f"{short_model_identifier}_{gene_name}", data_filepath, batch_size, lr, weight_decay, epochs, device,
                        all_labels, all_preds, mse, timestamp_run, validation_gene_name=gene_name, max_len=max_len, script_name=script_name, model_description=model_description)
            print(f"--- Finished cross-validation run for held-out gene: {gene_name} ---\n")

    print("All gene-wise cross-validation runs complete.")
    end_time_total = time.time()
    total_run_time = (end_time_total - start_time_total) / 60  # Show the run time in minutes.
    print(f"Total training and evaluation time for all genes: {total_run_time:.2f} minutes")


if __name__ == "__main__":
    main()