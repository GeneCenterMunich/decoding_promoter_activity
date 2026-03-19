# -*- coding: utf-8 -*-
"""
Created on Tue Mai 27 10:35:24 2025

@author: Christophe Jung

Ecd + Motif Model: The script extend the ModifiedModel to incorporate a more complex set of features. Beyond 'Ecd_Present',
these models include motif-specific data such as Motif ID (represented by a learnable embedding), PWM score.
This model is designed to handle a larger number of motif features, allowing for more nuanced predictions based on the presence and scores of multiple motifs.

In addition, the script implements the cross-validation strategy of training on data from all genes except one, and validating on that single held-out gene,
This approach is beneficial for assessing the model's generalisation capability to entirely new genes or promoter architectures that it has not
previously encountered during training
"""

#######################################################################################################################################################
# USER CONFIG
# Model hyper-parameters
batch_size=28  # Batch size for training and validation
lr=3e-5  # Learning rate
weight_decay=0.0003  # Weight decay
dropout=0.001  # Dropout rate for the model (set to zero for regression)
epochs= 25  # Number of training epochs
max_len = 130  # define max_len in the global scope

# Filepath to the dataset and model directory
model_name = r"/home/.../DNABERT-2-117M_model"    # DNABERT2 base model
data_filepath = r"/home/.../core_promoter_training_data_250604.xlsx"
motif_feature_file = r'/home/.../core_promoter_motifs_features.tsv'
# Note: Ensure the model directory contains the pre-trained model files (config.json, pytorch_model.bin, tokenizer files)
# User can select validation mode here:
validation_mode = 'random'  # Set to 'crossval' for gene-wise cross-validation, or 'random' for random split
# Define the test size for train-test split in random validation mode (ignored in cross-validation mode)
# Unactivated for gene-wise holdout validation, as the split is determined by the held-out gene.
test_size = 0.1
#######################################################################################################################################################


# imports
import torch, time, os, gc
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertConfig
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
from transformers.modeling_outputs import SequenceClassifierOutput
from scipy.stats import pearsonr, spearmanr


class ModifiedModel(torch.nn.Module):
    """
    A modified model class that incorporates additional features (Ecd_Present, motif features) into the DNABERT-based model.

        base_model: The pre-trained DNABERT model.
        ecd_embedding: Learnable embedding for the Ecd_Present feature.
        motif_embedding: Learnable embedding for the motif features.
        base_projection: Linear layer to project base model output to match hidden size.
        additional_layer1: Combines base output and Ecd embedding.
        additional_layer2: Further processes combined features.
        output_layer: Final regression output layer.
    """
    def __init__(self, base_model, hidden_size, num_motif_feature_columns):
        super(ModifiedModel, self).__init__()
        self.base_model = base_model
        self.base_projection = torch.nn.Linear(1, hidden_size)
        self.batch_norm1_base = torch.nn.BatchNorm1d(hidden_size)
        self.ecd_embedding = torch.nn.Embedding(2, hidden_size // 2)
        # Use the correct input size for all motif features (scores + positions)
        self.all_motifs_projection = torch.nn.Linear(num_motif_feature_columns, hidden_size // 2)
        # Adjust additional_layer1 input size: hidden_size (base) + hidden_size//2 (ecd) + hidden_size//2 (motifs) = hidden_size * 2
        self.additional_layer1 = torch.nn.Linear(hidden_size * 2, int(hidden_size))
        self.batch_norm1 = torch.nn.BatchNorm1d(int(hidden_size))
        self.additional_layer2 = torch.nn.Linear(int(hidden_size), hidden_size//2)
        self.batch_norm2 = torch.nn.BatchNorm1d(hidden_size//2)

        self.dropout = torch.nn.Dropout(p=dropout)
        self.classifier = torch.nn.Linear(hidden_size//2, 1)
        torch.nn.init.xavier_uniform_(self.base_projection.weight)
        torch.nn.init.xavier_uniform_(self.ecd_embedding.weight)
        torch.nn.init.xavier_uniform_(self.all_motifs_projection.weight)
        torch.nn.init.xavier_uniform_(self.additional_layer1.weight)
        torch.nn.init.xavier_uniform_(self.additional_layer2.weight)
        torch.nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, input_ids, ecd_feature, all_motif_scores): 
        # Get base model output (logits of shape: batch_size, 1)
        base_output = self.base_model(input_ids).logits
        # Project base model output to hidden_size
        base_output = self.base_projection(base_output)
        base_output = self.batch_norm1_base(base_output) # Apply BatchNorm

        # Get Ecd embedding
        ecd_embedded = self.ecd_embedding(ecd_feature)
        # Ensure ecd_embedded has the correct shape for concatenation
        ecd_embedded = ecd_embedded.squeeze(1)
        # Project all motif scores
        all_motifs_projected = self.all_motifs_projection(all_motif_scores)

        # Concatenate all features
        combined_input = torch.cat((base_output, ecd_embedded, all_motifs_projected), dim=1) # Shape: (batch_size, hidden_size * 2)

        # Pass through additional layers
        output = torch.relu(self.additional_layer1(combined_input))
        output = self.batch_norm1(output)
        output = torch.relu(self.additional_layer2(output))
        output = self.batch_norm2(output)
        output = self.dropout(output)
        output = self.classifier(output)

        return SequenceClassifierOutput(logits=output)
 
    
def load_model_and_tokenizer(model_name):
    """
    Load the pre-trained model and tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = BertConfig.from_pretrained(model_name)
    # Set dropout rate in the configuration (set to zero for regression)
    config.hidden_dropout_prob = dropout
    config.attention_probs_dropout_prob = dropout  # Set attention dropout to a very low value
    config.num_labels = 1  # Regression task with a single continuous value
    # Ensure the model head does not apply any activation (e.g., sigmoid/tanh) to the output logits for regression.
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
    Prepare the dataset and dataloaders for training and validation,
    with an option to hold out a specific gene for validation.
    Uses base model's 6mer and tokenizer logic for sequence input.
    """
    df = pd.read_excel(filepath)
    # Convert all string columns to uppercase
    df = df.apply(lambda col: col.str.upper() if col.dtype == "object" else col)
    # Ensure SequenceSample is string
    df['SequenceSample'] = df['SequenceSample'].astype(str)
    # Log2 conversion for NORM
    df['NORM'] = np.log2(np.clip(df['NORM'].astype(float), 5E-4, None))

    # Add a column indicating the presence of Ecd 
    df['Ecd'] = df['Ecd'].apply(lambda x: 1 if x == 'Yes' else 0)   

    # Only keep Block1 and Block7 from the main file
    motif_score_columns_main = ['Block1', 'Block7']
    for col in motif_score_columns_main:
        df[col] = df[col].fillna(0).replace('', 0).astype(float)

    # Read motif features file (fix for NameError)
    motif_df = pd.read_csv(motif_feature_file, sep='\t', dtype={"motif_id": str, "sequence_name": str, "strand": str, "score": float})
    # Build a list of all unique motif_ids (excluding Block1, Block7)
    motif_ids = sorted(set(motif_df['motif_id'].unique()) - set(motif_score_columns_main))
    # For each sample, build a dict of motif_id: avg_score and motif_id: max_start
    motif_scores_list = []
    motif_positions_list = []
    for idx, row in df.iterrows():
        sample_id = row['SampleID'] if 'SampleID' in row else row['SequenceID']
        sample_id = str(sample_id)
        motif_rows = motif_df[motif_df['sequence_name'].astype(str) == sample_id]
        motif_scores = {}
        motif_positions = {}
        for motif_id in motif_ids:
            scores = motif_rows[motif_rows['motif_id'] == motif_id]['score']
            starts = motif_rows[motif_rows['motif_id'] == motif_id]['start'] if 'start' in motif_rows.columns else pd.Series([])
            motif_scores[motif_id] = scores.astype(float).mean() if not scores.empty else 0.0
            motif_positions[motif_id] = starts.astype(float).mean() if not starts.empty else 0.0
        motif_scores_list.append(motif_scores)
        motif_positions_list.append(motif_positions)
    motif_scores_df = pd.DataFrame(motif_scores_list, index=df.index)
    motif_scores_df['Block1'] = df['Block1']
    motif_scores_df['Block7'] = df['Block7']
    motif_score_columns = ['Block1', 'Block7'] + [m for m in motif_ids]
    motif_scores_df = motif_scores_df[motif_score_columns]
    motif_positions_df = pd.DataFrame(motif_positions_list, index=df.index)
    motif_position_columns = [m for m in motif_ids]
    motif_positions_df = motif_positions_df[motif_position_columns]
    # Rename position columns to avoid duplication
    motif_positions_df = motif_positions_df.rename(columns={col: f"{col}_pos" for col in motif_positions_df.columns})
    # Concatenate scores and positions
    motif_features_df = pd.concat([motif_scores_df, motif_positions_df], axis=1)
    # Set all motif features to zero
    motif_features_df = motif_features_df.iloc[:, :2]

    print("Motif features DataFrame shape:", motif_features_df.shape)
    print("Motif features DataFrame columns:", motif_features_df.columns.tolist())
    # Add SequenceID column to motif_scores_df
    if 'SequenceID' in df.columns:
        motif_scores_df['SequenceID'] = df['SequenceID']
    # Export motif_scores_df to Excel
    from datetime import datetime
    # Scale motif features (scores + positions, excluding SequenceID if present)
    motif_feature_cols = [col for col in motif_features_df.columns if col not in ['SequenceID']]
    motif_features_matrix = motif_features_df[motif_feature_cols].values
    motif_features_excel_path = f"/home/.../results/motif_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    motif_features_df.to_excel(motif_features_excel_path, index=False)
    print(f"Motif features DataFrame exported to {motif_features_excel_path}")
    # Motif feature scaling: fit on train, transform val for cross-validation
    if validation_gene_name:
        val_df = df[df['Gene'] == validation_gene_name]
        train_df = df[df['Gene'] != validation_gene_name]
        scaler_motifs = StandardScaler()
        all_motif_features_train = scaler_motifs.fit_transform(motif_features_matrix[train_df.index])
        all_motif_features_val = scaler_motifs.transform(motif_features_matrix[val_df.index])
        X_train, y_train, ecd_train = train_df['SequenceSample'].values, train_df['NORM'].values, train_df['Ecd'].values
        X_val, y_val, ecd_val = val_df['SequenceSample'].values, val_df['NORM'].values, val_df['Ecd'].values
    else:
        # Split first, then fit scaler only on training motif features
        X_train, X_val, y_train, y_val, ecd_train, ecd_val, motif_features_train, motif_features_val = train_test_split(
            df['SequenceSample'].values,
            df['NORM'].values,
            df['Ecd'].values,
            motif_features_matrix,
            test_size=test_size , random_state=42
        )
        scaler_motifs = StandardScaler()
        all_motif_features_train = scaler_motifs.fit_transform(motif_features_train)
        all_motif_features_val = scaler_motifs.transform(motif_features_val)
        # to remove to use a standardscaler
        all_motif_features_train = motif_features_train
        all_motif_features_val = motif_features_val
        val_df = df

    class GeneExpressionDataset(Dataset):
        def __init__(self, sequences, expressions, tokenizer, max_len, ecd_features, all_motif_features_scaled, k=6):
            self.sequences = sequences
            self.expressions = expressions
            self.tokenizer = tokenizer
            self.max_len = max_len
            self.ecd_features = ecd_features
            self.all_motif_features_scaled = all_motif_features_scaled
            self.k = k
            # tokenization: k-merize after all cleaning
            self.kmer_sequences = [seq_to_kmers(seq, self.k) for seq in self.sequences]

        def __len__(self):
            return len(self.sequences)

        def __getitem__(self, item):
            kmers = self.kmer_sequences[item]
            tokens = ['[CLS]'] + kmers + ['[SEP]']
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            # Pad or truncate to max_len
            if len(input_ids) < self.max_len:
                input_ids = input_ids + [self.tokenizer.pad_token_id] * (self.max_len - len(input_ids))
            else:
                input_ids = input_ids[:self.max_len]
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            expression = self.expressions[item]
            ecd_feature = self.ecd_features[item]
            all_motif_features = self.all_motif_features_scaled[item]
            return input_ids, torch.tensor(expression, dtype=torch.float), torch.tensor(ecd_feature, dtype=torch.long), torch.tensor(all_motif_features, dtype=torch.float)

    df['Gene'] = df['Gene'].str.upper().str.replace(' ', '')
    # Use the already scaled motif scores from above
    if validation_gene_name:
        # all_motif_scores_train and all_motif_scores_val already set above
        X_train, y_train, ecd_train = train_df['SequenceSample'].values, train_df['NORM'].values, train_df['Ecd'].values
        X_val, y_val, ecd_val = val_df['SequenceSample'].values, val_df['NORM'].values, val_df['Ecd'].values
    else:
        # all_motif_scores_train and all_motif_scores_val already set above
        val_df = df

    train_dataset = GeneExpressionDataset(X_train, y_train, tokenizer, max_len, ecd_train, all_motif_features_train)
    val_dataset = GeneExpressionDataset(X_val, y_val, tokenizer, max_len, ecd_val, all_motif_features_val)
    num_motif_feature_columns = len(motif_feature_cols)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size)   
    print("############################################################################################################")
    return train_loader, val_loader, (X_val, y_val, val_df), num_motif_feature_columns
 

def train_model(model, model_name, train_loader, val_loader, optimizer, loss_fn, device, epochs, logger=None, scheduler=None):
    """
    Train the model with optional logging and learning rate scheduling.
    num_motif_feature_columns = motif_features_matrix.shape[1]

        model: The model to train.
        model_name (str): Name of the model for saving outputs.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        optimizer: Optimizer for training.
        loss_fn: Loss function for training.
        device: Device to run the training on (e.g., 'cuda' or 'cpu').
        epochs (int): Number of training epochs.
        logger: Optional logger for TensorBoard.
        scheduler: Optional learning rate scheduler.

    Returns:
        Training and validation losses for each epoch.
    """
    train_losses = []
    val_losses = []
    val_mses = []
    log_interval = 10  # Log every 10 steps
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
            # Ensure logits are accessed and squeezed for MSELoss
            loss = loss_fn(outputs.logits.view(-1), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if scheduler is not None:
                scheduler.step()
            if logger is not None:
                current_lr = optimizer.param_groups[0]['lr']
                global_step = epoch * len(train_loader) + i
                logger.add_scalar('LearningRate', current_lr, global_step)
            if logger is not None:
                current_lr = optimizer.param_groups[0]['lr']
                global_step = epoch * len(train_loader) + i
                if global_step % log_interval == 0:
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            logger.add_scalar(f'Gradients/{name}', param.grad.norm().item(), global_step)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        # Validation loss
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids, labels, ecd_feature, all_motif_scores = batch
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                ecd_feature = ecd_feature.to(device)
                all_motif_scores = all_motif_scores.to(device) 
                outputs = model(input_ids, ecd_feature, all_motif_scores)
                logits = outputs.logits.view(-1) 
                loss = loss_fn(logits, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        # Calculate true MSE for validation set
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids, labels, ecd_feature, all_motif_scores = batch
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                ecd_feature = ecd_feature.to(device)
                all_motif_scores = all_motif_scores.to(device)
                outputs = model(input_ids, ecd_feature, all_motif_scores)
                logits = outputs.logits.view(-1)
                val_preds.extend(logits.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
        from sklearn.metrics import mean_squared_error
        val_mse = mean_squared_error(val_targets, val_preds)
        val_mses.append(val_mse)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val MSE: {val_mse:.4f}")
        if logger is not None:
            logger.add_scalar('Loss/train', avg_train_loss, epoch)
            logger.add_scalar('Loss/val', avg_val_loss, epoch)
        model.train()
    return train_losses, val_losses, val_mses


def evaluate_model(model, device, val_loader, loss_fn, log_transform_targets=False, y_val_orig=None, metrics=False):
    """
    Evaluate the model on the validation dataset.
    If metrics=True, also return all statistics.
    """
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids, labels, ecd_feature, all_motif_scores = batch 
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            ecd_feature = ecd_feature.to(device)
            all_motif_scores = all_motif_scores.to(device) 
            outputs = model(input_ids, ecd_feature, all_motif_scores)
            logits = outputs.logits.view(-1) 
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            all_preds.extend(logits.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    mse = total_loss / len(val_loader)
    if metrics:
        mae = np.mean(np.abs(np.array(all_labels) - np.array(all_preds)))
        pearson_corr, pearson_p = pearsonr(all_labels, all_preds)
        spearman_corr, spearman_p = spearmanr(all_labels, all_preds)
        r2 = r2_score(all_labels, all_preds)
        metrics_dict = {
            'mse': mse,
            'mae': mae,
            'pearson_corr': pearson_corr,
            'pearson_p': pearson_p,
            'spearman_corr': spearman_corr,
            'spearman_p': spearman_p,
            'r2': r2
        }
        return all_preds, all_labels, mse, metrics_dict
    else:
        return all_preds, all_labels, mse


def plot_evaluation_results(all_preds, all_labels, model_name):
    """
    Generate and save evaluation plots (scatter plot and residuals histogram).

    Args:
        all_preds (list): Predicted values.
        all_labels (list): Actual values.
        model_name (str): Name of the model for saving plots.
    """
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
    plt.savefig(f"/home/.../results/{model_name}_pred_vs_actual{timestamp}.png")
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
    plt.savefig(f"/home/.../results/{model_name}_residuals_{timestamp}.png")
    plt.grid(True)
    plt.show()
    plt.close()


def export_results_to_excel(all_preds, all_labels, model_name, sequences=None, names=None, metrics=None):
    """
    Export evaluation results to an Excel file, with metrics if provided.
    """
    results_dict = {
        'Actual': all_labels,
        'Predicted': all_preds,
        'Residuals': np.array(all_preds) - np.array(all_labels)
    }  
    # Optional: Include sequences and names if provided
    if names is not None:
        results_dict['Name'] = names
    if sequences is not None:
        results_dict['Sequence'] = sequences
    results_df = pd.DataFrame(results_dict)
    # Add metrics as columns after DataFrame creation, filling only the first row
    if metrics is not None:
        for k, v in metrics.items():
            results_df[k] = [v] + [None]*(len(results_df)-1)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"/home/.../results/{model_name}_evaluation_results_{timestamp}.xlsx"
    results_df.to_excel(file_name, index=False)
    print(f"Results exported to {file_name}")


def save_model_with_timestamp(model, model_name):
    """
    Save the model with a timestamped filename (same as v3_5 logic).

    Args:
        model: The trained model.
        model_name (str): Name of the model for saving.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{model_name}_model_{timestamp}.pth"
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved to {model_filename}")


def predict(model, tokenizer, sequences, device, ecd_features_for_prediction, all_motif_scores_for_prediction):
    """
    Make predictions on new sequences.

    Args:
        model: The trained model.
        tokenizer: Tokenizer for encoding sequences.
        sequences (str or list): Input sequences for prediction.
        device: Device to run the predictions on.

    Returns:
        list: Predicted values.
    """
    model.eval()
    predictions = []
    with torch.no_grad():
        if isinstance(sequences, str):
            sequences = [sequences]

        encoding = tokenizer(sequences, truncation=True, padding='max_length', max_length=130, return_tensors="pt")
        input_ids = encoding['input_ids'].to(device)

        # Ensure ecd_features_for_prediction and all_motif_scores_for_prediction are tensors and on the correct device
        ecd_features_for_prediction = ecd_features_for_prediction.to(device)
        all_motif_scores_for_prediction = all_motif_scores_for_prediction.to(device)

        outputs = model(input_ids, ecd_features_for_prediction, all_motif_scores_for_prediction) # Pass updated arguments
        logits = outputs.logits.view(-1)
        predictions = logits.cpu().numpy()
    return predictions



def export_outliers_and_all_results(val_names, val_sequences, all_labels, all_preds, model_name, timestamp):
    """
    Export outliers and all results to text files.

    Args:
        val_names (list): Names of validation samples.
        val_sequences (list): Sequences of validation samples.
        all_labels (list): Actual values.
        all_preds (list): Predicted values.
        model_name (str): Name of the model for saving files.
        timestamp (str): Timestamp for filenames.
    """
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
    # Add a section for the 20% best predictions (lowest abs residuals)
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
    outlier_txt_path = f"/home/.../results/{model_name}_outliers_{timestamp}.txt"
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
    all_txt_path = f"/home/.../results/{model_name}_all_results_{timestamp}.txt"
    with open(all_txt_path, 'w') as f:
        f.write("\n".join(all_lines))
    print(f"All results info written to {all_txt_path}")


def export_runinfo(model, model_name, data_filepath, batch_size, lr, weight_decay, epochs, device, all_labels, all_preds, mse, timestamp, validation_gene_name=None):
    """
    Export model parameters and evaluation metrics to a text file.

    Args:
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
    """
    from sklearn.metrics import r2_score, mean_absolute_error
    params_txt_path = f"/home/.../results/{model_name}_runinfo_{timestamp}.txt"
    with open(params_txt_path, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Training data file: {data_filepath}\n")
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
        f.write(f"Dropout (hidden): 0.00001 (to update...)\n")
        f.write(f"Dropout (attention): 0.00001 (to update...)\n")
        r2 = r2_score(all_labels, all_preds)
        mae = mean_absolute_error(all_labels, all_preds)
        norm_mae = mae / (np.max(all_labels) - np.min(all_labels)) if (np.max(all_labels) - np.min(all_labels)) > 0 else float('nan')
        f.write(f"Validation MSE: {mse:.4f}\n")
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
    print(f"Run info written to {params_txt_path}")


def plot_training_validation_loss(train_losses, val_losses, epochs, model_name, output_dir):
    """
    Plot and save the training and validation loss curves.

    Args:
        train_losses (list): Training losses for each epoch.
        val_losses (list): Validation losses for each epoch.
        epochs (int): Number of training epochs.
        model_name (str): Name of the model for saving the plot.
        output_dir (str): Directory to save the plot.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(range(epochs), train_losses, marker='o', linestyle='-', color='b', label='Train Loss')
    plt.plot(range(epochs), val_losses, marker='s', linestyle='-', color='orange', label='Val Loss')
    plt.title('Training & Validation Loss over Epochs', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"/home/.../results/{model_name}_train_val_loss.png")
    plt.savefig(output_path)
    plt.show()
    plt.close()


def main():
    
    start_time_total = time.time()
    #torch.manual_seed(42) # Set a fixed seed for PyTorch operations

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # CUDA initialisation settings
    # if memory issues run to avoid fragmentation in linux:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
    torch.cuda.empty_cache()
    gc.collect()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Load the model and tokenizer
    base_dnabert_model, tokenizer = load_model_and_tokenizer(model_name)
  
    # Option to enable log10 transform
    log_transform_targets = False  # Set True to enable log10 transform

    # Read the entire dataset once to get all unique gene names.
    df_full = pd.read_excel(data_filepath)


    short_model_identifier = os.path.basename(model_name) # Extracts 'DNABERT-2-117M_model'

    if validation_mode == 'crossval':
        unique_genes = df_full['Gene'].str.upper().str.replace(' ', '').unique()
        print(f"Discovered {len(unique_genes)} unique genes for cross-validation: {unique_genes}")
        for gene_name in unique_genes[:]:
            train_loader, val_loader, val_data_tuple, num_motif_feature_columns = prepare_data(
                data_filepath, tokenizer, batch_size=batch_size, 
                log_transform_targets=log_transform_targets,
                validation_gene_name=gene_name
            )
            print(f"Motif feature columns (scores + positions): {num_motif_feature_columns}")
            model = ModifiedModel(base_dnabert_model, base_dnabert_model.config.hidden_size, num_motif_feature_columns)
            model.to(device)
            X_val, y_val, val_df = val_data_tuple
            print(f"Model refined to incorporate Ecd_Present feature and {num_motif_feature_columns} motif features")
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
            train_losses, val_losses, val_mses = train_model(model, model_name, train_loader, val_loader, optimizer, loss_fn, device, epochs=epochs, logger=logger_run, scheduler=scheduler)
            training_time = (time.time() - start_time)/60
            print(f"Training time for gene {gene_name}: {training_time:.2f} minutes")
            logger_run.close()
            timestamp_run = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir_run = os.path.join("/home/.../results", f"{short_model_identifier}_{gene_name}_{timestamp_run}")
            os.makedirs(output_dir_run, exist_ok=True)
            # Use val_mses for plotting and reporting
            plot_training_validation_loss(train_losses, val_mses, epochs, f"{short_model_identifier}_{gene_name}", output_dir_run)
            save_model_with_timestamp(model, os.path.join(output_dir_run, f"model_for_{gene_name}"))
            print(f"Evaluating model for gene {gene_name}...")
            start_time = time.time()
            all_preds, all_labels, mse, metrics_dict = evaluate_model(model, device, val_loader, loss_fn, log_transform_targets=log_transform_targets, y_val_orig=y_val if log_transform_targets else None, metrics=True)
            eval_time = (time.time() - start_time)/60
            print(f"Evaluation time for gene {gene_name}: {eval_time:.2f} minutes")
            print(f"Validation MSE for gene {gene_name}: {mse:.4f}")
            print(f"Pearson: {metrics_dict['pearson_corr']:.4f} (p={metrics_dict['pearson_p']:.2g})")
            print(f"Spearman: {metrics_dict['spearman_corr']:.4f} (p={metrics_dict['spearman_p']:.2g})")
            print(f"MAE: {metrics_dict['mae']:.4f}")
            print(f"R2: {metrics_dict['r2']:.4f}")
            plot_evaluation_results(all_preds, all_labels, f"{short_model_identifier}_{gene_name}")
            val_sequences_current_run = val_df['SequenceSample'].values
            val_names_current_run = val_df['SequenceID'].values
            export_results_to_excel(all_preds, all_labels, f"{short_model_identifier}_{gene_name}", 
                                    sequences=val_sequences_current_run, names=val_names_current_run, metrics=metrics_dict)
            export_outliers_and_all_results(val_names_current_run, val_sequences_current_run,
                                            all_labels, all_preds, f"{short_model_identifier}_{gene_name}", timestamp_run)
            export_runinfo(model, f"{short_model_identifier}_{gene_name}", r"/home/.../results/", batch_size, lr, weight_decay, epochs, device,
                        all_labels, all_preds, mse, timestamp_run, validation_gene_name=gene_name)
            print(f"--- Finished cross-validation run for held-out gene: {gene_name} ---\n")
        print("All gene-wise cross-validation runs complete.")
        end_time_total = time.time()
        total_run_time = (end_time_total - start_time_total) / 60  # Convert to minutes
        print(f"Total training and evaluation time for all genes: {total_run_time:.2f} minutes")
    elif validation_mode == 'random':
        print("\n--- Starting random train/validation split ---")
        train_loader, val_loader, val_data_tuple, num_motif_feature_columns = prepare_data(
            data_filepath, tokenizer, batch_size=batch_size, 
            log_transform_targets=log_transform_targets,
            validation_gene_name=None
        )
        print(f"Motif feature columns (scores + positions): {num_motif_feature_columns}")
        # Always use num_motif_feature_columns for motif feature input size
        print(f"Initializing ModifiedModel with motif feature input size: {num_motif_feature_columns}")
        model = ModifiedModel(base_dnabert_model, base_dnabert_model.config.hidden_size, num_motif_feature_columns)
        model.to(device)
        X_val, y_val, val_df = val_data_tuple
        print(f"Model refined to incorporate Ecd_Present feature and {num_motif_feature_columns} motif features")
        print(f"Number of training samples: {len(train_loader.dataset)}")
        print(f"Number of validation samples: {len(val_loader.dataset)}")
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()
        total_steps = len(train_loader) * epochs
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)
        log_dir_run = os.path.join("runs", short_model_identifier, "random_split", datetime.now().strftime("%Y%m%d_%H%M%S"))
        logger_run = SummaryWriter(log_dir_run)
        print(f"Training model (random split)...")
        start_time = time.time()
        train_losses, val_losses, val_mses = train_model(model, model_name, train_loader, val_loader, optimizer, loss_fn, device, epochs=epochs, logger=logger_run, scheduler=scheduler)
        training_time = (time.time() - start_time)/60
        print(f"Training time: {training_time:.2f} minutes")
        logger_run.close()
        timestamp_run = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir_run = os.path.join("/home/.../results", f"{short_model_identifier}_randomsplit_{timestamp_run}")
        os.makedirs(output_dir_run, exist_ok=True)
        plot_training_validation_loss(train_losses, val_mses, epochs, f"{short_model_identifier}_randomsplit", output_dir_run)
        save_model_with_timestamp(model, os.path.join(output_dir_run, f"model_randomsplit"))
        print(f"Evaluating model (random split)...")
        start_time = time.time()
        all_preds, all_labels, mse, metrics_dict = evaluate_model(model, device, val_loader, loss_fn, log_transform_targets=log_transform_targets, y_val_orig=y_val if log_transform_targets else None, metrics=True)
        eval_time = (time.time() - start_time)/60
        print(f"Evaluation time: {eval_time:.2f} minutes")
        print(f"Validation MSE: {mse:.4f}")
        print(f"Pearson: {metrics_dict['pearson_corr']:.4f} (p={metrics_dict['pearson_p']:.2g})")
        print(f"Spearman: {metrics_dict['spearman_corr']:.4f} (p={metrics_dict['spearman_p']:.2g})")
        print(f"MAE: {metrics_dict['mae']:.4f}")
        print(f"R2: {metrics_dict['r2']:.4f}")
        plot_evaluation_results(all_preds, all_labels, f"{short_model_identifier}_randomsplit")
        val_sequences_current_run = val_df['SequenceSample'].values
        val_names_current_run = val_df['SequenceID'].values
        export_results_to_excel(all_preds, all_labels, f"{short_model_identifier}_randomsplit", 
                                sequences=None , names=None , metrics=metrics_dict)
        export_outliers_and_all_results(val_names_current_run, val_sequences_current_run,
                                        all_labels, all_preds, f"{short_model_identifier}_randomsplit", timestamp_run)
        export_runinfo(model, f"{short_model_identifier}_randomsplit", r"/home/.../results/", batch_size, lr, weight_decay, epochs, device,
                    all_labels, all_preds, mse, timestamp_run, validation_gene_name=None)
        print(f"--- Finished random split training/validation ---\n")
        end_time_total = time.time()
        total_run_time = (end_time_total - start_time_total) / 60  # Convert to minutes
        print(f"Total training and evaluation time: {total_run_time:.2f} minutes")
    else:
        raise ValueError("validation_mode must be 'crossval' or 'random'")


if __name__ == "__main__":
    main()