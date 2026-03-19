DNABERT-2 Promoter Expression Regression

This script fine-tunes a DNABERT-2 model for gene promoter expression prediction from DNA sequences. It treats the task as a regression problem, where each input sequence is mapped to a single continuous target value representing promoter activity or expression strength.

The workflow covers the full training pipeline:

-loading a pretrained DNABERT-2 model and tokenizer
-reading promoter sequences and expression values from an Excel file
-converting sequences into 6-mers
-training a regression head on top of DNABERT-2
-evaluating performance on a validation split
-generating plots and exported result files
-saving the trained model, tokenizer, run metadata, and prediction summaries
-running example predictions on new sequences
-optionally exporting attention weights for downstream analysis


The script is designed for datasets where:
one column contains DNA sequences, 
one column contains measured expression values

It performs the following steps:

1. Load pretrained DNABERT-2

The script loads a pretrained DNABERT-2 model from a local directory:

model_name = "/home/.../DNABERT-2-117M_model"

It also modifies the model configuration to behave as a single-output regression model:

num_labels = 1

very small dropout values are applied


2. Read and preprocess the dataset

The dataset is read from an Excel file:

data_filepath = "/home/.../core_promoter_expressions.xlsx"

Expected columns include:

SequenceSample: DNA sequence

NORM: expression value

The script then:

converts NORM to float

clips values below 5e-3

applies log2 transformation

converts sequence strings to uppercase


3. Convert sequences to 6-mers

Each DNA sequence is split into overlapping 6-mers, which are used as the input tokens for DNABERT-2.

For example, a sequence like:

ATGCGTAA

becomes:

ATGCGT TGCGTA GCGTAA

The sequence is wrapped with:

[CLS] at the beginning

[SEP] at the end

Then it is padded or truncated to max_len=130.


4. Split data into train and validation sets

The script uses:

train_test_split(..., test_size=0.1, random_state=42)

So:

90% of the data is used for training

10% is used for validation


5. Train the model

Training is performed using:

AdamW optimizer, MSELoss as the regression loss

user-defined hyperparameters:
batch size
learning rate
weight decay
number of epochs

After each epoch, the model is evaluated on the validation set and the validation MSE is printed.


6. Evaluate the model

After training, the script computes predictions on the validation set and reports:
validation MSE
prediction/label shapes
evaluation time


7. Save outputs

The script creates a timestamped results folder and stores:

trained model weights
tokenizer files
training loss / validation MSE plot
predicted vs actual scatter plot
residual distribution plot
Excel file with evaluation results
text file with outliers
text file with all validation results
run information file with metadata and metrics
timestamped model checkpoint


8. Predict on new sequences

The script includes an example section that predicts expression values for three hard-coded sequences and can optionally save attention weights as a NumPy file.

Input requirements
Dataset format

Your Excel file must contain at least these columns:
Column name	Description
SequenceSample	DNA sequence string
NORM	Numeric expression value

Example:

SequenceSample	NORM
ATGCGT...	0.75
TTAGGC...	1.21
Model directory

model_name must point to a valid local or Hugging Face-compatible DNABERT-2 model directory containing the tokenizer and model files.


How to use:

1. Install dependencies

Create an environment and install the required packages.

pip install torch transformers scikit-learn pandas numpy matplotlib seaborn openpyxl


2. Edit the paths

Update these variables near the top of the script:

model_name = r"/path/to/DNABERT-2-117M_model"
data_filepath = r"/path/to/core_promoter_expressions.xlsx"

Also check hard-coded output paths later in the script such as:

"/home/.../results"
"/home/be-em/data/Core_Promoter_2015/results/"

These should be changed to valid directories on your machine.


3. Adjust hyperparameters

At the top of the script you can set:

batch_size = 16
lr = 1e-5
weight_decay = 0.0006
epochs = 25

Typical things to tune:

batch_size: increase if you have enough GPU memory

lr: lower values may be more stable for fine-tuning

epochs: increase if validation still improves


4. Run the script
python your_script_name.py

During execution, the script will:
load the model
preprocess data
print sample tokenization examples
train for the specified number of epochs
evaluate on the validation set
export plots and result files
Outputs


A timestamped result directory is created, typically like:

/home/.../results/DNABERT-2-117M_model_YYYYMMDD_HHMMSS/

Inside, you can expect files such as:
Model artifacts
pytorch_model.bin
tokenizer files saved by tokenizer.save_pretrained(...)
Training diagnostics
training loss and validation MSE plot


After updating the paths, simply run:

python train_dnabert_expression.py
Predicting on custom sequences

You can modify this block in main():

new_sequences = [
    "ACGT...",
    "TTGC...",
    "GGCA..."
]
predictions, attentions = predict(model, tokenizer, new_sequences, device)
print(predictions)

This will return one predicted expression value per sequence.


Notes on preprocessing:

Log transformation
The target column NORM is transformed using:
np.log2(np.clip(df['NORM'].astype(float), 5e-3, None))

This means:
values smaller than 0.005 are clipped
all values are then converted to log2
This is useful when expression values span a wide dynamic range and need stabilization for regression.
Uppercasing
All string columns are converted to uppercase. This helps standardize DNA sequence inputs before k-mer conversion.

Fixed input length:
The default max_len=130 means the final tokenized input contains at most:
128 k-mers
plus [CLS] and [SEP]
Longer sequences are truncated.


Hardware considerations

The script supports GPU if available:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
On GPU systems, it also clears CUDA cache before training.
For larger datasets or longer sequences, GPU training is strongly recommended.