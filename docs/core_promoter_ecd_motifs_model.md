This script trains an extended DNABERT-2 regression model for promoter expression prediction by combining DNA sequence embeddings with hand-crafted biological features.

In addition to the promoter sequence itself, the model incorporates:
-Ecd presence/absence as a binary regulatory feature
-motif-derived features from an external motif file
-optional gene-wise holdout cross-validation for testing generalization to unseen genes

The goal of this model is to improve expression prediction by enriching the sequence-based DNABERT representation with structured biological information related to transcription factor motifs and regulatory context.


What this script does:
1. Loads a pretrained DNABERT-2 model
The script starts from a pretrained DNABERT-2 backbone and adapts it for regression:
one continuous output (num_labels = 1)
configurable dropout
sequence tokenization through the DNABERT tokenizer

Unlike a standard DNABERT regression script, this model does not rely only on the base DNABERT output. Instead, it feeds the DNABERT prediction through an additional architecture that merges it with extra biological features.


2. Builds a hybrid model (ModifiedModel)
The core addition is the custom ModifiedModel class.

This model combines three sources of information:
a. DNABERT sequence output

The promoter sequence is tokenized and passed through the pretrained DNABERT model.
Its regression output is then projected into a larger hidden representation.

b. Ecd feature
The script uses the Ecd column from the dataset and converts it into a binary feature:
Yes → 1
other values → 0

This binary variable is then passed through a learnable embedding layer, so the model can learn a richer internal representation of Ecd presence/absence.

c. Motif features
The script reads a separate motif feature file and builds additional numeric features for each sequence.
These motif features are then projected with a linear layer and concatenated with the DNABERT-derived representation and the Ecd embedding.

Final architecture:
After concatenation, the merged feature vector is passed through additional fully connected layers with:
batch normalization
ReLU activations
dropout
final linear regression output

Biological features used:
-Ecd feature
The script explicitly models whether Ecd is present in a sample. This allows the model to capture expression differences linked to this regulatory context.

-Motif features
The script reads a tab-separated motif file:
motif_feature_file = "/path/to/core_promoter_motifs_features.tsv"
For each sequence/sample, it extracts motif-related information from this file.

The code is written to supportmotif identity, motif scores and motif positions

It computes per-sample motif summaries by matching the sample ID in the expression dataset with sequence_name in the motif file.

At the moment, the code constructs motif score and position tables, but then reduces the active motif feature matrix to only:
Block1
Block7
with this line:
motif_features_df = motif_features_df.iloc[:, :2]

So although the script is structured for a richer motif feature space, in its current form it effectively uses only the first two motif-related columns.

That detail is important, because the architecture is more general than the current active feature selection.

Validation strategies
One of the most important parts of this script is that it supports two validation modes.

1. Random split
If validation_mode = 'random'
the script performs a standard random train/validation split using:
train_test_split(..., test_size=test_size, random_state=42)
This is useful for routine model development and quick experiments.

2. Gene-wise cross-validation
If validation_mode = 'crossval' the script performs leave-one-gene-out validation: it identifies all unique genes in the dataset for each run, it trains on all genes except one.
it validates on the single held-out gene
This is much more biologically meaningful when you want to test whether the model generalizes to entirely unseen genes or promoter architectures.

That makes this script especially useful for assessing true generalization, rather than only interpolation within a mixed random split.


Input files
1. Main training dataset
Excel file:
data_filepath = "/path/to/core_promoter_training_data_250604.xlsx"

Expected columns include:
SequenceSample — DNA sequence
NORM — expression value
Ecd — categorical feature, expected to contain values like Yes
Gene — gene identifier used for cross-validation
SequenceID or SampleID — sequence/sample identifier used to match motif features
Block1
Block7

2. Motif feature file
TSV file:
motif_feature_file = "/path/to/core_promoter_motifs_features.tsv"

Expected columns include:
sequence_name
motif_id
score
optionally start
optionally strand
This file is used to derive motif-based feature vectors for each sample.


Compared with a simpler DNABERT regression script, this script adds several important modeling and experiment features:

1. Multi-input model
The model uses both sequence data and tabular biological features.

2. Gene-wise validation
This is a major methodological improvement for realistic generalization testing.

3. TensorBoard logging
The script logs training information using:
SummaryWriter including losses, learning rate, gradient norms

4. Cosine learning-rate schedule with warmup
Training uses get_cosine_schedule_with_warmup
This is more advanced than using a fixed learning rate throughout training.

5. Additional regression metrics
The script reports not only MSE, but also:
MAE
Pearson correlation
Spearman correlation
R²

This is very useful for biological prediction tasks where ranking and correlation matter, not only squared error.


Main parameters:
At the top of the script:
batch_size = 28
lr = 3e-5
weight_decay = 0.0003
dropout = 0.001
epochs = 25
max_len = 130

model_name = "/path/to/DNABERT-2-117M_model"
data_filepath = "/path/to/core_promoter_training_data_250604.xlsx"
motif_feature_file = "/path/to/core_promoter_motifs_features.tsv"

validation_mode = "random"   # or "crossval"
test_size = 0.1
Meaning

batch_size — number of samples per batch
lr — learning rate
weight_decay — optimizer regularization
dropout — dropout in the modified model and base config
epochs — number of training epochs
max_len — maximum token length
validation_mode — either random split or gene-wise cross-validation
test_size — validation fraction for random mode only


Usage:
Install dependencies
pip install torch transformers scikit-learn pandas numpy matplotlib seaborn scipy tensorboard openpyxl
Configure paths

Edit these variables in the script:

model_name = "/path/to/DNABERT-2-117M_model"
data_filepath = "/path/to/core_promoter_training_data_250604.xlsx"
motif_feature_file = "/path/to/core_promoter_motifs_features.tsv"

Also review the hard-coded result paths such as:
/home/.../results/
