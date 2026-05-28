# ViraExplorer - A Tool for Viral Genome Identification in Metagenomic Data

ViraExplorer is a tool designed to identify and analyze viral sequences from metagenomic data in human samples. An ensemble of Deep Learning models is applied to raw DNA sequences.

The framework combines:

* Multi-scale Convolutional Neural Networks (CNNs)
* Transformer Encoder layers
* Feature fusion through fully connected layers

The model operates directly on raw DNA nucleotide sequences using one-hot encoding, without requiring handcrafted biological features.

## Installation / Requirements

No specific version of Python/PyTorch is required. However, Python 3.12.13 was used along with the following libraries, as per the Google Colab environment from the 2026.04 release:

```bash
matplotlib == 3.10.0
numpy == 2.0.2
pandas == 2.2.2
scikit-learn == 1.6.1
torch == 2.10.0

```

## Usage

### Training

The entire training pipeline is implemented in the viraexplorer.py script.
The script automatically performs:

- Dataset loading
- DNA sequence preprocessing
- One-hot encoding of nucleotide sequences
- Online data augmentation
- Model training and validation
- Automatic checkpoint saving
- Early stopping
- Test set evaluation
- ROC and Precision-Recall curve generation

### Model Architecture
ViraExplorer is composed of three main branches:

1. Pattern Branch: a CNN branch designed to detect discriminative viral sequence motifs using:
    - Kernel sizes: 6, 12, 18, 24
    - 512 filters per convolution
    - Adaptive Max Pooling

2. Frequency Branch: a second CNN branch focused on capturing frequency-related sequence information using:
    - Kernel sizes: 6, 12, 18, 24
    - 256 filters per convolution
    - Adaptive Average Pooling

3. Transformer Branch: a Transformer Encoder branch used to model long-range dependencies in DNA sequences; the outputs of all branches are concatenated and passed through fully connected layers for binary classification (virus / non-virus), and the model includes:
    - Learnable positional embeddings
    - CLS token representation
    - Multi-head self-attention
    - 4 Transformer encoder layers
    - 8 attention heads

### Input Representation
DNA sequences are one-hot encoded

## Reproducibility

To reproduce the results, simply run:

```run_vira.slurm```

The SLURM script launches the complete training pipeline on GPU infrastructure.

## Dataset description

The original ViraMiner dataset was used with augmented datapoints. Dataset files are:

* fullset_train.csv
* fullset_validation.csv
* fullset_test.csv

Each dataset sample contains:

| Column | Description                      |
| ------ | -------------------------------- |
| 0	     | Sequence identifier              |
| 1	     | DNA sequence                     |
| 2	     | Binary label (virus / non-virus) |

<br>
(Tampuu A, Bzhalava Z, Dillner J, Vicente R (2019) ViraMiner: Deep learning on raw DNA sequences for identifying viral genomes in human samples. PLOS ONE 14(9): e0222271. https://doi.org/10.1371/journal.pone.0222271)

### Data Augmentation
Online augmentation is dynamically applied during training to improve generalization. Implemented augmentation techniques are:

* Reverse complement transformation
* Random positional shifts between -5 and +5 nucleotide

---
