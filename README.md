# ViraExplorer — Viral Genome Identification in Metagenomic Data

ViraExplorer is a deep learning framework for identifying viral sequences in metagenomic DNA data from human samples. It operates directly on raw nucleotide sequences using one-hot encoding, without requiring handcrafted biological features.

The model combines three parallel branches — two multi-scale CNNs and a Transformer encoder — whose representations are fused by a fully connected classifier for binary (virus / non-virus) prediction.

> **Result:** ViraExplorer achieves a test AUROC of **0.939** on the ViraMiner benchmark dataset, outperforming the ViraMiner baseline (AUROC = 0.923).

---

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Dataset](#dataset)
4. [Usage](#usage)
5. [Model Architecture](#model-architecture)
6. [Results](#results)
7. [Reference](#reference)

---

## Requirements

Python 3.12.13 was used along with the following library versions (based on Colab 2026.04 release `freeze`, see [googlecolab/backend-info](https://github.com/googlecolab/backend-info)):

| Library      | Version |
|--------------|---------|
| matplotlib   | 3.10.0  |
| numpy        | 2.0.2   |
| pandas       | 2.2.2   |
| scikit-learn | 1.6.1   |
| torch        | 2.10.0  |

Other versions may work but have not been tested.

---

## Installation

```bash
# 1. Clone this repository
git clone https://github.com/simionRiccard0/viraexplorer.git
cd viraexplorer

# 2. Install dependencies
pip install matplotlib==3.10.0 numpy==2.0.2 pandas==2.2.2 \
            scikit-learn==1.6.1 torch==2.10.0
```

---

## Dataset

ViraExplorer uses the original ViraMiner dataset (Tampuu et al., 2019), which consists of 300 bp DNA contigs labelled as viral or non-viral.

Download the dataset from the [ViraMiner repository](https://github.com/NeuroCSUT/ViraMiner) and place the files at the following paths (relative to the project root):

```
ViraMiner/data/DNA_data/fullset_train.csv
ViraMiner/data/DNA_data/fullset_validation.csv
ViraMiner/data/DNA_data/fullset_test.csv
```

Each CSV file has no header row and three columns:

| Column | Description                                |
|--------|--------------------------------------------|
| 0      | Sequence identifier                        |
| 1      | DNA sequence (300 bp)                      |
| 2      | Binary label: `1` = virus, `0` = non-virus |

---

## Usage

### Training

Run the full training pipeline with:

```bash
python viraexplorer.py
```

The script automatically handles:
- Data loading and one-hot encoding
- Online data augmentation (see [Model Architecture](#model-architecture))
- Model training with mixed-precision and gradient clipping
- Validation AUROC monitoring and early stopping
- Checkpoint saving (resumes automatically if interrupted)
- Test set evaluation with ROC and Precision-Recall curve generation

All outputs are saved to the directory defined by variable `SAVE_DIR` at the top of viraexplorer.py. Before running, update `SAVE_DIR` to an existing path on the user's system. The script will create it automatically with `os.makedirs(SAVE_DIR, exist_ok=True)` if it does not exist, but the parent path must be accessible to the current user.

Output files written to `SAVE_DIR`:

| File                       | Description                                  |
|----------------------------|----------------------------------------------|
| `best_viraexplorer_v2.pth` | Weights of the best validation checkpoint    |
| `checkpoint.pth`           | Full training state (model, optimizer, scheduler, epoch) |
| `test_evaluation.png`      | ROC and Precision-Recall curves on the test set |

### Resuming an Interrupted Run

Training resumes automatically from the last saved checkpoint — simply re-run `python viraexplorer.py`. No additional flags are needed.

### HPC / SLURM

To run on a GPU cluster:

```bash
sbatch run_vira.slurm
```

The SLURM script (`run_vira.slurm`, included in this repository) requests one GPU node and launches `viraexplorer.py`. Adjust the partition name and time limit to match your cluster configuration before submitting.

---

## Model Architecture

ViraExplorer processes each 300 bp contig through three parallel branches. Their outputs (each 512-dim) are concatenated into a 1536-dim vector and passed through a two-layer MLP classifier.

```
Input: 300 bp DNA sequence  →  one-hot encoding  →  (300 × 4) tensor
         │
         ├── PatternBranch     →  512-dim
         ├── FrequencyBranch   →  512-dim
         └── TransformerBranch →  512-dim
                                        │
                               concat → 1536-dim
                                        │
                                  MLP classifier
                                        │
                               logit → sigmoid → P(viral)
```

### PatternBranch (CNN + Max-Pooling)
Detects the presence of localised viral sequence motifs anywhere in the contig.
- Four parallel `Conv1d` layers with kernel sizes 6, 12, 18, 24 (512 filters each)
- Outputs channel-concatenated (2048 channels), batch-normalised, then globally max-pooled
- Projected to 512 dimensions via a linear layer + GELU + Dropout

### FrequencyBranch (CNN + Average-Pooling)
Captures the global nucleotide composition and k-mer frequency profile of the contig.
- Four parallel `Conv1d` layers with kernel sizes 6, 12, 18, 24 (256 filters each)
- Outputs channel-concatenated (1024 channels), batch-normalised, then globally average-pooled
- Projected to 512 dimensions via a linear layer + GELU + Dropout

### TransformerBranch (Transformer Encoder)
Models long-range dependencies between positions in the sequence.
- Linear projection of 4-dim one-hot input to 256-dim embeddings
- Learned positional embeddings (positions 1 … 300)
- Learnable CLS token prepended; its final hidden state is used as the sequence representation
- 4 Transformer encoder layers, 8 attention heads, pre-norm (`norm_first=True`)
- Projected to 512 dimensions via a linear layer

### Input Encoding
Each nucleotide is mapped to a 4-dim one-hot vector:

| Base | Encoding  |
|------|-----------|
| A    | [1, 0, 0, 0] |
| C    | [0, 1, 0, 0] |
| G    | [0, 0, 1, 0] |
| T    | [0, 0, 0, 1] |
| N    | [0, 0, 0, 0] |

Sequences are truncated to 300 bp or zero-padded on the right if shorter.

### Data Augmentation (training only)
Two augmentations are applied online and independently at each sample:
- **Reverse complement** — applied with 50% probability; the sequence is reverse-complemented (A↔T, C↔G) before encoding
- **Random positional shift** — a random integer in [−5, +5] bp is drawn; positive values trim leading bases, negative values prepend N-padding

### Training Details

| Setting              | Value                                      |
|----------------------|--------------------------------------------|
| Loss                 | Focal Loss (α=0.75, γ=2.0, label smoothing ε=0.05) |
| Optimizer            | AdamW (weight decay 1e-4)                  |
| CNN branch LR        | 1e-4                                       |
| Transformer LR       | 5e-5                                       |
| LR schedule          | Linear warm-up (5 epochs) → cosine annealing |
| Batch size           | 128                                        |
| Max epochs           | 120                                        |
| Early stopping       | Patience = 15 epochs on validation AUROC   |
| Gradient clipping    | Max-norm = 1.0                             |
| Mixed precision      | `torch.amp` (float16 forward pass)         |
| Transformer warm-up  | Frozen for first 5 epochs, then unfrozen   |

---

## Results

Performance on the ViraMiner test set:

| Model        | AUROC     | AUPRC     |
|--------------|-----------|-----------|
| ViraMiner    | 0.923     | —         |
| ViraExplorer | **0.939** | **0.569** |

---

## Reference

Tampuu A, Bzhalava Z, Dillner J, Vicente R (2019). *ViraMiner: Deep learning on raw DNA sequences for identifying viral genomes in human samples.* PLOS ONE 14(9): e0222271. https://doi.org/10.1371/journal.pone.0222271
