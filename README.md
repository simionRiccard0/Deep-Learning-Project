# ViraExplorer - A Tool for Viral Genome Identification in Metagenomic Data

ViraExplorer is a tool designed to identify and analyze viral sequences from metagenomic data in human samples. An ensemble of Deep Learning models is applied to raw DNA sequences.

## Installation / Requirements

No specific version of Python/PyTorch is required.
However, Python 3.12.13 was used along with the following libraries, as per the Google Colab environment from the 2026.04 release:

```bash
matplotlib == 3.10.0
numpy == 2.0.2
pandas == 2.2.2
scikit-learn == 1.6.1
torch == 2.10.0

```

## Usage

### Training

See the viraexplorer.py file. (SCRIVERE DETTAGLI)

## Reproduce Paper Results (Ensemble)

To reproduce the results, simply run:

```run_vira.slurm```

---

## Dataset description

The original ViraMiner dataset was used with augmented datapoints. (ADD DETAILS)
<br>
(Tampuu A, Bzhalava Z, Dillner J, Vicente R (2019) ViraMiner: Deep learning on raw DNA sequences for identifying viral genomes in human samples. PLOS ONE 14(9): e0222271. https://doi.org/10.1371/journal.pone.0222271)

---
