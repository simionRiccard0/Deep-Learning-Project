import os
import math
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)

SAVE_DIR        = "/home/simionricc/virproject/"
BEST_MODEL_PATH = os.path.join(SAVE_DIR, "best_viraexplorer_v2.pth")
CHECKPOINT_PATH = os.path.join(SAVE_DIR, "checkpoint.pth")

train = pd.read_csv('ViraMiner/data/DNA_data/fullset_train.csv', header=None)
val   = pd.read_csv('ViraMiner/data/DNA_data/fullset_validation.csv', header=None)
test  = pd.read_csv('ViraMiner/data/DNA_data/fullset_test.csv', header=None)

print("Train shape:", train.shape)
print("Sequence length:", len(train[1][0]))
print("\nClass balance:")
print(train[2].value_counts())

mapping = {
    "A":[1,0,0,0],
    "C":[0,1,0,0],
    "G":[0,0,1,0],
    "T":[0,0,0,1],
    "N":[0,0,0,0]
}

MAX_LEN = 300

def encode_sequence(seq, augment=False):
    complement = {'A':'T','T':'A','C':'G','G':'C','N':'N'}

    if augment and random.random() > 0.5:
        seq = ''.join(complement.get(b,'N') for b in reversed(seq))

    if augment:
        shift = random.randint(-5, 5)
        seq = seq[max(0,shift):] if shift > 0 else 'N'*(-shift) + seq

    seq = seq[:MAX_LEN]

    encoded = [mapping.get(b,[0,0,0,0]) for b in seq]

    while len(encoded) < MAX_LEN:
        encoded.append([0,0,0,0])

    return encoded

class DNADataset(Dataset):
    def __init__(self, dataframe, augment=False):
        self.sequences = dataframe[1].values
        self.labels    = dataframe[2].values
        self.augment   = augment

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x = torch.tensor(
            encode_sequence(self.sequences[idx], self.augment),
            dtype=torch.float32
        )
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y

train_loader = DataLoader(
    DNADataset(train, augment=True),
    batch_size=128,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    DNADataset(val),
    batch_size=128,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

test_loader = DataLoader(
    DNADataset(test),
    batch_size=64,
    shuffle=False,
    num_workers=2
)

class PatternBranch(nn.Module):
    def __init__(self, dropout):
        super().__init__()

        self.conv6  = nn.Conv1d(4, 512, 6,  padding='same')
        self.conv12 = nn.Conv1d(4, 512, 12, padding='same')
        self.conv18 = nn.Conv1d(4, 512, 18, padding='same')
        self.conv24 = nn.Conv1d(4, 512, 24, padding='same')

        total = 512 * 4

        self.bn   = nn.BatchNorm1d(total)
        self.pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(total,512),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x.permute(0,2,1)

        x = torch.cat([
            F.gelu(self.conv6(x)),
            F.gelu(self.conv12(x)),
            F.gelu(self.conv18(x)),
            F.gelu(self.conv24(x))
        ], dim=1)

        x = self.bn(x)

        return self.fc(self.pool(x).squeeze(-1))

class FrequencyBranch(nn.Module):
    def __init__(self, dropout):
        super().__init__()

        self.conv6  = nn.Conv1d(4, 256, 6,  padding='same')
        self.conv12 = nn.Conv1d(4, 256, 12, padding='same')
        self.conv18 = nn.Conv1d(4, 256, 18, padding='same')
        self.conv24 = nn.Conv1d(4, 256, 24, padding='same')

        total = 256 * 4

        self.bn   = nn.BatchNorm1d(total)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(total,512),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x.permute(0,2,1)

        x = torch.cat([
            F.gelu(self.conv6(x)),
            F.gelu(self.conv12(x)),
            F.gelu(self.conv18(x)),
            F.gelu(self.conv24(x))
        ], dim=1)

        x = self.bn(x)

        return self.fc(self.pool(x).squeeze(-1))

class TransformerBranch(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_layers=4,
        max_len=300,
        dropout=0.1
    ):
        super().__init__()

        self.embedding = nn.Linear(4, d_model)
        self.pos_enc   = nn.Embedding(max_len+1, d_model)

        self.cls_token = nn.Parameter(torch.randn(1,1,d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(d_model)
        self.out  = nn.Linear(d_model, 512)

    def forward(self, x):
        B, L, _ = x.shape

        x = self.embedding(x) + self.pos_enc(
            torch.arange(1, L+1, device=x.device).unsqueeze(0)
        )

        x = torch.cat([
            self.cls_token.expand(B,-1,-1),
            x
        ], dim=1)

        return self.out(
            self.norm(self.transformer(x))[:,0,:]
        )

class ViraExplorer(nn.Module):
    def __init__(self, cnn_dropout=0.25, transformer_dropout=0.1):
        super().__init__()

        self.pattern     = PatternBranch(cnn_dropout)
        self.frequency   = FrequencyBranch(cnn_dropout)
        self.transformer = TransformerBranch(dropout=transformer_dropout)

        self.fc = nn.Sequential(
            nn.Linear(1536,512),
            nn.GELU(),
            nn.Dropout(cnn_dropout),
            nn.LayerNorm(512),

            nn.Linear(512,128),
            nn.GELU(),
            nn.Dropout(cnn_dropout),

            nn.Linear(128,1)
        )

    def forward(self, x):
        return self.fc(torch.cat([
            self.pattern(x),
            self.frequency(x),
            self.transformer(x)
        ], dim=1))

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, smoothing=0.05):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, logits, targets):
        targets = targets.unsqueeze(1)

        targets_smooth = (
            targets * (1-self.smoothing)
            + 0.5 * self.smoothing
        )

        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets_smooth,
            reduction='none'
        )

        return (
            self.alpha
            * (1-torch.exp(-bce))**self.gamma
            * bce
        ).mean()

EPOCHS        = 120
WARMUP_EPOCHS = 5

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {device}")

model = ViraExplorer(
    cnn_dropout=0.25,
    transformer_dropout=0.1
).to(device)

criterion = FocalLoss(
    alpha=0.75,
    gamma=2.0,
    smoothing=0.05
)

transformer_params = list(model.transformer.parameters())

other_params = (
    list(model.pattern.parameters()) +
    list(model.frequency.parameters()) +
    list(model.fc.parameters())
)

def lr_lambda(epoch):
    if epoch < WARMUP_EPOCHS:
        return epoch / WARMUP_EPOCHS

    progress = (
        (epoch - WARMUP_EPOCHS)
        / (EPOCHS - WARMUP_EPOCHS)
    )

    return 0.5 * (
        1 + math.cos(math.pi * progress)
    )

optimizer = torch.optim.AdamW([
    {
        "params": other_params,
        "lr": 1e-4
    },
    {
        "params": transformer_params,
        "lr": 5e-5
    },
], weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda
)

scaler = torch.amp.GradScaler('cuda')

best_auc    = 0
patience    = 15
counter     = 0
START_EPOCH = 0

for p in model.transformer.parameters():
    p.requires_grad = False

if os.path.exists(CHECKPOINT_PATH):
    ckpt = torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )

    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    scheduler.load_state_dict(ckpt['scheduler_state'])

    best_auc    = ckpt['best_auc']
    counter     = ckpt['counter']
    START_EPOCH = ckpt['epoch'] + 1

    if START_EPOCH >= WARMUP_EPOCHS:
        for p in model.transformer.parameters():
            p.requires_grad = True

        print("Transformer already unfrozen (past warmup)")

    print(
        f"Resumed from epoch {START_EPOCH} "
        f"| Best AUC so far: {best_auc:.4f}"
    )

else:
    print("No checkpoint found — starting from scratch")

print(f"Starting from epoch {START_EPOCH}")

for epoch in range(START_EPOCH, EPOCHS):

    if epoch == WARMUP_EPOCHS:
        for p in model.transformer.parameters():
            p.requires_grad = True

        print("Transformer unfrozen")

    model.train()

    total_loss = 0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            loss = criterion(
                model(X_batch),
                y_batch
            )

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    scheduler.step()

    model.eval()

    preds = []
    targets_list = []

    with torch.no_grad():

        for X_batch, y_batch in val_loader:

            probs = torch.sigmoid(
                model(X_batch.to(device))
            )

            preds.extend(
                probs.cpu().numpy().flatten()
            )

            targets_list.extend(
                y_batch.numpy()
            )

    auc = roc_auc_score(
        targets_list,
        preds
    )

    print(
        f"Epoch {epoch+1:3d} | "
        f"Loss: {total_loss/len(train_loader):.4f} | "
        f"Val AUC: {auc:.4f}"
    )

    if auc > best_auc:

        best_auc = auc
        counter  = 0

        torch.save(
            model.state_dict(),
            BEST_MODEL_PATH
        )

        print(
            f"  Saved new best: {best_auc:.4f}"
        )

    torch.save({
        'epoch':           epoch,
        'model_state':     model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'best_auc':        best_auc,
        'counter':         counter,
    }, CHECKPOINT_PATH)

    counter += (
        0 if auc > best_auc else 1
    )

    if counter >= patience:

        print(
            f"Early stopping at epoch {epoch+1} | "
            f"Best Val AUC: {best_auc:.4f}"
        )

        break

print(
    f"\nTraining complete | "
    f"Best Val AUC: {best_auc:.4f}"
)

print("\nRunning test set evaluation...")

model.load_state_dict(
    torch.load(
        BEST_MODEL_PATH,
        map_location=device
    )
)

model.eval()

preds   = []
targets = []

with torch.no_grad():

    for X, y in test_loader:

        probs = torch.sigmoid(
            model(X.to(device))
        )

        preds.extend(
            probs.cpu().numpy().flatten()
        )

        targets.extend(
            y.numpy()
        )

preds   = np.array(preds)
targets = np.array(targets)

auroc = roc_auc_score(targets, preds)

auprc = average_precision_score(
    targets,
    preds
)

prec, rec, thresh = precision_recall_curve(
    targets,
    preds
)

f1s = 2 * prec * rec / (
    prec + rec + 1e-8
)

best_thresh = thresh[np.argmax(f1s)]

preds_bin = (
    preds >= best_thresh
).astype(int)

tn, fp, fn, tp = confusion_matrix(
    targets,
    preds_bin
).ravel()

print("=" * 45)

print(
    f"  Test AUROC        : {auroc:.4f}   "
    f"(ViraMiner: 0.923)"
)

print(f"  Test AUPRC        : {auprc:.4f}")
print(f"  Best threshold    : {best_thresh:.3f}")

print(
    f"  Precision (virus) : "
    f"{tp/(tp+fp):.4f}"
)

print(
    f"  Recall    (virus) : "
    f"{tp/(tp+fn):.4f}"
)

print(
    f"  F1        (virus) : "
    f"{2*tp/(2*tp+fp+fn):.4f}"
)

print("=" * 45)

print(classification_report(
    targets,
    preds_bin,
    target_names=["non-virus","virus"]
))

fig, axes = plt.subplots(
    1, 2,
    figsize=(12, 5)
)

fpr, tpr, _ = roc_curve(
    targets,
    preds
)

axes[0].plot(
    fpr,
    tpr,
    color='steelblue',
    lw=2,
    label=f'ViraExplorer (AUC={auroc:.3f})'
)

axes[0].plot(
    [0,1],
    [0,1],
    'k--',
    label='Random (AUC=0.500)'
)

axes[0].axvline(
    x=0.077,
    color='red',
    linestyle='--',
    alpha=0.7,
    label='ViraMiner AUC=0.923'
)

axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')

axes[0].set_title('ROC Curve — Test Set')

axes[0].legend()

axes[0].grid(alpha=0.3)

axes[1].plot(
    rec,
    prec,
    color='darkorange',
    lw=2,
    label=f'ViraExplorer (AP={auprc:.3f})'
)

axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')

axes[1].set_title(
    'Precision-Recall Curve — Test Set'
)

axes[1].legend()

axes[1].grid(alpha=0.3)

plt.tight_layout()

plot_path = os.path.join(
    SAVE_DIR,
    "test_evaluation.png"
)

plt.savefig(
    plot_path,
    dpi=150,
    bbox_inches='tight'
)

print(f"\nPlot saved to {plot_path}")
