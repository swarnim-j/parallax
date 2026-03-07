#!/usr/bin/env python3
import json
import os
from pathlib import Path

import esm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SEQ_LEN = 1022
EPOCHS = 10
LR = 1e-4
BATCH_SIZE = 8
VAL_SPLIT = 0.1
MODEL_NAME = "esm2_t6_8M_UR50D"
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "data/finetuned"))


class ProteinDataset(Dataset):
    def __init__(self, sequences: list[str], labels: list[int], alphabet):
        self.sequences = sequences
        self.labels = labels
        self.batch_converter = alphabet.get_batch_converter()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx][:MAX_SEQ_LEN], self.labels[idx]

    def collate(self, batch):
        seqs, labels = zip(*batch)
        data = [(f"p{i}", s) for i, s in enumerate(seqs)]
        _, _, tokens = self.batch_converter(data)
        return tokens, torch.tensor(labels, dtype=torch.float32)


class HazardModel(nn.Module):
    def __init__(self, esm_model, num_layers: int):
        super().__init__()
        self.esm = esm_model
        self.num_layers = num_layers
        hidden = esm_model.embed_dim
        self.head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, tokens):
        out = self.esm(tokens, repr_layers=[self.num_layers])
        emb = out["representations"][self.num_layers][:, 1:-1].mean(dim=1)
        return self.head(emb).squeeze(-1)


def load_data():
    hazard_meta = json.load(open("data/hazard_db/metadata.json"))
    benign_meta = json.load(open("data/benign_db/metadata.json"))

    sequences = hazard_meta["sequences"] + benign_meta["sequences"]
    labels = [1] * len(hazard_meta["sequences"]) + [0] * len(benign_meta["sequences"])

    idx = np.random.RandomState(42).permutation(len(sequences))
    sequences = [sequences[i] for i in idx]
    labels = [labels[i] for i in idx]

    split = int(len(sequences) * (1 - VAL_SPLIT))
    return sequences[:split], labels[:split], sequences[split:], labels[split:]


def evaluate(model, val_dl):
    model.eval()
    correct, total = 0, 0
    tp, fp, fn = 0, 0, 0
    with torch.no_grad():
        for tokens, labels in val_dl:
            tokens, labels = tokens.to(DEVICE), labels.to(DEVICE)
            preds = torch.sigmoid(model(tokens)) > 0.5
            targets = labels > 0.5
            correct += (preds == targets).sum().item()
            total += len(labels)
            tp += (preds & targets).sum().item()
            fp += (preds & ~targets).sum().item()
            fn += (~preds & targets).sum().item()
    acc = correct / total
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    return {"acc": acc, "precision": prec, "recall": rec}


def main():
    print(f"Device: {DEVICE}", flush=True)
    print(f"Loading {MODEL_NAME}...", flush=True)

    esm_model, alphabet = getattr(esm.pretrained, MODEL_NAME)()
    num_layers = int(MODEL_NAME.split("_t")[1].split("_")[0])
    model = HazardModel(esm_model, num_layers).to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}", flush=True)

    print("Loading data...", flush=True)
    train_seqs, train_labels, val_seqs, val_labels = load_data()
    print(f"Train: {len(train_seqs)} ({sum(train_labels)} hazard, {len(train_labels)-sum(train_labels)} benign)", flush=True)
    print(f"Val: {len(val_seqs)} ({sum(val_labels)} hazard, {len(val_labels)-sum(val_labels)} benign)", flush=True)

    train_ds = ProteinDataset(train_seqs, train_labels, alphabet)
    val_ds = ProteinDataset(val_seqs, val_labels, alphabet)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=train_ds.collate)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=val_ds.collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loss_fn = nn.BCEWithLogitsLoss()

    best_acc = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        n_batches = 0
        for tokens, labels in train_dl:
            tokens, labels = tokens.to(DEVICE), labels.to(DEVICE)
            logits = model(tokens)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        metrics = evaluate(model, val_dl)
        print(f"Epoch {epoch+1}/{EPOCHS}  loss={total_loss/n_batches:.4f}  val_acc={metrics['acc']:.3f}  prec={metrics['precision']:.3f}  rec={metrics['recall']:.3f}", flush=True)

        if metrics["acc"] > best_acc:
            best_acc = metrics["acc"]
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "esm_model_name": MODEL_NAME,
                "num_layers": num_layers,
                "metrics": metrics,
                "epoch": epoch + 1,
            }, OUTPUT_DIR / "best_model.pt")
            print(f"  Saved best model (acc={best_acc:.3f})", flush=True)

    print(f"\nBest val accuracy: {best_acc:.3f}", flush=True)
    print(f"Model saved to {OUTPUT_DIR}/best_model.pt", flush=True)


if __name__ == "__main__":
    main()
