#!/usr/bin/env python3
"""Re-embed with current model if needed, then train classifier."""
from pathlib import Path
import numpy as np
from parallax.db import HazardDB
from parallax.embed import Embedder
from parallax.classifier import train_classifier

embedder = Embedder()
dim = embedder.embed("MKVL").shape[0]
print(f"Model: {embedder.model_name}, dim: {dim}, device: {embedder.device}", flush=True)

for label, path in [("hazard", "data/hazard_db"), ("benign", "data/benign_db")]:
    db = HazardDB.load(path)
    if db.embeddings.shape[1] != dim:
        print(f"Re-embedding {label}: {len(db.sequences)} proteins ({db.embeddings.shape[1]}d -> {dim}d)...", flush=True)
        db.embeddings = embedder.embed_batch(db.sequences)
        db.save(path)
    else:
        print(f"{label}: {db.embeddings.shape} — already correct dim", flush=True)

hdb = HazardDB.load("data/hazard_db")
bdb = HazardDB.load("data/benign_db")
print(f"\nTraining on {len(hdb.names)} hazard + {len(bdb.names)} benign...", flush=True)

model, m = train_classifier(hdb.embeddings, bdb.embeddings, epochs=30)
model.save("data/classifier")
print(f"acc={m['val_acc']:.3f} prec={m['precision']:.3f} rec={m['recall']:.3f}", flush=True)
print("Done.", flush=True)
