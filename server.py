#!/usr/bin/env python3
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sklearn.manifold import TSNE

from parallax.classifier import HazardClassifier
from parallax.db import HazardDB
from parallax.embed import Embedder
from parallax.screen import Screener
from parallax.translate import is_dna, translate_dna

print("Loading...", flush=True)
embedder = Embedder()
hazard_db = HazardDB.load("data/hazard_db")
benign_db = HazardDB.load("data/benign_db") if Path("data/benign_db").exists() else None
classifier = HazardClassifier.load("data/classifier") if Path("data/classifier").exists() else None
screener = Screener(embedder, hazard_db, classifier)

# Sample subset for t-SNE visualization (full DB is too large to plot)
VIZ_SAMPLE = 100
rng = np.random.RandomState(42)
all_emb, all_labels, all_names = [], [], []

h_idx = rng.choice(len(hazard_db.names), min(VIZ_SAMPLE, len(hazard_db.names)), replace=False)
for i in h_idx:
    all_emb.append(hazard_db.embeddings[i]); all_labels.append("hazard"); all_names.append(hazard_db.names[i])
if benign_db:
    b_idx = rng.choice(len(benign_db.names), min(VIZ_SAMPLE, len(benign_db.names)), replace=False)
    for i in b_idx:
        all_emb.append(benign_db.embeddings[i]); all_labels.append("benign"); all_names.append(benign_db.names[i])

all_emb_np = np.stack(all_emb)
coords = TSNE(n_components=2, perplexity=min(30, len(all_emb_np) - 1), random_state=42).fit_transform(all_emb_np)
projection_points = [
    {"x": float(coords[i, 0]), "y": float(coords[i, 1]), "label": all_labels[i], "name": all_names[i]}
    for i in range(len(all_names))
]
print("Ready.", flush=True)

app = FastAPI(title="Parallax")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

ESMFOLD_ENDPOINT = "https://api.esmatlas.com/foldSequence/v1/pdb/"
MAX_FOLD_LENGTH = 400
ALLOWED_AA = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")


class ScreenRequest(BaseModel):
    sequence: str


def normalize_sequence(sequence: str) -> str:
    return sequence.strip().replace("\n", "").replace(" ", "").upper()


def protein_for_folding(sequence: str) -> tuple[str, str]:
    cleaned = normalize_sequence(sequence)
    if not cleaned:
        raise HTTPException(status_code=400, detail="Sequence is empty.")

    if is_dna(cleaned):
        proteins = translate_dna(cleaned)
        if not proteins:
            raise HTTPException(
                status_code=422,
                detail="No translatable ORF (minimum 50 aa) was found in the DNA sequence.",
            )
        protein = max(proteins, key=len)
        return protein, "dna"

    if any(char not in ALLOWED_AA for char in cleaned):
        raise HTTPException(
            status_code=422,
            detail="Protein sequence contains unsupported characters for folding.",
        )
    return cleaned, "protein"


@lru_cache(maxsize=128)
def fold_with_esmfold(protein_sequence: str) -> str:
    req = Request(
        ESMFOLD_ENDPOINT,
        data=protein_sequence.encode("utf-8"),
        method="POST",
        headers={"Content-Type": "text/plain"},
    )

    try:
        with urlopen(req, timeout=150) as resp:
            pdb_text = resp.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Folding service returned {exc.code}: {body[:240]}") from exc
    except URLError as exc:
        raise RuntimeError("Folding service is unreachable right now.") from exc

    if "ATOM" not in pdb_text:
        raise RuntimeError("Folding service returned an invalid PDB payload.")

    return pdb_text


def project_query(embedding: np.ndarray) -> tuple[float, float]:
    sims = all_emb_np @ embedding
    w = np.exp(sims * 10)
    w /= w.sum()
    return float(w @ coords[:, 0]), float(w @ coords[:, 1])


@app.post("/api/screen")
def screen_sequence(req: ScreenRequest):
    result = screener.screen(req.sequence)
    seq = normalize_sequence(req.sequence)
    if is_dna(seq):
        proteins = translate_dna(seq)
        query_emb = embedder.embed(proteins[0]) if proteins else embedder.embed(seq[:500])
    else:
        query_emb = embedder.embed(seq)
    qx, qy = project_query(query_emb)
    return JSONResponse({"result": asdict(result), "query_point": {"x": qx, "y": qy}})


@app.get("/api/embedding-space")
def embedding_space():
    return JSONResponse({"points": projection_points})


@app.post("/api/structure")
def structure_sequence(req: ScreenRequest):
    protein, input_type = protein_for_folding(req.sequence)

    truncated = len(protein) > MAX_FOLD_LENGTH
    fold_protein = protein[:MAX_FOLD_LENGTH] if truncated else protein

    try:
        pdb_text = fold_with_esmfold(fold_protein)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return JSONResponse(
        {
            "input_type": input_type,
            "protein_length": len(protein),
            "protein_sequence": protein,
            "pdb": pdb_text,
            "fold_source": "esmatlas-esmfold-v1",
            "truncated": truncated,
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
