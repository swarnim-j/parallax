#!/usr/bin/env python3
"""Generate ProteinMPNN variants for top hazard proteins and re-embed."""

import json
import subprocess
import sys
from pathlib import Path

import requests

ALPHAFOLD_API = "https://alphafold.ebi.ac.uk/api/prediction/{acc}"
TOP_N = 500
N_VARIANTS = 20
TEMP = "0.2"


def get_pdb_url(acc: str) -> str | None:
    try:
        resp = requests.get(ALPHAFOLD_API.format(acc=acc), timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            if data:
                return data[0].get("pdbUrl")
    except Exception:
        pass
    return None


def download_structure(acc: str, out_dir: Path) -> Path | None:
    pdb_path = out_dir / f"{acc}.pdb"
    if pdb_path.exists():
        return pdb_path
    url = get_pdb_url(acc)
    if not url:
        return None
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200 and "ATOM" in resp.text:
            pdb_path.write_text(resp.text)
            return pdb_path
    except Exception:
        pass
    return None


def run_mpnn(pdb_path: Path, out_dir: Path) -> list[str]:
    mpnn_dir = Path("ProteinMPNN")
    result_dir = out_dir / pdb_path.stem
    result_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(mpnn_dir / "protein_mpnn_run.py"),
        "--pdb_path", str(pdb_path),
        "--out_folder", str(result_dir),
        "--num_seq_per_target", str(N_VARIANTS),
        "--sampling_temp", TEMP,
        "--model_name", "v_48_020",
        "--seed", "42",
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=120)
    except Exception:
        return []

    seqs = []
    fasta_dir = result_dir / "seqs"
    if fasta_dir.exists():
        for fa in fasta_dir.glob("*.fa"):
            reading_seq = False
            for line in fa.read_text().split("\n"):
                if line.startswith(">"):
                    reading_seq = True
                    continue
                if reading_seq and line.strip():
                    seqs.append(line.strip())
                    reading_seq = False
    return seqs[1:]  # skip original


def main():
    meta_path = Path("data/hazard_db/metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)

    accs = list(dict.fromkeys(meta["source_ids"]))[:TOP_N]
    print(f"Generating variants for {len(accs)} hazard proteins...", flush=True)

    struct_dir = Path("data/structures")
    struct_dir.mkdir(parents=True, exist_ok=True)
    mpnn_out = Path("data/mpnn_out")

    all_variants = {}
    downloaded = 0
    total_variants = 0

    for i, acc in enumerate(accs):
        pdb = download_structure(acc, struct_dir)
        if not pdb:
            continue
        downloaded += 1
        variants = run_mpnn(pdb, mpnn_out)
        if variants:
            all_variants[acc] = variants
            total_variants += len(variants)

        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(accs)}: {downloaded} structures, {total_variants} variants", flush=True)

    print(f"\nTotal: {downloaded} structures, {total_variants} variants", flush=True)

    # Save variants as FASTA
    out_path = Path("data/mpnn_variants.json")
    with open(out_path, "w") as f:
        json.dump(all_variants, f)
    print(f"Saved to {out_path}", flush=True)

    # Re-embed with variants included
    print("\nRe-embedding hazard DB with synthetic variants...", flush=True)
    from parallax.embed import Embedder
    from parallax.db import HazardDB

    hazard_db = HazardDB.load("data/hazard_db")
    proteins = {}
    for name, seq, sid in zip(hazard_db.names, hazard_db.sequences, hazard_db.source_ids):
        proteins[name] = (seq, sid)

    for acc, seqs in all_variants.items():
        for j, seq in enumerate(seqs):
            proteins[f"mpnn_{acc}_{j}"] = (seq, f"{acc}_v{j}")

    print(f"  Total hazard proteins (real + synthetic): {len(proteins)}", flush=True)
    embedder = Embedder()
    db = HazardDB()
    db.build(proteins, embedder)
    db.save("data/hazard_db")
    print("  Saved updated hazard DB", flush=True)

    # Retrain classifier
    print("\nRetraining classifier...", flush=True)
    from parallax.classifier import train_classifier
    benign_db = HazardDB.load("data/benign_db")
    model, metrics = train_classifier(db.embeddings, benign_db.embeddings, epochs=30)
    model.save("data/classifier")
    print(f"  acc={metrics['val_acc']:.3f} prec={metrics['precision']:.3f} rec={metrics['recall']:.3f}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
