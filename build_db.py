#!/usr/bin/env python3
"""Fetch proteins, generate ProteinMPNN variants, embed, train classifier."""

import subprocess
import sys
from pathlib import Path

import requests

UNIPROT_BASE = "https://rest.uniprot.org/uniprotkb/search"
ALPHAFOLD_PDB = "https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v6.pdb"
PAGE_SIZE = 500

HAZARD_QUERY = "(keyword:KW-0800+OR+keyword:KW-0843+OR+keyword:KW-0046+OR+go:0090729)+AND+reviewed:true"
BENIGN_QUERY = "reviewed:true+AND+organism_id:9606+NOT+keyword:KW-0800+NOT+keyword:KW-0843"

# ProteinMPNN settings — top N hazards get synthetic variants
MPNN_TOP_N = 50
MPNN_VARIANTS = 20
MPNN_TEMP = "0.2"


def fetch_uniprot(query: str) -> dict[str, tuple[str, str]]:
    proteins = {}
    url = f"{UNIPROT_BASE}?query={query}&format=fasta&size={PAGE_SIZE}"
    while url:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        for block in resp.text.strip().split("\n>"):
            lines = block.strip().split("\n")
            if not lines or not lines[0]:
                continue
            header = lines[0].lstrip(">")
            seq = "".join(lines[1:])
            if not seq:
                continue
            parts = header.split("|")
            acc = parts[1] if len(parts) >= 3 else header.split()[0]
            name = parts[2].split(" OS=")[0].strip() if len(parts) >= 3 else acc
            proteins[name] = (seq, acc)
        link = resp.headers.get("Link", "")
        url = link.split(";")[0].strip("<>") if 'rel="next"' in link else None
    return proteins


def download_structure(acc: str, out_dir: Path) -> Path | None:
    pdb_path = out_dir / f"{acc}.pdb"
    if pdb_path.exists():
        return pdb_path
    try:
        resp = requests.get(ALPHAFOLD_PDB.format(acc=acc), timeout=30)
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
        "--num_seq_per_target", str(MPNN_VARIANTS),
        "--sampling_temp", MPNN_TEMP,
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
    return seqs[1:]  # skip original sequence (first entry)


def main():
    # Step 1: Fetch from UniProt
    print("Fetching hazard proteins...", flush=True)
    hazards = fetch_uniprot(HAZARD_QUERY)
    print(f"  {len(hazards)} hazard proteins", flush=True)

    print("Fetching benign proteins...", flush=True)
    benign = fetch_uniprot(BENIGN_QUERY)
    print(f"  {len(benign)} benign proteins", flush=True)

    # Step 2: ProteinMPNN variants for top hazards
    print(f"\nGenerating ProteinMPNN variants for top {MPNN_TOP_N} hazards...", flush=True)

    if not Path("ProteinMPNN").exists():
        subprocess.run(["git", "clone", "https://github.com/dauparas/ProteinMPNN.git"], check=True)

    struct_dir = Path("data/structures")
    struct_dir.mkdir(parents=True, exist_ok=True)
    mpnn_out = Path("data/mpnn_out")

    accs = [acc for _, (_, acc) in list(hazards.items())[:MPNN_TOP_N]]
    total_variants = 0

    for i, acc in enumerate(accs):
        pdb = download_structure(acc, struct_dir)
        if not pdb:
            continue
        variants = run_mpnn(pdb, mpnn_out)
        for j, seq in enumerate(variants):
            hazards[f"mpnn_{acc}_{j}"] = (seq, f"{acc}_v{j}")
            total_variants += 1
        print(f"  [{i+1}/{MPNN_TOP_N}] {acc}: {len(variants)} variants", flush=True)

    print(f"  Total synthetic variants: {total_variants}", flush=True)

    # Step 3: Embed
    print(f"\nEmbedding {len(hazards)} hazard + {len(benign)} benign proteins...", flush=True)
    from parallax.embed import Embedder
    from parallax.db import HazardDB

    embedder = Embedder()
    for label, proteins, path in [
        ("hazard", hazards, Path("data/hazard_db")),
        ("benign", benign, Path("data/benign_db")),
    ]:
        print(f"  {label}: {len(proteins)}...", flush=True)
        db = HazardDB()
        db.build(proteins, embedder)
        db.save(str(path))

    # Step 4: Train MLP classifier
    print("\nTraining classifier...", flush=True)
    from parallax.classifier import train_classifier
    hazard_db = HazardDB.load("data/hazard_db")
    benign_db = HazardDB.load("data/benign_db")
    model, metrics = train_classifier(hazard_db.embeddings, benign_db.embeddings, epochs=30)
    model.save("data/classifier")
    print(f"  acc={metrics['val_acc']:.3f} prec={metrics['precision']:.3f} rec={metrics['recall']:.3f}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
