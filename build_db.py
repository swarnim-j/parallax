#!/usr/bin/env python3
"""Fetch hazard + benign proteins from UniProt, download AlphaFold structures,
generate synthetic variants with ProteinMPNN, embed everything, and train classifier."""

import json
import subprocess
import sys
from pathlib import Path

import requests

UNIPROT_BASE = "https://rest.uniprot.org/uniprotkb/search"
ALPHAFOLD_PDB = "https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v4.pdb"
PAGE_SIZE = 500

HAZARD_QUERY = "(keyword:KW-0800+OR+keyword:KW-0843+OR+keyword:KW-0046+OR+go:0090729)+AND+reviewed:true"
BENIGN_QUERY = "keyword:KW-0597+AND+reviewed:true+AND+organism_id:9606"

MPNN_VARIANTS_PER_STRUCTURE = 5
MPNN_SAMPLING_TEMP = 0.2


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


def download_alphafold_structures(accessions: list[str], out_dir: Path, max_count: int = 2000) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for acc in accessions[:max_count]:
        pdb_path = out_dir / f"AF-{acc}-F1.pdb"
        if pdb_path.exists():
            downloaded.append(pdb_path)
            continue
        url = ALPHAFOLD_PDB.format(acc=acc)
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200 and "ATOM" in resp.text:
                pdb_path.write_text(resp.text)
                downloaded.append(pdb_path)
        except Exception:
            pass
    return downloaded


def generate_mpnn_variants(pdb_dir: Path, out_dir: Path, n_variants: int, temp: float) -> dict[str, list[str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    mpnn_dir = Path("ProteinMPNN")

    if not mpnn_dir.exists():
        print("  Cloning ProteinMPNN...", flush=True)
        subprocess.run(["git", "clone", "https://github.com/dauparas/ProteinMPNN.git"], check=True)

    variants = {}
    pdb_files = sorted(pdb_dir.glob("*.pdb"))
    print(f"  Running ProteinMPNN on {len(pdb_files)} structures...", flush=True)

    for pdb_path in pdb_files:
        acc = pdb_path.stem.replace("AF-", "").replace("-F1", "")
        result_dir = out_dir / acc
        result_dir.mkdir(exist_ok=True)

        cmd = [
            sys.executable, str(mpnn_dir / "protein_mpnn_run.py"),
            "--pdb_path", str(pdb_path),
            "--out_folder", str(result_dir),
            "--num_seq_per_target", str(n_variants),
            "--sampling_temp", str(temp),
            "--model_name", "v_48_020",
            "--seed", "42",
        ]

        try:
            subprocess.run(cmd, capture_output=True, timeout=120)
            fasta_dir = result_dir / "seqs"
            if fasta_dir.exists():
                seqs = []
                for fa in fasta_dir.glob("*.fa"):
                    for line in fa.read_text().split("\n"):
                        if line and not line.startswith(">"):
                            seqs.append(line.strip())
                if seqs:
                    variants[acc] = seqs[1:]  # skip first (original sequence)
        except Exception:
            pass

    return variants


def main():
    print("=== Step 1: Fetch proteins from UniProt ===", flush=True)

    print("Fetching hazard proteins...", flush=True)
    hazards = fetch_uniprot(HAZARD_QUERY)
    print(f"  Got {len(hazards)} hazard proteins", flush=True)

    print("Fetching benign proteins...", flush=True)
    benign = fetch_uniprot(BENIGN_QUERY)
    print(f"  Got {len(benign)} benign proteins", flush=True)

    print("\n=== Step 2: Download AlphaFold structures ===", flush=True)
    hazard_accs = [acc for _, (_, acc) in hazards.items()]
    structures_dir = Path("data/structures")
    pdb_files = download_alphafold_structures(hazard_accs, structures_dir)
    print(f"  Downloaded {len(pdb_files)} structures", flush=True)

    print("\n=== Step 3: Generate ProteinMPNN variants ===", flush=True)
    variants_dir = Path("data/mpnn_variants")
    variants = generate_mpnn_variants(structures_dir, variants_dir, MPNN_VARIANTS_PER_STRUCTURE, MPNN_SAMPLING_TEMP)
    total_variants = sum(len(v) for v in variants.values())
    print(f"  Generated {total_variants} synthetic variants from {len(variants)} structures", flush=True)

    # Add synthetic variants to hazard set
    for acc, seqs in variants.items():
        for i, seq in enumerate(seqs):
            hazards[f"synthetic_{acc}_{i}"] = (seq, f"{acc}_mpnn_{i}")

    print(f"\n=== Step 4: Embed all proteins ===", flush=True)
    print(f"  Total hazard (real + synthetic): {len(hazards)}", flush=True)
    print(f"  Total benign: {len(benign)}", flush=True)

    from parallax.embed import Embedder
    from parallax.db import HazardDB

    embedder = Embedder()

    for label, proteins, path in [
        ("hazard", hazards, Path("data/hazard_db")),
        ("benign", benign, Path("data/benign_db")),
    ]:
        print(f"  Embedding {len(proteins)} {label} proteins...", flush=True)
        db = HazardDB()
        db.build(proteins, embedder)
        db.save(str(path))
        print(f"  Saved to {path}", flush=True)

    print("\n=== Step 5: Train classifier ===", flush=True)
    from parallax.classifier import train_classifier
    hazard_db = HazardDB.load("data/hazard_db")
    benign_db = HazardDB.load("data/benign_db")
    model, metrics = train_classifier(hazard_db.embeddings, benign_db.embeddings, epochs=30)
    model.save("data/classifier")
    print(f"  val_acc={metrics['val_acc']:.3f} precision={metrics['precision']:.3f} recall={metrics['recall']:.3f}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
