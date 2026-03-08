# Parallax

Embedding-based biosecurity screening for AI-redesigned proteins.

Parallax uses ESM2 protein language model embeddings and nearest-neighbor search to detect hazardous proteins — even when they have been redesigned to evade traditional sequence-identity screening. It compares each query against a curated database of known hazards, runs an MLP classifier, and explains whether traditional BLAST-style screening would have caught the same threat.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                  Frontend  (Next.js / React)              │
│         sequence input · risk dashboard · 3D viewer       │
└────────────────────────────┬─────────────────────────────┘
                             │  HTTP
                             ▼
┌──────────────────────────────────────────────────────────┐
│                  Backend  (FastAPI / Uvicorn)              │
│                                                           │
│  POST /api/screen     → screening pipeline                │
│  POST /api/structure  → ESMFold 3D structure prediction   │
│  GET  /api/embedding-space → t-SNE projection             │
└──────┬──────────┬───────────────┬────────────────────────┘
       │          │               │
       ▼          ▼               ▼
   Embedder    HazardDB     HazardClassifier
   (ESM2)     (numpy +       (PyTorch MLP)
              metadata)
```

**Screening pipeline** (`parallax/screen.py`):

1. Normalize the input. If DNA, translate all six reading frames and extract ORFs ≥ 50 amino acids.
2. Embed each protein with ESM2 (mean-pooled, L2-normalized).
3. Score the embedding with the MLP classifier (hazard probability).
4. Query the hazard database for the *k*-nearest neighbors by cosine similarity.
5. Compute pairwise sequence identity against each hit using Biopython global alignment.
6. Flag the query if the classifier score > 0.85 or any hit exceeds 30 % sequence identity.
7. Generate a comparative explanation (Parallax vs. traditional screening).

## Project Structure

```
parallax/
├── server.py              # FastAPI application and endpoint definitions
├── build_db.py            # End-to-end pipeline: fetch data, generate variants, build DBs, train classifier
├── quick_train.py         # Re-train the classifier from existing DBs
├── train.py               # Optional ESM2 fine-tuning script
├── gen_variants.py        # Generate ProteinMPNN variants and rebuild DBs
├── parallax/
│   ├── embed.py           # ESM2 embedding (single, batch, sliding-window)
│   ├── db.py              # HazardDB: build, query, save/load
│   ├── classifier.py      # MLP classifier: architecture and training loop
│   ├── screen.py          # Screener: orchestrates the full screening pipeline
│   ├── translate.py       # DNA detection and six-frame translation
│   └── multiscale.py      # Experimental multi-scale screening (unused in main flow)
├── data/
│   ├── hazard_db/         # embeddings.npy + metadata.json
│   ├── benign_db/         # embeddings.npy + metadata.json
│   └── classifier/        # model.pt + config.json
├── web/                   # Next.js frontend
│   ├── app/
│   │   ├── page.tsx       # Single-page UI: input, results, 3D viewer
│   │   ├── layout.tsx     # Root layout and fonts
│   │   └── globals.css    # Styling and CSS variables
│   └── package.json
├── pyproject.toml         # Python dependencies (managed with uv)
└── sky-train.yaml         # Cloud training config (SkyPilot / GCP A100)
```

## Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- Node.js 18+

### 1. Install dependencies

```bash
# Python (from the project root)
uv sync

# Frontend
cd web
npm install
```

### 2. Build the databases

Before the server can screen anything, you need the hazard database. The full pipeline fetches proteins from UniProt, optionally generates ProteinMPNN variants, computes ESM2 embeddings, and trains the classifier:

```bash
uv run python build_db.py
```

This creates `data/hazard_db/`, `data/benign_db/`, and `data/classifier/`.

For a quicker iteration cycle (databases already exist, just re-train the classifier):

```bash
uv run python quick_train.py
```

### 3. Start the servers

```bash
# Backend (serves on http://localhost:8000)
uv run python server.py

# Frontend (serves on http://localhost:3000)
cd web
npm run dev
```

Open the frontend URL in your browser, paste a protein or DNA sequence, and click **Screen Sequence**.

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/screen` | Screen a protein or DNA sequence. Body: `{ "sequence": "MKFL..." }`. Returns risk score, nearest-neighbor hits, flagged status, and explanation. |
| `POST` | `/api/structure` | Predict 3D structure via ESMFold. Body: `{ "sequence": "MKFL..." }`. Returns PDB text (sequences truncated to 400 aa). |
| `GET` | `/api/embedding-space` | t-SNE projection of hazard and benign database entries. |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ESM_MODEL` | `esm2_t30_150M_UR50D` (GPU) / `esm2_t6_8M_UR50D` (CPU) | ESM2 model to use for embeddings |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Backend URL for the frontend |

## How It Works

Traditional biosecurity screening relies on BLAST-style sequence identity to flag dangerous proteins. This fails when a protein is redesigned — tools like ProteinMPNN can produce functional variants with low sequence identity to any known hazard.

Parallax addresses this gap by operating in **embedding space**. ESM2 captures functional and structural similarity that persists even when the raw sequence has diverged. The screening pipeline reports both the embedding-based risk score and the classical sequence identity, then explains the difference — highlighting cases where a redesigned hazard would slip past traditional methods.
