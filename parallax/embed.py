import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import esm
import numpy as np

MAX_SEQ_LEN = 1022


class Embedder:
    def __init__(self, model_name: str | None = None):
        default = "esm2_t30_150M_UR50D" if torch.cuda.is_available() else "esm2_t6_8M_UR50D"
        model_name = model_name or os.environ.get("ESM_MODEL", default)
        self.model_name = model_name
        self.model, self.alphabet = getattr(esm.pretrained, model_name)()
        self.model.eval()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.num_layers = int(model_name.split("_t")[1].split("_")[0])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

    def embed(self, sequence: str) -> np.ndarray:
        if len(sequence) <= MAX_SEQ_LEN:
            return self._embed_single(sequence)
        chunks = [sequence[i:i + MAX_SEQ_LEN] for i in range(0, len(sequence), MAX_SEQ_LEN)]
        embeds = [self._embed_single(c) for c in chunks]
        lengths = [len(c) for c in chunks]
        weighted = sum(e * l for e, l in zip(embeds, lengths)) / sum(lengths)
        return weighted / np.linalg.norm(weighted)

    def embed_batch(self, sequences: list[str]) -> np.ndarray:
        results = []
        for i, seq in enumerate(sequences):
            results.append(self.embed(seq))
            if (i + 1) % 500 == 0:
                print(f"    {i+1}/{len(sequences)} embedded", flush=True)
        if len(sequences) >= 500:
            print(f"    {len(sequences)}/{len(sequences)} embedded", flush=True)
        return np.stack(results)

    def embed_windows(self, sequence: str, window_size: int = 100, stride: int = 50) -> list[tuple[int, int, np.ndarray]]:
        if len(sequence) <= window_size:
            return [(0, len(sequence), self.embed(sequence))]
        windows = []
        for start in range(0, len(sequence) - window_size + 1, stride):
            end = start + window_size
            windows.append((start, end, self.embed(sequence[start:end])))
        if windows[-1][1] < len(sequence):
            windows.append((len(sequence) - window_size, len(sequence), self.embed(sequence[-window_size:])))
        return windows

    def _embed_single(self, sequence: str) -> np.ndarray:
        _, _, tokens = self.batch_converter([("_", sequence)])
        tokens = tokens.to(self.device)
        with torch.no_grad():
            out = self.model(tokens, repr_layers=[self.num_layers])
        emb = out["representations"][self.num_layers][0, 1:-1].mean(0)
        emb = emb.cpu().numpy().astype(np.float32)
        return emb / np.linalg.norm(emb)
