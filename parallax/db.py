import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Hit:
    name: str
    sequence: str
    similarity: float
    source_id: str


class HazardDB:
    def __init__(self):
        self.names: list[str] = []
        self.sequences: list[str] = []
        self.source_ids: list[str] = []
        self.embeddings: np.ndarray | None = None

    def build(self, proteins: dict[str, tuple[str, str]], embedder) -> None:
        self.names = []
        self.sequences = []
        self.source_ids = []

        for name, (sequence, source_id) in proteins.items():
            self.names.append(name)
            self.sequences.append(sequence)
            self.source_ids.append(source_id)

        self.embeddings = embedder.embed_batch(self.sequences)

    def query(self, embedding: np.ndarray, k: int = 5) -> list[Hit]:
        if self.embeddings is None:
            return []
        scores = self.embeddings @ embedding
        top_k = min(k, len(self.names))
        indices = np.argsort(scores)[::-1][:top_k]
        return [
            Hit(
                name=self.names[i],
                sequence=self.sequences[i],
                similarity=float(scores[i]),
                source_id=self.source_ids[i],
            )
            for i in indices
        ]

    def save(self, path: str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "embeddings.npy", self.embeddings)
        with open(path / "metadata.json", "w") as f:
            json.dump({
                "names": self.names,
                "sequences": self.sequences,
                "source_ids": self.source_ids,
            }, f)

    @classmethod
    def load(cls, path: str) -> "HazardDB":
        path = Path(path)
        db = cls()
        db.embeddings = np.load(path / "embeddings.npy")
        with open(path / "metadata.json") as f:
            meta = json.load(f)
        db.names = meta["names"]
        db.sequences = meta["sequences"]
        db.source_ids = meta["source_ids"]
        return db
