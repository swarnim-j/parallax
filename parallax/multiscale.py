from dataclasses import dataclass

import numpy as np

from .db import HazardDB, Hit
from .embed import Embedder


@dataclass
class ScaleHit:
    hit: Hit
    scale: str
    start: int
    end: int


def multiscale_screen(
    sequence: str,
    embedder: Embedder,
    db: HazardDB,
    window_size: int = 100,
    stride: int = 50,
    k: int = 3,
) -> list[ScaleHit]:
    results = []

    whole_emb = embedder.embed(sequence)
    for hit in db.query(whole_emb, k=k):
        results.append(ScaleHit(hit=hit, scale="whole", start=0, end=len(sequence)))

    if len(sequence) > window_size:
        windows = embedder.embed_windows(sequence, window_size, stride)
        for start, end, emb in windows:
            for hit in db.query(emb, k=1):
                results.append(ScaleHit(hit=hit, scale="window", start=start, end=end))

    return results
