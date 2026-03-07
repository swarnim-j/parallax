from dataclasses import dataclass, field

import numpy as np
from Bio.Align import PairwiseAligner

from .classifier import HazardClassifier
from .db import HazardDB
from .embed import Embedder
from .translate import is_dna, translate_dna

RISK_THRESHOLD = 0.85
SEQ_SCREEN_THRESHOLD = 0.30
_aligner = PairwiseAligner(mode="global", match_score=1, mismatch_score=-1, open_gap_score=-2, extend_gap_score=-0.5)


@dataclass
class ScreenHit:
    hazard_name: str
    embed_sim: float
    seq_sim: float
    differential: float
    scale: str
    start: int
    end: int


@dataclass
class ScreenResult:
    risk_score: float
    flagged: bool
    classifier_score: float = 0.0
    seq_screen_flagged: bool = False
    seq_screen_best_identity: float = 0.0
    seq_screen_best_match: str = ""
    hits: list[ScreenHit] = field(default_factory=list)
    explanation: str = ""
    input_type: str = "protein"
    proteins_screened: int = 0


def sequence_identity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    a, b = a[:500], b[:500]
    alignment = _aligner.align(a, b)[0]

    matches = 0
    aligned_len = 0
    for (a_start, a_end), (b_start, b_end) in zip(alignment.aligned[0], alignment.aligned[1]):
        seg_len = min(a_end - a_start, b_end - b_start)
        aligned_len += seg_len
        matches += sum(aa == bb for aa, bb in zip(a[a_start:a_end], b[b_start:b_end]))

    total_columns = len(a) + len(b) - aligned_len
    if total_columns <= 0:
        return 0.0
    return max(0.0, min(1.0, matches / total_columns))


class Screener:
    def __init__(self, embedder: Embedder, db: HazardDB, classifier: HazardClassifier | None = None):
        self.embedder = embedder
        self.db = db
        self.classifier = classifier

    def screen(self, sequence: str) -> ScreenResult:
        sequence = sequence.strip().replace("\n", "").replace(" ", "")

        if is_dna(sequence):
            proteins = translate_dna(sequence)
            input_type = "dna"
        else:
            proteins = [sequence]
            input_type = "protein"

        if not proteins:
            return ScreenResult(risk_score=0.0, flagged=False, explanation="No valid protein sequences found.",
                                input_type=input_type, proteins_screened=0)

        best_cls = 0.0
        best_seq_id = 0.0
        best_seq_match = ""
        all_hits = []

        for protein in proteins:
            embedding = self.embedder.embed(protein)

            if self.classifier:
                best_cls = max(best_cls, self.classifier.predict(embedding))

            top_hits = self.db.query(embedding, k=5)
            for hit in top_hits:
                seq_sim = sequence_identity(protein, hit.sequence)
                if seq_sim > best_seq_id:
                    best_seq_id = seq_sim
                    best_seq_match = hit.name
                all_hits.append(ScreenHit(
                    hazard_name=hit.name, embed_sim=hit.similarity,
                    seq_sim=seq_sim, differential=hit.similarity - seq_sim,
                    scale="whole", start=0, end=len(protein),
                ))

        all_hits.sort(key=lambda h: h.embed_sim, reverse=True)

        risk_score = float(best_cls) if self.classifier else 0.0
        flagged = bool(risk_score > RISK_THRESHOLD)
        seq_flagged = bool(best_seq_id > SEQ_SCREEN_THRESHOLD)
        top = all_hits[0] if all_hits else None

        if flagged and not seq_flagged:
            explanation = (
                f"Parallax flagged this as {risk_score:.0%} hazardous, but traditional sequence screening "
                f"would MISS it (best match: {best_seq_match}, {best_seq_id:.0%} identity — below detection threshold). "
                f"This is the gap Parallax fills."
            )
        elif flagged:
            explanation = (
                f"Classified as {risk_score:.0%} hazardous. "
                f"Nearest known hazard: {best_seq_match} ({best_seq_id:.0%} sequence identity). "
                f"Traditional screening would also catch this."
            )
        elif top:
            explanation = (
                f"Classified as {risk_score:.0%} hazardous — below threshold. "
                f"Nearest hazard: {best_seq_match}."
            )
        else:
            explanation = "No significant matches found."

        return ScreenResult(
            risk_score=risk_score, flagged=flagged,
            classifier_score=best_cls,
            seq_screen_flagged=seq_flagged,
            seq_screen_best_identity=best_seq_id,
            seq_screen_best_match=best_seq_match,
            hits=all_hits[:5] if flagged else all_hits[:3],
            explanation=explanation,
            input_type=input_type, proteins_screened=len(proteins),
        )
