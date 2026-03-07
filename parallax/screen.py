from dataclasses import dataclass, field

import numpy as np
from Bio.Align import PairwiseAligner

from .classifier import HazardClassifier
from .db import HazardDB
from .embed import Embedder
from .multiscale import multiscale_screen
from .translate import is_dna, translate_dna

RISK_THRESHOLD = 0.6
_aligner = PairwiseAligner(mode="global", match_score=1, mismatch_score=0, open_gap_score=0, extend_gap_score=0)


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
    hits: list[ScreenHit] = field(default_factory=list)
    explanation: str = ""
    input_type: str = "protein"
    proteins_screened: int = 0


def sequence_identity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    a, b = a[:500], b[:500]
    return _aligner.score(a, b) / max(len(a), len(b))


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

        best_classifier_score = 0.0
        all_hits = []

        for protein in proteins:
            embedding = self.embedder.embed(protein)

            if self.classifier:
                cls_score = self.classifier.predict(embedding)
                best_classifier_score = max(best_classifier_score, cls_score)

            for sh in multiscale_screen(protein, self.embedder, self.db):
                seq_sim = sequence_identity(protein[sh.start:sh.end], sh.hit.sequence)
                all_hits.append(ScreenHit(
                    hazard_name=sh.hit.name, embed_sim=sh.hit.similarity,
                    seq_sim=seq_sim, differential=sh.hit.similarity - seq_sim,
                    scale=sh.scale, start=sh.start, end=sh.end,
                ))

        all_hits.sort(key=lambda h: h.embed_sim, reverse=True)

        if self.classifier:
            risk_score = best_classifier_score
        elif all_hits:
            risk_score = max(0.0, min(1.0, all_hits[0].embed_sim - all_hits[0].seq_sim))
        else:
            risk_score = 0.0

        flagged = risk_score > RISK_THRESHOLD
        top = all_hits[0] if all_hits else None

        if flagged and top:
            explanation = (
                f"Classifier confidence: {risk_score:.0%} hazardous. "
                f"Nearest known hazard: {top.hazard_name} "
                f"(embed_sim={top.embed_sim:.2f}, seq_sim={top.seq_sim:.2f})."
            )
        elif top:
            explanation = (
                f"Classifier confidence: {risk_score:.0%} hazardous. "
                f"Nearest hazard: {top.hazard_name} "
                f"(embed_sim={top.embed_sim:.2f}, seq_sim={top.seq_sim:.2f}). "
                f"No significant risk detected."
            )
        else:
            explanation = "No significant matches found."

        return ScreenResult(
            risk_score=risk_score, flagged=flagged,
            classifier_score=best_classifier_score,
            hits=all_hits[:10], explanation=explanation,
            input_type=input_type, proteins_screened=len(proteins),
        )
