from Bio.Seq import Seq

MIN_ORF_LENGTH = 50


def is_dna(sequence: str) -> bool:
    cleaned = sequence.upper().replace("\n", "").replace(" ", "")
    dna_chars = sum(1 for c in cleaned if c in "ATCGN")
    return len(cleaned) > 0 and dna_chars / len(cleaned) > 0.8


def translate_dna(dna: str) -> list[str]:
    dna = dna.upper().replace("\n", "").replace(" ", "")
    seq = Seq(dna)
    rev_comp = seq.reverse_complement()
    proteins = []

    for frame in range(3):
        for strand in (seq, rev_comp):
            subseq = strand[frame:]
            subseq = subseq[:len(subseq) - len(subseq) % 3]
            if len(subseq) < 3:
                continue
            protein = str(subseq.translate())
            for orf in protein.split("*"):
                if len(orf) >= MIN_ORF_LENGTH:
                    proteins.append(orf)

    return proteins
