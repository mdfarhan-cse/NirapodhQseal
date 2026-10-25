"""2-bit DNA mapping: 00->A, 01->C, 10->G, 11->T."""

import numpy as np

_BASES = np.frombuffer(b"ACGT", np.uint8)
_LUT = np.full(256, 255, np.uint8)
for _i, _b in enumerate("ACGT"):
    _LUT[ord(_b)] = _i


def bits_to_dna(bits):
    bits = np.asarray(bits, np.uint8)
    if len(bits) % 2:
        bits = np.concatenate([bits, [0]])
    idx = (bits[0::2] << 1) | bits[1::2]
    return _BASES[idx].tobytes().decode("ascii")


def dna_to_bits(seq):
    v = _LUT[np.frombuffer(seq.encode("ascii"), np.uint8)]
    if (v == 255).any():
        raise ValueError("non-ACGT symbol")
    out = np.empty(2 * len(v), np.uint8)
    out[0::2] = v >> 1
    out[1::2] = v & 1
    return out


def gc_content(seq):
    return (seq.count("G") + seq.count("C")) / len(seq) if seq else 0.0
