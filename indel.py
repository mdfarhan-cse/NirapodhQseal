"""Indel-aware DNA channel (markers + CRC-8 + Reed-Solomon erasures + block
repetition) and the two comparison codes (RS-only, repetition-only)."""

import numpy as np
from reedsolo import RSCodec

from . import config as C

_B = "ACGT"
_D = {c: i for i, c in enumerate(_B)}
MARKER = "TTAATTAACCGGCCGG"
_ML = len(MARKER)


def bytes_to_dna(bs):
    return "".join(_B[(b >> s) & 3] for b in bs for s in (6, 4, 2, 0))


def dna_to_bytes(seq):
    out = bytearray()
    for i in range(0, len(seq) // 4 * 4, 4):
        v = 0
        for j in range(4):
            v = (v << 2) | _D.get(seq[i + j], 0)
        out.append(v)
    return bytes(out)


def indel_channel(seq, p, rng):
    """Each base is deleted with prob. p; after each kept base a random base
    is inserted with prob. p."""
    out = []
    for ch in seq:
        if rng.random() < p:
            continue
        out.append(ch)
        if rng.random() < p:
            out.append(_B[int(rng.integers(0, 4))])
    return "".join(out)


def _crc8(b):
    c = 0
    for x in b:
        c ^= x
        for _ in range(8):
            c = ((c << 1) ^ 0x07) & 0xFF if c & 0x80 else (c << 1) & 0xFF
    return c


def enc_indel(msg, blk=C.INDEL_BLOCK, nsym=C.INDEL_RS_NSYM, rep=C.INDEL_REPEAT):
    cw = bytes(RSCodec(nsym).encode(msg))
    frames = []
    for bi in range(0, len(cw), blk):
        ch = cw[bi:bi + blk]
        i = bi // blk
        fr = bytes([i]) + ch + bytes([_crc8(bytes([i]) + ch)])
        frames.append(MARKER + bytes_to_dna(fr))
    return "".join(frames) * rep, dict(cwlen=len(cw), nsym=nsym, blk=blk)


def _markers(seq, tol=3):
    pos, i, n = [], 0, len(seq)
    while i <= n - _ML:
        if sum(1 for a, b in zip(seq[i:i + _ML], MARKER) if a == b) >= _ML - tol:
            pos.append(i)
            i += _ML
        else:
            i += 1
    return pos


def dec_indel(seq, meta):
    cl, ns, blk = meta["cwlen"], meta["nsym"], meta["blk"]
    buf, erased = bytearray(cl), set(range(cl))
    pos = _markers(seq)
    fl = (blk + 2) * 4
    for k, ps in enumerate(pos):
        st = ps + _ML
        en = pos[k + 1] if k + 1 < len(pos) else len(seq)
        body = seq[st:en]
        if len(body) != fl:
            continue
        fb = dna_to_bytes(body)
        i, ch, rc = fb[0], fb[1:1 + blk], fb[1 + blk]
        if _crc8(bytes([i]) + ch) != rc:
            continue
        for j, by in enumerate(ch):
            if i * blk + j < cl and i * blk + j in erased:
                buf[i * blk + j] = by
                erased.discard(i * blk + j)
    try:
        return bytes(RSCodec(ns).decode(bytes(buf), erase_pos=sorted(erased))[0])
    except Exception:
        return None


def enc_rs(msg, nsym=C.INDEL_RS_NSYM):
    return bytes_to_dna(bytes(RSCodec(nsym).encode(msg))), nsym


def dec_rs(seq, nsym):
    try:
        return bytes(RSCodec(nsym).decode(dna_to_bytes(seq))[0])
    except Exception:
        return None


def enc_rep(msg, r=C.INDEL_REP_ONLY):
    bits = np.unpackbits(np.frombuffer(msg, np.uint8))
    return bytes_to_dna(np.packbits(np.tile(bits, r)).tobytes()), (len(bits), r)


def dec_rep(seq, meta):
    nb, r = meta
    bits = np.unpackbits(np.frombuffer(dna_to_bytes(seq), np.uint8))
    if len(bits) < r * nb:
        bits = np.pad(bits, (0, r * nb - len(bits)))
    return np.packbits((bits[: r * nb].reshape(r, nb).sum(0) > r / 2).astype(np.uint8)).tobytes()
