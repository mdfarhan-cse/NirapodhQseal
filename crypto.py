"""Keystream encryption (Eq. 16), integrity tag, and the sealed position map."""

import hashlib
import hmac
import struct

import numpy as np


def keystream(k, tag, nbits):
    """kappa = SHA-256(k || tag || t) for t = 0, 1, ... (Eq. 16)."""
    nblocks = -(-nbits // 256)
    out = b"".join(hashlib.sha256(k + tag + t.to_bytes(8, "big")).digest() for t in range(nblocks))
    return np.unpackbits(np.frombuffer(out, np.uint8))[:nbits]


def encrypt(bits, k, tag):
    return (bits ^ keystream(k, tag, len(bits))).astype(np.uint8)


def mac(k_mac, tag, c_bits):
    msg = tag + struct.pack(">Q", len(c_bits)) + np.packbits(c_bits).tobytes()
    return hmac.new(k_mac, msg, hashlib.sha256).digest()


def mac_ok(k_mac, tag, c_bits, t):
    return hmac.compare_digest(mac(k_mac, tag, c_bits), t)


# ------------------------------------------------------------ position map
def _serialize(pi):
    parts = [struct.pack(">I", len(pi))]
    for name in sorted(pi):
        a = np.ascontiguousarray(pi[name])
        nm = name.encode()
        dt = a.dtype.str.encode()
        parts.append(struct.pack(">H", len(nm)) + nm + struct.pack(">H", len(dt)) + dt)
        parts.append(struct.pack(">B", a.ndim) + b"".join(struct.pack(">Q", s) for s in a.shape))
        raw = a.tobytes()
        parts.append(struct.pack(">Q", len(raw)) + raw)
    return b"".join(parts)


def _deserialize(buf):
    off = 0
    (n,) = struct.unpack_from(">I", buf, off); off += 4
    pi = {}
    for _ in range(n):
        (ln,) = struct.unpack_from(">H", buf, off); off += 2
        name = buf[off:off + ln].decode(); off += ln
        (ld,) = struct.unpack_from(">H", buf, off); off += 2
        dt = np.dtype(buf[off:off + ld].decode()); off += ld
        (nd,) = struct.unpack_from(">B", buf, off); off += 1
        shape = struct.unpack_from(">" + "Q" * nd, buf, off); off += 8 * nd
        (lr,) = struct.unpack_from(">Q", buf, off); off += 8
        pi[name] = np.frombuffer(buf[off:off + lr], dt).reshape(shape).copy(); off += lr
    return pi


def _pi_stream(k_pi, tag, nbytes):
    """SHAKE-256 (FIPS 202) extendable output as the Pi keystream."""
    return np.frombuffer(hashlib.shake_256(k_pi + b"PI|" + tag).digest(nbytes), np.uint8)


def seal_pi(pi, k_pi, k_mac, tag):
    """Encrypt-then-MAC the position map so it can travel as side information
    without revealing where the payload sits."""
    raw = np.frombuffer(_serialize(pi), np.uint8)
    ct = (raw ^ _pi_stream(k_pi, tag, len(raw))).tobytes()
    t = hmac.new(k_mac, b"PI|" + tag + ct, hashlib.sha256).digest()
    return ct + t


def open_pi(sealed, k_pi, k_mac, tag):
    """Returns the position map, or None if the integrity check fails."""
    ct, t = sealed[:-32], sealed[-32:]
    if not hmac.compare_digest(hmac.new(k_mac, b"PI|" + tag + ct, hashlib.sha256).digest(), t):
        return None
    raw = (np.frombuffer(ct, np.uint8) ^ _pi_stream(k_pi, tag, len(ct))).tobytes()
    return _deserialize(raw)
