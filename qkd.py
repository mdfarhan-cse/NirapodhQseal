"""Simulated E91 key establishment (Algorithm 1 of the manuscript).

A session runs: entangled-pair measurement on both sides -> sifting ->
CHSH test with a finite-sample lower bound -> QBER estimation on a
disclosed sample -> Cascade reconciliation -> verification hash ->
finite-key Toeplitz privacy amplification -> HKDF key derivation.
Alice and Bob each derive their keys from their own strings; the session
records whether they agree.

Randomness is drawn from independent seeded streams (not from the cover),
which keeps the simulation reproducible; a hardware source would replace
the private stream with physical measurement outcomes.
"""

import hashlib
import hmac
import math

import numpy as np
from scipy.signal import fftconvolve

from . import config as C


def h2(x):
    x = min(max(float(x), 1e-15), 1 - 1e-15)
    return -x * math.log2(x) - (1 - x) * math.log2(1 - x)


# ---------------------------------------------------------------- physics
def _correlated(E, rng, n):
    a = rng.integers(0, 2, n) * 2 - 1
    flip = rng.random(n) > (1 + E) / 2
    return a, np.where(flip, -a, a)


def measure(n_pairs, noise, eve_frac, rng):
    """Phi+ pairs measured at random angles. Correlation E(a,b) = V cos 2(a-b).
    An intercept-resend Eve measures a fraction of pairs at a random angle."""
    V = 1.0 - noise
    a_key = rng.integers(0, 2, n_pairs).astype(bool)
    b_key = rng.integers(0, 2, n_pairs).astype(bool)
    a_ang = np.where(a_key, rng.choice(C.E91_A_KEY, n_pairs), rng.choice(C.E91_A_TEST, n_pairs))
    b_ang = np.where(b_key, rng.choice(C.E91_B_KEY, n_pairs), rng.choice(C.E91_B_TEST, n_pairs))
    eve = rng.random(n_pairs) < eve_frac
    a_out = np.empty(n_pairs, np.int8)
    b_out = np.empty(n_pairs, np.int8)
    hp = ~eve
    if hp.any():
        E = V * np.cos(2 * np.radians(a_ang[hp] - b_ang[hp]))
        a_out[hp], b_out[hp] = _correlated(E, rng, int(hp.sum()))
    if eve.any():
        m = int(eve.sum())
        theta = rng.choice([0.0, 22.5, 45.0, 67.5], m)
        e = rng.integers(0, 2, m) * 2 - 1
        Ea = V * np.cos(2 * np.radians(a_ang[eve] - theta))
        Eb = V * np.cos(2 * np.radians(theta - b_ang[eve]))
        a_out[eve] = np.where(rng.random(m) > (1 + Ea) / 2, -e, e)
        b_out[eve] = np.where(rng.random(m) > (1 + Eb) / 2, -e, e)
    return a_ang, b_ang, a_key, b_key, a_out, b_out


def chsh(a_ang, b_ang, a_key, b_key, a_out, b_out):
    """S = E(0,22.5) - E(0,67.5) + E(45,22.5) + E(45,67.5) with a Hoeffding
    lower bound on |S| (each correlator lies in [-1, 1])."""
    test = (~a_key) & (~b_key)
    pairs = [(0.0, 22.5, 1), (0.0, 67.5, -1), (45.0, 22.5, 1), (45.0, 67.5, 1)]
    S, dev, counts = 0.0, 0.0, []
    for a, b, sgn in pairs:
        m = test & (np.abs(a_ang - a) < 1e-9) & (np.abs(b_ang - b) < 1e-9)
        n = int(m.sum())
        counts.append(n)
        if n == 0:
            return float("nan"), float("nan"), counts
        S += sgn * float(np.mean(a_out[m].astype(np.int32) * b_out[m]))
        dev += math.sqrt(2.0 * math.log(4.0 / C.EPS_PE) / n)
    return S, abs(S) - dev, counts


# ---------------------------------------------------------- reconciliation
def cascade(alice, bob, q_est, rng, passes=C.CASCADE_PASSES):
    """Cascade with back-tracking. Every parity Alice reveals is counted as
    leaked. Returns Bob's corrected string and the number of leaked bits."""
    bob = bob.copy()
    n = len(alice)
    k1 = max(4, int(math.ceil(0.73 / max(q_est, 1e-3))))
    perm, pos, ks, apar, bpar = [], [], [], [], []
    leak = 0
    for p in range(passes):
        pm = np.arange(n) if p == 0 else rng.permutation(n)
        k = min(k1 * (2 ** p), n)
        nb = -(-n // k)
        idx = np.full(nb * k, -1, np.int64)
        idx[:n] = pm
        blocks = idx.reshape(nb, k)
        valid = blocks >= 0
        pa = (np.where(valid, alice[np.maximum(blocks, 0)], 0).sum(1) & 1).astype(np.int8)
        pb = (np.where(valid, bob[np.maximum(blocks, 0)], 0).sum(1) & 1).astype(np.int8)
        ps = np.empty(n, np.int64)
        ps[pm] = np.arange(n)
        perm.append(pm); pos.append(ps); ks.append(k); apar.append(pa); bpar.append(pb)
        leak += nb
        work = [(p, int(b)) for b in np.nonzero(pa != pb)[0]]
        while work:
            q, b = work.pop()
            if apar[q][b] == bpar[q][b]:
                continue
            members = perm[q][b * ks[q]:(b + 1) * ks[q]]
            lo, hi = 0, len(members)
            while hi - lo > 1:
                mid = (lo + hi) // 2
                leak += 1
                if (int(alice[members[lo:mid]].sum()) & 1) != (int(bob[members[lo:mid]].sum()) & 1):
                    hi = mid
                else:
                    lo = mid
            i = members[lo]
            bob[i] ^= 1
            for r in range(p + 1):
                bb = int(pos[r][i] // ks[r])
                bpar[r][bb] ^= 1
                if apar[r][bb] != bpar[r][bb]:
                    work.append((r, bb))
    return bob, leak


def toeplitz_hash(seed_bits, x, m):
    """y = T x mod 2 for the m x n Toeplitz matrix defined by seed_bits
    (length m + n - 1), computed by FFT convolution."""
    n = len(x)
    full = fftconvolve(seed_bits.astype(np.float64), x.astype(np.float64))
    return (np.rint(full[n - 1:n - 1 + m]).astype(np.int64) & 1).astype(np.uint8)


def hkdf(ikm, salt, info, length=32):
    prk = hmac.new(salt, ikm, hashlib.sha256).digest()
    out, t, i = b"", b"", 1
    while len(out) < length:
        t = hmac.new(prk, t + info + bytes([i]), hashlib.sha256).digest()
        out += t
        i += 1
    return out[:length]


# ------------------------------------------------------------------ session
def session(private_rng, public_rng, n_pairs=C.E91_PAIRS, noise=C.E91_NOISE, eve_frac=0.0):
    """One full key session. Returns a dict of statistics and, if it does not
    abort, Alice's and Bob's derived keys."""
    a_ang, b_ang, a_key, b_key, a_out, b_out = measure(n_pairs, noise, eve_frac, private_rng)
    S, S_lower, chsh_counts = chsh(a_ang, b_ang, a_key, b_key, a_out, b_out)
    keymask = a_key & b_key & (np.abs(a_ang - b_ang) < 1e-9)
    alice = ((a_out[keymask] + 1) // 2).astype(np.uint8)
    bob = ((b_out[keymask] + 1) // 2).astype(np.uint8)
    n_sift = len(alice)

    order = public_rng.permutation(n_sift)
    n_pe = max(1, int(round(C.PE_FRACTION * n_sift)))
    pe, keep = order[:n_pe], np.sort(order[n_pe:])
    q_hat = float(np.mean(alice[pe] != bob[pe]))
    q_up = q_hat + math.sqrt(math.log(1.0 / C.EPS_PE) / (2.0 * n_pe))
    st = dict(n_pairs=n_pairs, noise=noise, eve_frac=eve_frac, chsh_S=S, chsh_abs_S=abs(S),
              chsh_S_lower=S_lower, chsh_test_pairs=int(sum(chsh_counts)), n_sifted=n_sift,
              n_pe=n_pe, qber_hat=q_hat, qber_upper=q_up, true_qber=float(np.mean(alice != bob)),
              asymptotic_fraction=max(0.0, 1 - 2 * h2(q_hat)))
    st["chsh_verified"] = bool(abs(S) > 2.0) if S == S else False
    if not st["chsh_verified"] or q_hat > C.QBER_MAX:
        st.update(abort=True, abort_reason="chsh" if not st["chsh_verified"] else "qber")
        return st

    xa, xb = alice[keep], bob[keep]
    xb_rec, leak_ec = cascade(xa, xb, q_hat, public_rng)
    vsalt = public_rng.bytes(16)
    vhash = lambda v: hmac.new(vsalt, np.packbits(v).tobytes(), hashlib.sha256).digest()[: C.EC_VERIFY_BITS // 8]
    ec_ok = vhash(xa) == vhash(xb_rec)
    leak = leak_ec + C.EC_VERIFY_BITS
    n = len(xa)
    m = int(math.floor(n * (1 - h2(q_up)) - leak - 2 * math.log2(1.0 / C.EPS_PA)))
    st.update(n_key_string=n, leak_ec=leak_ec, leak_total=leak, ec_verified=bool(ec_ok),
              residual_errors=int(np.sum(xa != xb_rec)), secret_len=max(m, 0),
              secret_fraction=max(m, 0) / n_sift if n_sift else 0.0)
    if not ec_ok:
        st.update(abort=True, abort_reason="ec_verify")
        return st
    if m < C.KEY_BITS:
        st.update(abort=True, abort_reason="insufficient_secret_key")
        return st

    tseed = public_rng.integers(0, 2, n + m - 1).astype(np.uint8)
    salt = public_rng.bytes(32)
    s_a = np.packbits(toeplitz_hash(tseed, xa, m)).tobytes()
    s_b = np.packbits(toeplitz_hash(tseed, xb_rec, m)).tobytes()
    keys_a = {lab: hkdf(s_a, salt, f"NirapodQSeal {lab}".encode()) for lab in ("k", "mac", "pi")}
    keys_b = {lab: hkdf(s_b, salt, f"NirapodQSeal {lab}".encode()) for lab in ("k", "mac", "pi")}
    st.update(abort=False, abort_reason="", keys_agree=keys_a == keys_b,
              key_fingerprint=hashlib.sha256(keys_a["k"]).hexdigest()[:16],
              keys_alice=keys_a, keys_bob=keys_b, s_alice=s_a, hkdf_salt=salt)
    return st


def derive_keys(s, salt):
    return {lab: hkdf(s, salt, f"NirapodQSeal {lab}".encode()) for lab in ("k", "mac", "pi")}
