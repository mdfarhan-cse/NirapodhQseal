"""Zone classification, key-bound permutation, embedding routes and methods.

Methods (all embed the same DNA-recoded ciphertext bits):
  LSB           LSB matching at key-ordered random positions over the image
  DCT           nearest-point QIM on 10 mid-frequency AC coefficients, all blocks
  UNIWARD-cost  UNIWARD cost, pixels filled in ascending-cost order by LSB matching
  NirapodQSeal  keyed zone map Zk routes each 8x8 block: D -> DCT-QIM,
                S -> UNIWARD-cost LSB matching, L -> keyed LSB matching
Every method returns (stego, position map Pi); extraction reads only Pi.
"""

import hashlib
import math

import cv2
import numpy as np
import pywt

from . import config as C

_MFI = np.array([p[0] for p in C.MID_FREQ])
_MFJ = np.array([p[1] for p in C.MID_FREQ])
NCOEF = len(C.MID_FREQ)
ROUTES = ("dct", "su", "lsb")
DEFAULT_ROUTE_MAP = {0: "dct", 1: "su", 2: "lsb"}


def keyed_rng(k, label):
    return np.random.default_rng(int.from_bytes(hashlib.sha256(k + b"|" + label.encode()).digest(), "big"))


# ---------------------------------------------------------------- zones
def complexity_map(gray):
    """psi(b) = Laplacian variance + std + mean squared db2 HH energy (8x8)."""
    H, W = gray.shape
    hb, wb = H // 8, W // 8
    gf = gray[:hb * 8, :wb * 8].astype(np.float32)
    lap = cv2.Laplacian(gf, cv2.CV_32F)
    lap_var = lap.reshape(hb, 8, wb, 8).var(axis=(1, 3))
    gry_std = gf.reshape(hb, 8, wb, 8).std(axis=(1, 3))
    _, (_, _, HH) = pywt.dwt2(gf, "db2")
    HH2 = cv2.resize(HH ** 2, (wb * 8, hb * 8), interpolation=cv2.INTER_LINEAR)
    hh = HH2.reshape(hb, 8, wb, 8).mean(axis=(1, 3))
    return (lap_var + gry_std + hh).astype(np.float64)


def base_zone_map(gray):
    psi = complexity_map(gray)
    t_lo, t_hi = np.percentile(psi, 33), np.percentile(psi, 66)
    z = np.zeros(psi.shape, np.uint8)
    z[psi >= t_lo] = 1
    z[psi >= t_hi] = 2
    return z, float(t_lo), float(t_hi)


def permute_zone_map(z0, k, gamma):
    """Algorithm 3: PRNG seeded by SHA-256(k); shuffle labels of a gamma-fraction."""
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(k).digest(), "big"))
    flat = z0.reshape(-1).copy()
    n = len(flat)
    n_sel = int(round(gamma * n))
    if n_sel >= 2:
        sel = rng.choice(n, size=n_sel, replace=False)
        lab = flat[sel].copy()
        rng.shuffle(lab)
        flat[sel] = lab
    return flat.reshape(z0.shape)


# ---------------------------------------------------------------- UNIWARD
def _uniward_filters():
    w = pywt.Wavelet("db8")
    lo, hi = np.array(w.dec_lo), np.array(w.dec_hi)
    return [np.outer(lo, hi), np.outer(hi, lo), np.outer(hi, hi)]


_UF = _uniward_filters()


def uniward_cost(gray, sigma=C.UNIWARD_SIGMA):
    X = gray.astype(np.float64)
    rho = np.zeros_like(X)
    for F in _UF:
        R = cv2.filter2D(X, -1, F, borderType=cv2.BORDER_REFLECT)
        rho += cv2.filter2D(1.0 / (sigma + np.abs(R)), -1, np.abs(F), borderType=cv2.BORDER_REFLECT)
    return rho


# ---------------------------------------------------------------- DCT-QIM
def _dct_matrix():
    k = np.arange(8).reshape(-1, 1)
    n = np.arange(8).reshape(1, -1)
    D = np.cos(np.pi * (2 * n + 1) * k / 16.0)
    D[0, :] *= math.sqrt(1 / 8)
    D[1:, :] *= math.sqrt(2 / 8)
    return D


_D8 = _dct_matrix()


def _block_rc(bids, wb):
    by, bx = bids // wb, bids % wb
    r = by[:, None] * 8 + np.arange(8)[None, :]
    c = bx[:, None] * 8 + np.arange(8)[None, :]
    return r, c


def _gather(img, bids, wb):
    r, c = _block_rc(bids, wb)
    return img[r[:, :, None], c[:, None, :], :].astype(np.float64)       # (n,8,8,C)


def _scatter(img, bids, wb, blocks):
    r, c = _block_rc(bids, wb)
    img[r[:, :, None], c[:, None, :], :] = blocks


def _dct2(X):
    return np.einsum("ij,njkc,lk->nilc", _D8, X, _D8, optimize=True)


def _idct2(F):
    return np.einsum("ji,njkc,kl->nilc", _D8, F, _D8, optimize=True)


def _q(u):
    return np.floor(u + 0.5).astype(np.int64)


def qim_set(c, bits, delta):
    """Nearest lattice point of step delta whose index parity equals the bit."""
    u = c / delta
    q = _q(u)
    mism = (q & 1) != bits
    step = np.where(u >= q, 1, -1)
    return np.where(mism, q + step, q) * delta


def qim_get(c, delta):
    return (_q(c / delta) & 1).astype(np.uint8)


def _to_u8(Y):
    return np.clip(np.rint(Y), 0, 255).astype(np.uint8)


def dct_route_embed(img, bits, bids, delta):
    """Embeds bits into blocks `bids` (already in keyed order), 10 coefficients
    x C channels per block. Saturation casualties are relocated to unused
    blocks of the same route. Returns Pi rows [block_id, channel, coef_idx]."""
    H, W, Ch = img.shape
    wb = W // 8
    n = len(bits)
    per = Ch * NCOEF
    nbk = -(-n // per)
    used, spares = bids[:nbk], bids[nbk:]
    j = np.arange(n)
    b, r = j // per, j % per
    ch, ci = r // NCOEF, r % NCOEF
    F = _dct2(_gather(img, used, wb))
    F[b, _MFI[ci], _MFJ[ci], ch] = qim_set(F[b, _MFI[ci], _MFJ[ci], ch], bits.astype(np.int64), delta)
    _scatter(img, used, wb, _to_u8(_idct2(F)))
    pos = np.stack([used[b], ch, ci], 1).astype(np.int32)

    fail = np.nonzero(dct_route_extract(img, pos, delta) != bits)[0]
    slots = [(int(s), c) for s in spares for c in range(Ch)]
    si = 0
    for p in fail:
        done = False
        while si < len(slots) and not done:
            sb, sc = slots[si]; si += 1
            blk = _gather(img, np.array([sb]), wb)
            Fb = _dct2(blk)
            Fb[0, _MFI[0], _MFJ[0], sc] = qim_set(np.array([Fb[0, _MFI[0], _MFJ[0], sc]]),
                                                  np.array([int(bits[p])]), delta)[0]
            _scatter(img, np.array([sb]), wb, _to_u8(_idct2(Fb)))
            cand = np.array([[sb, sc, 0]], np.int32)
            if dct_route_extract(img, cand, delta)[0] == bits[p]:
                pos[p] = cand[0]
                done = True
        if not done:
            raise RuntimeError("DCT route: no spare coefficient left for relocation")
    return pos


def dct_route_extract(img, pos, delta):
    if len(pos) == 0:
        return np.zeros(0, np.uint8)
    wb = img.shape[1] // 8
    ub, inv = np.unique(pos[:, 0], return_inverse=True)
    F = _dct2(_gather(img, ub.astype(np.int64), wb))
    ci = pos[:, 2].astype(np.int64)
    return qim_get(F[inv, _MFI[ci], _MFJ[ci], pos[:, 1]], delta)


# ---------------------------------------------------------------- spatial
def lsb_match(flat, idx, bits, rng):
    """y = x if LSB(x) == b else x +/- 1 (random sign, kept inside [0,255])."""
    v = flat[idx].astype(np.int16)
    mism = (v & 1) != bits
    s = rng.integers(0, 2, len(idx)).astype(np.int16) * 2 - 1
    s[v == 0] = 1
    s[v == 255] = -1
    v[mism] += s[mism]
    flat[idx] = v.astype(np.uint8)


def _block_pixels(bids, wb, W):
    if len(bids) == 0:
        return np.zeros(0, np.int64)
    r, c = _block_rc(bids, wb)
    return (r[:, :, None] * W + c[:, None, :]).reshape(-1)


def _samples(px, n, Ch):
    k = np.arange(n, dtype=np.int64)
    return px[: -(-n // Ch)][k // Ch] * Ch + k % Ch


# ---------------------------------------------------------------- capacity
def route_blocks(zk, route_map=DEFAULT_ROUTE_MAP):
    flat = zk.reshape(-1)
    return {r: np.nonzero(np.isin(flat, [z for z, rr in route_map.items() if rr == r]))[0].astype(np.int64)
            for r in ROUTES}


def capacity(method, shape, zk=None, route_map=DEFAULT_ROUTE_MAP, delta_headroom=C.DCT_HEADROOM):
    H, W, Ch = shape
    nb = (H // 8) * (W // 8)
    if method in ("LSB", "UNIWARD-cost"):
        return H * W * Ch
    if method == "DCT":
        return int(math.floor(delta_headroom * nb * Ch * NCOEF))
    rb = route_blocks(zk, route_map)
    return sum(route_caps(rb, Ch).values())


def route_caps(rb, Ch):
    return {"dct": int(math.floor(C.DCT_HEADROOM * len(rb["dct"]) * Ch * NCOEF)),
            "su": len(rb["su"]) * 64 * Ch, "lsb": len(rb["lsb"]) * 64 * Ch}


def split_bits(n, caps):
    """Proportional to capacity; overflow spills to routes with room (manuscript Sec. 4)."""
    tot = sum(caps.values())
    n_dct = min(caps["dct"], int(round(n * caps["dct"] / tot)))
    n_su = min(caps["su"], int(round(n * caps["su"] / tot)))
    n_lsb = n - n_dct - n_su
    if n_lsb > caps["lsb"]:
        spill = n_lsb - caps["lsb"]
        add = min(spill, caps["dct"] - n_dct)
        n_dct += add
        n_su += spill - add
    n_su = min(n_su, caps["su"])
    n_lsb = n - n_dct - n_su
    return {"dct": n_dct, "su": n_su, "lsb": n_lsb}


# ---------------------------------------------------------------- methods
def embed(method, cover, bits, k, cost=None, zk=None, route_map=DEFAULT_ROUTE_MAP, delta=C.DCT_DELTA):
    """Returns (stego, pi, info). `cost` is the UNIWARD cost of the cover's
    gray plane; `zk` the keyed zone map (NirapodQSeal only)."""
    H, W, Ch = cover.shape
    wb = W // 8
    n = len(bits)
    if n > capacity(method, cover.shape, zk, route_map):
        raise ValueError("payload exceeds capacity")
    stego = cover.copy()
    flat = stego.reshape(-1)
    sign = keyed_rng(k, "sign|" + method)
    order = keyed_rng(k, "order|" + method)
    info = {}

    if method == "LSB":
        idx = order.choice(H * W * Ch, size=n, replace=False).astype(np.int64)
        lsb_match(flat, idx, bits, sign)
        return stego, {"lsb": idx}, info
    if method == "UNIWARD-cost":
        px = np.argsort(cost.reshape(-1), kind="stable")
        idx = _samples(px, n, Ch)
        lsb_match(flat, idx, bits, sign)
        return stego, {"su": idx}, info
    if method == "DCT":
        bids = order.permutation((H // 8) * wb).astype(np.int64)
        pos = dct_route_embed(stego, bits, bids, delta)
        return stego, {"dct": pos}, info

    # NirapodQSeal
    rb = route_blocks(zk, route_map)
    caps = route_caps(rb, Ch)
    nsplit = split_bits(n, caps)
    info.update({f"bits_{r}": nsplit[r] for r in ROUTES})
    info.update({f"blocks_{r}": len(rb[r]) for r in ROUTES})
    off = 0
    pi = {}
    if nsplit["dct"]:
        bids = rb["dct"][order.permutation(len(rb["dct"]))]
        pi["dct"] = dct_route_embed(stego, bits[:nsplit["dct"]], bids, delta)
        off = nsplit["dct"]
    if nsplit["su"]:
        px = _block_pixels(rb["su"], wb, W)
        px = px[np.argsort(cost.reshape(-1)[px], kind="stable")]
        idx = _samples(px, nsplit["su"], Ch)
        lsb_match(flat, idx, bits[off:off + nsplit["su"]], sign)
        pi["su"] = idx
        off += nsplit["su"]
    if nsplit["lsb"]:
        px = _block_pixels(rb["lsb"], wb, W)
        px = px[order.permutation(len(px))]
        idx = _samples(px, nsplit["lsb"], Ch)
        lsb_match(flat, idx, bits[off:off + nsplit["lsb"]], sign)
        pi["lsb"] = idx
    return stego, pi, info


def extract(stego, pi, delta=C.DCT_DELTA):
    """Reads Pi in embedding order: DCT, then UNIWARD-cost, then LSB route."""
    flat = stego.reshape(-1)
    out = []
    if "dct" in pi:
        out.append(dct_route_extract(stego, pi["dct"], delta))
    if "su" in pi:
        out.append((flat[pi["su"]] & 1).astype(np.uint8))
    if "lsb" in pi:
        out.append((flat[pi["lsb"]] & 1).astype(np.uint8))
    return np.concatenate(out) if out else np.zeros(0, np.uint8)


# ---------------------------------------------------------------- JPEG-robust
_JPEG_LUMA = np.array([[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
                       [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
                       [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
                       [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]], np.float64)
JPEG_POS = ((0, 1), (1, 0), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2), (0, 3), (3, 0))
_JI = np.array([p[0] for p in JPEG_POS])
_JJ = np.array([p[1] for p in JPEG_POS])


def jpeg_qtable(quality):
    s = 5000 / quality if quality < 50 else 200 - 2 * quality
    return np.clip(np.floor((_JPEG_LUMA * s + 50) / 100), 1, 255)


def _bgr_to_ycc(img):
    B, G, R = (img[..., i].astype(np.float64) for i in range(3))
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = 128 - 0.168736 * R - 0.331264 * G + 0.5 * B
    Cr = 128 + 0.5 * R - 0.418688 * G - 0.081312 * B
    return Y, Cb, Cr


def _ycc_to_bgr(Y, Cb, Cr):
    R = Y + 1.402 * (Cr - 128)
    G = Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128)
    B = Y + 1.772 * (Cb - 128)
    return np.clip(np.rint(np.stack([B, G, R], -1)), 0, 255).astype(np.uint8)


def jpeg_steps(quality=C.JPEG_DESIGN_QF, qf=C.JPEG_STEP_FACTOR):
    Q = jpeg_qtable(quality)
    return np.maximum(2.0, qf * Q[_JI, _JJ])


def embed_jpeg_robust(cover, bits, k, zk, quality=C.JPEG_DESIGN_QF, qf=C.JPEG_STEP_FACTOR):
    """NirapodQSeal-JPEG: nearest-point QIM on 9 low-frequency luma DCT
    coefficients of the D and S blocks of Zk, step = qf x JPEG(quality) luma
    quantiser. Only luma (JPEG's YCbCr) is modified; the stego stays colour."""
    H, W, _ = cover.shape
    wb = W // 8
    Y, Cb, Cr = _bgr_to_ycc(cover)
    steps = jpeg_steps(quality, qf)
    bids = np.nonzero(zk.reshape(-1) <= 1)[0].astype(np.int64)
    bids = bids[keyed_rng(k, "robust|jpeg").permutation(len(bids))]
    n = len(bits)
    nbk = -(-n // len(JPEG_POS))
    if nbk > len(bids):
        raise ValueError("payload exceeds JPEG-robust capacity")
    used = bids[:nbk]
    j = np.arange(n)
    b, ci = j // len(JPEG_POS), j % len(JPEG_POS)
    F = _dct2(_gather(Y[..., None], used, wb))
    F[b, _JI[ci], _JJ[ci], 0] = qim_set(F[b, _JI[ci], _JJ[ci], 0], bits.astype(np.int64), steps[ci])
    Yw = Y[..., None].copy()
    _scatter(Yw, used, wb, _idct2(F))
    stego = _ycc_to_bgr(Yw[..., 0], Cb, Cr)
    return stego, {"jpeg": np.stack([used[b], ci], 1).astype(np.int32)}


def extract_jpeg_robust(img, pi, quality=C.JPEG_DESIGN_QF, qf=C.JPEG_STEP_FACTOR):
    wb = img.shape[1] // 8
    Y = _bgr_to_ycc(img)[0][..., None]
    pos = pi["jpeg"]
    ub, inv = np.unique(pos[:, 0], return_inverse=True)
    F = _dct2(_gather(Y, ub.astype(np.int64), wb))
    ci = pos[:, 1].astype(np.int64)
    return qim_get(F[inv, _JI[ci], _JJ[ci], 0], jpeg_steps(quality, qf)[ci])


def embed_robust(method, cover, bits, k, zk=None, delta=C.ROBUST_DELTA):
    """Robust-mode embedders for the attack study: DCT-QIM with a wide step,
    over all blocks (DCT) or over the D and S blocks of Zk (NirapodQSeal-R)."""
    H, W, Ch = cover.shape
    order = keyed_rng(k, "robust|" + method)
    stego = cover.copy()
    if method == "NirapodQSeal-R":
        bids = np.nonzero(zk.reshape(-1) <= 1)[0].astype(np.int64)
    else:
        bids = np.arange((H // 8) * (W // 8), dtype=np.int64)
    bids = bids[order.permutation(len(bids))]
    pos = dct_route_embed(stego, bits, bids, delta)
    return stego, {"dct": pos}
