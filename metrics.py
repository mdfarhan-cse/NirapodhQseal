"""Fidelity and targeted-steganalysis metrics. The RS, SPA and WS estimators
work on a single plane; for colour images they are applied to each channel
and averaged, so no detector sees a blended (gray) version of the changes."""

import math

import cv2
import numpy as np
from skimage.metrics import structural_similarity


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10.0 * math.log10(255.0 ** 2 / mse), mse


def ssim(a, b):
    """Mean SSIM over channels with scikit-image's default definition
    (7x7 uniform window, sample covariance, K1=0.01, K2=0.03, reflect
    padding, border of 3 excluded), computed with OpenCV box filters.
    Equal to skimage.metrics.structural_similarity to float precision."""
    win, pad = 7, 3
    n = win * win
    cov_norm = n / (n - 1)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    vals = []
    for c in range(a.shape[2]):
        X = a[:, :, c].astype(np.float64)
        Y = b[:, :, c].astype(np.float64)
        f = lambda Z: cv2.boxFilter(Z, -1, (win, win), normalize=True, borderType=cv2.BORDER_REFLECT)
        ux, uy = f(X), f(Y)
        vx = cov_norm * (f(X * X) - ux * ux)
        vy = cov_norm * (f(Y * Y) - uy * uy)
        vxy = cov_norm * (f(X * Y) - ux * uy)
        S = ((2 * ux * uy + C1) * (2 * vxy + C2)) / ((ux ** 2 + uy ** 2 + C1) * (vx + vy + C2))
        vals.append(S[pad:-pad, pad:-pad].mean())
    return float(np.mean(vals))


def ssim_reference(a, b):
    return float(structural_similarity(a, b, channel_axis=2, data_range=255))


def _flip_f1(x):
    return x ^ 1


def _flip_fm1(x):
    return ((x + 1) ^ 1) - 1


def _rs_counts(g, mask):
    H, W = g.shape
    W4 = (W // 4) * 4
    grp = g[:, :W4].reshape(H, W4 // 4, 4)
    f = lambda x: np.abs(np.diff(x, axis=2)).sum(axis=2)
    f0 = f(grp)
    m = mask.reshape(1, 1, 4)
    ff = f(np.where(m == 1, _flip_f1(grp), grp))
    fn = f(np.where(m == 1, _flip_fm1(grp), grp))
    return int((ff > f0).sum()), int((ff < f0).sum()), int((fn > f0).sum()), int((fn < f0).sum())


def rs(plane):
    """RS analysis (Fridrich, Goljan, Du): estimated LSB embedding rate."""
    g = plane.astype(np.int32)
    mask = np.array([0, 1, 1, 0])
    R0, S0, Rn0, Sn0 = _rs_counts(g, mask)
    R1, S1, Rn1, Sn1 = _rs_counts(_flip_f1(g), mask)
    d0, d1, d0n, d1n = R0 - S0, R1 - S1, Rn0 - Sn0, Rn1 - Sn1
    a, b, c = 2 * (d1 + d0), d0n - d1n - d1 - 3 * d0, d0 - d0n
    if abs(a) < 1e-9:
        x = (-c / b) if abs(b) > 1e-9 else 0.0
    else:
        disc = max(b * b - 4 * a * c, 0.0)
        x1, x2 = (-b + math.sqrt(disc)) / (2 * a), (-b - math.sqrt(disc)) / (2 * a)
        x = x1 if abs(x1) <= abs(x2) else x2
    p = x / (x - 0.5) if abs(x - 0.5) > 1e-9 else 0.0
    return float(np.clip(abs(p), 0.0, 1.0))


def spa(plane):
    """Sample pairs analysis (Dumitrescu, Wu, Wang): estimated LSB rate."""
    g = plane.astype(np.int32).reshape(-1)
    g = g[: len(g) // 2 * 2]
    u, v = g[0::2], g[1::2]
    ue = u % 2 == 0
    X = np.sum((ue & (u < v)) | (~ue & (u > v)))
    Y = np.sum((ue & (u > v)) | (~ue & (u < v)))
    Cw = np.sum((u >> 1) == (v >> 1))
    n = len(u)
    a, b, c = 0.5 * Cw, 2 * X - n, Y - X
    if abs(a) < 1e-9:
        p = 0.0 if abs(b) < 1e-9 else -c / b
    else:
        disc = max(b * b - 4 * a * c, 0.0)
        p = min(abs((-b + math.sqrt(disc)) / (2 * a)), abs((-b - math.sqrt(disc)) / (2 * a)))
    return float(np.clip(p, 0.0, 1.0))


def ws(plane):
    """Weighted-stego estimator (uniform weights): estimated change rate x2."""
    s = plane.astype(np.float64)
    k = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.float64) / 4.0
    mu = cv2.filter2D(s, -1, k, borderType=cv2.BORDER_REFLECT)
    flip = s + 1 - 2 * np.mod(s, 2)
    return float(np.clip(abs(np.mean((s - flip) * (s - mu))) * 2.0, 0.0, 1.0))


def detectors(img):
    """Per-channel RS/SPA/WS averaged over channels."""
    out = {"RS": [], "SPA": [], "WS": []}
    for c in range(img.shape[2]):
        p = img[:, :, c]
        out["RS"].append(rs(p)); out["SPA"].append(spa(p)); out["WS"].append(ws(p))
    return {k: float(np.mean(v)) for k, v in out.items()}


def change_rate(cover, stego):
    return float(np.mean(cover != stego))
