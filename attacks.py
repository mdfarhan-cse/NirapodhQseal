"""Channel attacks applied to the stego image. Random attacks draw from a
logged stream, so every attacked image can be regenerated."""

import cv2
import numpy as np

from . import config as C


def jpeg(img, qf):
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(qf)])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def gaussian(img, sigma, rng):
    return np.clip(np.rint(img.astype(np.float64) + rng.normal(0.0, sigma, img.shape)), 0, 255).astype(np.uint8)


def median(img, k=C.MEDIAN_K):
    return cv2.medianBlur(img, k)


def salt_pepper(img, p, rng):
    out = img.copy()
    m = rng.random(img.shape[:2])
    out[m < p / 2] = 0
    out[(m >= p / 2) & (m < p)] = 255
    return out


def attack_list():
    """(family, level) pairs used for the robustness curves."""
    return ([("none", 0)] + [("jpeg", q) for q in C.JPEG_QF] + [("gauss", s) for s in C.GAUSS_SIGMA]
            + [("median", C.MEDIAN_K), ("saltpepper", C.SALT_PEPPER_P)])


def apply(img, family, level, rng):
    if family == "none":
        return img.copy()
    if family == "jpeg":
        return jpeg(img, level)
    if family == "gauss":
        return gaussian(img, level, rng)
    if family == "median":
        return median(img, int(level))
    if family == "saltpepper":
        return salt_pepper(img, level, rng)
    raise ValueError(family)
