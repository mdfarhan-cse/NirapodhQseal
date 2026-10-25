"""Dataset discovery, the frozen image manifest, and cover loading."""

import hashlib
import zlib
from pathlib import Path

import cv2
import numpy as np

from . import config as C


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def discover(data_root):
    """Colour covers of native square size in C.SIZES. A 3-channel file whose
    channels are identical is treated as grayscale and excluded."""
    root = Path(data_root)
    files = sorted({p for e in ("*.tif", "*.tiff", "*.TIF", "*.TIFF") for p in root.rglob(e)})
    rows = []
    for p in files:
        im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if im is None or im.ndim != 3 or im.shape[2] != 3:
            continue
        h, w = im.shape[:2]
        if h != w or h not in C.SIZES:
            continue
        if (im[..., 0] == im[..., 1]).all() and (im[..., 1] == im[..., 2]).all():
            continue
        rel = p.relative_to(root).parts
        rows.append(dict(image=p.stem, file=p.name, volume=rel[0] if len(rel) > 1 else "",
                         size=int(h), path=str(p), sha256=file_sha256(p)))
    rows.sort(key=lambda r: (r["size"], r["image"]))
    return rows


def image_key(name):
    """Stable integer id of an image, independent of discovery order."""
    return zlib.crc32(name.encode())


def load_cover(path):
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    return np.ascontiguousarray(im)


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def rng_for(*keys):
    """Independent, reproducible random stream for a fixed spawn key."""
    return np.random.default_rng(np.random.SeedSequence(C.MASTER_SEED, spawn_key=tuple(int(k) for k in keys)))
