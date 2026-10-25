"""Run manifest, code fingerprint, environment record and atomic CSV writes."""

import datetime
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import pandas as pd

from . import config as C

PKG_DIR = Path(__file__).resolve().parent


def code_hash():
    h = hashlib.sha256()
    for p in sorted(PKG_DIR.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def environment():
    import cv2, numpy, scipy, skimage, pywt, reedsolo
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "processor": platform.processor(), "cpu_count": os.cpu_count(),
            "numpy": numpy.__version__, "scipy": scipy.__version__, "opencv": cv2.__version__,
            "scikit_image": skimage.__version__, "pywavelets": pywt.__version__,
            "pandas": pd.__version__, "reedsolo": getattr(reedsolo, "__version__", "unknown")}


def init_run(out_dir, images, force=False):
    """Creates the manifest on first use; on resume, refuses to mix results
    from a different code version or image set unless force=True."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mf = out_dir / "manifest.json"
    ch = code_hash()
    img_ids = [(r["image"], r["sha256"]) for r in images]
    if mf.exists():
        m = json.loads(mf.read_text())
        if not force and (m["code_sha256"] != ch or [tuple(x) for x in m["image_ids"]] != img_ids):
            raise RuntimeError("results folder belongs to a different code version or image set; "
                               "use a new output folder (or force=True to override)")
        m.setdefault("resumed", []).append(datetime.datetime.now().isoformat(timespec="seconds"))
    else:
        m = {"run_id": "run_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
             "created": datetime.datetime.now().isoformat(timespec="seconds"),
             "code_sha256": ch, "package_version": __import__("nirapodqseal_v2").__version__,
             "config": C.as_dict(), "environment": environment(), "image_ids": img_ids,
             "images": images}
    mf.write_text(json.dumps(m, indent=2))
    pd.DataFrame(images).to_csv(out_dir / "image_manifest.csv", index=False)
    return m


def write_csv(rows, path):
    """Atomic write: a file either exists complete or not at all."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, path)
