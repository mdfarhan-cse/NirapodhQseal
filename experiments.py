"""The study's experiments. Per-image experiments (E1-E4, E6) return rows for
one cover; E5 runs once. Every row carries the image id and hash, the key
session's spawn keys and the parameters, so any number can be traced back
and regenerated (see reproduce_variant.py).

E1 main grid          fidelity, steganalysis, recovery, runtime (all methods, rates, sessions)
E2 robustness         fragility at base settings + matched-message robust mode under attacks
E3 ablation           gamma sweep (with adversary reconstruction) and route removal
E4 security           wrong-key test, tamper detection, NIST SP 800-22 on keystream/ciphertext
E5 QKD                intercept-resend sweep and finite-key secret length vs pairs
E6 indel channel      indel-aware DNA vs RS-only vs repetition, payload carried in the image
"""

import hashlib
from time import perf_counter

import numpy as np

from . import attacks as A
from . import config as C
from . import crypto, data, dna, indel, metrics, nist, qkd
from . import embed as E

QKD_COLS = ("chsh_S", "chsh_abs_S", "chsh_S_lower", "chsh_verified", "chsh_test_pairs", "n_sifted",
            "n_pe", "qber_hat", "qber_upper", "true_qber", "leak_ec", "leak_total", "residual_errors",
            "secret_len", "secret_fraction", "asymptotic_fraction", "abort", "abort_reason",
            "keys_agree", "key_fingerprint")


# ---------------------------------------------------------------- shared
class Cover:
    """Everything derived from the cover alone, computed once per image."""

    def __init__(self, row):
        self.row = row
        self.name = row["image"]
        self.key = data.image_key(self.name)
        self.img = data.load_cover(row["path"])
        self.H, self.W, self.Ch = self.img.shape
        self.gray = data.to_gray(self.img)
        t = perf_counter()
        self.z0, self.tau_lo, self.tau_hi = E.base_zone_map(self.gray)
        self.t_zone = perf_counter() - t
        self.cost = E.uniward_cost(self.gray)
        self.hash = hashlib.sha256(self.img.tobytes()).digest()
        self.cover_det = metrics.detectors(self.img)

    def ident(self):
        return dict(image=self.name, volume=self.row["volume"], size=self.row["size"],
                    image_sha256=self.row["sha256"])

    def n_bits(self, rate):
        return int(rate * self.H * self.W * self.Ch) // 8 * 8

    def tag(self, label):
        return self.hash + f"|{self.name}|{label}".encode()


def key_session(cov, s):
    spawn_priv = (C.STREAM["qkd_private"], cov.key, s)
    spawn_pub = (C.STREAM["qkd_public"], cov.key, s)
    t = perf_counter()
    st = qkd.session(data.rng_for(*spawn_priv), data.rng_for(*spawn_pub))
    st["t_qkd_s"] = perf_counter() - t
    st["spawn_private"] = str(spawn_priv)
    st["spawn_public"] = str(spawn_pub)
    return st


def qkd_fields(st):
    out = {c: st.get(c) for c in QKD_COLS}
    out.update(session_spawn_private=st["spawn_private"], session_spawn_public=st["spawn_public"])
    return out


def payload(cov, rate_idx, n_bits):
    rng = data.rng_for(C.STREAM["payload"], cov.key, rate_idx)
    return np.unpackbits(rng.integers(0, 256, n_bits // 8, dtype=np.uint8))


def transmit(method, cov, m_bits, st, zk, tag, timings=None):
    """Full Stage B (Alice) + Stage C (Bob) for one method."""
    ka, kb = st["keys_alice"], st["keys_bob"]
    T = timings if timings is not None else {}
    t = perf_counter(); c = crypto.encrypt(m_bits, ka["k"], tag); t_mac = crypto.mac(ka["mac"], tag, c)
    T["t_encrypt_s"] = perf_counter() - t
    t = perf_counter(); seq = dna.bits_to_dna(c); b = dna.dna_to_bits(seq)[: len(c)]
    T["t_dna_s"] = perf_counter() - t
    t = perf_counter(); stego, pi, info = E.embed(method, cov.img, b, ka["k"], cost=cov.cost, zk=zk)
    T["t_embed_s"] = perf_counter() - t
    t = perf_counter(); sealed = crypto.seal_pi(pi, ka["pi"], ka["mac"], tag); T["t_seal_pi_s"] = perf_counter() - t
    t = perf_counter(); pi_r = crypto.open_pi(sealed, kb["pi"], kb["mac"], tag); T["t_open_pi_s"] = perf_counter() - t
    t = perf_counter(); ext = E.extract(stego, pi_r); T["t_extract_s"] = perf_counter() - t
    t = perf_counter()
    c_hat = dna.dna_to_bits(dna.bits_to_dna(ext))[: len(c)]
    ok_mac = crypto.mac_ok(kb["mac"], tag, c_hat, t_mac)
    m_hat = crypto.encrypt(c_hat, kb["k"], tag)
    T["t_decrypt_verify_s"] = perf_counter() - t
    rec = dict(ber=float(np.mean(m_hat != m_bits)), exact=bool(np.array_equal(m_hat, m_bits)),
               mac_ok=bool(ok_mac), pi_bytes=len(sealed), gc_content=dna.gc_content(seq), dna_bases=len(seq))
    return stego, pi, sealed, c, t_mac, info, rec


# ---------------------------------------------------------------- E1
def e1_main(row):
    cov = Cover(row)
    rows = []
    base = cov.ident()
    base.update(tau_low=cov.tau_lo, tau_high=cov.tau_hi, t_zone_s=cov.t_zone,
                **{f"cover_{k}": v for k, v in cov.cover_det.items()})
    for s in range(C.N_SESSIONS):
        st = key_session(cov, s)
        q = qkd_fields(st)
        if st["abort"]:
            for ri, rate in enumerate(C.RATES_BPC):
                for m in C.METHODS:
                    rows.append({**base, **q, "session": s, "rate_bpc": rate, "method": m, "status": "key_abort"})
            continue
        t = perf_counter()
        zk = E.permute_zone_map(cov.z0, st["keys_alice"]["k"], C.GAMMA)
        t_perm = perf_counter() - t
        agree = float(np.mean(zk == cov.z0))
        for ri, rate in enumerate(C.RATES_BPC):
            n = cov.n_bits(rate)
            m_bits = payload(cov, ri, n)
            tag = cov.tag(f"s{s}|r{rate}")
            for m in C.METHODS:
                r = {**base, **q, "session": s, "rate_bpc": rate, "bpp": rate * cov.Ch,
                     "payload_bits": n, "payload_kB": n / 8 / 1024, "method": m,
                     "capacity_bits": E.capacity(m, cov.img.shape, zk), "t_qkd_s": st["t_qkd_s"]}
                if n > r["capacity_bits"]:
                    rows.append({**r, "status": "skip", "skip_reason": "payload exceeds method capacity"})
                    continue
                T = {}
                stego, pi, sealed, c, t_mac, info, rec = transmit(m, cov, m_bits, st, zk if m == "NirapodQSeal" else None,
                                                                  tag, T)
                t = perf_counter(); p, mse = metrics.psnr(cov.img, stego); T["t_psnr_s"] = perf_counter() - t
                t = perf_counter(); ss = metrics.ssim(cov.img, stego); T["t_ssim_s"] = perf_counter() - t
                t = perf_counter(); det = metrics.detectors(stego); T["t_detectors_s"] = perf_counter() - t
                if m == "NirapodQSeal":
                    r.update(t_permute_s=t_perm, zk_z0_agreement=agree, gamma=C.GAMMA)
                rows.append({**r, **info, **rec, **T, "status": "ok", "psnr": p, "mse": mse, "ssim": ss,
                             "change_rate": metrics.change_rate(cov.img, stego), **det,
                             **{f"delta_{k}": det[k] - cov.cover_det[k] for k in det}})
    return rows


# ---------------------------------------------------------------- E2
def _majority(bits, n, r):
    return (bits[: n * r].reshape(r, n).sum(0) > r / 2).astype(np.uint8)


def e2_robustness(row):
    cov = Cover(row)
    st = key_session(cov, 0)
    if st["abort"]:
        return [{**cov.ident(), "status": "key_abort"}]
    ka, kb = st["keys_alice"], st["keys_bob"]
    zk = E.permute_zone_map(cov.z0, ka["k"], C.GAMMA)
    rows = []
    atk = A.attack_list()

    # fragility at the base settings
    ri = C.RATES_BPC.index(C.FRAGILITY_RATE)
    n = cov.n_bits(C.FRAGILITY_RATE)
    m_bits = payload(cov, ri, n)
    tag = cov.tag(f"s0|r{C.FRAGILITY_RATE}")
    for m in C.METHODS:
        if n > E.capacity(m, cov.img.shape, zk):
            continue
        stego, pi, sealed, *_ = transmit(m, cov, m_bits, st, zk if m == "NirapodQSeal" else None, tag)
        pi_r = crypto.open_pi(sealed, kb["pi"], kb["mac"], tag)
        for ai, (fam, lev) in enumerate(atk):
            att = A.apply(stego, fam, lev, data.rng_for(C.STREAM["attack"], cov.key, ai))
            c_hat = E.extract(att, pi_r)
            m_hat = crypto.encrypt(c_hat, kb["k"], tag)
            rows.append({**cov.ident(), "part": "fragility", "method": m, "rate_bpc": C.FRAGILITY_RATE,
                         "attack": fam, "level": lev, "ber": float(np.mean(m_hat != m_bits)),
                         "exact": bool(np.array_equal(m_hat, m_bits)), "status": "ok"})

    # matched message, robust mode
    msg = data.rng_for(C.STREAM["payload"], cov.key, 99).integers(0, 2, C.ROBUST_MSG_BITS).astype(np.uint8)
    tag = cov.tag("robust")
    c = crypto.encrypt(msg, ka["k"], tag)
    code = np.tile(c, C.ROBUST_REPEAT)
    for m in ("LSB", "UNIWARD-cost", "DCT-R", "NirapodQSeal-R", "NirapodQSeal-JPEG"):
        if m == "NirapodQSeal-JPEG":
            stego, pi = E.embed_jpeg_robust(cov.img, code, ka["k"], zk)
            read = lambda img, pi=pi: E.extract_jpeg_robust(img, pi)
        elif m in ("DCT-R", "NirapodQSeal-R"):
            stego, pi = E.embed_robust(m, cov.img, code, ka["k"], zk=zk)
            read = lambda img, pi=pi: E.extract(img, pi, delta=C.ROBUST_DELTA)
        else:
            stego, pi, _ = E.embed(m, cov.img, code, ka["k"], cost=cov.cost)
            read = lambda img, pi=pi: E.extract(img, pi)
        p, _ = metrics.psnr(cov.img, stego)
        for ai, (fam, lev) in enumerate(atk):
            att = A.apply(stego, fam, lev, data.rng_for(C.STREAM["attack"], cov.key, ai))
            dec = _majority(read(att), C.ROBUST_MSG_BITS, C.ROBUST_REPEAT)
            m_hat = crypto.encrypt(dec, kb["k"], tag)
            rows.append({**cov.ident(), "part": "robust", "method": m, "msg_bits": C.ROBUST_MSG_BITS,
                         "repeat": C.ROBUST_REPEAT, "psnr_clean": p, "attack": fam, "level": lev,
                         "bit_accuracy": float(np.mean(m_hat == msg)),
                         "exact": bool(np.array_equal(m_hat, msg)), "status": "ok"})
    return rows


# ---------------------------------------------------------------- E3
def e3_ablation(row):
    cov = Cover(row)
    st = key_session(cov, 0)
    if st["abort"]:
        return [{**cov.ident(), "status": "key_abort"}]
    k = st["keys_alice"]["k"]
    variants = [(f"gamma={g}", g, E.DEFAULT_ROUTE_MAP) for g in C.GAMMA_SWEEP]
    variants += [("no_DCT_route", C.GAMMA, {0: "su", 1: "su", 2: "lsb"}),
                 ("no_UNIWARD_route", C.GAMMA, {0: "dct", 1: "lsb", 2: "lsb"}),
                 ("no_LSB_route", C.GAMMA, {0: "dct", 1: "su", 2: "su"})]
    rows = []
    for rate in C.ABLATION_RATES:
        ri = C.RATES_BPC.index(rate)
        n = cov.n_bits(rate)
        bits = payload(cov, ri, n)
        for name, g, rmap in variants:
            zk = E.permute_zone_map(cov.z0, k, g)
            r = {**cov.ident(), "rate_bpc": rate, "variant": name, "gamma": g,
                 "route_map": str(rmap), "status": "ok"}
            if n > E.capacity("NirapodQSeal", cov.img.shape, zk, rmap):
                rows.append({**r, "status": "skip"})
                continue
            stego, pi, info = E.embed("NirapodQSeal", cov.img, bits, k, cost=cov.cost, zk=zk, route_map=rmap)
            p, _ = metrics.psnr(cov.img, stego)
            z0_stego, _, _ = E.base_zone_map(data.to_gray(stego))
            rows.append({**r, **info, "exact": bool(np.array_equal(E.extract(stego, pi), bits)),
                         "psnr": p, "ssim": metrics.ssim(cov.img, stego), **metrics.detectors(stego),
                         "adversary_acc_from_stego": float(np.mean(z0_stego == zk)),
                         "adversary_acc_from_cover": float(np.mean(cov.z0 == zk))})
    return rows


# ---------------------------------------------------------------- E4
def e4_security(row):
    cov = Cover(row)
    st = key_session(cov, 0)
    if st["abort"]:
        return [{**cov.ident(), "status": "key_abort"}]
    ka, kb = st["keys_alice"], st["keys_bob"]
    zk = E.permute_zone_map(cov.z0, ka["k"], C.GAMMA)
    rng = data.rng_for(C.STREAM["security"], cov.key)
    ri = C.RATES_BPC.index(C.SECURITY_RATE)
    n = cov.n_bits(C.SECURITY_RATE)
    m_bits = payload(cov, ri, n)
    tag = cov.tag(f"s0|r{C.SECURITY_RATE}")
    stego, pi, sealed, c, t_mac, _, rec = transmit("NirapodQSeal", cov, m_bits, st, zk, tag)
    rows = []
    base = {**cov.ident(), "status": "ok"}

    # wrong key: one bit of the distilled secret flipped, all subkeys re-derived
    s = bytearray(st["s_alice"])
    pos = int(rng.integers(0, 8 * len(s)))
    s[pos // 8] ^= 1 << (7 - pos % 8)
    kw = qkd.derive_keys(bytes(s), st["hkdf_salt"])
    pi_w = crypto.open_pi(sealed, kw["pi"], kw["mac"], tag)
    c_hat = E.extract(stego, pi)  # worst case: adversary somehow holds the true Pi
    m_w = crypto.encrypt(c_hat, kw["k"], tag)
    zk_w = E.permute_zone_map(cov.z0, kw["k"], C.GAMMA)
    rows.append({**base, "test": "wrong_key_1bit", "flipped_bit": pos, "pi_rejected": pi_w is None,
                 "mac_rejected": not crypto.mac_ok(kw["mac"], tag, c_hat, t_mac),
                 "ber_with_true_pi": float(np.mean(m_w != m_bits)),
                 "zk_agreement_wrong_key": float(np.mean(zk_w == zk)), "correct_key_exact": rec["exact"]})

    # tamper: one embedded sample flipped / one random pixel flipped / one byte of sealed Pi flipped
    spatial = np.concatenate([pi.get("su", np.zeros(0, np.int64)), pi.get("lsb", np.zeros(0, np.int64))])
    for trial in range(20):
        for kind in ("embedded_sample", "random_sample"):
            st2 = stego.copy().reshape(-1)
            i = int(spatial[rng.integers(0, len(spatial))]) if kind == "embedded_sample" else int(rng.integers(0, st2.size))
            st2[i] ^= 1
            c2 = dna.dna_to_bits(dna.bits_to_dna(E.extract(st2.reshape(stego.shape), pi)))[: len(c)]
            rows.append({**base, "test": f"tamper_{kind}", "trial": trial, "flat_index": i,
                         "payload_changed": bool(not np.array_equal(c2, c)),
                         "mac_rejected": not crypto.mac_ok(kb["mac"], tag, c2, t_mac)})
        sp = bytearray(sealed)
        j = int(rng.integers(0, len(sp)))
        sp[j] ^= 1 << int(rng.integers(0, 8))
        rows.append({**base, "test": "tamper_sealed_pi", "trial": trial, "byte_index": j,
                     "pi_rejected": crypto.open_pi(bytes(sp), kb["pi"], kb["mac"], tag) is None})

    # NIST SP 800-22 on the keystream and on the ciphertext
    ks = crypto.keystream(ka["k"], cov.tag("NIST"), C.NIST_KEYSTREAM_BITS)
    for src, bits in (("keystream", ks), ("ciphertext", c[: C.NIST_KEYSTREAM_BITS])):
        for test, (pmin, ok) in nist.run_all(bits, C.NIST_ALPHA).items():
            rows.append({**base, "test": "nist", "source": src, "n_bits": len(bits), "nist_test": test,
                         "p_value_min": pmin, "nist_pass": ok})
    return rows


# ---------------------------------------------------------------- E5
def e5_qkd():
    rows = []
    for fi, f in enumerate(C.EVE_FRACTIONS):
        for t in range(C.EVE_TRIALS):
            sp = (C.STREAM["eve"], 1, fi, t); su = (C.STREAM["eve"], 2, fi, t)
            st = qkd.session(data.rng_for(*sp), data.rng_for(*su), eve_frac=f)
            rows.append({"part": "eavesdrop", "eve_fraction": f, "trial": t, "spawn_private": str(sp),
                         **{c: st.get(c) for c in QKD_COLS}})
    for ni, npairs in enumerate(C.FINITE_KEY_PAIRS):
        for t in range(C.FINITE_KEY_TRIALS):
            sp = (C.STREAM["finite"], 1, ni, t); su = (C.STREAM["finite"], 2, ni, t)
            tt = perf_counter()
            st = qkd.session(data.rng_for(*sp), data.rng_for(*su), n_pairs=npairs)
            rows.append({"part": "finite_key", "n_pairs": npairs, "trial": t, "spawn_private": str(sp),
                         "t_session_s": perf_counter() - tt, **{c: st.get(c) for c in QKD_COLS},
                         "n_key_string": st.get("n_key_string")})
    return rows


# ---------------------------------------------------------------- E6
def e6_indel(row):
    cov = Cover(row)
    st = key_session(cov, 0)
    if st["abort"]:
        return [{**cov.ident(), "status": "key_abort"}]
    ka, kb = st["keys_alice"], st["keys_bob"]
    zk = E.permute_zone_map(cov.z0, ka["k"], C.GAMMA)
    tag = cov.tag("indel")
    msg_bits = np.unpackbits(np.frombuffer(C.INDEL_MSG, np.uint8))
    c = np.packbits(crypto.encrypt(msg_bits, ka["k"], tag)).tobytes()
    schemes = {"indel_aware_DNA": indel.enc_indel(c), "RS_only": indel.enc_rs(c), "repetition_only": indel.enc_rep(c)}
    decoders = {"indel_aware_DNA": indel.dec_indel, "RS_only": indel.dec_rs, "repetition_only": indel.dec_rep}
    rows = []
    for name, (seq, meta) in schemes.items():
        bits = dna.dna_to_bits(seq)
        stego, pi, _ = E.embed("NirapodQSeal", cov.img, bits, ka["k"], cost=cov.cost, zk=zk)
        carried = dna.bits_to_dna(E.extract(stego, pi))
        lossless = carried == seq
        for ri, p in enumerate(C.INDEL_RATES):
            ok = 0
            for t in range(C.INDEL_TRIALS):
                recv = indel.indel_channel(carried, p, data.rng_for(C.STREAM["indel"], cov.key, ri, t))
                out = decoders[name](recv, meta)
                if out is not None and len(out) == len(c):
                    m_hat = crypto.encrypt(np.unpackbits(np.frombuffer(out, np.uint8)), kb["k"], tag)
                    ok += int(np.array_equal(m_hat, msg_bits))
            rows.append({**cov.ident(), "scheme": name, "indel_rate": p, "trials": C.INDEL_TRIALS,
                         "exact_recovery": ok / C.INDEL_TRIALS, "dna_bases": len(seq),
                         "image_carry_lossless": lossless, "status": "ok"})
    return rows


PER_IMAGE = {"E1_main": e1_main, "E2_robustness": e2_robustness, "E3_ablation": e3_ablation,
             "E4_security": e4_security, "E6_indel": e6_indel}
