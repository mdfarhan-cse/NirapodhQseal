"""A subset of the NIST SP 800-22 statistical tests (rev. 1a), implemented
from the specification. Each test returns one or more p-values; a sequence
passes a test at level alpha when every p-value >= alpha."""

import math

import numpy as np
from scipy.special import erfc, gammaincc
from scipy.stats import norm


def frequency(x):
    s = abs(int(np.sum(2 * x.astype(np.int64) - 1))) / math.sqrt(len(x))
    return [float(erfc(s / math.sqrt(2)))]


def block_frequency(x, M=128):
    N = len(x) // M
    pi = x[: N * M].reshape(N, M).mean(1)
    chi2 = 4 * M * np.sum((pi - 0.5) ** 2)
    return [float(gammaincc(N / 2, chi2 / 2))]


def runs(x):
    n = len(x)
    pi = x.mean()
    if abs(pi - 0.5) >= 2 / math.sqrt(n):
        return [0.0]
    v = 1 + int(np.sum(x[1:] != x[:-1]))
    return [float(erfc(abs(v - 2 * n * pi * (1 - pi)) / (2 * math.sqrt(2 * n) * pi * (1 - pi))))]


def longest_run(x):
    n = len(x)
    if n >= 750_000:
        M, edges, probs = 10_000, (10, 11, 12, 13, 14, 15, 16), (0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727)
    elif n >= 6272:
        M, edges, probs = 128, (4, 5, 6, 7, 8, 9), (0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124)
    else:
        M, edges, probs = 8, (1, 2, 3, 4), (0.2148, 0.3672, 0.2305, 0.1875)
    N = n // M
    blocks = x[: N * M].reshape(N, M)
    longest = np.zeros(N, np.int64)
    run = np.zeros(N, np.int64)
    for j in range(M):
        run = np.where(blocks[:, j] == 1, run + 1, 0)
        longest = np.maximum(longest, run)
    K = len(probs) - 1
    cls = np.clip(np.searchsorted(edges, longest, side="left"), 0, K)
    nu = np.bincount(cls, minlength=K + 1)
    chi2 = np.sum((nu - N * np.array(probs)) ** 2 / (N * np.array(probs)))
    return [float(gammaincc(K / 2, chi2 / 2))]


def spectral(x):
    n = len(x)
    X = 2 * x.astype(np.float64) - 1
    mod = np.abs(np.fft.fft(X))[: n // 2]
    T = math.sqrt(math.log(1 / 0.05) * n)
    N0, N1 = 0.95 * n / 2, np.sum(mod < T)
    d = (N1 - N0) / math.sqrt(n * 0.95 * 0.05 / 4)
    return [float(erfc(abs(d) / math.sqrt(2)))]


def _pattern_counts(x, m):
    n = len(x)
    if m == 0:
        return np.array([n])
    ext = np.concatenate([x, x[: m - 1]]).astype(np.int64)
    code = np.zeros(n, np.int64)
    for j in range(m):
        code = (code << 1) | ext[j:j + n]
    return np.bincount(code, minlength=2 ** m)


def serial(x, m=3):
    n = len(x)
    psi = lambda mm: (2 ** mm / n) * np.sum(_pattern_counts(x, mm).astype(np.float64) ** 2) - n if mm > 0 else 0.0
    p0, p1, p2 = psi(m), psi(m - 1), psi(m - 2)
    d1, d2 = p0 - p1, p0 - 2 * p1 + p2
    return [float(gammaincc(2 ** (m - 2), d1 / 2)), float(gammaincc(2 ** (m - 3), d2 / 2))]


def approximate_entropy(x, m=8):
    n = len(x)

    def phi(mm):
        c = _pattern_counts(x, mm).astype(np.float64) / n
        c = c[c > 0]
        return float(np.sum(c * np.log(c)))

    apen = phi(m) - phi(m + 1)
    chi2 = 2 * n * (math.log(2) - apen)
    return [float(gammaincc(2 ** (m - 1), chi2 / 2))]


def cumulative_sums(x):
    n = len(x)
    X = 2 * x.astype(np.int64) - 1
    out = []
    for seq in (X, X[::-1]):
        z = int(np.max(np.abs(np.cumsum(seq))))
        k1 = np.arange(int((-n / z + 1) // 4), int((n / z - 1) // 4) + 1)
        k2 = np.arange(int((-n / z - 3) // 4), int((n / z - 1) // 4) + 1)
        sq = math.sqrt(n)
        p = (1 - np.sum(norm.cdf((4 * k1 + 1) * z / sq) - norm.cdf((4 * k1 - 1) * z / sq))
             + np.sum(norm.cdf((4 * k2 + 3) * z / sq) - norm.cdf((4 * k2 + 1) * z / sq)))
        out.append(float(p))
    return out


TESTS = {"frequency": frequency, "block_frequency": block_frequency, "runs": runs,
         "longest_run": longest_run, "spectral_dft": spectral, "serial": serial,
         "approximate_entropy": approximate_entropy, "cumulative_sums": cumulative_sums}


def run_all(bits, alpha):
    """NIST default block lengths (serial m=16, approximate entropy m=10),
    reduced for short sequences to satisfy the tests' size conditions.
    Validated against the SP 800-22 reference p-values for the bits of e."""
    x = np.asarray(bits, np.uint8)
    lg = int(math.floor(math.log2(len(x))))
    kw = {"serial": {"m": max(2, min(16, lg - 3))},
          "approximate_entropy": {"m": max(2, min(10, lg - 6))}}
    res = {}
    for name, fn in TESTS.items():
        ps = fn(x, **kw.get(name, {}))
        res[name] = (min(ps), all(p >= alpha for p in ps))
    return res
