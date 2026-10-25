"""Every study parameter in one place. The run manifest stores a copy of
these values, so each result can be traced to the exact settings used."""

MASTER_SEED = 20261016          # root of every random stream in the study

# ---- data -----------------------------------------------------------------
SIZES = (512, 1024, 2250)       # native square sizes used (colour covers only)
EXPECTED_N_IMAGES = 43

# ---- payload --------------------------------------------------------------
RATES_BPC = (0.05, 0.1, 0.3, 0.5)   # bits per colour channel sample
N_SESSIONS = 3                      # independent E91 key sessions per image

# ---- methods --------------------------------------------------------------
METHODS = ("LSB", "DCT", "UNIWARD-cost", "NirapodQSeal")
DCT_DELTA = 8.0
DCT_HEADROOM = 0.90             # fraction of DCT capacity used; rest = relocation spares
MID_FREQ = ((0, 1), (1, 0), (1, 1), (0, 2), (2, 0), (1, 2), (2, 1), (2, 2), (0, 3), (3, 0))
GAMMA = 1.0                     # key-bound permutation fraction (deployed setting)
UNIWARD_SIGMA = 1.0

# ---- E91 key layer --------------------------------------------------------
E91_PAIRS = 40_000
E91_NOISE = 0.02                # depolarising noise; visibility V = 1 - noise
E91_A_KEY = (22.5, 45.0)
E91_B_KEY = (22.5, 45.0)
E91_A_TEST = (0.0, 45.0)
E91_B_TEST = (22.5, 67.5)
QBER_MAX = 0.11                 # abort level Q_max
PE_FRACTION = 0.10              # sifted bits disclosed for parameter estimation
EPS_PE = 1e-10                  # failure prob. of the QBER / CHSH confidence bounds
EPS_PA = 1e-10                  # privacy-amplification security parameter
EPS_EC = 1e-10                  # error-verification failure probability
EC_VERIFY_BITS = 64             # hash length used to verify reconciliation
CASCADE_PASSES = 4
KEY_BITS = 256

# ---- robustness (E2) ------------------------------------------------------
ROBUST_DELTA = 24.0
ROBUST_REPEAT = 9
ROBUST_MSG_BITS = 800
JPEG_DESIGN_QF = 75             # quantiser the JPEG-robust step is locked to
JPEG_STEP_FACTOR = 3.0          # QIM step = factor x JPEG luma quantiser
FRAGILITY_RATE = 0.05
JPEG_QF = (95, 90, 80, 70, 60, 50, 40)
GAUSS_SIGMA = (0.5, 1.0, 2.0, 3.0, 5.0)
MEDIAN_K = 3
SALT_PEPPER_P = 0.01

# ---- ablation (E3) --------------------------------------------------------
ABLATION_RATES = (0.1, 0.3)
GAMMA_SWEEP = (0.0, 0.25, 0.5, 0.75, 1.0)

# ---- security (E4) --------------------------------------------------------
NIST_KEYSTREAM_BITS = 1_000_000
NIST_ALPHA = 0.01
SECURITY_RATE = 0.1

# ---- QKD sweeps (E5) ------------------------------------------------------
EVE_FRACTIONS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
EVE_TRIALS = 20
FINITE_KEY_PAIRS = (10_000, 20_000, 40_000, 100_000, 200_000, 500_000, 1_000_000)
FINITE_KEY_TRIALS = 5

# ---- indel DNA channel (E6) -----------------------------------------------
INDEL_MSG = (b"NirapodQSeal quantum-keyed DNA archival channel. " * 3)[:96]
INDEL_RATES = (0.0, 0.01, 0.02, 0.03, 0.05)
INDEL_TRIALS = 6
INDEL_RS_NSYM = 160
INDEL_BLOCK = 3
INDEL_REPEAT = 6
INDEL_REP_ONLY = 7

# ---- stream identifiers (fix the spawn keys of every random stream) -------
STREAM = {"qkd_private": 1, "qkd_public": 2, "payload": 3, "attack": 4,
          "security": 5, "eve": 6, "finite": 7, "indel": 8, "lsbsign": 9}


def as_dict():
    return {k: (list(v) if isinstance(v, tuple) else v)
            for k, v in globals().items() if k.isupper() and not isinstance(v, bytes)}
