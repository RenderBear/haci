r"""Shared pipeline hyperparameters, module inits, and script defaults."""

from __future__ import annotations

from types import SimpleNamespace

# ── L0: split-channel harmonic projection (lum / chroma Naka–Rushton) ─────────
L0 = SimpleNamespace(
    OFFSETS=[
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    ],
    ETA_LUM=0.01,
    ETA_CHR=0.01,
    GAMMA=1.0,
    EPS=1e-6,
    NOTCH_ENABLED=True,
    NOTCH_HALF_WIDTH=4,
    NOTCH_OMEGA_N_INIT=1.0 / 8,
    NOTCH_SIGMA_N_INIT=1.0 / 32,
    NOTCH_D_INIT=0.8,
    # Contrast gain control (L0GainControl): each direction's drive is divided by η² + its own
    # energy + λ·G_r ∗ (energy pooled over directions). λ starts ≈ 0, i.e. the per-direction L0.
    # G_r is truncated at POOL_RADIUS px. Coarser offsets (2 and 4 px) were tried here and their
    # learned weights stayed ≈ 0.02, so they were removed.
    POOL_LAMBDA_INIT=1e-5,
    POOL_SIGMA_INIT=2.0,
    POOL_RADIUS=16,
)

# ── L1: patch geometry + z₂ orientation bins (von Mises on θ_p) ───────────
L1 = SimpleNamespace(
    PATCH_SIZE=5,
    PATCH_OVERLAP=3,
    BORDER_PATCH_MAX_FRAC=0.2,
    EPS=1e-15,
    NUM_ORIENT_BINS=8,
    KAPPA_VM_INIT=2.0,
)

# ── Seed: η_z NR on |Z|, then co-circular + surround + divisive readout (η_readout) ─
SEED = SimpleNamespace(
    EPS=1e-9,
    ETA_Z_INIT=10.0,
    BETA_SEED_INIT=0.5,
    BETA_ASSOC_INIT=0.5,
    ETA_READOUT_INIT=0.30,
    LAMBDA_INIT=0.5,
    SIGMA_F_INIT=2.0,
    SIGMA_F_MIN=1.0,
    KAPPA_A_INIT=8.0,
    BETA_GAP_INIT=0.5,
    # Co-circular association field: bins couple along circles whose curvature is within the
    # learned tolerance σ_κ (1/cell). σ_κ ≈ 0 at init is the same-bin collinear term.
    SIGMA_KAPPA_INIT=1e-2,
    FACIL_RADIUS=5,
    # Readout pool: each orientation bin divided by the same bin's neighbouring cells (centre
    # excluded), Gaussian σ_s in cells. Replaced the sin²Δθ cross-orientation pool, which at broad κ_vm
    # read an edge's own energy; σ_s ≈ 1 ranked candidates best on the trained model.
    SURROUND_RADIUS=10,
    SIGMA_S_INIT=1.0,
    # Flank suppression: the readout is also divided by λ_f·HM(F+, F-)², the same bin's support on
    # either side of the cell along its normal (Gaussian σ_fl in cells, κ tuning to the normal), cells
    # that share pixels with the centre left out. The harmonic mean needs both flanks occupied
    # (texture, gratings), so a lone edge, or one beside a single parallel edge, is spared.
    # λ_f ≈ 0 at init is the model without it.
    FLANK_LAMBDA_INIT=1e-2,
    FLANK_SIGMA_INIT=2.0,
    FLANK_SIGMA_MIN=1.0,
    FLANK_KAPPA=2.0,
    FLANK_RADIUS=5,
    RHO_STE_TAU=0.1,
)

# ── Render: back-projection with learned 1D kernels ───────────────────────────
RENDER = SimpleNamespace(
    DEPOSIT_HALF_WIDTH_STRIDES=2.0,
    DEPOSIT_HALF_WIDTH_MIN=4,
    DEPOSIT_HALF_WIDTH_MAX=24,

    SIGMA_PERP_INIT=0.6,
    SIGMA_PAR_INIT=2.0,
    RAMP_CUTOFF_INIT=0.5,

    KAPPA_MAX_INIT=0.1,
    EXT_MAX_INIT=1.0,
    DELTA_N_MAX_INIT=1.0,
    ALPHA_RANGE_INIT=0.5,

    BIN_GATE_TEMP_INIT=0.08,

    CORR_HIDDEN=12,

    DEPOSIT_ENVELOPE_SIGMA=0.0,
    SIGMA_PAR_MAX=32.0,
    SIGMA_PERP_MAX=8.0,
    SPLAT_RADIUS_SIGMAS=3.0,
    THETA_SMOOTH_PASSES=0,
    THINNING_IN=30,
    THINNING_HIDDEN=12,
)

# ── Training ───────────────────────────────────────────────────────────────
TRAIN = SimpleNamespace(
    LR=5e-2,
    EPOCHS=50,
    BATCH_SIZE=4,
    GRAD_CLIP=1.0,
    NUM_WORKERS=0,
    GT_MIN_AGREEMENT=0.0,
    L0_CACHE_VERSION=3,
)

# ── Evaluation & inference defaults ───────────────────────────────────────────
EVAL = SimpleNamespace(
    DEFAULT_THRESHOLD=0.5,
    THRESHOLD_COUNT=99,
    MAX_DIST_FRAC=0.0075,
)

INFER = SimpleNamespace(
    SHAPE_THETA_BINS=12,
)

# ── Diagnostics / matplotlib styling ────────────────────────────────────────
VIZ = SimpleNamespace(
    BG="#0e0e0e",
    PANEL_BG="#111111",
    FG="#dddddd",
    ACCENT="#888888",
    EPS=1e-15,
    COMPASS={
        (-1, -1): "NW",
        (-1, 0): "N",
        (-1, 1): "NE",
        (0, -1): "W",
        (0, 1): "E",
        (1, -1): "SW",
        (1, 0): "S",
        (1, 1): "SE",
    },
    GRID_POS={
        (-1, -1): (0, 0),
        (-1, 0): (0, 1),
        (-1, 1): (0, 2),
        (0, -1): (1, 0),
        (0, 1): (1, 2),
        (1, -1): (2, 0),
        (1, 0): (2, 1),
        (1, 1): (2, 2),
    },
)