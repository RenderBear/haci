r"""Cell-grid contour seed — per-orientation-bin NR, co-circular association field, same-orientation surround, flank suppression, readout"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as Fn

try:
    from params import L1, SEED
except Exception:  # pragma: no cover
    class L1:  # type: ignore
        NUM_ORIENT_BINS = 8
        KAPPA_VM_INIT = 2.0

    class SEED:  # type: ignore
        EPS = 1e-6
        ETA_Z_INIT = 0.30
        SURROUND_RADIUS = 5
        RHO_STE_TAU = 0.1


_ETA_Z_INIT = float(getattr(SEED, "ETA_Z_INIT", 0.30))
_BETA_SEED_INIT = float(getattr(SEED, "BETA_SEED_INIT", 0.5))
_BETA_ASSOC_INIT = float(getattr(SEED, "BETA_ASSOC_INIT", 0.5))
_ETA_READOUT_INIT = float(
    getattr(SEED, "ETA_READOUT_INIT", getattr(SEED, "ETA_INIT", 0.30))
)
_LAMBDA_INIT = float(getattr(SEED, "LAMBDA_INIT", 0.5))
_SIGMA_F_INIT = float(getattr(SEED, "SIGMA_F_INIT", 1.3))
_SIGMA_S_INIT = float(getattr(SEED, "SIGMA_S_INIT", 1.0))
_KAPPA_VM_INIT = float(getattr(L1, "KAPPA_VM_INIT", 2.0))
_K = int(getattr(L1, "NUM_ORIENT_BINS", 8))
_FACIL_RADIUS = int(getattr(SEED, "FACIL_RADIUS", 2))
_SURROUND_RADIUS = int(getattr(SEED, "SURROUND_RADIUS", 5))
_RHO_STE_TAU = float(getattr(SEED, "RHO_STE_TAU", 0.1))
_SIGMA_F_MIN = float(getattr(SEED, "SIGMA_F_MIN", 1.0))
_KAPPA_A_INIT = float(getattr(SEED, "KAPPA_A_INIT", 8.0))
_BETA_GAP_INIT = float(getattr(SEED, "BETA_GAP_INIT", 0.5))
_SIGMA_KAPPA_INIT = float(getattr(SEED, "SIGMA_KAPPA_INIT", 1e-2))
_FLANK_LAMBDA_INIT = float(getattr(SEED, "FLANK_LAMBDA_INIT", 1e-2))
_FLANK_SIGMA_INIT = float(getattr(SEED, "FLANK_SIGMA_INIT", 2.0))
_FLANK_SIGMA_MIN = float(getattr(SEED, "FLANK_SIGMA_MIN", 1.0))
_FLANK_KAPPA = float(getattr(SEED, "FLANK_KAPPA", 2.0))
_FLANK_RADIUS = int(getattr(SEED, "FLANK_RADIUS", 5))
# Cells within this Chebyshev distance share pixels with the centre cell (d·S < P), so they carry the
# cell's own edge and are left out of the flanks.
_P = int(getattr(L1, "PATCH_SIZE", 5))
_FLANK_EXCLUDE = math.ceil(_P / (_P - int(getattr(L1, "PATCH_OVERLAP", 3)))) - 1


def _inv_softplus(x: float) -> float:
    x = max(float(x), 1e-8)
    if x > 20.0:
        return x
    return math.log(math.expm1(x))


def orientation_bin_centers(K: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.arange(K, device=device, dtype=dtype) * (math.pi / float(K))


def _reflect_conv_hwk(
    x_hwk: torch.Tensor,
    weight: torch.Tensor,
    *,
    groups: int = 1,
) -> torch.Tensor:
    k = int(weight.shape[-1])
    pad = k // 2
    x = x_hwk.permute(2, 0, 1).unsqueeze(0)
    x_pad = Fn.pad(x, (pad, pad, pad, pad), mode="reflect")
    y = Fn.conv2d(x_pad, weight, groups=groups)
    return y.squeeze(0).permute(1, 2, 0)


def _gaussian_exponent(
    d2: torch.Tensor,
    sigma_sq: torch.Tensor,
) -> torch.Tensor:
    return torch.exp(-d2 / (2.0 * sigma_sq.clamp_min(1e-12)))


def cocircular_bipole_bins(
    rho_nr: torch.Tensor,
    bar_theta: torch.Tensor,
    *,
    sigma_f: torch.Tensor,
    kappa_a: torch.Tensor,
    sigma_kappa: torch.Tensor,
    radius: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Support ahead of and behind each cell along bin k's tangent, read co-circularly across bins.

    w(δ) = exp(-|δ|²/2σ_f²) · exp(κ_a(cos 2ψ - 1)), ψ = angle between δ and the tangent, so reach
    (σ_f) and direction tuning (κ_a) are separate knobs. Each lobe is normalised by its own weight
    sum: a cell inside a long contour sees f_fwd ≈ f_bwd ≈ the contour's ρ.

    A neighbour at δ lies on the circle tangent to θ_k with curvature κ = 2 sin ψ / |δ|, and the
    circle's tangent there is θ_k + 2ψ (Parent & Zucker). Bin k reads the neighbour's response at
    θ_k + s·2ψ, linearly interpolated between the two nearest bins, with s = σ_κ² / (σ_κ² + κ²):
    turns implied by curvatures within the tolerance σ_κ (per cell) are expected, sharper ones are
    read as straight continuation. At σ_κ = 0 this is the same-bin collinear read. The K×K bank
    of kernels is built from the parameters and applied as one fixed-radius conv.
    """
    nH, nW, K = rho_nr.shape
    dtype, dev = rho_nr.dtype, rho_nr.device
    r = torch.arange(-radius, radius + 1, device=dev, dtype=dtype)
    dy, dx = torch.meshgrid(r, r, indexing="ij")
    d2 = dy * dy + dx * dx
    ct = torch.cos(bar_theta).view(K, 1, 1)
    st = torch.sin(bar_theta).view(K, 1, 1)
    t = dy * ct + dx * st
    n = -dy * st + dx * ct
    cos2phi = (t * t - n * n) / d2.clamp_min(1.0)
    radial = torch.exp(-d2 / (2.0 * (sigma_f * sigma_f).clamp_min(eps)))
    w = radial * torch.exp(kappa_a * (cos2phi - 1.0))
    on_fwd = t > 1e-6
    on_bwd = t < -1e-6

    # Turn of the tangent along the circle, 2ψ with ψ = atan(n/t) in (−π/2, π/2); it is the same
    # for both lobes. t = 0 lies in neither lobe.
    turn = 2.0 * torch.atan(n / torch.where(on_fwd | on_bwd, t, torch.ones_like(t)))
    kappa_sq = 4.0 * n * n / (d2 * d2).clamp_min(1.0)
    sk2 = sigma_kappa * sigma_kappa
    shrink = sk2 / (sk2 + kappa_sq + eps)
    # Read position in bin units. k is added as an integer so σ_κ = 0 reads exactly bin k.
    k_idx = torch.arange(K, device=dev, dtype=dtype).view(K, 1, 1)
    pos = k_idx + shrink * turn * (K / math.pi)
    j0 = torch.floor(pos)
    frac = (pos - j0).unsqueeze(-1)
    j0 = torch.remainder(j0, K).unsqueeze(-1)
    j1 = torch.remainder(j0 + 1.0, K)
    jj = torch.arange(K, device=dev, dtype=dtype).view(1, 1, 1, K)
    read = (1.0 - frac) * (jj == j0).to(dtype) + frac * (jj == j1).to(dtype)  # (K, ks, ks, K_in)

    fwd = (w * on_fwd).unsqueeze(-1) * read
    bwd = (w * on_bwd).unsqueeze(-1) * read
    bank = torch.cat([fwd, bwd], dim=0).permute(0, 3, 1, 2)  # (2K, K_in, ks, ks)
    both = _reflect_conv_hwk(rho_nr, bank)
    # Each kernel's read weights sum to 1 over the input bins, so the lobe norms are unchanged.
    f_fwd = both[..., :K] / ((w * on_fwd).sum(dim=(-2, -1)).view(1, 1, K) + eps)
    f_bwd = both[..., K:] / ((w * on_bwd).sum(dim=(-2, -1)).view(1, 1, K) + eps)
    return f_fwd, f_bwd


def surround_bins_same(
    rho_nr: torch.Tensor,
    *,
    sigma_s: torch.Tensor,
    radius: int,
    eps: float,
) -> torch.Tensor:
    """Same-orientation pool: each bin's Gaussian-weighted mean over the neighbouring cells, the cell
    itself excluded. The pool is tuned exactly like the bins (κ_vm), so a cell's own orientation never
    reaches another channel's pool, and the readout codes a cell relative to its same-orientation context.
    """
    nH, nW, K = rho_nr.shape
    dtype, dev = rho_nr.dtype, rho_nr.device
    sigma_sq = (sigma_s * sigma_s).clamp_min(eps)
    radius = int(radius)
    if radius <= 0:
        return torch.zeros_like(rho_nr)
    ks = 2 * radius + 1

    # The surround is a Gaussian over every offset but the centre. exp(-(dy² + dx²)/2σ²) factors
    # into a column pass and a row pass; the centre tap has weight 1 and is subtracted afterwards.
    r = torch.arange(-radius, radius + 1, device=dev, dtype=dtype)
    g = _gaussian_exponent(r * r, sigma_sq)
    w_acc = g.sum() ** 2 - 1.0

    x = rho_nr.permute(2, 0, 1).unsqueeze(0)
    x = Fn.pad(x, (radius, radius, radius, radius), mode="reflect")
    x = Fn.conv2d(x, g.view(1, 1, ks, 1).expand(K, 1, ks, 1), groups=K)
    x = Fn.conv2d(x, g.view(1, 1, 1, ks).expand(K, 1, 1, ks), groups=K)
    smoothed = x.squeeze(0).permute(1, 2, 0) - rho_nr
    return smoothed / (w_acc + eps)


def flank_bins(
    rho_nr: torch.Tensor,
    bar_theta: torch.Tensor,
    *,
    sigma: torch.Tensor,
    kappa: float,
    radius: int,
    exclude: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same-orientation support on either side of each cell along bin k's normal (its flanks).

    w(δ) = exp(-|δ|²/2σ²) · exp(κ(cos 2ψ_n - 1)), ψ_n = angle between δ and the normal. The sides
    n > 0 and n < 0 are normalised separately, like the association lobes. Cells within Chebyshev
    distance `exclude` share pixels with the centre cell and are left out, so a lone edge, which
    spreads into its overlapping neighbours, does not flank itself.
    """
    nH, nW, K = rho_nr.shape
    dtype, dev = rho_nr.dtype, rho_nr.device
    r = torch.arange(-radius, radius + 1, device=dev, dtype=dtype)
    dy, dx = torch.meshgrid(r, r, indexing="ij")
    d2 = dy * dy + dx * dx
    ct = torch.cos(bar_theta).view(K, 1, 1)
    st = torch.sin(bar_theta).view(K, 1, 1)
    t = dy * ct + dx * st
    n = -dy * st + dx * ct
    cos2psi_n = (n * n - t * t) / d2.clamp_min(1.0)
    radial = torch.exp(-d2 / (2.0 * (sigma * sigma).clamp_min(eps)))
    w = radial * torch.exp(kappa * (cos2psi_n - 1.0))
    far = torch.maximum(dy.abs(), dx.abs()) > exclude
    w_pos = w * (far & (n > 1e-6)).to(dtype)
    w_neg = w * (far & (n < -1e-6)).to(dtype)

    # One group per bin: bin k's two side kernels read only bin k.
    bank = torch.stack([w_pos, w_neg], dim=1).reshape(2 * K, 1, 2 * radius + 1, 2 * radius + 1)
    both = _reflect_conv_hwk(rho_nr, bank, groups=K).reshape(nH, nW, K, 2)
    f_pos = both[..., 0] / (w_pos.sum(dim=(-2, -1)).view(1, 1, K) + eps)
    f_neg = both[..., 1] / (w_neg.sum(dim=(-2, -1)).view(1, 1, K) + eps)
    return f_pos, f_neg


def rho_ste_hard_max(rho_out: torch.Tensor, tau: float) -> torch.Tensor:
    w = Fn.softmax(rho_out / float(tau), dim=-1)
    hard = rho_out.max(dim=-1).values
    soft = (w * rho_out).sum(dim=-1)
    return hard - soft.detach() + soft


def soft_theta_double_angle(
    weights: torch.Tensor,
    bar_theta: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    w = weights.clamp_min(0.0)
    kd = w.dim()
    br = bar_theta.to(device=w.device, dtype=w.dtype)
    c2 = torch.cos(2.0 * br).reshape(*([1] * (kd - 1)), -1)
    s2 = torch.sin(2.0 * br).reshape(*([1] * (kd - 1)), -1)
    num_c = (w * c2).sum(dim=-1)
    num_s = (w * s2).sum(dim=-1)
    return 0.5 * torch.atan2(num_s, num_c + float(eps))


class ContourSeed(nn.Module):

    def __init__(
        self,
        eps: float = SEED.EPS,
        eta_z_init: float = _ETA_Z_INIT,
        beta_seed_init: float = _BETA_SEED_INIT,
        beta_assoc_init: float = _BETA_ASSOC_INIT,
        eta_readout_init: float = _ETA_READOUT_INIT,
        lambda_init: float = _LAMBDA_INIT,
        sigma_f_init: float = _SIGMA_F_INIT,
        sigma_s_init: float = _SIGMA_S_INIT,
        kappa_vm_init: float = _KAPPA_VM_INIT,
        num_orient_bins: int = _K,
        facil_radius: int = _FACIL_RADIUS,
        cross_surround_radius: int = _SURROUND_RADIUS,
        rho_ste_tau: float = _RHO_STE_TAU,
        kappa_a_init: float = _KAPPA_A_INIT,
        beta_gap_init: float = _BETA_GAP_INIT,
        sigma_kappa_init: float = _SIGMA_KAPPA_INIT,
        flank_lambda_init: float = _FLANK_LAMBDA_INIT,
        flank_sigma_init: float = _FLANK_SIGMA_INIT,
        flank_kappa: float = _FLANK_KAPPA,
        flank_radius: int = _FLANK_RADIUS,
        *,
        surround_radius: int | None = None,
        **kw,
    ):
        super().__init__()
        _ = kw
        if surround_radius is not None:
            cross_surround_radius = int(surround_radius)
        self.eps = float(eps)
        self.K = int(num_orient_bins)
        self.facil_radius = int(facil_radius)
        self.cross_surround_radius = int(cross_surround_radius)
        self.rho_ste_tau = float(rho_ste_tau)
        self.flank_kappa = float(flank_kappa)
        self.flank_radius = int(flank_radius)
        self.flank_exclude = _FLANK_EXCLUDE

        dev0 = torch.device("cpu")
        tb = orientation_bin_centers(self.K, dev0, torch.float32)
        self.register_buffer("theta_bins", tb, persistent=False)

        self._kappa_vm_raw = nn.Parameter(torch.tensor(_inv_softplus(kappa_vm_init)))
        self._eta_z_raw = nn.Parameter(torch.tensor(_inv_softplus(eta_z_init)))
        self._beta_seed_raw = nn.Parameter(torch.tensor(_inv_softplus(beta_seed_init)))
        self._beta_assoc_raw = nn.Parameter(torch.tensor(_inv_softplus(beta_assoc_init)))
        self._eta_readout_raw = nn.Parameter(torch.tensor(_inv_softplus(eta_readout_init)))
        self._lambda_raw = nn.Parameter(torch.tensor(_inv_softplus(lambda_init)))
        # σ_f = floor + softplus(raw): unlike a clamp, the gradient never dies at the floor.
        self.sigma_f_min = _SIGMA_F_MIN
        self._sigma_f_raw = nn.Parameter(torch.tensor(_inv_softplus(sigma_f_init - self.sigma_f_min)))
        self._kappa_a_raw = nn.Parameter(torch.tensor(_inv_softplus(kappa_a_init)))
        self._beta_gap_raw = nn.Parameter(torch.tensor(_inv_softplus(beta_gap_init)))
        self._sigma_s_raw = nn.Parameter(torch.tensor(_inv_softplus(sigma_s_init)))
        # Curvature tolerance of the association field, in 1/cell; ≈ 0 starts collinear.
        self._sigma_kappa_raw = nn.Parameter(torch.tensor(_inv_softplus(sigma_kappa_init)))
        # Flank suppression weight (≈ 0 starts without it) and reach in cells, floored like σ_f.
        self._flank_lambda_raw = nn.Parameter(torch.tensor(_inv_softplus(flank_lambda_init)))
        self.flank_sigma_min = _FLANK_SIGMA_MIN
        self._flank_sigma_raw = nn.Parameter(
            torch.tensor(_inv_softplus(flank_sigma_init - self.flank_sigma_min))
        )

    @property
    def kappa_vm(self) -> torch.Tensor:
        return Fn.softplus(self._kappa_vm_raw).view(())

    @property
    def eta_z(self) -> torch.Tensor:
        return Fn.softplus(self._eta_z_raw).view(())

    @property
    def beta_seed(self) -> torch.Tensor:
        return Fn.softplus(self._beta_seed_raw).view(())

    @property
    def beta_assoc(self) -> torch.Tensor:
        return Fn.softplus(self._beta_assoc_raw).view(())

    @property
    def eta_readout(self) -> torch.Tensor:
        return Fn.softplus(self._eta_readout_raw).view(())

    @property
    def lam(self) -> torch.Tensor:
        return Fn.softplus(self._lambda_raw).view(())

    @property
    def sigma_f(self) -> torch.Tensor:
        return self.sigma_f_min + Fn.softplus(self._sigma_f_raw).view(())

    @property
    def kappa_a(self) -> torch.Tensor:
        return Fn.softplus(self._kappa_a_raw).view(())

    @property
    def beta_gap(self) -> torch.Tensor:
        return Fn.softplus(self._beta_gap_raw).view(())

    @property
    def sigma_kappa(self) -> torch.Tensor:
        return Fn.softplus(self._sigma_kappa_raw).view(())

    @property
    def sigma_s(self) -> torch.Tensor:
        return Fn.softplus(self._sigma_s_raw).view(()).clamp_min(0.3)

    @property
    def flank_lambda(self) -> torch.Tensor:
        return Fn.softplus(self._flank_lambda_raw).view(())

    @property
    def flank_sigma(self) -> torch.Tensor:
        return self.flank_sigma_min + Fn.softplus(self._flank_sigma_raw).view(())

    @property
    def surround_radius(self) -> int:
        return self.cross_surround_radius

    @property
    def beta(self) -> torch.Tensor:
        return self.beta_seed

    @property
    def kappa(self) -> torch.Tensor:
        return self.beta_assoc

    def forward(
        self,
        cells_flat,
        return_surface_diags: bool = False,
        **kw,
    ):
        _ = kw
        device = next(self.parameters()).device
        nH, nW = int(cells_flat["nH"]), int(cells_flat["nW"])
        N = nH * nW
        eps = self.eps

        is_border = cells_flat["is_border"].to(device).reshape(nH, nW).bool()
        ok = (~is_border).to(torch.float32)

        if "rho_bin" not in cells_flat:
            raise ValueError("cells_flat must include rho_bin (from L1).")

        rho_bin = cells_flat["rho_bin"].to(device).float()
        if rho_bin.dim() == 2:
            rho_bin = rho_bin.reshape(nH, nW, -1)
        K = rho_bin.shape[-1]
        if K != self.K:
            raise ValueError(f"cells_flat rho_bin last dim K={K} != seed.K={self.K}")

        ax_bin = cells_flat["ax_bin"].to(device).float()
        ay_bin = cells_flat["ay_bin"].to(device).float()
        if ax_bin.dim() == 2:
            ax_bin = ax_bin.reshape(nH, nW, K)
            ay_bin = ay_bin.reshape(nH, nW, K)

        bar_theta = self.theta_bins.to(device=device, dtype=rho_bin.dtype)

        R = rho_bin * ok.unsqueeze(-1)
        Rsq = R * R
        eta_z_sq = self.eta_z * self.eta_z
        rho_nr = (Rsq / (Rsq + eta_z_sq + eps)) * ok.unsqueeze(-1)

        f_fwd, f_bwd = cocircular_bipole_bins(
            rho_nr,
            bar_theta,
            sigma_f=self.sigma_f,
            kappa_a=self.kappa_a,
            sigma_kappa=self.sigma_kappa,
            radius=self.facil_radius,
            eps=eps,
        )
        rho_assoc = 0.5 * (f_fwd + f_bwd)
        # Gain: association-field support scales the cell's own response, so it cannot create edges
        # past line ends. Bridge: harmonic mean of the two lobes, non-zero only when support
        # exists on both sides, so it fills gaps without extending ends.
        bridge = 2.0 * f_fwd * f_bwd / (f_fwd + f_bwd + eps)
        gain = self.beta_seed + self.beta_assoc * rho_assoc
        e = rho_nr * gain + self.beta_gap * bridge

        # Divisive pool from the same orientation channel's neighbours; facilitation stays in e.
        S = surround_bins_same(
            rho_nr,
            sigma_s=self.sigma_s,
            radius=self.cross_surround_radius,
            eps=eps,
        )

        # Flank suppression: same-orientation contours on both sides along the normal (texture,
        # gratings). The harmonic mean is non-zero only when both flanks are occupied.
        f_pos, f_neg = flank_bins(
            rho_nr,
            bar_theta,
            sigma=self.flank_sigma,
            kappa=self.flank_kappa,
            radius=self.flank_radius,
            exclude=self.flank_exclude,
            eps=eps,
        )
        flank = 2.0 * f_pos * f_neg / (f_pos + f_neg + eps)

        e2 = e * e
        eta_r = self.eta_readout * self.eta_readout
        pool = self.lam * (S * S)
        denom_r = e2 + eta_r + pool + eps + self.flank_lambda * (flank * flank)
        rho_out = (e2 / denom_r) * ok.unsqueeze(-1)

        rho_cell = rho_ste_hard_max(rho_out, self.rho_ste_tau)
        theta_grid = soft_theta_double_angle(rho_out, bar_theta, eps)
        k_star = rho_out.argmax(dim=-1)
        ax_star = torch.gather(ax_bin, dim=-1, index=k_star.unsqueeze(-1)).squeeze(-1)
        ay_star = torch.gather(ay_bin, dim=-1, index=k_star.unsqueeze(-1)).squeeze(-1)

        rho_flat = rho_cell.reshape(N)
        theta_flat = theta_grid.reshape(N, 1)

        cf_out = dict(cells_flat)
        cf_out["rho_nr"] = rho_nr.mean(dim=-1)
        cf_out["rho_seed"] = rho_nr.mean(dim=-1)
        cf_out["rho_nr_bins"] = rho_nr
        cf_out["rho_assoc"] = (rho_assoc * ok.unsqueeze(-1)).mean(dim=-1)
        cf_out["rho_assoc_bins"] = rho_assoc * ok.unsqueeze(-1)
        cf_out["fac"] = cf_out["rho_assoc"]
        cf_out["exc"] = (e * ok.unsqueeze(-1)).mean(dim=-1)
        cf_out["sur"] = (S * ok.unsqueeze(-1)).mean(dim=-1)
        cf_out["sur_bins"] = S * ok.unsqueeze(-1)
        cf_out["flank_bins"] = flank * ok.unsqueeze(-1)
        cf_out["drive"] = rho_bin.mean(dim=-1) * ok
        cf_out["g_R"] = cf_out["exc"]
        cf_out["g_E"] = cf_out["sur"]
        cf_out["rho_out_bins"] = rho_out
        cf_out["theta"] = theta_flat
        cf_out["cx_z2"] = ax_star.reshape(N)
        cf_out["cy_z2"] = ay_star.reshape(N)

        branch = torch.zeros(N, device=device, dtype=torch.long)
        z1 = torch.zeros(N, 1, device=device, dtype=rho_flat.dtype)

        diags = None
        if return_surface_diags:
            ra = rho_flat[~is_border.reshape(N)]
            rc = cf_out["rho_assoc"].reshape(-1)[~is_border.reshape(N)]
            diags = {
                "rho_assoc": cf_out["rho_assoc"].detach().cpu().numpy(),
                "iter_stats": [{
                    "rho_mean": float(rho_flat.mean().detach()),
                    "rho_max": float(rho_flat.max().detach()),
                    "mid_band_frac": float(
                        ((ra > 0.3) & (ra < 0.7)).float().mean().detach()
                    ) if ra.numel() else 0.0,
                    "n_interior": int(ok.sum().item()),
                    "fac_mean": float(rc.mean().detach()) if rc.numel() else 0.0,
                    "assoc_mean": float(rc.mean().detach()) if rc.numel() else 0.0,
                    "sur_mean": float(cf_out["sur"].mean().detach()),
                }],
            }

        return rho_flat, branch, rho_flat, z1, z1, cf_out, diags


AndGateSeed = ContourSeed
CellSeed = ContourSeed