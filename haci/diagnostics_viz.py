r"""Shared matplotlib diagnostics: reusable plots and infer.py figure helpers."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image

from params import L0, VIZ


def rho_heatmap_cmap():
    cdict = {
        "red": [(0.0, 0.0, 0.0), (0.33, 1.0, 1.0), (0.66, 1.0, 1.0), (1.0, 1.0, 1.0)],
        "green": [(0.0, 0.0, 0.0), (0.33, 0.0, 0.0), (0.66, 1.0, 1.0), (1.0, 1.0, 1.0)],
        "blue": [(0.0, 0.0, 0.0), (0.33, 0.0, 0.0), (0.66, 0.0, 0.0), (1.0, 1.0, 1.0)],
    }
    return mcolors.LinearSegmentedColormap("rho_bryw", cdict)


def apply_border_zero(g, is_border):
    out = np.asarray(g, dtype=np.float64).copy()
    b = np.asarray(is_border, dtype=bool)
    if out.shape != b.shape:
        return out
    out[b] = 0.0
    return out


def viz_l0_pinwheel(h, img, out_path):

    H, W, N = h.shape
    fig, axes = plt.subplots(3, 3, figsize=(18, 18), facecolor=VIZ.BG)
    fig.suptitle(
        "L0 pinwheel (directional contrast)",
        fontsize=10,
        color=VIZ.FG,
        fontfamily="monospace",
    )
    for ax in axes.ravel():
        ax.set_facecolor(VIZ.PANEL_BG)
        ax.axis("off")
    axes[1, 1].imshow(np.clip(img, 0.0, 1.0))
    axes[1, 1].set_title("input", fontsize=8, color=VIZ.FG, fontfamily="monospace")
    h_max = max(h.max(), VIZ.EPS)
    for k, (dy, dx) in enumerate(L0.OFFSETS):
        r, c = VIZ.GRID_POS[(dy, dx)]
        ax = axes[r, c]
        ax.imshow(h[:, :, k], cmap="hot", vmin=0, vmax=h_max)
        ax.set_title(
            f"h_{VIZ.COMPASS[(dy, dx)]} (k={k})",
            fontsize=8,
            color=VIZ.FG,
            fontfamily="monospace",
        )
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor=VIZ.BG)
    plt.close(fig)


def _interior_vmin_vmax(arr: np.ndarray, interior: np.ndarray) -> tuple[float, float]:
    if not interior.any():
        return 0.0, 1.0
    v = arr[interior]
    vmin = float(np.min(v))
    vmax = float(np.max(v))
    if vmin == vmax:
        vmax = vmin + max(abs(vmin), 1.0) * 1e-12 + 1e-20
    return vmin, vmax


def viz_l1_rho_masses(
    panels: list[tuple[np.ndarray, str]],
    is_border: np.ndarray,
    out_path: str,
    suptitle: str,
) -> None:

    cmap = "coolwarm"
    interior = ~np.asarray(is_border, dtype=bool)
    n = len(panels)
    fig_h = max(5.2 * n, 5.2)

    fig, axes = plt.subplots(n, 1, figsize=(5.5, fig_h), facecolor=VIZ.BG)
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, (panel, label) in zip(axes_flat, panels):
        arr = apply_border_zero(panel, is_border)
        ax.set_facecolor(VIZ.PANEL_BG)
        ax.axis("off")
        vmin, vmax = _interior_vmin_vmax(arr, interior)
        im = ax.imshow(
            arr,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(
            f"{label}\ninterior min={vmin:.4g}  max={vmax:.4g}",
            fontsize=9,
            color=VIZ.FG,
            fontfamily="monospace",
        )
        cbar = fig.colorbar(
            im,
            ax=ax,
            fraction=0.035,
            pad=0.02,
            shrink=0.82,
        )
        cbar.outline.set_edgecolor(VIZ.ACCENT)
        cbar.ax.tick_params(colors=VIZ.FG, labelsize=7)

    fig.suptitle(
        f"{suptitle}  (each row: own min→max; red=max, blue=min)",
        fontsize=10,
        color=VIZ.FG,
        fontfamily="monospace",
        y=0.995,
    )
    fig.text(
        0.5,
        0.008,
        "Legend: coolwarm on interior cells — blue = minimum, red = maximum "
        "(separate scale per panel; border cells masked).",
        ha="center",
        va="bottom",
        fontsize=8,
        color=VIZ.FG,
        fontfamily="monospace",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor=VIZ.BG)
    plt.close(fig)


def save_rho_png(pix_proj: np.ndarray, out_path):

    m = max(float(pix_proj.max()), VIZ.EPS)
    cmap = rho_heatmap_cmap()
    rgba = cmap(np.clip(pix_proj / m, 0.0, 1.0))
    Image.fromarray((rgba[..., :3] * 255).astype(np.uint8), "RGB").save(out_path)


def viz_infer_l0_pinwheel(
    h_np: np.ndarray, img_pinwheel: np.ndarray, out_path: str
) -> None:

    viz_l0_pinwheel(h_np, img_pinwheel, out_path)


def viz_infer_h2m(
    h2m_lum: np.ndarray,
    h2m_chr: np.ndarray,
    h2m: np.ndarray,
    out_path: str,
) -> None:
    panels = [
        (np.asarray(h2m_lum, dtype=np.float64), r"$h_{2m}^{\mathrm{lum}}$"),
        (np.asarray(h2m_chr, dtype=np.float64), r"$h_{2m}^{\mathrm{chr}}$"),
        (np.asarray(h2m, dtype=np.float64), r"$h_{2m}$"),
    ]
    cmap = rho_heatmap_cmap()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=VIZ.BG)
    for ax, (arr, label) in zip(axes, panels):
        ax.set_facecolor(VIZ.PANEL_BG)
        ax.axis("off")
        m = max(float(np.max(arr)), VIZ.EPS)
        ax.imshow(
            np.clip(arr / m, 0.0, 1.0),
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        ax.set_title(
            f"{label}\nraw max={m:.4g}",
            fontsize=9,
            color=VIZ.FG,
            fontfamily="monospace",
        )
    fig.suptitle(
        r"L0 $h_{2m}$ — luminance, chrominance, combined (per-panel max → 0…1)",
        fontsize=10,
        color=VIZ.FG,
        fontfamily="monospace",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor=VIZ.BG)
    plt.close(fig)


def viz_infer_l1_rho_masses(
    rho_total: np.ndarray,
    rho_peak: np.ndarray,
    is_border: np.ndarray,
    out_path: str,
) -> None:

    viz_l1_rho_masses(
        [
            (rho_total, r"$\rho_{\mathrm{total}} = \sum|z_2|$"),
            (rho_peak, r"$|Z_2| = \left|\sum z_2\right|$"),
        ],
        is_border,
        out_path,
        suptitle=r"L1 z₂ moments ($\rho_{\mathrm{total}}$, $|Z_2|$ per cell)",
    )


def viz_infer_rho(
    rho_after_nr: np.ndarray,
    rho_after_suppress: np.ndarray,
    is_border: np.ndarray,
    out_path: str,
    *,
    n_bins: int = 50,
) -> None:
    x_max = 1.0

    def _prep(g: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        gg = apply_border_zero(np.asarray(g, dtype=np.float64), is_border)
        ib_g = np.asarray(is_border, dtype=bool)
        if ib_g.ndim == 1 and gg.ndim == 2:
            ib_g = ib_g.reshape(gg.shape)
        flat = gg.ravel()[~ib_g.ravel()]
        return gg, flat, int(flat.size)

    rows = [
        (
            rho_after_nr,
            r"row 1 — $\rho_{\mathrm{NR}} = |Z|^2/(|Z|^2+\eta_z^2)$ (after $\eta_z$ NR)",
        ),
        (
            rho_after_suppress,
            r"row 2 — $\rho$ after association field + surround ($e^2/(e^2+\eta_{\mathrm{readout}}^2+\lambda S^2+\lambda_f F^2)$; splat input)",
        ),
    ]

    fig = plt.figure(figsize=(14.0, 9.2), facecolor=VIZ.BG)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], width_ratios=[2.4, 1.0, 1.0], hspace=0.38, wspace=0.32)
    cmap = rho_heatmap_cmap()

    for ri, (rho_g, row_title) in enumerate(rows):
        g, flat, n_cells = _prep(rho_g)
        m = max(float(np.max(g)), VIZ.EPS)

        ax_m = fig.add_subplot(gs[ri, 0])
        ax_h = fig.add_subplot(gs[ri, 1])
        ax_c = fig.add_subplot(gs[ri, 2])

        ax_m.set_facecolor(VIZ.PANEL_BG)
        ax_m.imshow(
            np.clip(g / m, 0.0, 1.0),
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        ax_m.set_title(
            rf"{row_title} — map  raw max={m:.4g}",
            fontsize=9,
            color=VIZ.FG,
            fontfamily="monospace",
        )
        ax_m.axis("off")

        ax_h.set_facecolor(VIZ.PANEL_BG)
        if n_cells > 0:
            ax_h.hist(
                flat,
                bins=n_bins,
                range=(0.0, x_max),
                color=VIZ.ACCENT,
                edgecolor=VIZ.PANEL_BG,
                linewidth=0.3,
            )
        ax_h.set_xlim(0.0, x_max)
        ax_h.set_title(
            rf"interior histogram  n={n_cells}",
            fontsize=9,
            color=VIZ.FG,
            fontfamily="monospace",
        )
        ax_h.set_xlabel(r"$\rho$", fontsize=8, color=VIZ.FG)
        ax_h.set_ylabel("count", fontsize=8, color=VIZ.FG)
        ax_h.tick_params(colors=VIZ.FG, labelsize=7)
        for s in ax_h.spines.values():
            s.set_color(VIZ.ACCENT)

        ax_c.set_facecolor(VIZ.PANEL_BG)
        if n_cells > 0:
            xs = np.sort(flat.astype(np.float64, copy=False))
            ys = np.arange(1, n_cells + 1, dtype=np.float64)
            ax_c.plot(xs, ys, color=VIZ.FG, linewidth=1.2)
        ax_c.set_xlim(0.0, x_max)
        ax_c.set_ylim(0.0, float(max(n_cells, 1)))
        ax_c.set_title("empirical CDF", fontsize=9, color=VIZ.FG, fontfamily="monospace")
        ax_c.set_xlabel(r"$\rho$", fontsize=8, color=VIZ.FG)
        ax_c.set_ylabel("cumulative count", fontsize=8, color=VIZ.FG)
        ax_c.grid(True, alpha=0.25, color=VIZ.ACCENT)
        ax_c.tick_params(colors=VIZ.FG, labelsize=7)
        for s in ax_c.spines.values():
            s.set_color(VIZ.ACCENT)

    fig.suptitle(
        r"Cell $\rho$ — NR on $|Z|$ then association field + surround suppression",
        fontsize=10,
        color=VIZ.FG,
        fontfamily="monospace",
    )
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor=VIZ.BG)
    plt.close(fig)


def viz_infer_geometry(
    rho_assoc: np.ndarray,
    surround: np.ndarray,
    is_border: np.ndarray,
    out_path: str,
    *,
    n_bins: int = 50,
) -> None:
    eps = float(VIZ.EPS)

    def _prep(g: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, float]:
        gg = apply_border_zero(np.asarray(g, dtype=np.float64), is_border)
        ib_g = np.asarray(is_border, dtype=bool)
        if ib_g.ndim == 1 and gg.ndim == 2:
            ib_g = ib_g.reshape(gg.shape)
        flat = gg.ravel()[~ib_g.ravel()]
        n_cells = int(flat.size)
        if n_cells > 0:
            x_max = max(float(np.max(flat)), eps)
        else:
            x_max = 1.0
        return gg, flat, n_cells, x_max

    rows = [
        (
            rho_assoc,
            r"row 1 — $\rho_{\mathrm{assoc}}$ (association-field support)",
            r"$\rho_{\mathrm{assoc}}$",
        ),
        (
            surround,
            r"row 2 — $S$ (same-orientation surround pool; divisive term)",
            r"$S$",
        ),
    ]

    fig = plt.figure(figsize=(14.0, 9.2), facecolor=VIZ.BG)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], width_ratios=[2.4, 1.0, 1.0], hspace=0.38, wspace=0.32)
    cmap = rho_heatmap_cmap()

    for ri, (rho_g, row_title, xlabel) in enumerate(rows):
        g, flat, n_cells, x_max = _prep(rho_g)
        m = max(float(np.max(g)), eps)

        ax_m = fig.add_subplot(gs[ri, 0])
        ax_h = fig.add_subplot(gs[ri, 1])
        ax_c = fig.add_subplot(gs[ri, 2])

        ax_m.set_facecolor(VIZ.PANEL_BG)
        ax_m.imshow(
            np.clip(g / m, 0.0, 1.0),
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        ax_m.set_title(
            rf"{row_title} — map  raw max={m:.4g}",
            fontsize=9,
            color=VIZ.FG,
            fontfamily="monospace",
        )
        ax_m.axis("off")

        ax_h.set_facecolor(VIZ.PANEL_BG)
        if n_cells > 0:
            ax_h.hist(
                flat,
                bins=n_bins,
                range=(0.0, x_max),
                color=VIZ.ACCENT,
                edgecolor=VIZ.PANEL_BG,
                linewidth=0.3,
            )
        ax_h.set_xlim(0.0, x_max)
        ax_h.set_title(
            rf"interior histogram  n={n_cells}",
            fontsize=9,
            color=VIZ.FG,
            fontfamily="monospace",
        )
        ax_h.set_xlabel(xlabel, fontsize=8, color=VIZ.FG)
        ax_h.set_ylabel("count", fontsize=8, color=VIZ.FG)
        ax_h.tick_params(colors=VIZ.FG, labelsize=7)
        for s in ax_h.spines.values():
            s.set_color(VIZ.ACCENT)

        ax_c.set_facecolor(VIZ.PANEL_BG)
        if n_cells > 0:
            xs = np.sort(flat.astype(np.float64, copy=False))
            ys = np.arange(1, n_cells + 1, dtype=np.float64)
            ax_c.plot(xs, ys, color=VIZ.FG, linewidth=1.2)
        ax_c.set_xlim(0.0, x_max)
        ax_c.set_ylim(0.0, float(max(n_cells, 1)))
        ax_c.set_title("empirical CDF", fontsize=9, color=VIZ.FG, fontfamily="monospace")
        ax_c.set_xlabel(xlabel, fontsize=8, color=VIZ.FG)
        ax_c.set_ylabel("cumulative count", fontsize=8, color=VIZ.FG)
        ax_c.grid(True, alpha=0.25, color=VIZ.ACCENT)
        ax_c.tick_params(colors=VIZ.FG, labelsize=7)
        for s in ax_c.spines.values():
            s.set_color(VIZ.ACCENT)

    fig.suptitle(
        r"Seed geometry — association-field support $\rho_{\mathrm{assoc}}$ and same-orientation surround $S$",
        fontsize=10,
        color=VIZ.FG,
        fontfamily="monospace",
    )
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor=VIZ.BG)
    plt.close(fig)


def viz_infer_base_edges_overlay(
    base_rgb: np.ndarray,
    edges_u8: np.ndarray,
    out_path: str,
    *,
    edge_rgb: tuple[float, float, float] = (0.0, 1.0, 0.95),
    edge_weight: float = 0.88,
) -> None:

    base = np.clip(np.asarray(base_rgb, dtype=np.float32), 0.0, 1.0)
    if base.ndim != 3 or base.shape[2] != 3:
        raise ValueError("base_rgb must be H×W×3")
    em = np.asarray(edges_u8)
    if em.ndim == 3 and em.shape[2] == 1:
        em = em[..., 0]
    mask = em.astype(np.float32) >= 127.5
    ec = np.array(edge_rgb, dtype=np.float32).reshape(1, 1, 3)
    w = float(np.clip(edge_weight, 0.0, 1.0))
    out = base.copy()
    if mask.any():
        m = mask[..., np.newaxis]
        out = np.where(m, (1.0 - w) * base + w * ec, base)
    Image.fromarray((np.clip(out, 0.0, 1.0) * 255).astype(np.uint8), "RGB").save(
        out_path
    )


def viz_infer_shape_readout(
    theta_map: np.ndarray,
    bmap: np.ndarray,
    *,
    threshold: float,
    theta_bins: int = 12,
    out_path: str,
) -> None:

    th = np.asarray(theta_map, dtype=np.float32)
    edge_mask = np.asarray(bmap, dtype=np.float32) >= float(threshold)
    valid = edge_mask & np.isfinite(th)
    rgb = np.zeros((*th.shape, 3), dtype=np.float32)
    if valid.any():
        n_bins = max(2, int(theta_bins))

        th_mod = np.mod(th[valid], np.pi)
        bidx = np.floor((th_mod / np.pi) * n_bins).astype(np.int64)
        bidx = np.clip(bidx, 0, n_bins - 1)
        palette = plt.get_cmap("tab20", n_bins)(np.arange(n_bins))[..., :3]
        rgb_valid = palette[bidx]
        rgb[valid] = rgb_valid.astype(np.float32)
    Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8), "RGB").save(
        out_path
    )
