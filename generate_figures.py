"""Generate schematic and experimental figures (PDF and EPS) from recorded JSON results."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.path import Path as MplPath
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
import numpy as np
from scipy.stats import friedmanchisquare, studentized_range

ROOT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ROOT_DIR / "results"
FIG_DIR = ROOT_DIR / "figures"


def _blend(fg: str, alpha: float, bg: str = "#FFFFFF") -> Tuple[float, float, float]:
    """Composite fg over bg with alpha for vector EPS compatibility."""
    f = np.asarray(to_rgb(fg), dtype=float)
    b = np.asarray(to_rgb(bg), dtype=float)
    c = alpha * f + (1.0 - alpha) * b
    return (float(c[0]), float(c[1]), float(c[2]))


def _save_fig(fig: plt.Figure, pdf_name: str, eps_index: int) -> None:
    """Save PDF and EPS figures into figures/."""
    fig.savefig(FIG_DIR / pdf_name)
    fig.savefig(FIG_DIR / f"{eps_index}.eps", format="eps")


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
    "font.size": 9.0,
    "axes.titlesize": 9.6,
    "axes.labelsize": 9.0,
    "xtick.labelsize": 8.2,
    "ytick.labelsize": 8.2,
    "legend.fontsize": 7.8,
    "legend.framealpha": 1.0,
    "legend.facecolor": "#FFFFFF",
    "figure.dpi": 300,
    "savefig.bbox": "tight",
})

DATASETS_ORDER = [
    ("electricity", "Electricity"),
    ("solar", "Solar-Energy"),
    ("uci501", "Beijing-Air"),
    ("energy", "EnergyData"),
    ("sml", "SML2010"),
    ("airquality", "AirQuality"),
    ("etth1", "ETTh1"),
    ("etth2", "ETTh2"),
]

MAIN_METHODS = [
    "FGB-IPN (Full)",
    "ECQR",
    "MQR-NN",
    "ACHF",
    "CAFHT",
    "DAQD-NN",
    "MSDIPN",
]

SHORT_NAMES = {
    "FGB-IPN (Full)": "FGB-IPN",
    "ECQR": "ECQR",
    "MQR-NN": "MQR-NN",
    "ACHF": "ACHF",
    "CAFHT": "CAFHT",
    "DAQD-NN": "DAQD-NN",
    "MSDIPN": "MSDIPN",
    "FGB-NoPsi-TSK-ACI": "FGB-NoPsi",
    "SoftStat-GB-IPN": "SoftStat-GB",
    "Hard-GB-IPN": "Hard-GB",
    "VarCluster-TSK-ACI": "VarCluster",
    "TopM-Select-TSK-ACI": "TopM-Sel",
    "PCA-TSK-ACI": "PCA-TSK",
    "Raw-AllDim-TSK-ACI": "Raw-AllDim",
    "FGB-Global-ACI": "FGB-Global",
    "FGB-TSK-Static": "FGB-Static",
}


def _rbox(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    fc: str = "#FFFFFF",
    ec: str = "#333333",
    lw: float = 1.1,
    r: float = 0.08,
    ls: str = "-",
    alpha: float = 1.0,
    zorder: int = 1,
):
    """Draw a rounded rectangle whose visual outer bounds match (x, y, w, h) closely."""
    pad = 0.01
    p = FancyBboxPatch(
        (x + pad, y + pad),
        w - 2 * pad,
        h - 2 * pad,
        boxstyle=f"round,pad={pad},rounding_size={r}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        linestyle=ls,
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(p)
    return p


def _ortho_arrow(
    ax,
    pts: Sequence[Tuple[float, float]],
    col: str = "#2C3E50",
    lw: float = 1.2,
    ls: str = "-",
    ms: float = 9.5,
    zorder: int = 5,
) -> None:
    """Draw a strictly straight (horizontal/vertical) single or multi-segment orthogonal arrow."""
    assert len(pts) >= 2
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        assert abs(x0 - x1) < 1e-6 or abs(y0 - y1) < 1e-6, (
            f"Non-orthogonal segment: {(x0, y0)} -> {(x1, y1)}"
        )

    if len(pts) == 2:
        p = FancyArrowPatch(
            pts[0],
            pts[1],
            arrowstyle="-|>",
            mutation_scale=ms,
            linewidth=lw,
            linestyle=ls,
            color=col,
            shrinkA=0.0,
            shrinkB=0.0,
            zorder=zorder,
        )
        ax.add_patch(p)
    else:
        codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(pts) - 1)
        path = MplPath(pts, codes)
        p = FancyArrowPatch(
            path=path,
            arrowstyle="-|>",
            mutation_scale=ms,
            linewidth=lw,
            linestyle=ls,
            color=col,
            shrinkA=0.0,
            shrinkB=0.0,
            zorder=zorder,
        )
        ax.add_patch(p)


def draw_fig1_framework() -> None:
    """Draw a 100% planar, strictly orthogonal, zero-overlap structural schematic of FGB-IPN."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14.1, 5.15))
    ax.set_xlim(0.0, 14.1)
    ax.set_ylim(0.0, 5.15)
    ax.axis("off")

    # Stage headers at top (y = 4.92), leaving y = 4.58 as a clean planar corridor
    # for the Target Linear Skip connection into (+).
    y_box_bot, y_box_h = 0.18, 4.18  # Stage outer boxes span y: 0.18 .. 4.36

    # ==========================================================================
    # 0. INPUT WINDOW (x: 0.14 .. 1.94, y: 0.18 .. 4.36)
    # Top = Target y_t (connects cleanly via top corridor y=4.58 -> (+));
    # Bottom = Exogenous Sensors X_t (connects cleanly right into Stage 1).
    # ==========================================================================
    _rbox(ax, 0.14, y_box_bot, 1.80, y_box_h, fc="#F8F9F9", ec="#566573", lw=1.25, r=0.09, zorder=1)
    ax.text(1.04, 4.86, "Input Window", ha="center", va="bottom",
            fontsize=9.3, fontweight="bold", color="#1C2833")

    # 0a. TOP: Target series box (x: 0.26 .. 1.82, y: 3.16 .. 4.22)
    _rbox(ax, 0.26, 3.16, 1.56, 1.06, fc="#FEF9E7", ec="#B7950B", lw=0.95, r=0.06, zorder=2)
    ax.text(1.04, 4.10, "Target $\\mathbf{y}_t \\in \\mathbb{R}^L$",
            ha="center", va="top", fontsize=7.7, fontweight="bold", color="#7D6608")
    t_y = np.linspace(0.36, 1.72, 80)
    y_wave = 0.16 * np.sin((t_y - 0.36) * 6.0) + 0.05 * np.cos((t_y - 0.36) * 13.5)
    ax.plot(t_y, 3.54 + y_wave, color="#B7950B", lw=1.35, zorder=3)

    # 0b. BOTTOM: Exogenous sensor panel (x: 0.26 .. 1.82, y: 0.32 .. 2.98)
    _rbox(ax, 0.26, 0.32, 1.56, 2.66, fc="#FFFFFF", ec="#ABB2B9", lw=0.9, r=0.06, zorder=2)
    ax.text(1.04, 2.86, "Sensors $\\mathbf{X}_t \\in \\mathbb{R}^{d \\times L}$",
            ha="center", va="top", fontsize=7.8, fontweight="bold", color="#1C2833")

    t_grid = np.linspace(0.36, 1.72, 85)
    base_y_pos = [2.32, 1.78, 1.08]
    cols_in = ["#2874A6", "#148F77", "#C0392B"]
    for idx, (by, col) in enumerate(zip(base_y_pos, cols_in)):
        wave = 0.11 * np.sin((t_grid - 0.36) * (6.8 + idx * 1.4) + idx)
        if idx == 2:
            spike = np.zeros_like(t_grid)
            spike[44] = 0.34  # Peak at y ~ 1.42 (well below green wave min 1.67)
            spike[45] = -0.20
            wave = wave + spike
        ax.plot(t_grid, by + wave, color=col, lw=1.15, zorder=3)

    spike_x = float(t_grid[44])
    spike_y = 1.40
    _ortho_arrow(ax, [(spike_x + 0.30, spike_y), (spike_x + 0.05, spike_y)],
                 col="#C0392B", lw=0.9, ms=7.0, zorder=6)
    ax.text(spike_x + 0.33, spike_y, "spike", ha="left", va="center",
            fontsize=6.9, fontweight="bold", color="#C0392B")
    ax.text(1.04, 0.52, "$\\vdots$  ($d$ noisy channels)", ha="center", va="center",
            fontsize=7.1, color="#566573")

    # ==========================================================================
    # 1. STAGE 1: DYNAMIC FUZZY GRANULAR-BALL (x: 2.34 .. 5.92, y: 0.18 .. 4.36)
    # ==========================================================================
    _rbox(ax, 2.34, y_box_bot, 3.58, y_box_h, fc="#EBF5FB", ec="#1F618D", lw=1.35, r=0.09, zorder=1)
    ax.text(4.13, 4.86, "Stage 1: Dynamic Fuzzy Granular-Ball", ha="center", va="bottom",
            fontsize=9.3, fontweight="bold", color="#154360")

    # Orthogonal bus from Sensors X_t (1.82, 2.00) -> Stage 1 rFFT+Coh (y=3.10) & Cauchy (y=0.96)
    ax.plot([1.82, 2.12], [2.00, 2.00], color="#2C3E50", lw=1.2, zorder=4)
    ax.add_patch(Circle((2.12, 2.00), 0.032, facecolor="#2C3E50", edgecolor="none", zorder=6))
    _ortho_arrow(ax, [(2.12, 2.00), (2.12, 3.10), (2.48, 3.10)], col="#2C3E50", lw=1.2, ms=8.5)
    _ortho_arrow(ax, [(2.12, 2.00), (2.12, 0.96), (2.48, 0.96)], col="#2C3E50", lw=1.15, ms=8.5)

    # 1a. rFFT + Coh mini-panel (x: 2.48 .. 3.46, y: 1.94 .. 4.22)
    _rbox(ax, 2.48, 1.94, 0.98, 2.28, fc="#FFFFFF", ec="#5DADE2", lw=0.95, r=0.06, zorder=2)
    ax.text(2.97, 4.10, "rFFT + Coh", ha="center", va="top",
            fontsize=7.5, fontweight="bold", color="#1F618D")
    bar_x = np.linspace(2.61, 3.33, 6)
    bar_h = [0.82, 0.62, 0.46, 0.31, 0.21, 0.13]
    for bx, bh in zip(bar_x, bar_h):
        ax.add_patch(Rectangle((bx - 0.04, 2.58), 0.08, bh,
                               facecolor="#3498DB", edgecolor="#1F618D", lw=0.55, zorder=3))
    ax.plot([2.55, 3.39], [2.58, 2.58], color="#1F618D", lw=0.7, zorder=3)
    ax.text(2.97, 2.24, "$\\mathbf{v}_j,\\; \\mathrm{Rel}_j$", ha="center", va="center",
            fontsize=7.6, color="#1B4F72")

    _ortho_arrow(ax, [(3.46, 3.10), (3.74, 3.10)], col="#1F618D", lw=1.15, ms=8.5)

    # 1b. JSG Fuzzy Balls 2D visual panel (x: 3.74 .. 5.78, y: 1.94 .. 4.22)
    _rbox(ax, 3.74, 1.94, 2.04, 2.28, fc="#FFFFFF", ec="#2E86C1", lw=1.0, r=0.06, zorder=2)
    ax.text(4.76, 4.10, "JSG Fuzzy Balls ($d \\to M$)", ha="center", va="top",
            fontsize=7.5, fontweight="bold", color="#154360")

    balls_spec = [
        (4.28, 3.18, 0.30, "#3498DB"),
        (5.02, 3.22, 0.28, "#1ABC9C"),
        (4.60, 2.58, 0.29, "#9B59B6"),
    ]
    rng = np.random.default_rng(7)
    for cx, cy, rad, bcol in balls_spec:
        ax.add_patch(Circle((cx, cy), rad * 1.18, facecolor=_blend(bcol, 0.14), edgecolor=_blend(bcol, 0.65),
                            ls="--", lw=0.8, zorder=2.4))
        ax.add_patch(Circle((cx, cy), rad * 0.82, facecolor=_blend(bcol, 0.26), edgecolor=bcol,
                            lw=1.0, zorder=2.7))
        ax.scatter([cx], [cy], color=bcol, s=20, marker="x", linewidths=1.3, zorder=5)
        pts = rng.normal(loc=[cx, cy], scale=rad * 0.34, size=(6, 2))
        ax.scatter(pts[:, 0], pts[:, 1], color=_blend(bcol, 0.85), s=10, zorder=4)

    ax.text(4.04, 3.72, "$\\mu_{j,m}^{\\mathrm{spec}}$", ha="center", va="center",
            fontsize=7.3, color="#1F618D")
    ax.scatter([5.45], [2.52], color="#C0392B", s=28, marker="^", zorder=5)
    ax.text(5.41, 2.18, "$\\mu_{t,j}^{\\mathrm{dyn}}\\!\\to\\!0$", ha="center", va="center",
            fontsize=6.9, color="#C0392B")

    # 1c. Cauchy Consistency block (x: 2.48 .. 3.94, y: 0.32 .. 1.64)
    _rbox(ax, 2.48, 0.32, 1.46, 1.32, fc="#FFFFFF", ec="#2E86C1", lw=0.95, r=0.06, zorder=2)
    ax.text(3.21, 1.53, "Cauchy Consistency", ha="center", va="top",
            fontsize=7.2, fontweight="bold", color="#154360")
    cx_grid = np.linspace(-2.5, 2.5, 65)
    cauchy_y = 1.0 / (1.0 + cx_grid ** 2)
    ax.plot([2.62, 3.80], [0.72, 0.72], color="#BDC3C7", lw=0.65, zorder=3)
    ax.plot(2.66 + (cx_grid + 2.5) * 0.22, 0.72 + cauchy_y * 0.50,
            color="#C0392B", lw=1.35, zorder=4)
    ax.text(3.21, 0.50, "$\\mu_{t,j}^{\\mathrm{dyn}}=(1+(e/\\kappa)^2)^{-1}$",
            ha="center", va="center", fontsize=6.6, color="#1C2833")

    # 1d & 1e. Two DFGB outputs (x: 4.30 .. 5.78)
    _rbox(ax, 4.30, 1.08, 1.48, 0.56, fc="#D6EAF8", ec="#1F618D", lw=0.95, r=0.05, zorder=2)
    ax.text(5.04, 1.36, "Prototypes $\\tilde{\\mathbf{X}}_t \\in \\mathbb{R}^{M \\times L}$",
            ha="center", va="center", fontsize=7.2, fontweight="bold", color="#154360")

    _rbox(ax, 4.30, 0.32, 1.48, 0.56, fc="#FDEDEC", ec="#C0392B", lw=0.95, r=0.05, zorder=2)
    ax.text(5.04, 0.60, "Inconsist. $\\psi_t^{(m)} \\in [0, 1)$",
            ha="center", va="center", fontsize=7.1, fontweight="bold", color="#922B21")

    _ortho_arrow(ax, [(5.04, 1.94), (5.04, 1.64)], col="#1F618D", lw=1.15, ms=8.5)
    _ortho_arrow(ax, [(3.94, 0.96), (4.12, 0.96), (4.12, 1.36), (4.30, 1.36)],
                 col="#2E86C1", lw=1.05, ms=8.0)
    _ortho_arrow(ax, [(4.12, 0.96), (4.12, 0.60), (4.30, 0.60)],
                 col="#C0392B", lw=1.05, ms=8.0)
    ax.add_patch(Circle((4.12, 0.96), 0.03, facecolor="#2C3E50", edgecolor="none", zorder=6))

    # ==========================================================================
    # 2. STAGE 2: SPECTRAL-TSK NETWORK (x: 6.40 .. 9.96, y: 0.18 .. 4.36)
    # Both sub-panels aligned cleanly at x: 6.54 .. 9.82!
    # ==========================================================================
    _rbox(ax, 6.40, y_box_bot, 3.56, y_box_h, fc="#FEF9E7", ec="#B7950B", lw=1.35, r=0.09, zorder=1)
    ax.text(8.18, 4.86, "Stage 2: Spectral-TSK Network", ha="center", va="bottom",
            fontsize=9.3, fontweight="bold", color="#7D6608")

    # 2a. Upper Sub-panel: Full-Spectrum Complex Mixing (x: 6.54 .. 9.82, y: 2.48 .. 4.22)
    _rbox(ax, 6.54, 2.48, 3.28, 1.74, fc="#FFFFFF", ec="#D4AC0D", lw=1.0, r=0.06, zorder=2)
    ax.text(7.96, 4.10, "Full-Spectrum Complex Mixing ($O(M)$)",
            ha="center", va="top", fontsize=7.5, fontweight="bold", color="#7D6608")

    # Dynamic Inconsistency Gate circle g_{t,m} at (6.88, 3.34)
    ax.add_patch(Circle((6.88, 3.34), 0.17, facecolor="#FCF3CF", edgecolor="#B7950B", lw=1.05, zorder=4))
    ax.text(6.88, 3.34, "$g_{t,m}$", ha="center", va="center",
            fontsize=7.1, fontweight="bold", color="#7D6608", zorder=5)

    # Complex filter 4x4 colored matrix (x: 7.24 .. 7.80, y: 3.06 .. 3.62)
    gx0, gy0, cs = 7.24, 3.06, 0.14
    grid_cols = [
        ["#2980B9", "#5DADE2", "#AED6F1", "#E67E22"],
        ["#5DADE2", "#1ABC9C", "#F4D03F", "#EB984E"],
        ["#85C1E9", "#48C9B0", "#2ECC71", "#F5B041"],
        ["#AED6F1", "#A3E4D7", "#82E0AA", "#FAD7A0"],
    ]
    for ri in range(4):
        for ci in range(4):
            ax.add_patch(Rectangle((gx0 + ci * cs, gy0 + ri * cs), cs, cs,
                                   facecolor=grid_cols[ri][ci], edgecolor="#FFFFFF", lw=0.5, zorder=3))
    ax.text(gx0 + 2 * cs, 2.78, "$\\mathbf{W}^{\\mathrm{re}}+i\\mathbf{W}^{\\mathrm{im}}$",
            ha="center", va="center", fontsize=6.9, color="#1C2833")

    # 2-layer Spectral MLP graph (x: 8.14 .. 8.96)
    l1_y = [3.66, 3.45, 3.23, 3.02]
    l2_y = [3.54, 3.34, 3.14]
    x_l1, x_l2, x_out = 8.14, 8.56, 8.94
    for y1 in l1_y:
        for y2 in l2_y:
            ax.plot([x_l1, x_l2], [y1, y2], color="#BDC3C7", lw=0.65, zorder=3)
    for y2 in l2_y:
        ax.plot([x_l2, x_out], [y2, 3.34], color="#BDC3C7", lw=0.75, zorder=3)
    for y1 in l1_y:
        ax.add_patch(Circle((x_l1, y1), 0.058, facecolor="#5DADE2", edgecolor="#1B4F72", lw=0.7, zorder=4))
    for y2 in l2_y:
        ax.add_patch(Circle((x_l2, y2), 0.058, facecolor="#48C9B0", edgecolor="#0E6251", lw=0.7, zorder=4))
    ax.add_patch(Circle((x_out, 3.34), 0.068, facecolor="#F4D03F", edgecolor="#7D6608", lw=0.8, zorder=4))
    ax.text(8.54, 2.70, "Spectral MLP", ha="center", va="center", fontsize=7.0, color="#566573")

    # Sum (+) node at (9.48, 3.34) — receives Target Linear Skip from TOP (9.48, 4.58 -> 3.48)!
    ax.add_patch(Circle((9.48, 3.34), 0.14, facecolor="#EAFAF1", edgecolor="#1E8449", lw=1.05, zorder=4))
    ax.text(9.48, 3.34, "$+$", ha="center", va="center",
            fontsize=9.2, fontweight="bold", color="#1E8449", zorder=5)

    _ortho_arrow(ax, [(7.05, 3.34), (7.24, 3.34)], col="#2C3E50", lw=1.05, ms=8.0)
    _ortho_arrow(ax, [(7.80, 3.34), (8.07, 3.34)], col="#2C3E50", lw=1.05, ms=8.0)
    _ortho_arrow(ax, [(9.01, 3.34), (9.34, 3.34)], col="#2C3E50", lw=1.05, ms=8.0)

    # Dedicated TOP corridor for Target Linear Skip (y = 4.56, zero crossings anywhere!)
    _ortho_arrow(ax, [(1.04, 4.22), (1.04, 4.56), (9.48, 4.56), (9.48, 3.48)],
                 col="#B7950B", lw=1.15, ls="--", ms=8.5, zorder=4)
    ax.text(5.26, 4.60, "Target Linear Skip ($\\mathbf{y}_t$)",
            ha="center", va="bottom", fontsize=7.3, style="italic", color="#7D6608")

    # 2b. Lower Sub-panel: Ellipsoidal TSK Fuzzy Head (x: 6.54 .. 9.82, y: 0.32 .. 2.26)
    _rbox(ax, 6.54, 0.32, 3.28, 1.94, fc="#FFFFFF", ec="#D4AC0D", lw=1.0, r=0.06, zorder=2)
    ax.text(8.18, 2.15, "Ellipsoidal TSK Fuzzy Head ($R$ Rules)",
            ha="center", va="top", fontsize=7.5, fontweight="bold", color="#7D6608")

    # State vector u_t capsule (x: 6.72 .. 7.00, y: 0.78 .. 1.72)
    _rbox(ax, 6.72, 0.78, 0.28, 0.94, fc="#F5EEF8", ec="#8E44AD", lw=0.8, r=0.04, zorder=3)
    for uy in np.linspace(0.90, 1.60, 5):
        ax.add_patch(Circle((6.86, uy), 0.045, facecolor="#D2B4DE", edgecolor="#6C3483", lw=0.65, zorder=4))
    ax.text(6.86, 0.54, "$\\mathbf{u}_t$", ha="center", va="center",
            fontsize=7.6, fontweight="bold", color="#6C3483")

    # R=3 Fuzzy Granule Rule boxes (x: 7.36 .. 7.88, y_centers: 1.68, 1.25, 0.82)
    rule_ys = [1.68, 1.25, 0.82]
    gx = np.linspace(-2.0, 2.0, 35)
    gy_bell = np.exp(-0.5 * gx ** 2)

    ax.plot([7.00, 7.18], [1.25, 1.25], color="#8E44AD", lw=0.95, zorder=3)
    ax.plot([7.18, 7.18], [rule_ys[-1], rule_ys[0]], color="#8E44AD", lw=0.95, zorder=3)
    for ry in rule_ys:
        _ortho_arrow(ax, [(7.18, ry), (7.36, ry)], col="#8E44AD", lw=0.9, ms=7.0, zorder=4)
        _rbox(ax, 7.36, ry - 0.14, 0.52, 0.28, fc="#F5EEF8", ec="#7D3C98", lw=0.85, r=0.04, zorder=4)
        ax.plot(7.42 + (gx + 2.0) * 0.10, ry - 0.09 + gy_bell * 0.18,
                color="#8E44AD", lw=1.1, zorder=5)
    ax.text(7.62, 0.50, "Antecedents $w_r$", ha="center", va="center",
            fontsize=6.8, color="#6C3483")

    # Consequent blocks (x: 8.16 .. 8.86) + 3-to-1 collector bus -> Sigma at (9.42, 1.25)
    for r_i, ry in enumerate(rule_ys):
        _ortho_arrow(ax, [(7.88, ry), (8.16, ry)], col="#7D3C98", lw=0.9, ms=7.0, zorder=4)
        _rbox(ax, 8.16, ry - 0.13, 0.70, 0.26, fc="#EBF5FB", ec="#2471A3", lw=0.85, r=0.04, zorder=4)
        ax.text(8.51, ry, f"$s_{r_i+1}^{{\\pm}} > 0$", ha="center", va="center",
                fontsize=6.8, color="#1B4F72", zorder=5)
        ax.plot([8.86, 9.06], [ry, ry], color="#2471A3", lw=0.95, zorder=3)

    ax.plot([9.06, 9.06], [rule_ys[-1], rule_ys[0]], color="#2471A3", lw=0.95, zorder=3)
    _ortho_arrow(ax, [(9.06, 1.25), (9.25, 1.25)], col="#2471A3", lw=0.95, ms=7.5, zorder=4)

    ax.add_patch(Circle((9.42, 1.25), 0.17, facecolor="#FCF3CF", edgecolor="#B7950B", lw=1.05, zorder=4))
    ax.text(9.42, 1.25, "$\\Sigma$", ha="center", va="center",
            fontsize=9.2, fontweight="bold", color="#7D6608", zorder=5)

    # Planar non-crossing orthogonal bus between Stage 1 and Stage 2:
    _ortho_arrow(ax, [(5.78, 1.36), (6.08, 1.36), (6.08, 3.34), (6.71, 3.34)],
                 col="#1F618D", lw=1.25, ms=9.0, zorder=5)
    _ortho_arrow(ax, [(5.78, 0.60), (6.24, 0.60), (6.24, 2.98), (6.88, 2.98), (6.88, 3.17)],
                 col="#C0392B", lw=1.0, ls="--", ms=8.0, zorder=5)
    _ortho_arrow(ax, [(6.24, 1.25), (6.72, 1.25)], col="#C0392B", lw=1.2, ms=8.5, zorder=5)
    ax.add_patch(Circle((6.24, 1.25), 0.03, facecolor="#C0392B", edgecolor="none", zorder=6))

    # ==========================================================================
    # 3. STAGE 3: FGB-ACI CALIBRATION (x: 10.94 .. 13.96, y: 0.18 .. 4.36)
    # Wide gap x: 9.96 .. 10.94 gives generous clear space for \hat{y}_{t+h} & \hat{s}_{t+h}^\pm, w_t!
    # ==========================================================================
    _rbox(ax, 10.94, y_box_bot, 3.02, y_box_h, fc="#EAFAF1", ec="#1E8449", lw=1.35, r=0.09, zorder=1)
    ax.text(12.45, 4.86, "Stage 3: FGB-ACI Calibration", ha="center", va="bottom",
            fontsize=9.3, fontweight="bold", color="#145A32")

    # 3a. TOP: Calibrated Prediction Interval Fan Plot (x: 11.08 .. 13.82, y: 2.48 .. 4.22)
    _rbox(ax, 11.08, 2.48, 2.74, 1.74, fc="#FFFFFF", ec="#1E8449", lw=1.0, r=0.06, zorder=2)
    ax.text(12.45, 4.11, "Interval $\\hat{I}_{t+h} = [\\hat{L}_{t+h},\\, \\hat{U}_{t+h}]$",
            ha="center", va="top", fontsize=7.4, fontweight="bold", color="#145A32")

    tx_hist = np.linspace(11.22, 12.16, 30)
    ty_hist = 3.18 + 0.21 * np.sin((tx_hist - 11.22) * 6.8)
    tx_fut = np.linspace(12.16, 13.68, 42)
    ty_pred = 3.18 + 0.21 * np.sin((tx_fut - 11.22) * 6.8) + 0.06 * (tx_fut - 12.16)
    band_up = ty_pred + 0.14 + 0.13 * (tx_fut - 12.16)
    band_lo = ty_pred - (0.11 + 0.09 * (tx_fut - 12.16))

    poly_pts = np.vstack([
        np.column_stack([tx_fut, band_up]),
        np.column_stack([tx_fut[::-1], band_lo[::-1]]),
    ])
    ax.add_patch(Polygon(poly_pts, closed=True, facecolor=_blend("#82E0AA", 0.45), edgecolor="none", zorder=3))
    ax.plot(tx_fut, band_up, color="#1E8449", lw=0.9, ls="--", zorder=4)
    ax.plot(tx_fut, band_lo, color="#1E8449", lw=0.9, ls="--", zorder=4)
    ax.plot(tx_hist, ty_hist, color="#2C3E50", lw=1.3, zorder=4)
    ax.plot(tx_fut, ty_pred, color="#C0392B", lw=1.4, zorder=5)
    ax.plot([12.16, 12.16], [2.60, 3.86], color="#7F8C8D", lw=0.85, ls=":", zorder=4)

    obs_x = tx_fut[5::8]
    obs_y = ty_pred[5::8] + np.array([0.05, -0.04, 0.10, -0.05, 0.07])
    ax.scatter(obs_x, obs_y, color="#1C2833", s=9, zorder=6)

    # 3b. MIDDLE: Rule-Weighted Quantile Q_t (left) & Unclipped ACI Update (right)
    _rbox(ax, 11.08, 1.46, 1.26, 0.68, fc="#D4EFDF", ec="#1E8449", lw=0.95, r=0.05, zorder=2)
    ax.text(11.71, 1.80, "Rule-Weighted\nQuantile $Q_t$", ha="center", va="center",
            fontsize=7.0, fontweight="bold", color="#145A32")

    _rbox(ax, 12.56, 1.46, 1.26, 0.68, fc="#FADBD8", ec="#C0392B", lw=0.95, r=0.05, zorder=2)
    ax.text(13.19, 1.80, "Unclipped ACI\n$\\alpha_{t+1}$ Update", ha="center", va="center",
            fontsize=7.0, fontweight="bold", color="#922B21")

    # 3c. BOTTOM: Causal Score Buffer & Delay h (x: 11.08 .. 13.82, y: 0.32 .. 1.20)
    _rbox(ax, 11.08, 0.32, 2.74, 0.88, fc="#FFFFFF", ec="#27AE60", lw=0.95, r=0.06, zorder=2)
    ax.text(12.45, 1.11, "Causal Score Buffer ($E_\\tau$) & Delay $h$",
            ha="center", va="top", fontsize=7.1, fontweight="bold", color="#145A32")
    cell_x0 = 11.22
    for ci in range(7):
        is_realized = ci < 5
        fc_c = "#A9DFBF" if is_realized else "#E5E7E9"
        ec_c = "#1E8449" if is_realized else "#95A5A6"
        ax.add_patch(Rectangle((cell_x0 + ci * 0.355, 0.59), 0.315, 0.27,
                               facecolor=fc_c, edgecolor=ec_c, lw=0.8,
                               hatch="" if is_realized else "///", zorder=3))
    ax.text(11.92, 0.44, "Realized ($\\tau+h \\leq t$)", ha="center", va="center",
            fontsize=6.5, color="#145A32")
    ax.text(13.33, 0.44, "Gap $h$", ha="center", va="center",
            fontsize=6.5, color="#566573")

    # Stage 2 -> Stage 3 strictly straight planar orthogonal connectors (zero crossings, wide gap!):
    # 1) Point forecast \hat{y}_{t+h}: dead-straight horizontal at y = 3.34
    _ortho_arrow(ax, [(9.62, 3.34), (11.08, 3.34)], col="#1E8449", lw=1.25, ms=9.0, zorder=5)
    ax.text(10.45, 3.47, "$\\hat{y}_{t+h}$", ha="center", va="bottom",
            fontsize=8.0, fontweight="bold", color="#1C2833")

    # 2) Scales & rule weights from Sigma (9.59, 1.25):
    #    Label sits cleanly in open gap x in [10.02, 10.70], vertical spine at x = 10.74!
    ax.plot([9.59, 10.74], [1.25, 1.25], color="#7D3C98", lw=1.2, zorder=4)
    ax.text(10.35, 1.37, "$\\hat{s}_{t+h}^{\\pm},\\, \\mathbf{w}_t$", ha="center", va="bottom",
            fontsize=7.6, fontweight="bold", color="#6C3483")
    _ortho_arrow(ax, [(10.74, 1.25), (10.74, 2.86), (11.08, 2.86)],
                 col="#7D3C98", lw=1.15, ms=8.5, zorder=5)
    _ortho_arrow(ax, [(10.74, 1.80), (11.08, 1.80)],
                 col="#7D3C98", lw=1.15, ms=8.5, zorder=5)
    _ortho_arrow(ax, [(10.74, 1.25), (10.74, 0.76), (11.08, 0.76)],
                 col="#7D3C98", lw=1.1, ms=8.5, zorder=5)
    ax.add_patch(Circle((10.74, 1.25), 0.03, facecolor="#7D3C98", edgecolor="none", zorder=6))
    ax.add_patch(Circle((10.74, 1.80), 0.03, facecolor="#7D3C98", edgecolor="none", zorder=6))

    # Stage 3 internal vertical & horizontal straight connectors:
    _ortho_arrow(ax, [(11.71, 1.20), (11.71, 1.46)], col="#1E8449", lw=1.05, ms=8.0, zorder=5)
    _ortho_arrow(ax, [(11.71, 2.14), (11.71, 2.48)], col="#1E8449", lw=1.1, ms=8.5, zorder=5)
    ax.text(11.85, 2.30, "$Q_t$", ha="left", va="center",
            fontsize=7.2, fontweight="bold", color="#145A32")

    _ortho_arrow(ax, [(13.19, 2.48), (13.19, 2.14)], col="#C0392B", lw=1.05, ls="--", ms=8.5, zorder=5)
    ax.text(13.33, 2.30, "$\\mathrm{err}_k$", ha="left", va="center",
            fontsize=7.1, fontweight="bold", color="#922B21")

    _ortho_arrow(ax, [(12.56, 1.80), (12.34, 1.80)], col="#C0392B", lw=1.05, ms=8.0, zorder=5)

    _save_fig(fig, "fig1_framework.pdf", 1)
    plt.close(fig)


def load_json(path: Path) -> Dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _sci_tex(val: float) -> str:
    """Format scientific notation cleanly in LaTeX math ($a \\times 10^{b}$)."""
    exp = int(np.floor(np.log10(abs(val))))
    mant = val / (10.0 ** exp)
    return f"{mant:.2f} \\times 10^{{{exp}}}"


def compute_stats_and_fig2_fig6() -> Dict:
    main_data = {}
    for ds, _ in DATASETS_ORDER:
        p = RESULTS_DIR / f"main_{ds}_clean.json"
        if p.exists():
            main_data[ds] = load_json(p)

    avail_ds = [ds for ds, _ in DATASETS_ORDER if ds in main_data]
    K = len(MAIN_METHODS)
    N = len(avail_ds)

    wink_mat = np.zeros((N, K))
    cov_err_mat = np.zeros((N, K))
    pinaw_mat = np.zeros((N, K))
    rmse_mat = np.zeros((N, K))

    for i, ds in enumerate(avail_ds):
        res = main_data[ds]["results"]
        for j, m in enumerate(MAIN_METHODS):
            mm = res[m]["metrics_mean"]
            wink_mat[i, j] = mm["winkler"]
            cov_err_mat[i, j] = mm["abs_cov_err"]
            pinaw_mat[i, j] = mm["pinaw"]
            rmse_mat[i, j] = mm["rmse"]

    rel_wink = wink_mat / wink_mat.min(axis=1, keepdims=True)

    def _friedman_nemenyi(score_mat: np.ndarray):
        ranks = np.zeros_like(score_mat)
        for i in range(score_mat.shape[0]):
            order = np.argsort(score_mat[i])
            r = np.empty(score_mat.shape[1])
            r[order] = np.arange(1, score_mat.shape[1] + 1)
            ranks[i] = r
        avg_ranks = ranks.mean(axis=0)
        stat, pval = friedmanchisquare(*[score_mat[:, j] for j in range(score_mat.shape[1])])
        q_alpha = float(studentized_range.ppf(0.95, K, np.inf) / np.sqrt(2.0))
        cd = float(q_alpha * np.sqrt(K * (K + 1) / (6.0 * N)))
        return {
            "friedman_stat": float(stat),
            "p_value": float(pval),
            "cd_005": cd,
            "avg_ranks": {SHORT_NAMES[MAIN_METHODS[j]]: float(avg_ranks[j]) for j in range(K)},
        }

    stats_wink = _friedman_nemenyi(wink_mat)
    stats_cov = _friedman_nemenyi(cov_err_mat)
    stats_rmse = _friedman_nemenyi(rmse_mat)

    # --- Fig 2: Calibration--Sharpness Trade-off & Aggregate Summary ---
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.65))
    colors = ["#C0392B", "#2980B9", "#16A085", "#8E44AD", "#D35400", "#7F8C8D", "#2C3E50"]
    markers = ["*", "o", "s", "^", "D", "v", "P"]

    ax = axes[0]
    xs_cov = cov_err_mat.mean(axis=0) * 100.0
    ys_rw = rel_wink.mean(axis=0)
    x_span = max(xs_cov.max() - xs_cov.min(), 1.0)
    y_span = max(ys_rw.max() - ys_rw.min(), 0.2)
    for j, m in enumerate(MAIN_METHODS):
        sz = 195 if j == 0 else 72
        ax.scatter(xs_cov[j], ys_rw[j], s=sz, color=colors[j], marker=markers[j], zorder=4)
        if m == "ACHF":
            dy_off = 0.048 * y_span
        elif m == "CAFHT":
            dy_off = -0.058 * y_span
        else:
            dy_off = (0.025 if j % 2 == 0 else -0.04) * y_span
        ax.text(
            xs_cov[j] + 0.030 * x_span,
            ys_rw[j] + dy_off,
            SHORT_NAMES[m],
            fontsize=8.0,
            color=colors[j],
            fontweight="bold" if j == 0 else "normal",
        )
    ax.set_xlim(max(0.0, xs_cov.min() - 0.12 * x_span), xs_cov.max() + 0.30 * x_span)
    ax.set_ylim(max(0.95, ys_rw.min() - 0.12 * y_span), ys_rw.max() + 0.16 * y_span)
    ax.set_xlabel("Mean $|\\mathrm{PICP} - 95\\%|$ (%)")
    ax.set_ylabel("Mean Relative Winkler Score")
    ax.set_title("(a) Calibration\u2013Sharpness Trade-off")
    ax.grid(True, linestyle="--", color=_blend("#B0B0B0", 0.4))

    ax = axes[1]
    x_pos = np.arange(K)
    w_bar = 0.36
    ax.bar(x_pos - w_bar / 2, ys_rw, width=w_bar, color="#2E86C1", label="Mean Rel. Winkler")
    ax.set_ylim(0, ys_rw.max() * 1.28)
    ax2 = ax.twinx()
    ax2.bar(x_pos + w_bar / 2, xs_cov, width=w_bar, color=_blend("#E67E22", 0.86),
            label="Mean $|\\mathrm{PICP} - 95\\%|$ (%)")
    ax2.set_ylim(0, max(xs_cov.max() * 1.28, 3.0))
    ax.set_xticks(x_pos)
    ax.set_xticklabels([SHORT_NAMES[m] for m in MAIN_METHODS], rotation=25, ha="right")
    ax.set_ylabel("Mean Relative Winkler Score", color="#2E86C1")
    ax2.set_ylabel("Mean $|\\mathrm{PICP} - 95\\%|$ (%)", color="#D35400")
    ax.set_title("(b) Aggregate Comparison")
    ax.grid(True, axis="y", linestyle="--", color=_blend("#B0B0B0", 0.35))
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", frameon=True, fontsize=7.6)

    fig.tight_layout()
    _save_fig(fig, "fig2_efficiency_tradeoff.pdf", 2)
    plt.close(fig)

    # --- Fig 6: Friedman + Nemenyi CD Diagrams ---
    p_wink_tex = _sci_tex(stats_wink["p_value"])
    p_cov_tex = _sci_tex(stats_cov["p_value"])
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.15))
    for ax, st, title in [
        (axes[0], stats_wink, f"(a) Winkler Score ($p = {p_wink_tex}$)"),
        (axes[1], stats_cov, f"(b) $|\\mathrm{{PICP}} - 0.95|$ ($p = {p_cov_tex}$)"),
    ]:
        items = sorted(st["avg_ranks"].items(), key=lambda kv: kv[1])
        names = [kv[0] for kv in items]
        rks = [kv[1] for kv in items]
        y_pos = np.arange(len(names))
        bar_cols = [_blend("#C0392B", 0.92) if n == "FGB-IPN" else _blend("#5DADE2", 0.92) for n in names]
        ax.barh(y_pos, rks, color=bar_cols, height=0.55, zorder=3)
        best_r = rks[0]
        cd = st["cd_005"]
        ax.axvline(best_r + cd, color="#2C3E50", linestyle="--", lw=1.2,
                   label=f"Best + CD ({cd:.2f})", zorder=2)
        for yi, rv in enumerate(rks):
            ax.text(
                rv + 0.10,
                yi,
                f"{rv:.2f}",
                va="center",
                fontsize=8.1,
                zorder=5,
                bbox=dict(boxstyle="round,pad=0.08", facecolor="white", edgecolor="none"),
            )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlim(0, K + 1.6)
        ax.set_xlabel("Average Rank")
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=7.6)
        ax.grid(True, axis="x", linestyle="--", color=_blend("#B0B0B0", 0.35), zorder=1)

    fig.tight_layout()
    _save_fig(fig, "fig6_nemenyi_cd.pdf", 6)
    plt.close(fig)

    return {
        "winkler": stats_wink,
        "abs_cov_err": stats_cov,
        "rmse": stats_rmse,
        "mean_rel_winkler": {SHORT_NAMES[MAIN_METHODS[j]]: float(ys_rw[j]) for j in range(K)},
        "mean_abs_cov_err_pct": {SHORT_NAMES[MAIN_METHODS[j]]: float(xs_cov[j]) for j in range(K)},
        "mean_pinaw": {SHORT_NAMES[MAIN_METHODS[j]]: float(pinaw_mat[:, j].mean()) for j in range(K)},
    }


def draw_fig3_ablation() -> None:
    ds_list = [
        ("electricity", "Electricity"),
        ("solar", "Solar-Energy"),
        ("uci501", "Beijing-Air"),
        ("energy", "EnergyData"),
    ]
    variants = [
        "FGB-IPN (Full)",
        "FGB-NoPsi-TSK-ACI",
        "Hard-GB-IPN",
        "VarCluster-TSK-ACI",
        "TopM-Select-TSK-ACI",
        "PCA-TSK-ACI",
        "Raw-AllDim-TSK-ACI",
    ]
    avail = [(k, lbl) for k, lbl in ds_list if (RESULTS_DIR / f"ablation_{k}_noisy_30.json").exists()]
    if not avail:
        return

    fig, axes = plt.subplots(1, len(avail), figsize=(3.05 * len(avail), 3.25))
    if len(avail) == 1:
        axes = [axes]
    palette = ["#C0392B", "#D98880", "#E67E22", "#16A085", "#5DADE2", "#8E44AD", "#34495E"]

    for ax, (ds, lbl) in zip(axes, avail):
        d = load_json(RESULTS_DIR / f"ablation_{ds}_noisy_30.json")
        v_present = [v for v in variants if v in d["results"]]
        vals = [d["results"][v]["metrics_mean"]["winkler"] for v in v_present]
        stds = [d["results"][v]["metrics_std"]["winkler"] for v in v_present]
        x = np.arange(len(v_present))
        ax.bar(x, vals, yerr=stds, capsize=2.8, color=[_blend(c, 0.92) for c in palette[: len(v_present)]])
        top_val = max(v + s for v, s in zip(vals, stds))
        ax.set_ylim(0, top_val * 1.12)
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT_NAMES[v] for v in v_present], rotation=32, ha="right", fontsize=7.6)
        ax.set_title(lbl)
        if ax is axes[0]:
            ax.set_ylabel("Winkler Score")
        ax.grid(True, axis="y", linestyle="--", color=_blend("#B0B0B0", 0.35))

    fig.tight_layout()
    _save_fig(fig, "fig3_ablation_noisy.pdf", 3)
    plt.close(fig)


def draw_fig4_stress() -> None:
    ds_list = [("electricity", "(a) Electricity"), ("uci501", "(b) Beijing-Air")]
    avail = [(k, lbl) for k, lbl in ds_list if (RESULTS_DIR / f"stress_{k}.json").exists()]
    if not avail:
        return

    fig, axes = plt.subplots(1, len(avail), figsize=(5.2 * len(avail), 3.45))
    if len(avail) == 1:
        axes = [axes]
    methods = ["FGB-IPN (Full)", "Hard-GB-IPN", "VarCluster-TSK-ACI", "PCA-TSK-ACI", "CAFHT"]
    styles = [
        ("#C0392B", "-", "*", 2.2),
        ("#E67E22", "--", "s", 1.6),
        ("#16A085", "-.", "o", 1.6),
        ("#8E44AD", "-.", "^", 1.6),
        ("#2980B9", ":", "D", 1.6),
    ]

    for ax, (ds, lbl) in zip(axes, avail):
        d = load_json(RESULTS_DIR / f"stress_{ds}.json")
        xs = [int(lv * 100) for lv in d["levels"]]
        keys = [f"{x}%" for x in xs]
        all_ys = []
        for m, (col, ls, mk, lw) in zip(methods, styles):
            if m not in d["curve"][keys[0]]:
                continue
            ys = [d["curve"][k][m]["winkler"] for k in keys]
            all_ys.extend(ys)
            ax.plot(xs, ys, color=col, linestyle=ls, marker=mk, linewidth=lw,
                    markersize=5.6, label=SHORT_NAMES[m])
        ax.set_xlabel("Corrupted Sensor Ratio (%)")
        ax.set_ylabel("Winkler Score")
        ax.set_title(lbl)
        ax.set_xticks(xs)
        if all_ys:
            ymin, ymax = min(all_ys), max(all_ys)
            span = max(ymax - ymin, 10.0)
            ax.set_ylim(ymin - 0.08 * span, ymax + 0.32 * span)
        ax.grid(True, linestyle="--", color=_blend("#B0B0B0", 0.4))
        ax.legend(loc="upper left", ncol=3, frameon=True, fontsize=7.4)

    fig.tight_layout()
    _save_fig(fig, "fig4_stress_curve.pdf", 4)
    plt.close(fig)


def draw_fig5_sensitivity() -> None:
    ds_list = [("electricity", "Electricity"), ("uci501", "Beijing-Air")]
    avail = [(k, lbl) for k, lbl in ds_list if (RESULTS_DIR / f"sensitivity_{k}.json").exists()]
    if not avail:
        return

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.1))
    cols = ["#C0392B", "#2980B9"]
    mks = ["o", "s"]
    panel_ymax = [1.0, 1.0, 1.0]

    for idx, (ds, lbl) in enumerate(avail):
        s = load_json(RESULTS_DIR / f"sensitivity_{ds}.json")["sensitivity"]
        for p_idx, (ax, param_key, xlabel, title) in enumerate([
            (axes[0], "lookback", "Window $L$", "(a) Window $L$"),
            (axes[1], "max_balls", "Ball Cap $M_{\\max}$", "(b) Ball Cap $M_{\\max}$"),
            (axes[2], "num_rules", "Rule Count $R$", "(c) Rule Count $R$"),
        ]):
            ks = sorted(s[param_key].keys(), key=lambda x: int(x))
            xs = [int(k) for k in ks]
            ys = np.array([s[param_key][k]["metrics_mean"]["winkler"] for k in ks])
            ys_rel = ys / ys.min()
            panel_ymax[p_idx] = max(panel_ymax[p_idx], float(ys_rel.max()))
            ax.plot(xs, ys_rel, color=cols[idx], marker=mks[idx], lw=1.8, markersize=5.5, label=lbl)
            ax.set_xlabel(xlabel)
            ax.set_title(title)
            ax.set_xticks(xs)
            ax.grid(True, linestyle="--", color=_blend("#B0B0B0", 0.4))
            if p_idx == 0:
                ax.set_ylabel("Relative Winkler")

    legend_locs = ["upper left", "upper right", "upper left"]
    for p_idx, ax in enumerate(axes):
        span = max(panel_ymax[p_idx] - 1.0, 0.02)
        ax.set_ylim(1.0 - 0.08 * span, panel_ymax[p_idx] + 0.24 * span)
        ax.legend(loc=legend_locs[p_idx], frameon=True, fontsize=7.6)

    fig.tight_layout()
    _save_fig(fig, "fig5_sensitivity.pdf", 5)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    draw_fig1_framework()
    stats = compute_stats_and_fig2_fig6()
    draw_fig3_ablation()
    draw_fig4_stress()
    draw_fig5_sensitivity()
    with open(RESULTS_DIR / "friedman_nemenyi_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
