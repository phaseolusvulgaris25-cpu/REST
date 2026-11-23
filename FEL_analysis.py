# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 04:03:53 2026

@author: fulya
"""


# -*- coding: utf-8 -*-
"""
Gibbs Free Energy Landscape (PC1-PC2)
White Background + Black 3D Mesh Grid
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# =====================================================
# 1. LOAD XVG DATA
# =====================================================

def load_xvg_two_datasets_as_pc1_pc2(path):

    set_id = 0
    pc1, pc2 = [], []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:

        for line in f:

            line = line.strip()

            if not line or line.startswith(("@", "#")):
                continue

            if line.startswith("&"):
                set_id += 1
                continue

            parts = line.split()

            if len(parts) < 2:
                continue

            try:
                y = float(parts[1])
            except ValueError:
                continue

            if set_id == 0:
                pc1.append(y)

            elif set_id == 1:
                pc2.append(y)

    n = min(len(pc1), len(pc2))

    if n == 0:
        raise ValueError(
            "XVG dosyasından PC1 ve PC2 verileri okunamadı."
        )

    return np.array(pc1[:n]), np.array(pc2[:n])


# =====================================================
# 2. FREE ENERGY CALCULATION
# =====================================================

def free_energy_surface(
    x,
    y,
    bins=70,
    T=300.0,
    smooth_sigma=1.0,
    max_dG=12.0
):

    H, xedges, yedges = np.histogram2d(
        x, y, bins=bins
    )

    P = H / np.sum(H)

    # Zero probability bins
    P[P == 0] = np.nan

    kB = 8.314462618e-3  # kJ/mol/K

    dG = -kB * T * np.log(P)

    # Relative free energy
    dG -= np.nanmin(dG)

    # Energy cutoff
    dG[dG > max_dG] = max_dG

    # Gaussian smoothing
    if smooth_sigma:

        try:
            from scipy.ndimage import gaussian_filter

            valid = np.isfinite(dG)

            filled = np.where(
                valid,
                dG,
                max_dG
            )

            dG = gaussian_filter(
                filled,
                sigma=smooth_sigma
            )

        except ImportError:
            print("SciPy not found. Smoothing skipped.")

    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])

    X, Y = np.meshgrid(
        xc,
        yc,
        indexing="ij"
    )

    return (X, Y), dG


# =====================================================
# 3. 3D GIBBS FREE ENERGY LANDSCAPE
# =====================================================

def plot_gibbs_landscape_3d(X, Y, dG, out_png):

    # -------------------------------------------------
    # COLOR MAP
    # -------------------------------------------------

    rainbow_energy = LinearSegmentedColormap.from_list(
        "rainbow_energy",
        [
            "#fdd835",  # Yellow - low energy
            "#8e24aa",  # Purple
            "#3949ab",  # Blue
            "#1e88e5",  # Light blue
            "#00897b",  # Teal
            "#7cb342",  # Green
            "#b71c1c"   # Red - high energy
        ],
        N=256
    )

    # -------------------------------------------------
    # FIGURE AND WHITE BACKGROUND
    # -------------------------------------------------

    fig = plt.figure(
        figsize=(8.5, 7.0),
        facecolor="white"
    )

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    # White background
    ax.set_facecolor("white")

    # White 3D axis panes
    ax.xaxis.set_pane_color((1, 1, 1, 1))
    ax.yaxis.set_pane_color((1, 1, 1, 1))
    ax.zaxis.set_pane_color((1, 1, 1, 1))

    # -------------------------------------------------
    # SURFACE
    # -------------------------------------------------

    dGm = np.ma.masked_invalid(dG)

    surf = ax.plot_surface(
        X,
        Y,
        dGm,
        cmap=rainbow_energy,
        edgecolor="#222222",
        linewidth=0.25,
        antialiased=True,
        rstride=2,
        cstride=2,
        shade=True
    )

    # -------------------------------------------------
    # ENERGY FLOOR
    # -------------------------------------------------

    zmin = np.nanmin(dG)
    zmax = np.nanmax(dG)

    zfloor = zmin - 0.08 * (zmax - zmin)

    ax.contourf(
        X,
        Y,
        dGm,
        zdir="z",
        offset=zfloor,
        levels=40,
        cmap=rainbow_energy
    )

    # -------------------------------------------------
    # AXIS LABELS
    # -------------------------------------------------

    ax.set_xlabel(
        "PC1",
        labelpad=9,
        fontsize=11,
        color="black"
    )

    ax.set_ylabel(
        "PC2",
        labelpad=9,
        fontsize=11,
        color="black"
    )

    ax.set_zlabel(
        "G (kJ/mol)",
        labelpad=9,
        fontsize=11,
        color="black"
    )

    ax.set_title(
        "FEL (PC1–PC2)--TG2-LDN",
        pad=0.5,
        fontsize=17,
        color="black"
    )

    # -------------------------------------------------
    # DARK GRID
    # -------------------------------------------------

    ax.grid(True)

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:

        axis._axinfo["grid"]["color"] = (
            0.15, 0.15, 0.15, 0.8
        )

        axis._axinfo["grid"]["linewidth"] = 0.6

        axis._axinfo["grid"]["linestyle"] = "-"

    # Axis lines
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:

        axis._axinfo["axisline"]["color"] = (
            0.1, 0.1, 0.1, 1
        )

    # Tick colors
    ax.tick_params(
        axis="both",
        colors="black",
        labelsize=9
    )

    # -------------------------------------------------
    # 3D VIEW
    # -------------------------------------------------

    ax.view_init(
        elev=30,
        azim=-60
    )

    ax.set_zlim(
        zfloor,
        zmax
    )

    # -------------------------------------------------
    # COLOR BAR
    # -------------------------------------------------

    cbar = fig.colorbar(
        surf,
        ax=ax,
        shrink=0.72,
        pad=0.12
    )

    cbar.set_label(
        "",
        fontsize=10,
        color="black"
    )

    cbar.ax.tick_params(
        colors="black",
        labelsize=9
    )

    # -------------------------------------------------
    # LAYOUT
    # -------------------------------------------------

    plt.tight_layout()

    # -------------------------------------------------
    # SAVE HIGH-RESOLUTION FIGURE
    # -------------------------------------------------

    plt.savefig(
        out_png,
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="white"
    )

    print(f"Saved: {out_png}")

    plt.show()


# =====================================================
# 4. MAIN
# =====================================================

if __name__ == "__main__":

    data_file = (
        r"proj_1_prot.xvg"
    )

    out_png = (
        r"gibbs_energy_landscape.png"
    )

    pc1, pc2 = load_xvg_two_datasets_as_pc1_pc2(
        data_file
    )

    print("Number of frames:", len(pc1))

    (X, Y), dG = free_energy_surface(
        pc1,
        pc2,
        bins=70,
        T=300.0,
        smooth_sigma=1.0,
        max_dG=12.0
    )

    plot_gibbs_landscape_3d(
        X,
        Y,
        dG,
        out_png
    )
