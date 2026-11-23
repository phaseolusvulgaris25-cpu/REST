# -*- coding: utf-8 -*-
"""
Created on Sat May 23 10:59:54 2026

@author: fulya
"""

# -*- coding: utf-8 -*-

import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib as mpl
import os

# =========================
# STYLE
# =========================

mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['axes.linewidth'] = 1.2

# =========================
# FONT SIZES
# =========================

LABEL_SIZE = 18
TICK_SIZE = 14

# =========================
# DSSP SIMPLIFICATION
# =========================

# 0 = Helix
# 1 = Sheet
# 2 = Turn/Bend
# 3 = Coil

dssp_map = {
    'H': 0,
    'G': 0,
    'I': 0,

    'E': 1,
    'B': 1,

    'T': 2,
    'S': 2,

    '-': 3
}

# =========================
# LOAD
# =========================

traj = md.load(
    "mdnowater.nc",
    top="no_water.pdb"
)

chain_A = traj.topology.select("chainid 0")
traj_chainA = traj.atom_slice(chain_A)

# =========================
# DSSP
# =========================

dssp = md.compute_dssp(traj_chainA)

ss_numeric = np.array([
    [dssp_map.get(s, 3) for s in frame]
    for frame in dssp
])

# =========================
# PROFESSIONAL COLORS
# =========================

colors = [
    "#B22222",  # Helix dark red
    "#1D3557",  # Sheet navy
    "#6C757D",  # Turn/Bend gray
    "#F8F9FA"   # Coil very light
]

cmap = ListedColormap(colors)

# =========================
# TIME
# =========================

time_ns = np.arange(ss_numeric.shape[0]) / 10

# =========================
# FIGURE
# =========================

fig, ax = plt.subplots(figsize=(13, 4.2))

im = ax.imshow(
    ss_numeric.T,
    aspect='auto',
    cmap=cmap,
    interpolation='nearest',
    origin='lower',
    extent=[
        time_ns[0],
        time_ns[-1],
        1,
        ss_numeric.shape[1]
    ]
)

# =========================
# LABELS
# =========================

ax.set_xlabel(
    "Time (ns)",
    fontsize=LABEL_SIZE,
    fontweight='bold'
)

ax.set_ylabel(
    "Residue Number",
    fontsize=LABEL_SIZE,
    fontweight='bold'
)

# =========================
# TICKS
# =========================

ax.tick_params(
    axis='both',
    labelsize=TICK_SIZE,
    width=1.2,
    length=5
)

# Daha temiz tickler
ax.set_xticks(np.arange(0, 110, 20))
ax.set_yticks(np.arange(0, 160, 20))

# =========================
# SPINES
# =========================

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# =========================
# COLORBAR
# =========================

cbar = plt.colorbar(
    im,
    pad=0.015,
    shrink=0.92
)

cbar.set_ticks([0,1,2,3])

cbar.set_ticklabels([
    'Helix',
    'Sheet',
    'Turn/Bend',
    'Coil'
])

cbar.ax.tick_params(labelsize=12)

# =========================
# SAVE
# =========================

plt.tight_layout()

desktop = os.path.join(
    os.path.expanduser("~"),
    "Desktop"
)

outfile = os.path.join(
    desktop,
    "DSSP.png"
)

plt.savefig(
    outfile,
    dpi=1200,
    bbox_inches='tight',
    transparent=True
)

print("Saved:", outfile)

plt.show()