"""
Generate a diagram showing the components of a single node in the Criticality Engine.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.set_xlim(-0.5, 12.5)
ax.set_ylim(1.5, 7.5)
ax.set_aspect('equal')
ax.axis('off')

# Colors
colors = {
    'pbit': '#E3F2FD',
    'pbit_border': '#1976D2',
    'mirror': '#FFF3E0',
    'mirror_border': '#F57C00',
    'sample': '#E8F5E9',
    'sample_border': '#388E3C',
    'dac': '#FCE4EC',
    'dac_border': '#C2185B',
    'neighbor': '#F3E5F5',
    'neighbor_border': '#7B1FA2',
}

# Title
ax.text(6, 7.2, 'Anatomy of a Single Node', fontsize=14, fontweight='bold', ha='center')

# Central p-bit cell
pbit = FancyBboxPatch((4.5, 3.5), 3, 2, boxstyle="round,pad=0.05",
                       facecolor=colors['pbit'], edgecolor=colors['pbit_border'], linewidth=3)
ax.add_patch(pbit)
ax.text(6, 5, 'P-bit Cell', ha='center', va='center', fontsize=11, fontweight='bold', color=colors['pbit_border'])
ax.text(6, 4.4, 'cross-coupled inverters', ha='center', va='center', fontsize=9, style='italic')
ax.text(6, 3.9, '$x_i(t)$', ha='center', va='center', fontsize=11)

# Current mirror inputs (left side)
for i, label in enumerate(['$J_{i1}$', '$J_{i2}$', '...', '$J_{id}$']):
    y = 5.5 - i * 0.6
    mirror_in = FancyBboxPatch((1.5, y-0.25), 1.8, 0.5, boxstyle="round,pad=0.02",
                                facecolor=colors['mirror'], edgecolor=colors['mirror_border'], linewidth=1.5)
    ax.add_patch(mirror_in)
    ax.text(2.4, y, label, ha='center', va='center', fontsize=9)
    # Arrow to p-bit
    ax.annotate('', xy=(4.5, 4.5), xytext=(3.3, y),
                arrowprops=dict(arrowstyle='->', color=colors['mirror_border'], lw=1.5))

ax.text(2.4, 6.2, 'Current Mirror\nInputs ($d$ total)', ha='center', va='center', fontsize=9,
        fontweight='bold', color=colors['mirror_border'])

# Current mirror outputs (right side)
for i, label in enumerate(['$J_{1i}$', '$J_{2i}$', '...', '$J_{di}$']):
    y = 5.5 - i * 0.6
    mirror_out = FancyBboxPatch((8.7, y-0.25), 1.8, 0.5, boxstyle="round,pad=0.02",
                                 facecolor=colors['mirror'], edgecolor=colors['mirror_border'], linewidth=1.5)
    ax.add_patch(mirror_out)
    ax.text(9.6, y, label, ha='center', va='center', fontsize=9)
    # Arrow from p-bit
    ax.annotate('', xy=(8.7, y), xytext=(7.5, 4.5),
                arrowprops=dict(arrowstyle='->', color=colors['mirror_border'], lw=1.5))

ax.text(9.6, 6.2, 'Current Mirror\nOutputs ($d$ total)', ha='center', va='center', fontsize=9,
        fontweight='bold', color=colors['mirror_border'])

# Sample-and-hold (top)
sample = FancyBboxPatch((4.5, 6), 3, 0.8, boxstyle="round,pad=0.02",
                         facecolor=colors['sample'], edgecolor=colors['sample_border'], linewidth=2)
ax.add_patch(sample)
ax.text(6, 6.4, 'Sample & Hold + Register', ha='center', va='center', fontsize=9, fontweight='bold', color=colors['sample_border'])
ax.annotate('', xy=(6, 6), xytext=(6, 5.5),
            arrowprops=dict(arrowstyle='->', color=colors['sample_border'], lw=2))

# DAC (bottom)
dac = FancyBboxPatch((4.5, 1.8), 3, 0.8, boxstyle="round,pad=0.02",
                      facecolor=colors['dac'], edgecolor=colors['dac_border'], linewidth=2)
ax.add_patch(dac)
ax.text(6, 2.2, 'Bias DAC ($b_i$)', ha='center', va='center', fontsize=9, fontweight='bold', color=colors['dac_border'])
ax.annotate('', xy=(6, 3.5), xytext=(6, 2.6),
            arrowprops=dict(arrowstyle='->', color=colors['dac_border'], lw=2))

# Neighbor nodes (ghosted)
for pos, label in [((0.3, 4.5), '$x_1$'), ((0.3, 3.5), '$x_2$'), ((11.2, 4.5), '$x_3$'), ((11.2, 3.5), '$x_4$')]:
    neighbor = Circle(pos, 0.4, facecolor=colors['neighbor'], edgecolor=colors['neighbor_border'],
                      linewidth=1.5, alpha=0.7)
    ax.add_patch(neighbor)
    ax.text(pos[0], pos[1], label, ha='center', va='center', fontsize=8, color=colors['neighbor_border'])

ax.text(0.3, 5.3, 'Neighbors', ha='center', fontsize=8, color=colors['neighbor_border'], style='italic')
ax.text(11.2, 5.3, 'Neighbors', ha='center', fontsize=8, color=colors['neighbor_border'], style='italic')

plt.tight_layout()
plt.savefig('paper_figures/node_anatomy.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved paper_figures/node_anatomy.png")
