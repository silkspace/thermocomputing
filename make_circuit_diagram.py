"""
Generate circuit diagrams for the Hardware Realization section.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Colors
colors = {
    'nmos': '#2196F3',
    'pmos': '#F44336',
    'wire': '#424242',
    'vdd': '#4CAF50',
    'gnd': '#795548',
    'signal': '#FF9800',
    'node': '#9C27B0',
}

# ============== Panel A: P-bit Cell ==============
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('(a) Stochastic P-bit Cell', fontsize=12, fontweight='bold', pad=10)

# VDD rail
ax.plot([1, 9], [9, 9], color=colors['vdd'], linewidth=3)
ax.text(5, 9.3, '$V_{DD}$', ha='center', fontsize=10, color=colors['vdd'])

# GND rail
ax.plot([1, 9], [1, 1], color=colors['gnd'], linewidth=3)
ax.text(5, 0.5, 'GND', ha='center', fontsize=10, color=colors['gnd'])

# Inverter 1 (left)
# PMOS
ax.add_patch(Rectangle((2.5, 6.5), 1.5, 1.5, facecolor='white', edgecolor=colors['pmos'], linewidth=2))
ax.text(3.25, 7.25, 'P', ha='center', va='center', fontsize=9, color=colors['pmos'])
ax.plot([3.25, 3.25], [8, 9], color=colors['wire'], linewidth=1.5)
# NMOS
ax.add_patch(Rectangle((2.5, 2), 1.5, 1.5, facecolor='white', edgecolor=colors['nmos'], linewidth=2))
ax.text(3.25, 2.75, 'N', ha='center', va='center', fontsize=9, color=colors['nmos'])
ax.plot([3.25, 3.25], [1, 2], color=colors['wire'], linewidth=1.5)
# Connect PMOS to NMOS
ax.plot([3.25, 3.25], [3.5, 6.5], color=colors['wire'], linewidth=1.5)
# Output node
ax.plot([3.25, 4.5], [5, 5], color=colors['signal'], linewidth=2)
ax.add_patch(Circle((3.25, 5), 0.15, facecolor=colors['node'], edgecolor='white'))

# Inverter 2 (right)
# PMOS
ax.add_patch(Rectangle((6, 6.5), 1.5, 1.5, facecolor='white', edgecolor=colors['pmos'], linewidth=2))
ax.text(6.75, 7.25, 'P', ha='center', va='center', fontsize=9, color=colors['pmos'])
ax.plot([6.75, 6.75], [8, 9], color=colors['wire'], linewidth=1.5)
# NMOS
ax.add_patch(Rectangle((6, 2), 1.5, 1.5, facecolor='white', edgecolor=colors['nmos'], linewidth=2))
ax.text(6.75, 2.75, 'N', ha='center', va='center', fontsize=9, color=colors['nmos'])
ax.plot([6.75, 6.75], [1, 2], color=colors['wire'], linewidth=1.5)
# Connect
ax.plot([6.75, 6.75], [3.5, 6.5], color=colors['wire'], linewidth=1.5)
# Output node
ax.plot([5.5, 6.75], [5, 5], color=colors['signal'], linewidth=2)
ax.add_patch(Circle((6.75, 5), 0.15, facecolor=colors['node'], edgecolor='white'))

# Cross-coupling
ax.annotate('', xy=(2.5, 7.25), xytext=(5.5, 5),
            arrowprops=dict(arrowstyle='-', color=colors['wire'], lw=1.5,
                           connectionstyle='arc3,rad=0.3'))
ax.annotate('', xy=(2.5, 2.75), xytext=(5.5, 5),
            arrowprops=dict(arrowstyle='-', color=colors['wire'], lw=1.5,
                           connectionstyle='arc3,rad=-0.3'))
ax.annotate('', xy=(7.5, 7.25), xytext=(4.5, 5),
            arrowprops=dict(arrowstyle='-', color=colors['wire'], lw=1.5,
                           connectionstyle='arc3,rad=-0.3'))
ax.annotate('', xy=(7.5, 2.75), xytext=(4.5, 5),
            arrowprops=dict(arrowstyle='-', color=colors['wire'], lw=1.5,
                           connectionstyle='arc3,rad=0.3'))

# Labels
ax.text(3.25, 5.5, '$x_i$', fontsize=11, ha='center', color=colors['node'])
ax.text(6.75, 5.5, '$\\bar{x}_i$', fontsize=11, ha='center', color=colors['node'])

# Noise annotation
ax.annotate('thermal\nnoise', xy=(5, 4), fontsize=9, ha='center',
            color='#666666', style='italic')
ax.annotate('', xy=(4, 4.5), xytext=(6, 4.5),
            arrowprops=dict(arrowstyle='<->', color='#999999', lw=1))

# ============== Panel B: Current Mirror Coupling ==============
ax = axes[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('(b) Current Mirror Coupling', fontsize=12, fontweight='bold', pad=10)

# VDD
ax.plot([1, 9], [9, 9], color=colors['vdd'], linewidth=3)
ax.text(5, 9.3, '$V_{DD}$', ha='center', fontsize=10, color=colors['vdd'])

# Reference transistor (left)
ax.add_patch(Rectangle((1.5, 5.5), 1.5, 1.5, facecolor='white', edgecolor=colors['pmos'], linewidth=2))
ax.text(2.25, 6.25, 'M1', ha='center', va='center', fontsize=9)
ax.plot([2.25, 2.25], [7, 9], color=colors['wire'], linewidth=1.5)

# Mirror transistor (right)
ax.add_patch(Rectangle((7, 5.5), 1.5, 1.5, facecolor='white', edgecolor=colors['pmos'], linewidth=2))
ax.text(7.75, 6.25, 'M2', ha='center', va='center', fontsize=9)
ax.plot([7.75, 7.75], [7, 9], color=colors['wire'], linewidth=1.5)

# Gate connection
ax.plot([2.25, 2.25], [5.5, 4.5], color=colors['wire'], linewidth=1.5)
ax.plot([2.25, 7.75], [4.5, 4.5], color=colors['wire'], linewidth=1.5)
ax.plot([7.75, 7.75], [4.5, 5.5], color=colors['wire'], linewidth=1.5)
# Diode connection on M1
ax.plot([1.5, 2.25], [6.25, 6.25], color=colors['wire'], linewidth=1.5)
ax.plot([1.5, 1.5], [6.25, 4.5], color=colors['wire'], linewidth=1.5)
ax.plot([1.5, 2.25], [4.5, 4.5], color=colors['wire'], linewidth=1.5)

# Input current
ax.annotate('', xy=(2.25, 3.5), xytext=(2.25, 5.5),
            arrowprops=dict(arrowstyle='->', color=colors['signal'], lw=2))
ax.text(2.8, 4.5, '$I_{in}$\n$\\propto x_i$', fontsize=10, color=colors['signal'])

# Output current
ax.annotate('', xy=(7.75, 3.5), xytext=(7.75, 5.5),
            arrowprops=dict(arrowstyle='->', color=colors['signal'], lw=2))
ax.text(8.3, 4.5, '$I_{out}$\n$= J_{ij} I_{in}$', fontsize=10, color=colors['signal'])

# W/L ratio annotation
ax.text(5, 7, '$\\frac{W_2/L_2}{W_1/L_1} = J_{ij}$', fontsize=11, ha='center',
        bbox=dict(boxstyle='round', facecolor='#FFF9C4', edgecolor='#FBC02D'))

# Node labels
ax.text(2.25, 3, 'from $x_i$', fontsize=9, ha='center', color='#666666')
ax.text(7.75, 3, 'to $x_j$', fontsize=9, ha='center', color='#666666')

# ============== Panel C: Dual-Bank DAC ==============
ax = axes[2]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('(c) Dual-Bank Reconfiguration', fontsize=12, fontweight='bold', pad=10)

# Bank A (active)
ax.add_patch(FancyBboxPatch((1, 6), 3.5, 2.5, boxstyle="round,pad=0.02",
                             facecolor='#E8F5E9', edgecolor='#4CAF50', linewidth=2))
ax.text(2.75, 8, 'Bank A', ha='center', fontsize=10, fontweight='bold', color='#4CAF50')
ax.text(2.75, 7, 'ACTIVE', ha='center', fontsize=9, color='#4CAF50')
ax.text(2.75, 6.3, '$\\theta_A$', ha='center', fontsize=11)

# Bank B (shadow)
ax.add_patch(FancyBboxPatch((5.5, 6), 3.5, 2.5, boxstyle="round,pad=0.02",
                             facecolor='#FFF3E0', edgecolor='#FF9800', linewidth=2))
ax.text(7.25, 8, 'Bank B', ha='center', fontsize=10, fontweight='bold', color='#FF9800')
ax.text(7.25, 7, 'SHADOW', ha='center', fontsize=9, color='#FF9800')
ax.text(7.25, 6.3, '$\\theta_B$', ha='center', fontsize=11)

# Multiplexer
ax.add_patch(Polygon([(4, 4), (6, 4), (5.5, 2.5), (4.5, 2.5)],
                     facecolor='#E3F2FD', edgecolor='#1976D2', linewidth=2))
ax.text(5, 3.4, 'MUX', ha='center', fontsize=9, fontweight='bold', color='#1976D2')

# Connections
ax.plot([2.75, 4.5], [6, 4], color=colors['wire'], linewidth=1.5)
ax.plot([7.25, 5.5], [6, 4], color=colors['wire'], linewidth=1.5)

# Output to T-array
ax.annotate('', xy=(5, 1.5), xytext=(5, 2.5),
            arrowprops=dict(arrowstyle='->', color=colors['signal'], lw=2))
ax.text(5, 1, 'to T-array', ha='center', fontsize=10, color=colors['signal'])

# Bank select signal
ax.annotate('', xy=(6.5, 3.25), xytext=(8, 3.25),
            arrowprops=dict(arrowstyle='->', color='#9C27B0', lw=1.5))
ax.text(8.2, 3.25, 'SEL', fontsize=9, color='#9C27B0')

# Gradient update arrow
ax.annotate('', xy=(7.25, 9), xytext=(7.25, 8.5),
            arrowprops=dict(arrowstyle='->', color='#FF9800', lw=2))
ax.text(7.25, 9.3, '$\\Delta\\theta$\nfrom Gradient Engine', ha='center', fontsize=8, color='#FF9800')

# Swap annotation
ax.annotate('atomic\nswap', xy=(5, 5.2), fontsize=9, ha='center',
            color='#666666', style='italic',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='#BDBDBD', alpha=0.8))

plt.tight_layout()
plt.savefig('circuit_diagrams.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved circuit_diagrams.png")
