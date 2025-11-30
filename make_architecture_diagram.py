"""
Generate a clean schematic diagram of the Criticality Engine processor architecture.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np

# Set up figure
fig, ax = plt.subplots(1, 1, figsize=(13, 8))
ax.set_xlim(0, 13)
ax.set_ylim(0, 8)
ax.set_aspect('equal')
ax.axis('off')

# Colors - refined palette
colors = {
    'tarray': '#E3F2FD',
    'tarray_border': '#1565C0',
    'measure': '#FFF8E1',
    'measure_border': '#FF8F00',
    'stats': '#F3E5F5',
    'stats_border': '#7B1FA2',
    'gradient': '#E8F5E9',
    'gradient_border': '#2E7D32',
    'reconfig': '#FFEBEE',
    'reconfig_border': '#C62828',
}

def draw_box(ax, x, y, w, h, label, sublabel, color, border_color, fontsize=11):
    box = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.02,rounding_size=0.2",
                          facecolor=color, edgecolor=border_color, linewidth=2.5)
    ax.add_patch(box)
    ax.text(x + w/2, y + h - 0.3, label, ha='center', va='top',
            fontsize=fontsize, fontweight='bold', color=border_color)
    if sublabel:
        ax.text(x + w/2, y + 0.5, sublabel, ha='center', va='center',
                fontsize=9, color='#424242', linespacing=1.2)

# Title
ax.text(6.5, 7.7, 'Criticality Engine: Processor Architecture',
        ha='center', fontsize=15, fontweight='bold', color='#212121')

# Layout: 3 columns
# Left: Reconfiguration (tall)
# Center: Measurement (top) + T-array (bottom)
# Right: Local Stats (top) + Gradient (bottom)

# Center column
# Measurement Plane
draw_box(ax, 4, 5.8, 4.5, 1.2, 'Measurement Plane',
         'Sample & hold at $\\{t_k\\}$',
         colors['measure'], colors['measure_border'])

# T-array
tarray_box = FancyBboxPatch((4, 2.5), 4.5, 3,
                             boxstyle="round,pad=0.02,rounding_size=0.2",
                             facecolor=colors['tarray'], edgecolor=colors['tarray_border'], linewidth=2.5)
ax.add_patch(tarray_box)
ax.text(6.25, 5.2, 'Thermodynamic Array', ha='center', va='top',
        fontsize=11, fontweight='bold', color=colors['tarray_border'])

# Node grid (4x2) - tighter spacing
for i in range(4):
    for j in range(2):
        cx = 4.7 + i * 1.0
        cy = 3.5 + j * 0.8
        circle = Circle((cx, cy), 0.2, facecolor='white',
                        edgecolor=colors['tarray_border'], linewidth=1.5, zorder=3)
        ax.add_patch(circle)
        if i < 3:
            ax.plot([cx + 0.2, cx + 0.8], [cy, cy],
                   color=colors['tarray_border'], linewidth=1.2, alpha=0.5, zorder=1)
        if j < 1:
            ax.plot([cx, cx], [cy + 0.2, cy + 0.6],
                   color=colors['tarray_border'], linewidth=1.2, alpha=0.5, zorder=1)

# Langevin equation inside T-array box (bottom)
ax.text(6.25, 2.75, '$dx = -\\nabla V\\,dt + \\sqrt{2kT}\\,dW$',
        ha='center', fontsize=9, color='#546E7A', style='italic')

# Right column
# Local Statistics
draw_box(ax, 9, 4.2, 3.5, 2.8, 'Local Statistics',
         'Correlators\n$m_i, C_{ij}$',
         colors['stats'], colors['stats_border'])

# Gradient Engine
draw_box(ax, 9, 1, 3.5, 2.8, 'Gradient Engine',
         'Compute $\\Delta\\theta$\nfrom learning rules',
         colors['gradient'], colors['gradient_border'])

# Left column
# Reconfiguration (full height to match right side)
draw_box(ax, 0.5, 1, 3, 6, 'Reconfiguration',
         'Dual-bank DACs\nBias banks $\\{\\mathbf{b}_k\\}$\nCoupling $J$',
         colors['reconfig'], colors['reconfig_border'])

# Arrows - clean flow
arrow_kw = dict(arrowstyle='->', mutation_scale=15, linewidth=2)

# ① T-array → Measurement (up)
ax.annotate('', xy=(6.25, 5.8), xytext=(6.25, 5.5),
            arrowprops=dict(**arrow_kw, color=colors['measure_border']))
ax.text(6.6, 5.65, 'states', fontsize=8, color=colors['measure_border'])

# ② Measurement → Local Stats (curved right)
ax.annotate('', xy=(9, 5.8), xytext=(8.5, 6.4),
            arrowprops=dict(**arrow_kw, color=colors['stats_border'],
                          connectionstyle='arc3,rad=-0.15'))
ax.text(9.0, 6.5, '$x_i(t_k)$', fontsize=9, color=colors['stats_border'])

# ③ Local Stats → Gradient (down)
ax.annotate('', xy=(10.75, 3.8), xytext=(10.75, 4.2),
            arrowprops=dict(**arrow_kw, color=colors['gradient_border']))
ax.text(11.0, 4.0, 'stats', fontsize=8, color=colors['gradient_border'])

# ④ Gradient → Reconfig (across bottom)
ax.annotate('', xy=(3.5, 1.8), xytext=(9, 1.8),
            arrowprops=dict(**arrow_kw, color=colors['reconfig_border']))
ax.text(6.25, 2.05, '$\\Delta J, \\Delta\\mathbf{b}$', ha='center',
        fontsize=10, color=colors['reconfig_border'])

# ⑤ Reconfig → T-array (right)
ax.annotate('', xy=(4, 4), xytext=(3.5, 4),
            arrowprops=dict(**arrow_kw, color=colors['reconfig_border']))
ax.text(3.75, 4.35, 'params', fontsize=8, ha='center', color=colors['reconfig_border'])

# Cycle legend (single line, bottom)
ax.text(6.5, 0.35, '① Sample → ② Measure → ③ Statistics → ④ Gradients → ⑤ Reconfigure',
        ha='center', fontsize=9, color='#757575', style='italic')

plt.tight_layout()
plt.savefig('paper_figures/architecture_schematic.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved paper_figures/architecture_schematic.png")
