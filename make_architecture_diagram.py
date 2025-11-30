"""
Generate a clean schematic diagram of the Criticality Engine processor architecture.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

# Set up figure
fig, ax = plt.subplots(1, 1, figsize=(14, 9))
ax.set_xlim(0, 14)
ax.set_ylim(0, 9)
ax.set_aspect('equal')
ax.axis('off')

# Colors - refined palette
colors = {
    'tarray': '#E3F2FD',      # Light blue - physics
    'tarray_border': '#1565C0',
    'measure': '#FFF8E1',      # Light amber - observation
    'measure_border': '#FF8F00',
    'stats': '#F3E5F5',        # Light purple - computation
    'stats_border': '#7B1FA2',
    'gradient': '#E8F5E9',     # Light green - learning
    'gradient_border': '#2E7D32',
    'reconfig': '#FFEBEE',     # Light red - control
    'reconfig_border': '#C62828',
}

def draw_box(ax, x, y, w, h, label, sublabel, color, border_color, fontsize=12):
    """Draw a rounded box with label."""
    box = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.02,rounding_size=0.25",
                          facecolor=color, edgecolor=border_color, linewidth=2.5)
    ax.add_patch(box)
    ax.text(x + w/2, y + h - 0.35, label, ha='center', va='top',
            fontsize=fontsize, fontweight='bold', color=border_color)
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.15, sublabel, ha='center', va='center',
                fontsize=9, color='#424242', linespacing=1.3)

# Title
ax.text(7, 8.7, 'Criticality Engine: Processor Architecture',
        ha='center', va='top', fontsize=16, fontweight='bold', color='#212121')

# 1. T-array (center, large)
tarray_box = FancyBboxPatch((4.5, 3.8), 5, 3.2,
                             boxstyle="round,pad=0.02,rounding_size=0.25",
                             facecolor=colors['tarray'], edgecolor=colors['tarray_border'], linewidth=2.5)
ax.add_patch(tarray_box)
ax.text(7, 6.65, 'Thermodynamic Array', ha='center', va='top',
        fontsize=13, fontweight='bold', color=colors['tarray_border'])

# Draw nodes inside T-array - cleaner grid (smaller, higher)
for i in range(4):
    for j in range(2):
        cx = 5.5 + i * 1.1
        cy = 5.5 + j * 0.7
        circle = Circle((cx, cy), 0.17, facecolor='white',
                        edgecolor=colors['tarray_border'], linewidth=1.5, zorder=3)
        ax.add_patch(circle)
        if i < 3:
            ax.plot([cx + 0.17, cx + 0.93], [cy, cy],
                   color=colors['tarray_border'], linewidth=1.2, alpha=0.4, zorder=1)
        if j < 1:
            ax.plot([cx, cx], [cy + 0.17, cy + 0.53],
                   color=colors['tarray_border'], linewidth=1.2, alpha=0.4, zorder=1)

# Equation below the nodes
ax.text(7, 4.3, '$dx = -\\nabla V\\, dt + \\sqrt{2kT}\\,dW$', ha='center', va='center',
        fontsize=10, color='#424242', style='italic')

# 2. Measurement Plane (above T-array)
draw_box(ax, 4.5, 7.3, 5, 1.1, 'Measurement Plane',
         'Sample & hold at times $\\{t_k\\}$',
         colors['measure'], colors['measure_border'])

# 3. Local Statistics Layer (right of T-array)
draw_box(ax, 10, 4.3, 3.5, 2.7, 'Local Statistics',
         'Correlators\n$m_i = \\langle x_i \\rangle$\n$C_{ij} = \\langle x_i x_j \\rangle$',
         colors['stats'], colors['stats_border'])

# 4. Gradient Engine (below statistics)
draw_box(ax, 10, 1.2, 3.5, 2.5, 'Gradient Engine',
         'Compute $\\Delta\\theta$\nfrom learning rules',
         colors['gradient'], colors['gradient_border'])

# 5. Reconfiguration Fabric (left of T-array)
draw_box(ax, 0.5, 3.8, 3.5, 3.2, 'Reconfiguration',
         'Dual-bank DACs\nBias banks $\\{\\mathbf{b}_k\\}$\nCoupling matrix $J$',
         colors['reconfig'], colors['reconfig_border'])

# Arrows showing data flow - cleaner positioning
arrow_style = dict(arrowstyle='->', mutation_scale=18, linewidth=2.5)

# T-array -> Measurement (up)
ax.annotate('', xy=(7, 7.3), xytext=(7, 7.0),
            arrowprops=dict(**arrow_style, color=colors['measure_border']))
ax.text(7.25, 7.15, 'states', fontsize=9, color=colors['measure_border'], va='center')

# Measurement -> Statistics (right-down curve)
ax.annotate('', xy=(10, 5.9), xytext=(9.5, 7.8),
            arrowprops=dict(**arrow_style, color=colors['stats_border'],
                          connectionstyle='arc3,rad=-0.2'))
ax.text(10.1, 7.0, '$x_i(t_k)$', fontsize=10, color=colors['stats_border'])

# Statistics -> Gradient (down)
ax.annotate('', xy=(11.75, 3.7), xytext=(11.75, 4.3),
            arrowprops=dict(**arrow_style, color=colors['gradient_border']))
ax.text(12.0, 4.0, 'stats', fontsize=9, color=colors['gradient_border'], va='center')

# Gradient -> Reconfig (long arrow across bottom)
ax.annotate('', xy=(4, 2.45), xytext=(10, 2.45),
            arrowprops=dict(**arrow_style, color=colors['reconfig_border']))
ax.text(7, 2.75, '$\\Delta J,\\, \\Delta \\mathbf{b}$', fontsize=11,
        ha='center', color=colors['reconfig_border'])

# Reconfig -> T-array (right)
ax.annotate('', xy=(4.5, 5.4), xytext=(4.0, 5.4),
            arrowprops=dict(**arrow_style, color=colors['reconfig_border']))
ax.text(4.25, 5.7, 'params', fontsize=9, color=colors['reconfig_border'], ha='center')

# Reconfig receives initial data (from below)
ax.annotate('', xy=(2.25, 3.8), xytext=(2.25, 3.3),
            arrowprops=dict(**arrow_style, color='#78909C'))
ax.text(2.25, 3.1, 'data init', fontsize=9, color='#607D8B', ha='center')

# Cycle number indicators (subtle, positioned to not overlap)
cycle_style = dict(fontsize=9, fontweight='bold', ha='center', va='center',
                   bbox=dict(boxstyle='circle,pad=0.12', facecolor='white',
                            edgecolor='#BDBDBD', linewidth=1))
ax.text(6.6, 7.6, '①', color='#9E9E9E', **cycle_style)
ax.text(9.8, 6.7, '②', color='#9E9E9E', **cycle_style)
ax.text(12.0, 3.55, '③', color='#9E9E9E', **cycle_style)
ax.text(6.5, 2.45, '④', color='#9E9E9E', **cycle_style)
ax.text(4.2, 5.0, '⑤', color='#9E9E9E', **cycle_style)

# Legend at bottom - cleaner format
ax.text(7, 0.5, '① Measure states  →  ② Sample trajectories  →  ③ Compute statistics  →  ④ Update parameters  →  ⑤ Reconfigure',
        ha='center', fontsize=10, color='#616161', style='italic')

plt.tight_layout()
plt.savefig('paper_figures/architecture_schematic.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved paper_figures/architecture_schematic.png")
