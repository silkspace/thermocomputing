"""
Generate a schematic diagram of the Criticality Engine processor architecture.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

# Set up figure
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.set_aspect('equal')
ax.axis('off')

# Colors
colors = {
    'tarray': '#E8F4FD',      # Light blue - physics
    'tarray_border': '#1976D2',
    'measure': '#FFF3E0',      # Light orange - observation
    'measure_border': '#F57C00',
    'stats': '#F3E5F5',        # Light purple - computation
    'stats_border': '#7B1FA2',
    'gradient': '#E8F5E9',     # Light green - learning
    'gradient_border': '#388E3C',
    'reconfig': '#FFEBEE',     # Light red - control
    'reconfig_border': '#D32F2F',
    'arrow': '#424242',
    'text': '#212121',
}

def draw_box(ax, x, y, w, h, label, sublabel, color, border_color, fontsize=12):
    """Draw a rounded box with label."""
    box = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.02,rounding_size=0.3",
                          facecolor=color, edgecolor=border_color, linewidth=2.5)
    ax.add_patch(box)
    ax.text(x + w/2, y + h - 0.4, label, ha='center', va='top',
            fontsize=fontsize, fontweight='bold', color=border_color)
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.2, sublabel, ha='center', va='center',
                fontsize=9, color=colors['text'], style='italic',
                wrap=True)

def draw_arrow(ax, start, end, color='#424242', style='->'):
    """Draw an arrow between two points."""
    arrow = FancyArrowPatch(start, end,
                            arrowstyle=style,
                            mutation_scale=15,
                            color=color, linewidth=2)
    ax.add_patch(arrow)

# Title
ax.text(7, 9.7, 'Criticality Engine: Processor Architecture',
        ha='center', va='top', fontsize=16, fontweight='bold')

# 1. T-array (center, large)
draw_box(ax, 4.5, 4.5, 5, 3, 'Thermodynamic Array',
         'N stochastic nodes\nLangevin dynamics\n$dx = -\\nabla V dt + \\sqrt{2kT}dW$',
         colors['tarray'], colors['tarray_border'], fontsize=13)

# Draw some nodes inside T-array to illustrate
for i in range(4):
    for j in range(3):
        cx = 5.2 + i * 1.0
        cy = 5.0 + j * 0.7
        circle = Circle((cx, cy), 0.15, facecolor='white',
                        edgecolor=colors['tarray_border'], linewidth=1.5)
        ax.add_patch(circle)
        # Add connections between adjacent nodes
        if i < 3:
            ax.plot([cx + 0.15, cx + 0.85], [cy, cy],
                   color=colors['tarray_border'], linewidth=1, alpha=0.5)
        if j < 2:
            ax.plot([cx, cx], [cy + 0.15, cy + 0.55],
                   color=colors['tarray_border'], linewidth=1, alpha=0.5)

# 2. Measurement Plane (above T-array)
draw_box(ax, 4.5, 8, 5, 1.2, 'Measurement Plane',
         'Sample & hold at times $\\{t_k\\}$',
         colors['measure'], colors['measure_border'])

# 3. Local Statistics Layer (right of T-array)
draw_box(ax, 10, 4.5, 3.5, 3, 'Local Statistics',
         'Correlators\n$m_i = \\langle x_i \\rangle$\n$C_{ij} = \\langle x_i x_j \\rangle$',
         colors['stats'], colors['stats_border'])

# 4. Gradient Engine (below statistics)
draw_box(ax, 10, 1.5, 3.5, 2.5, 'Gradient Engine',
         'Compute $\\Delta\\theta$\nfrom learning rules',
         colors['gradient'], colors['gradient_border'])

# 5. Reconfiguration Fabric (left of T-array)
draw_box(ax, 0.5, 4.5, 3.5, 3, 'Reconfiguration',
         'Dual-bank DACs\nBias banks $\\{\\mathbf{b}_k\\}$\nCoupling matrix $J$',
         colors['reconfig'], colors['reconfig_border'])

# 6. Data input (bottom left)
draw_box(ax, 0.5, 1.5, 3.5, 2.5, 'Data / Labels',
         'Training samples\nClass labels $k$',
         '#ECEFF1', '#607D8B')

# Arrows showing data flow
# T-array -> Measurement
draw_arrow(ax, (7, 7.5), (7, 8), colors['measure_border'])
ax.text(7.2, 7.75, 'states', fontsize=8, color=colors['measure_border'])

# Measurement -> Statistics
draw_arrow(ax, (9.5, 8.5), (10, 6.5), colors['stats_border'])
ax.text(9.9, 7.6, '$x_i(t_k)$', fontsize=9, color=colors['stats_border'])

# Statistics -> Gradient
draw_arrow(ax, (11.75, 4.5), (11.75, 4), colors['gradient_border'])
ax.text(12, 4.25, 'stats', fontsize=8, color=colors['gradient_border'])

# Gradient -> Reconfig
draw_arrow(ax, (10, 2.75), (4, 2.75), colors['reconfig_border'])
ax.text(7, 3, '$\\Delta J, \\Delta \\mathbf{b}$', fontsize=9,
        ha='center', color=colors['reconfig_border'])

# Reconfig -> T-array
draw_arrow(ax, (4, 6), (4.5, 6), colors['reconfig_border'])
ax.text(3.8, 6.3, 'params', fontsize=8, color=colors['reconfig_border'], ha='right')

# Data -> T-array (initialization)
draw_arrow(ax, (2.25, 4), (5, 4.5), colors['arrow'])
ax.text(3.5, 4.5, 'init', fontsize=8, color=colors['arrow'])

# Data -> Gradient (labels for contrastive)
draw_arrow(ax, (4, 2.75), (10, 2.75), colors['gradient_border'], style='<-')

# Cycle annotation
ax.annotate('', xy=(1, 8), xytext=(3, 8),
            arrowprops=dict(arrowstyle='->', color='#9E9E9E', lw=1.5,
                           connectionstyle='arc3,rad=-0.3'))
ax.text(2, 8.7, 'Learning\nCycle', ha='center', fontsize=9,
        color='#757575', style='italic')

# Legend / key operations
legend_y = 0.8
ax.text(0.5, legend_y, 'Operation cycle:', fontsize=10, fontweight='bold')
ax.text(0.5, legend_y - 0.4, '① Initialize T-array with data', fontsize=9)
ax.text(4.5, legend_y - 0.4, '② Evolve under Langevin dynamics', fontsize=9)
ax.text(9, legend_y - 0.4, '③ Measure trajectories', fontsize=9)
ax.text(0.5, legend_y - 0.8, '④ Compute local statistics', fontsize=9)
ax.text(4.5, legend_y - 0.8, '⑤ Update parameters via gradient', fontsize=9)
ax.text(9, legend_y - 0.8, '⑥ Atomic reconfiguration', fontsize=9)

plt.tight_layout()
plt.savefig('architecture_schematic.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved architecture_schematic.png")
