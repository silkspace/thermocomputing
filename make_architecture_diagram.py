"""
Generate a compact schematic diagram of the Criticality Engine processor architecture.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.set_xlim(0, 12)
ax.set_ylim(0, 6)
ax.set_aspect('equal')
ax.axis('off')

# Colors
colors = {
    'tarray': '#E3F2FD', 'tarray_border': '#1565C0',
    'measure': '#FFF8E1', 'measure_border': '#FF8F00',
    'stats': '#F3E5F5', 'stats_border': '#7B1FA2',
    'gradient': '#E8F5E9', 'gradient_border': '#2E7D32',
    'reconfig': '#FFEBEE', 'reconfig_border': '#C62828',
}

def box(x, y, w, h, label, sublabel, color, border):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                        facecolor=color, edgecolor=border, linewidth=2)
    ax.add_patch(b)
    ax.text(x + w/2, y + h - 0.2, label, ha='center', va='top',
            fontsize=10, fontweight='bold', color=border)
    ax.text(x + w/2, y + 0.25, sublabel, ha='center', va='center',
            fontsize=8, color='#424242')

# Title
ax.text(6, 5.7, 'Criticality Engine: Processor Architecture',
        ha='center', fontsize=13, fontweight='bold', color='#212121')

# Compact layout - all boxes sized to content
# Center: T-array with nodes
tarray = FancyBboxPatch((3.5, 1.8), 5, 2.4, boxstyle="round,pad=0.02,rounding_size=0.15",
                         facecolor=colors['tarray'], edgecolor=colors['tarray_border'], linewidth=2)
ax.add_patch(tarray)
ax.text(6, 3.95, 'Thermodynamic Array', ha='center', va='top',
        fontsize=10, fontweight='bold', color=colors['tarray_border'])

# 5x2 node grid - compact
for i in range(5):
    for j in range(2):
        cx = 4.0 + i * 0.9
        cy = 2.5 + j * 0.7
        circle = Circle((cx, cy), 0.18, facecolor='white',
                        edgecolor=colors['tarray_border'], linewidth=1.2, zorder=3)
        ax.add_patch(circle)
        if i < 4:
            ax.plot([cx + 0.18, cx + 0.72], [cy, cy],
                   color=colors['tarray_border'], linewidth=1, alpha=0.5, zorder=1)
        if j < 1:
            ax.plot([cx, cx], [cy + 0.18, cy + 0.52],
                   color=colors['tarray_border'], linewidth=1, alpha=0.5, zorder=1)

ax.text(6, 2.0, '$dx = -\\nabla V\\,dt + \\sqrt{2kT}\\,dW$',
        ha='center', fontsize=8, color='#546E7A', style='italic')

# Top: Measurement (compact)
box(4.2, 4.4, 3.6, 0.9, 'Measurement Plane', 'Sample & hold at $\\{t_k\\}$',
    colors['measure'], colors['measure_border'])

# Right top: Local Stats (compact)
box(9, 3.2, 2.7, 1.4, 'Local Statistics', 'Correlators $m_i, C_{ij}$',
    colors['stats'], colors['stats_border'])

# Right bottom: Gradient (compact)
box(9, 1.4, 2.7, 1.4, 'Gradient Engine', 'Compute $\\Delta\\theta$',
    colors['gradient'], colors['gradient_border'])

# Left: Reconfig (compact)
box(0.3, 1.4, 2.7, 3.2, 'Reconfiguration', 'Dual-bank DACs\n$\\{\\mathbf{b}_k\\}, J$',
    colors['reconfig'], colors['reconfig_border'])

# Arrows - tight
arrow_kw = dict(arrowstyle='->', mutation_scale=12, linewidth=1.5)

# T-array → Measure
ax.annotate('', xy=(6, 4.4), xytext=(6, 4.2),
            arrowprops=dict(**arrow_kw, color=colors['measure_border']))

# Measure → Stats
ax.annotate('', xy=(9, 4.2), xytext=(7.8, 4.85),
            arrowprops=dict(**arrow_kw, color=colors['stats_border'],
                          connectionstyle='arc3,rad=-0.2'))
ax.text(8.7, 4.7, '$x_i(t_k)$', fontsize=8, color=colors['stats_border'])

# Stats → Gradient
ax.annotate('', xy=(10.35, 2.8), xytext=(10.35, 3.2),
            arrowprops=dict(**arrow_kw, color=colors['gradient_border']))

# Gradient → Reconfig
ax.annotate('', xy=(3, 1.7), xytext=(9, 1.7),
            arrowprops=dict(**arrow_kw, color=colors['reconfig_border']))
ax.text(6, 1.4, '$\\Delta J, \\Delta\\mathbf{b}$', ha='center',
        fontsize=9, color=colors['reconfig_border'])

# Reconfig → T-array
ax.annotate('', xy=(3.5, 3), xytext=(3, 3),
            arrowprops=dict(**arrow_kw, color=colors['reconfig_border']))
ax.text(3.25, 3.3, 'params', fontsize=7, ha='center', color=colors['reconfig_border'])

# Compact legend
ax.text(6, 0.7, '① Sample → ② Measure → ③ Statistics → ④ Gradients → ⑤ Reconfigure',
        ha='center', fontsize=8, color='#9E9E9E', style='italic')

plt.tight_layout()
plt.savefig('paper_figures/architecture_schematic.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved paper_figures/architecture_schematic.png")
