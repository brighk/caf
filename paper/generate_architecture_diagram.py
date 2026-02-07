#!/usr/bin/env python3
"""
Generate CAF architecture diagram as publication-ready figure.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as patches

# Set publication style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11

fig, ax = plt.subplots(figsize=(12, 4))
ax.set_xlim(0, 12)
ax.set_ylim(0, 4)
ax.axis('off')

# Define box positions
il_box = FancyBboxPatch((0.5, 1.5), 2.5, 1.5,
                        boxstyle="round,pad=0.1",
                        edgecolor='black',
                        facecolor='#E8F4F8',
                        linewidth=2)

fvl_box = FancyBboxPatch((4.0, 1.5), 2.5, 1.5,
                         boxstyle="round,pad=0.1",
                         edgecolor='black',
                         facecolor='#F0E6F6',
                         linewidth=2)

de_box = FancyBboxPatch((7.5, 1.5), 2.5, 1.5,
                        boxstyle="round,pad=0.1",
                        edgecolor='black',
                        facecolor='#E8F8E8',
                        linewidth=2)

kb_box = FancyBboxPatch((4.0, 0.2), 2.5, 1.0,
                        boxstyle="round,pad=0.1",
                        edgecolor='black',
                        facecolor='#FFF4E6',
                        linewidth=2)

# Add boxes to plot
ax.add_patch(il_box)
ax.add_patch(fvl_box)
ax.add_patch(de_box)
ax.add_patch(kb_box)

# Add text labels
ax.text(1.75, 2.5, 'Inference Layer', ha='center', va='center',
        fontsize=12, fontweight='bold')
ax.text(1.75, 2.0, 'LLM Proposer', ha='center', va='center',
        fontsize=10, style='italic')

ax.text(5.25, 2.5, 'Formal Verification', ha='center', va='center',
        fontsize=12, fontweight='bold')
ax.text(5.25, 2.0, 'Layer', ha='center', va='center',
        fontsize=12, fontweight='bold')
ax.text(5.25, 1.7, 'Semantic Parser + KB', ha='center', va='center',
        fontsize=9, style='italic')

ax.text(8.75, 2.5, 'Deterministic', ha='center', va='center',
        fontsize=12, fontweight='bold')
ax.text(8.75, 2.0, 'Executive', ha='center', va='center',
        fontsize=12, fontweight='bold')
ax.text(8.75, 1.7, 'SCM + Entailment', ha='center', va='center',
        fontsize=9, style='italic')

ax.text(5.25, 0.7, 'RDF Knowledge Base', ha='center', va='center',
        fontsize=11, fontweight='bold')

# Add forward arrows
# IL -> FVL
arrow1 = FancyArrowPatch((3.0, 2.25), (4.0, 2.25),
                        arrowstyle='->', mutation_scale=20,
                        linewidth=2, color='black')
ax.add_patch(arrow1)

# FVL -> DE
arrow2 = FancyArrowPatch((6.5, 2.25), (7.5, 2.25),
                        arrowstyle='->', mutation_scale=20,
                        linewidth=2, color='black')
ax.add_patch(arrow2)

# FVL -> KB
arrow3 = FancyArrowPatch((5.25, 1.5), (5.25, 1.2),
                        arrowstyle='->', mutation_scale=20,
                        linewidth=2, color='black')
ax.add_patch(arrow3)

# Feedback arrow: DE -> IL (dashed, curved)
# Draw it in segments to go around the top
arrow4 = FancyArrowPatch((8.75, 3.0), (1.75, 3.0),
                        arrowstyle='->', mutation_scale=20,
                        linewidth=2, color='#D62728',
                        linestyle='--',
                        connectionstyle="arc3,rad=0")
ax.add_patch(arrow4)

# Add "Constraints" label above feedback arrow
ax.text(5.25, 3.3, 'Constraints', ha='center', va='bottom',
        fontsize=11, color='#D62728', fontweight='bold')

# Add connecting lines for the feedback arrow
ax.plot([8.75, 8.75], [2.25, 3.0], 'r--', linewidth=2)
ax.plot([1.75, 1.75], [3.0, 2.25], 'r--', linewidth=2)

plt.tight_layout()

# Save as both PDF and PNG
plt.savefig('paper/figures/architecture_overview.pdf', dpi=300, bbox_inches='tight')
plt.savefig('paper/figures/architecture_overview.png', dpi=300, bbox_inches='tight')

print("✓ Architecture diagram saved:")
print("  - paper/figures/architecture_overview.pdf")
print("  - paper/figures/architecture_overview.png")

plt.close()
