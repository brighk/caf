import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

fig, ax = plt.subplots(figsize=(12, 5))
ax.set_xlim(0, 12)
ax.set_ylim(0, 5)
ax.axis("off")

def add_box(x, y, w, h, text, fontsize=12, lw=1.8):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=lw,
        facecolor="white",
        edgecolor="black"
    )
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize)

def add_arrow(x1, y1, x2, y2, lw=1.8):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="->",
        mutation_scale=14,
        linewidth=lw,
        color="black"
    )
    ax.add_patch(arr)

# Section headers
ax.text(2.0, 4.65, "Inference Layer", ha="center", va="center", fontsize=13, fontweight="bold")
ax.text(6.0, 4.65, "Formal Verification Layer", ha="center", va="center", fontsize=13, fontweight="bold")
ax.text(10.0, 4.65, "Deterministic Executive", ha="center", va="center", fontsize=13, fontweight="bold")

# Vertical dividers
ax.add_line(Line2D([4, 4], [0.4, 4.35], linewidth=1.2, color="black", alpha=0.4))
ax.add_line(Line2D([8, 8], [0.4, 4.35], linewidth=1.2, color="black", alpha=0.4))

# Boxes
add_box(0.7, 2.9, 2.6, 1.0, "LLM Proposer")

add_box(4.5, 2.9, 3.0, 1.0, "Semantic Parser + KB")
add_box(4.5, 1.3, 3.0, 1.0, "RDF Knowledge Base")
add_box(4.5, 0.7, 3.0, 0.45, "Constraints", fontsize=11)

add_box(8.7, 2.9, 2.6, 1.0, "SCM + Entailment")
add_box(8.7, 1.3, 2.6, 1.0, "Deterministic Executive")

# Main pipeline arrows
add_arrow(3.35, 3.4, 4.45, 3.4)   # LLM -> Semantic Parser
add_arrow(7.55, 3.4, 8.65, 3.4)   # Semantic Parser -> SCM + Entailment

# Internal verification arrows
add_arrow(6.0, 2.85, 6.0, 2.35)   # Parser -> RDF KB
add_arrow(6.0, 2.3, 6.0, 1.85)    # (visual spacing)
add_arrow(6.0, 1.25, 6.0, 1.18)   # RDF KB -> Constraints

# Executive arrows
add_arrow(10.0, 2.85, 10.0, 2.35) # SCM + Entailment -> DE
add_arrow(11.3, 1.8, 11.7, 1.8)   # DE -> Response
ax.text(11.82, 1.8, "Response", ha="left", va="center", fontsize=12)


out_path = "caf_pipeline_figure2.png"
plt.tight_layout()
plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.show()
