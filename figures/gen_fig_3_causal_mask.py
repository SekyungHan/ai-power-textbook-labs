import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import font_manager as fm
import os

# Korean font setup - try multiple paths
font_paths = [
    os.path.expanduser('~/fonts/google_korean/NanumGothic.ttf'),
    os.path.expanduser('~/fonts/google_korean/NanumGothicBold.ttf'),
    os.path.expanduser('~/tmp_fonts/PAPERLOGY-5MEDIUM.TTF'),
]
font_loaded = False
for font_path in font_paths:
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        if not font_loaded:
            font_prop = fm.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = font_prop.get_name()
            font_loaded = True
if not font_loaded:
    plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

tokens = ["전력", "계통", "안정도", "분석", "결과"]
n = 5

# Build causal attention matrix with softmax-like values per row
np.random.seed(42)
raw = np.random.rand(n, n) * 2 + 0.5  # positive raw scores

# Apply causal mask and row-wise softmax
attention = np.full((n, n), np.nan)
for i in range(n):
    row_vals = raw[i, :i+1]
    exp_vals = np.exp(row_vals - np.max(row_vals))
    softmax_vals = exp_vals / exp_vals.sum()
    attention[i, :i+1] = softmax_vals

# For display: lower triangle gets attention values, upper gets -inf
display_vals = np.full((n, n), -np.inf)
for i in range(n):
    for j in range(i+1):
        display_vals[i, j] = attention[i, j]

# Custom colormap: white to teal (#1B7A8A)
cmap = LinearSegmentedColormap.from_list("teal_seq", ["#FFFFFF", "#1B7A8A"])

# For plotting heatmap: masked region shown as grey
plot_data = np.where(np.isnan(attention), np.nan, attention)

fig, ax = plt.subplots(figsize=(6, 5), dpi=150)

# Draw masked cells (upper triangle) as grey
for i in range(n):
    for j in range(i+1, n):
        ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor='#D5D8DC', edgecolor='white', linewidth=1.5))

# Draw attention cells (lower triangle including diagonal)
for i in range(n):
    for j in range(i+1):
        val = attention[i, j]
        color = cmap(val)
        ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor='white', linewidth=1.5))

# Annotate cells
for i in range(n):
    for j in range(n):
        if j <= i:
            val = attention[i, j]
            # Dark text on light cells, light text on dark cells
            text_color = 'white' if val > 0.55 else '#2C3E50'
            ax.text(j + 0.5, i + 0.5, f"{val:.2f}", ha='center', va='center',
                    fontsize=11, fontweight='bold', color=text_color)
        else:
            ax.text(j + 0.5, i + 0.5, "-\u221E", ha='center', va='center',
                    fontsize=11, color='#7F8C8D', fontweight='bold')

ax.set_xlim(0, n)
ax.set_ylim(n, 0)
ax.set_xticks([i + 0.5 for i in range(n)])
ax.set_yticks([i + 0.5 for i in range(n)])
ax.set_xticklabels(tokens, fontsize=11, fontweight='bold')
ax.set_yticklabels(tokens, fontsize=11, fontweight='bold')
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')

# Remove default spines, add outer border
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(length=0)

ax.set_title("Masked Self-Attention (Causal Mask)", fontsize=14, fontweight='bold',
             color='#2C3E50', pad=20, fontfamily='sans-serif')

# Add colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)
cbar.set_label("Attention Score", fontsize=10, color='#2C3E50')
cbar.ax.tick_params(labelsize=9)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_3_causal_mask.png"),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Done")
