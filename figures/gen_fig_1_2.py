import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import ListedColormap
import os

# Font setup - try multiple paths
font_paths = [
    os.path.expanduser('~/fonts/google_korean/NanumGothic.ttf'),
    os.path.expanduser('~/tmp_fonts/PAPERLOGY-5MEDIUM.TTF'),
]
font_loaded = False
for font_path in font_paths:
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        font_prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
        font_loaded = True
        break
if not font_loaded:
    plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Colors
NAVY = '#2C3E50'
TEAL = '#1B7A8A'
AMBER = '#D4984A'
CORAL = '#C75C3A'

# XOR data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor('white')

# Adjust spacing to leave room for arrow
fig.subplots_adjust(wspace=0.35)

# === LEFT PANEL: XOR linear inseparability ===
ax1.set_title('XOR: \uc120\ud615 \ubd84\ub9ac \ubd88\uac00', fontsize=15, fontweight='bold', color=NAVY, pad=12)

for i in range(4):
    color = AMBER if y[i] == 0 else TEAL
    label = 'Class 0' if y[i] == 0 else 'Class 1'
    # Avoid duplicate legend entries
    if i < 2:
        ax1.scatter(X[i, 0], X[i, 1], c=color, s=200, zorder=5, edgecolors=NAVY,
                    linewidths=1.5, label=label)
    else:
        ax1.scatter(X[i, 0], X[i, 1], c=color, s=200, zorder=5, edgecolors=NAVY, linewidths=1.5)

# Dashed line showing failed linear separation
x_line = np.linspace(-0.3, 1.3, 100)
ax1.plot(x_line, np.full_like(x_line, 0.5), '--', color=CORAL, linewidth=2, alpha=0.8, label='\uc120\ud615 \uacbd\uacc4')

# Red X marks showing failure
ax1.plot(0.5, 0.5, 'x', color=CORAL, markersize=18, markeredgewidth=3, zorder=6)

ax1.set_xlabel('x\u2081', fontsize=14)
ax1.set_ylabel('x\u2082', fontsize=14)
ax1.set_xlim(-0.3, 1.3)
ax1.set_ylim(-0.3, 1.3)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10, loc='upper right')

# === RIGHT PANEL: MLP nonlinear decision boundary ===
ax2.set_title('MLP: \ube44\uc120\ud615 \uacb0\uc815 \uacbd\uacc4', fontsize=15, fontweight='bold', color=NAVY, pad=12)

# Create a simple MLP-like decision boundary using a trained-style function
# Simulate XOR decision boundary: region where XOR output ~ 1
xx, yy = np.meshgrid(np.linspace(-0.3, 1.3, 300), np.linspace(-0.3, 1.3, 300))

# Approximate XOR with a smooth function: high when exactly one input is ~1
# Using a formula that mimics trained MLP output
z1 = 1 / (1 + np.exp(-12 * (xx + yy - 0.5)))   # x1 OR x2
z2 = 1 / (1 + np.exp(-12 * (xx + yy - 1.5)))   # x1 AND x2
zz = z1 - z2  # XOR approximation

# Color map: amber for class 0 region, teal for class 1 region
cmap_bg = ListedColormap([AMBER + '40', TEAL + '40'])
ax2.contourf(xx, yy, zz, levels=[0, 0.5, 1.0], colors=[AMBER, TEAL], alpha=0.2)
ax2.contour(xx, yy, zz, levels=[0.5], colors=[NAVY], linewidths=2, linestyles='--')

for i in range(4):
    color = AMBER if y[i] == 0 else TEAL
    ax2.scatter(X[i, 0], X[i, 1], c=color, s=200, zorder=5, edgecolors=NAVY, linewidths=1.5)

ax2.set_xlabel('x\u2081', fontsize=14)
ax2.set_ylabel('x\u2082', fontsize=14)
ax2.set_xlim(-0.3, 1.3)
ax2.set_ylim(-0.3, 1.3)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)

# Arrow between panels
fig.text(0.50, 0.5, '\u2192', fontsize=36, ha='center', va='center',
         fontweight='bold', color=NAVY, transform=fig.transFigure)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(OUT_DIR, 'fig_1_2_xor_mlp.png')
fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

size = os.path.getsize(output_path)
print(f"Saved: {output_path} ({size:,} bytes)")
