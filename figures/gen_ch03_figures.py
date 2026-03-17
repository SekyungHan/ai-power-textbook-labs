"""
Chapter 3 figures for AI Power Systems textbook.
Transformer & LLM: attention heatmap, positional encoding, scaling laws, OPF comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.colors import LinearSegmentedColormap
import warnings
import os
warnings.filterwarnings('ignore')

# === Style Setup ===
NAVY = '#2C3E50'
TEAL = '#1B7A8A'
AMBER = '#D4984A'
SAGE = '#5A7D6A'
CORAL = '#C75C3A'

# Try to register Korean fonts
font_paths = [
    os.path.expanduser('~/fonts/google_korean/NanumGothic.ttf'),
    os.path.expanduser('~/fonts/google_korean/NanumGothicBold.ttf'),
]
for fp in font_paths:
    if os.path.exists(fp):
        fontManager.addfont(fp)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['NanumGothic', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': NAVY,
    'axes.labelcolor': NAVY,
    'xtick.color': NAVY,
    'ytick.color': NAVY,
    'text.color': NAVY,
    'axes.grid': False,
    'grid.alpha': 0.3,
    'grid.color': '#cccccc',
    'lines.linewidth': 1.8,
})

OUT = os.path.dirname(os.path.abspath(__file__)) + '/'
DPI = 200

np.random.seed(42)


# ============================================================
# Fig 3.2 -- Attention Weight Heatmap
# ============================================================
def fig_3_2_attention_heatmap():
    tokens = ["전력", "계통의", "안정도", "분석을", "위한", "시뮬레이션", "결과를", "검증"]
    n = len(tokens)

    # Build a plausible attention pattern
    attn = np.random.uniform(0.01, 0.08, (n, n))

    # Diagonal: self-attention is moderate
    for i in range(n):
        attn[i, i] = np.random.uniform(0.15, 0.30)

    # Semantic links
    pairs = [(0, 1), (1, 0), (2, 3), (3, 2), (5, 6), (6, 5),
             (0, 2), (2, 0), (4, 5), (3, 5), (6, 7), (7, 6)]
    for i, j in pairs:
        attn[i, j] = np.random.uniform(0.18, 0.35)

    attn[2, 0] = 0.28
    attn[2, 1] = 0.25

    # Normalize rows to sum to 1 (softmax-like)
    attn = attn / attn.sum(axis=1, keepdims=True)

    # Custom colormap from white to teal
    cmap = LinearSegmentedColormap.from_list('teal_seq', ['#FFFFFF', '#B0D4DB', TEAL, NAVY])

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(attn, cmap=cmap, aspect='equal', vmin=0, vmax=0.35)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tokens, fontsize=12)
    ax.set_yticklabels(tokens, fontsize=12)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    # Annotate each cell
    for i in range(n):
        for j in range(n):
            val = attn[i, j]
            color = 'white' if val > 0.2 else NAVY
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=9, color=color, fontweight='bold' if val > 0.15 else 'normal')

    ax.set_xlabel('Key 토큰', fontsize=12, labelpad=10)
    ax.set_ylabel('Query 토큰', fontsize=12)
    ax.set_title('그림 3.2  셀프 어텐션 가중치 히트맵\n(전력 계통 문장 예시)',
                 fontsize=14, fontweight='bold', pad=20)

    # Thin colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('어텐션 가중치', fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    # Clean borders
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT + 'fig_3_2_attention_heatmap.png', dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print('[OK] fig_3_2_attention_heatmap.png')


# ============================================================
# Fig 3.5 -- Positional Encoding Visualization
# ============================================================
def fig_3_5_positional_encoding():
    max_pos = 50
    d_model = 64

    pe = np.zeros((max_pos, d_model))
    position = np.arange(max_pos)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7),
                              gridspec_kw={'width_ratios': [1.4, 1]})

    # --- Left: 2D heatmap ---
    ax = axes[0]
    cmap = LinearSegmentedColormap.from_list('navy_amber', [NAVY, '#FFFFFF', AMBER])
    im = ax.imshow(pe.T, cmap=cmap, aspect='auto', interpolation='nearest',
                    vmin=-1, vmax=1)
    ax.set_xlabel('위치 (position)', fontsize=12)
    ax.set_ylabel('임베딩 차원 (dimension)', fontsize=12)
    ax.set_title('위치 인코딩 히트맵', fontsize=13, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('인코딩 값', fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # --- Right: individual dimension curves ---
    ax2 = axes[1]
    dims_to_show = [0, 1, 4, 5, 16, 17, 48, 49]
    colors_cycle = [TEAL, TEAL, AMBER, AMBER, SAGE, SAGE, CORAL, CORAL]
    styles = ['-', '--', '-', '--', '-', '--', '-', '--']

    for dim, c, ls in zip(dims_to_show, colors_cycle, styles):
        label = f'd={dim} ({"sin" if dim % 2 == 0 else "cos"})'
        ax2.plot(range(max_pos), pe[:, dim], color=c, linestyle=ls,
                 linewidth=1.5, alpha=0.85, label=label)

    ax2.set_xlabel('위치 (position)', fontsize=12)
    ax2.set_ylabel('인코딩 값', fontsize=12)
    ax2.set_title('차원별 위치 인코딩 곡선', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=8, ncol=2, loc='upper right', framealpha=0.9)
    ax2.set_ylim(-1.15, 1.15)
    ax2.axhline(0, color='#cccccc', linewidth=0.5, zorder=0)
    ax2.grid(True, alpha=0.2)

    for spine in ['top', 'right']:
        ax2.spines[spine].set_visible(False)

    fig.suptitle('그림 3.5  사인/코사인 위치 인코딩', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(OUT + 'fig_3_5_positional_encoding.png', dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print('[OK] fig_3_5_positional_encoding.png')


# ============================================================
# Fig 3.7 -- Scaling Laws
# ============================================================
def fig_3_7_scaling_laws():
    # Model data: (name, params_billions, loss_approx)
    models = [
        ('GPT-2\n(117M)',   0.117,  3.30),
        ('GPT-2\n(345M)',   0.345,  2.85),
        ('GPT-2\n(774M)',   0.774,  2.65),
        ('GPT-2\n(1.5B)',   1.5,    2.50),
        ('GPT-3\n(6.7B)',   6.7,    2.10),
        ('GPT-3\n(13B)',    13.0,   1.95),
        ('GPT-3\n(175B)',   175.0,  1.55),
        ('Chinchilla\n(70B)', 70.0, 1.60),
        ('LLaMA\n(7B)',     7.0,    1.92),
        ('LLaMA\n(13B)',    13.0,   1.80),
        ('LLaMA\n(65B)',    65.0,   1.58),
        ('GPT-4급\n(~1.8T)', 1800., 1.15),
    ]

    # Chinchilla-optimal line (compute-matched)
    chinchilla_pts = [
        ('', 0.4, 2.95),
        ('', 1.0, 2.60),
        ('', 3.0, 2.20),
        ('', 10., 1.90),
        ('', 30., 1.68),
        ('', 70., 1.60),
        ('', 200., 1.42),
        ('', 500., 1.28),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Main scaling fit line
    params_arr = np.array([m[1] for m in models])
    loss_arr = np.array([m[2] for m in models])

    # Log-linear fit
    log_p = np.log10(params_arr)
    coeffs = np.polyfit(log_p, loss_arr, 1)
    fit_x = np.logspace(-1, 3.5, 200)
    fit_y = np.polyval(coeffs, np.log10(fit_x))

    ax.plot(fit_x, fit_y, '--', color='#aaaaaa', linewidth=1.2, zorder=1,
            label='로그-선형 피팅')

    # Chinchilla optimal line
    ch_params = np.array([c[1] for c in chinchilla_pts])
    ch_loss = np.array([c[2] for c in chinchilla_pts])
    ax.plot(ch_params, ch_loss, '-', color=AMBER, linewidth=2.0, alpha=0.8,
            zorder=2, label='Chinchilla-최적 경로')
    ax.scatter(ch_params, ch_loss, color=AMBER, s=20, zorder=3, alpha=0.6)

    # Main model scatter
    for name, p, l in models:
        color = TEAL
        marker = 'o'
        size = 60
        if 'Chinchilla' in name:
            color = AMBER
            marker = 'D'
            size = 80
        elif 'LLaMA' in name:
            color = SAGE
            marker = 's'
            size = 60
        elif 'GPT-4' in name:
            color = CORAL
            marker = '*'
            size = 150

        ax.scatter(p, l, color=color, marker=marker, s=size, zorder=5, edgecolors='white', linewidth=0.5)

        # Annotation
        offset = (8, 5)
        if 'GPT-4' in name:
            offset = (-15, -18)
        elif 'GPT-3\n(175B)' in name:
            offset = (-10, -18)
        elif 'Chinchilla' in name:
            offset = (8, -15)

        ax.annotate(name, (p, l), textcoords='offset points', xytext=offset,
                    fontsize=7.5, color=NAVY, ha='left', va='bottom')

    ax.set_xscale('log')
    ax.set_xlabel('모델 파라미터 수 (십억)', fontsize=12)
    ax.set_ylabel('검증 손실 (Validation Loss)', fontsize=12)
    ax.set_title('그림 3.7  LLM 스케일링 법칙: 파라미터 수 vs 성능',
                 fontsize=14, fontweight='bold', pad=15)

    ax.set_xlim(0.08, 5000)
    ax.set_ylim(0.9, 3.6)

    # Custom legend markers
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=TEAL, markersize=8, label='GPT-2/3 시리즈'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=AMBER, markersize=8, label='Chinchilla'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=SAGE, markersize=8, label='LLaMA 시리즈'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor=CORAL, markersize=12, label='GPT-4급'),
        Line2D([0], [0], linestyle='-', color=AMBER, linewidth=2, label='Chinchilla-최적 경로'),
        Line2D([0], [0], linestyle='--', color='#aaaaaa', linewidth=1.2, label='로그-선형 피팅'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.95)

    ax.grid(True, alpha=0.15, which='both')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT + 'fig_3_7_scaling_laws.png', dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print('[OK] fig_3_7_scaling_laws.png')


# ============================================================
# Fig 3.10 -- LLM vs Solver OPF Results Comparison
# ============================================================
def fig_3_10_llm_opf_comparison():
    # Simulated OPF results for a 5-generator system
    gen_labels = ['$P_{G1}$', '$P_{G2}$', '$P_{G3}$', '$P_{G4}$', '$P_{G5}$']
    solver_pg = np.array([76.2, 48.5, 35.0, 22.8, 17.5])   # MW
    llm_pg    = np.array([74.8, 49.1, 36.2, 23.5, 16.4])    # MW

    bus_labels = [f'Bus {i}' for i in range(1, 8)]
    solver_v = np.array([1.060, 1.045, 1.010, 0.998, 1.020, 0.985, 1.012])
    llm_v    = np.array([1.058, 1.042, 1.015, 1.001, 1.018, 0.990, 1.008])

    fig, axes = plt.subplots(1, 3, figsize=(15, 6.5),
                              gridspec_kw={'width_ratios': [1, 1, 0.8]})

    # --- (a) Generator Output Comparison ---
    ax = axes[0]
    x = np.arange(len(gen_labels))
    w = 0.32
    ax.bar(x - w/2, solver_pg, w, color=TEAL, label='수치 솔버 (ACOPF)', edgecolor='white', linewidth=0.5)
    ax.bar(x + w/2, llm_pg, w, color=AMBER, label='LLM 예측', edgecolor='white', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(gen_labels, fontsize=11)
    ax.set_ylabel('유효전력 출력 (MW)', fontsize=11)
    ax.set_title('(a) 발전기 출력 비교', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.set_ylim(0, 95)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.grid(True, axis='y', alpha=0.2)

    # --- (b) Bus Voltage Comparison ---
    ax2 = axes[1]
    x2 = np.arange(len(bus_labels))
    ax2.bar(x2 - w/2, solver_v, w, color=TEAL, label='수치 솔버', edgecolor='white', linewidth=0.5)
    ax2.bar(x2 + w/2, llm_v, w, color=AMBER, label='LLM 예측', edgecolor='white', linewidth=0.5)

    # Voltage limits
    ax2.axhline(1.05, color=CORAL, linewidth=1.0, linestyle='--', alpha=0.7, label='상한 (1.05 pu)')
    ax2.axhline(0.95, color=CORAL, linewidth=1.0, linestyle='--', alpha=0.7, label='하한 (0.95 pu)')
    ax2.axhspan(0.95, 1.05, alpha=0.04, color=SAGE)

    ax2.set_xticks(x2)
    ax2.set_xticklabels(bus_labels, fontsize=10, rotation=30)
    ax2.set_ylabel('전압 크기 (pu)', fontsize=11)
    ax2.set_title('(b) 모선 전압 비교', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8, loc='lower left', ncol=2)
    ax2.set_ylim(0.93, 1.08)
    for spine in ['top', 'right']:
        ax2.spines[spine].set_visible(False)
    ax2.grid(True, axis='y', alpha=0.2)

    # --- (c) Summary Metrics ---
    ax3 = axes[2]
    ax3.axis('off')
    ax3.set_title('(c) 성능 요약', fontsize=12, fontweight='bold', pad=15)

    table_data = [
        ['지표', '솔버', 'LLM'],
        ['총 발전비용', '$5,823', '$5,892'],
        ['최대 전압 위반', '0.000 pu', '0.008 pu'],
        ['제약 만족률', '100.0%', '97.1%'],
        ['비용 오차', '\u2014', '+1.2%'],
    ]

    table = ax3.table(cellText=table_data[1:],
                       colLabels=table_data[0],
                       cellLoc='center',
                       loc='center',
                       colWidths=[0.38, 0.31, 0.31])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    # Style header
    for j in range(3):
        cell = table[0, j]
        cell.set_facecolor(NAVY)
        cell.set_text_props(color='white', fontweight='bold', fontsize=10)
        cell.set_edgecolor('white')

    # Style body
    for i in range(1, 5):
        for j in range(3):
            cell = table[i, j]
            if j == 0:
                cell.set_facecolor('#f5f5f5')
                cell.set_text_props(fontweight='bold', fontsize=9)
            elif j == 1:
                cell.set_facecolor('#eef6f7')
            else:
                cell.set_facecolor('#fdf5ec')
            cell.set_edgecolor('#e0e0e0')

    fig.suptitle('그림 3.10  LLM 기반 OPF 예측 vs 수치 솔버 비교',
                 fontsize=14, fontweight='bold', y=1.03)
    fig.tight_layout()
    fig.savefig(OUT + 'fig_3_10_llm_opf_comparison.png', dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print('[OK] fig_3_10_llm_opf_comparison.png')


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print('=== Chapter 3 Figures ===')
    fig_3_2_attention_heatmap()
    fig_3_5_positional_encoding()
    fig_3_7_scaling_laws()
    fig_3_10_llm_opf_comparison()
    print('=== Done ===')
