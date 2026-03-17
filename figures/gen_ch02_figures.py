"""
Chapter 2 figures for AI Power Systems textbook.
Embedding, Autoencoder, and Generative Models.
Generates 6 matplotlib figures with a professional academic style.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.patches import FancyArrowPatch
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
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.color': '#cccccc',
    'lines.linewidth': 1.8,
})

OUT = os.path.dirname(os.path.abspath(__file__)) + '/'
DPI = 200

np.random.seed(42)


# ============================================================
# Helper: generate realistic hourly load data for N days
# ============================================================
def generate_load_profile(n_days=7, noise_level=0.05):
    """Generate realistic hourly power demand data."""
    hours = np.arange(n_days * 24)
    t_hour = hours % 24
    day_of_week = (hours // 24) % 7  # 0=Mon

    # Base load pattern (normalized)
    base = 0.5 + 0.3 * np.sin(2 * np.pi * (t_hour - 6) / 24)
    # Morning peak
    base += 0.15 * np.exp(-0.5 * ((t_hour - 9) / 2) ** 2)
    # Evening peak
    base += 0.2 * np.exp(-0.5 * ((t_hour - 19) / 2.5) ** 2)
    # Weekend reduction
    is_weekend = (day_of_week >= 5).astype(float)
    base *= (1.0 - 0.15 * is_weekend)
    # Scale to MW range
    load = 800 + 400 * base + noise_level * 400 * np.random.randn(len(hours))
    return hours, load


# ============================================================
# Fig 2.3: Embedding Space Visualization
# ============================================================
def fig_2_3_embedding_space():
    print("Generating fig_2_3_embedding_space.png ...")

    # Define semantic clusters with Korean power system terms
    clusters = {
        '발전 설비': {
            'words': ['발전기', '터빈', '보일러', '증기', '회전자', '고정자', '여자기'],
            'center': (-3.0, 2.0), 'color': TEAL
        },
        '송변전 설비': {
            'words': ['변압기', '모선', '송전선', '애자', '철탑', '개폐기', 'GIS'],
            'center': (2.5, 2.5), 'color': AMBER
        },
        '부하/수요': {
            'words': ['부하', '수요', '전력량', '피크', '역률', '수용가', '전력거래'],
            'center': (0.0, -3.0), 'color': SAGE
        },
        '보호 계전': {
            'words': ['계전기', '차단기', '보호', '과전류', 'OCR', 'UVLS', '재폐로'],
            'center': (-2.5, -1.0), 'color': CORAL
        },
    }

    fig, ax = plt.subplots(figsize=(10, 7), dpi=DPI)

    for label, info in clusters.items():
        cx, cy = info['center']
        n = len(info['words'])
        xs = cx + np.random.randn(n) * 0.7
        ys = cy + np.random.randn(n) * 0.7

        ax.scatter(xs, ys, c=info['color'], s=120, alpha=0.7,
                   edgecolors='white', linewidths=0.8, zorder=3)

        for i, w in enumerate(info['words']):
            ax.annotate(w, (xs[i], ys[i]), fontsize=9,
                        ha='center', va='bottom',
                        xytext=(0, 6), textcoords='offset points',
                        color=info['color'], fontweight='bold')

        # Cluster label with circle
        from matplotlib.patches import Ellipse
        ell = Ellipse((cx, cy), width=4.0, height=3.5, fill=False,
                      edgecolor=info['color'], linewidth=1.5, linestyle='--',
                      alpha=0.5, zorder=1)
        ax.add_patch(ell)
        ax.text(cx, cy + 2.0, label, ha='center', va='bottom',
                fontsize=11, fontweight='bold', color=info['color'],
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor=info['color'], alpha=0.8))

    # Draw a semantic arrow between related clusters
    ax.annotate('', xy=(1.5, 2.0), xytext=(-2.0, 1.5),
                arrowprops=dict(arrowstyle='->', color='#999999',
                                lw=1.2, ls='--'))
    ax.text(-0.3, 2.2, '의미적 관계', ha='center', fontsize=8,
            color='#999999', style='italic')

    ax.set_xlabel('임베딩 차원 1')
    ax.set_ylabel('임베딩 차원 2')
    ax.set_title('전력 계통 용어의 임베딩 공간 (t-SNE 시각화)', fontsize=14, fontweight='bold')
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(OUT + 'fig_2_3_embedding_space.png', dpi=DPI, bbox_inches='tight',
                facecolor='white')
    plt.close()


# ============================================================
# Fig 2.5: PCA vs Autoencoder Reconstruction
# ============================================================
def fig_2_5_pca_vs_ae():
    print("Generating fig_2_5_pca_vs_ae.png ...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=DPI)

    # Left: Reconstruction error vs latent dimensions
    dims = [2, 4, 8, 16, 32]
    # Simulated MSE values (PCA worse than AE, gap widens at low dims)
    pca_mse = [0.42, 0.28, 0.15, 0.08, 0.04]
    ae_mse = [0.25, 0.13, 0.06, 0.03, 0.015]

    ax = axes[0]
    ax.plot(dims, pca_mse, 'o-', color=CORAL, label='PCA', markersize=7)
    ax.plot(dims, ae_mse, 's-', color=TEAL, label='오토인코더', markersize=7)
    ax.set_xlabel('잠재 차원 수')
    ax.set_ylabel('재구성 오차 (MSE)')
    ax.set_title('(a) 잠재 차원에 따른 재구성 오차', fontsize=12, fontweight='bold')
    ax.set_xticks(dims)
    ax.legend(framealpha=0.9)
    ax.set_yscale('log')
    ax.set_ylim(0.01, 0.6)

    # Right: Example reconstruction of a nonlinear signal
    ax = axes[1]
    t = np.linspace(0, 24, 200)
    # Original: nonlinear daily load curve
    original = (0.6 + 0.3 * np.sin(2 * np.pi * (t - 6) / 24)
                + 0.15 * np.exp(-0.5 * ((t - 9) / 2) ** 2)
                + 0.2 * np.exp(-0.5 * ((t - 19) / 2.5) ** 2))
    original = 800 + 400 * original

    # PCA reconstruction (misses peaks)
    pca_recon = 800 + 400 * (0.6 + 0.3 * np.sin(2 * np.pi * (t - 6) / 24)
                              + 0.1 * np.sin(4 * np.pi * t / 24))

    # AE reconstruction (much closer)
    ae_recon = original + np.random.randn(len(t)) * 8

    ax.plot(t, original, '-', color=NAVY, linewidth=2.0, label='원본', zorder=3)
    ax.plot(t, pca_recon, '--', color=CORAL, linewidth=1.5, label='PCA (d=4)', alpha=0.85)
    ax.plot(t, ae_recon, ':', color=TEAL, linewidth=1.5, label='AE (d=4)', alpha=0.85)
    ax.set_xlabel('시각 (h)')
    ax.set_ylabel('전력 수요 (MW)')
    ax.set_title('(b) 재구성 비교 (잠재 차원=4)', fontsize=12, fontweight='bold')
    ax.legend(framealpha=0.9)
    ax.set_xlim(0, 24)

    # Annotate the peak region where PCA fails
    ax.annotate('PCA가 피크를\n놓치는 구간',
                xy=(19, 1060), xytext=(17, 920),
                arrowprops=dict(arrowstyle='->', color=CORAL, lw=1.2),
                fontsize=9, color=CORAL, ha='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor=CORAL, alpha=0.85))

    plt.tight_layout()
    plt.savefig(OUT + 'fig_2_5_pca_vs_ae.png', dpi=DPI, bbox_inches='tight',
                facecolor='white')
    plt.close()


# ============================================================
# Fig 2.6: VAE Latent Space
# ============================================================
def fig_2_6_vae_latent_space():
    print("Generating fig_2_6_vae_latent_space.png ...")

    fig = plt.figure(figsize=(12, 6), dpi=DPI)

    # Left: 2D latent space scatter
    ax1 = fig.add_subplot(121)

    # Generate latent codes for different load patterns
    n_per = 50
    patterns = {
        '평일 여름': {'center': (-1.5, 1.5), 'color': CORAL},
        '평일 겨울': {'center': (1.5, 1.5), 'color': TEAL},
        '주말 여름': {'center': (-1.5, -1.5), 'color': AMBER},
        '주말 겨울': {'center': (1.5, -1.5), 'color': SAGE},
    }

    for label, info in patterns.items():
        cx, cy = info['center']
        zx = cx + np.random.randn(n_per) * 0.5
        zy = cy + np.random.randn(n_per) * 0.5
        ax1.scatter(zx, zy, c=info['color'], s=30, alpha=0.6,
                    edgecolors='none', label=label)

    ax1.set_xlabel('잠재 변수 $z_1$')
    ax1.set_ylabel('잠재 변수 $z_2$')
    ax1.set_title('(a) VAE 잠재 공간', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', framealpha=0.9, fontsize=9)
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-4, 4)
    ax1.axhline(0, color='#cccccc', linewidth=0.8, zorder=0)
    ax1.axvline(0, color='#cccccc', linewidth=0.8, zorder=0)

    # Right: Grid of decoded load patterns from latent space
    ax2 = fig.add_subplot(122)

    grid_n = 5
    z1_vals = np.linspace(-2.5, 2.5, grid_n)
    z2_vals = np.linspace(-2.5, 2.5, grid_n)

    t = np.linspace(0, 24, 48)
    cell_w = 1.0 / grid_n
    cell_h = 1.0 / grid_n

    for i, z2 in enumerate(reversed(z2_vals)):
        for j, z1 in enumerate(z1_vals):
            # Generate load pattern based on latent position
            base = 0.5 + 0.25 * np.sin(2 * np.pi * (t - 6) / 24)
            # Weekend effect (z1 > 0 = more weekend-like)
            weekend_factor = 1.0 - 0.15 * (1 / (1 + np.exp(-z1)))
            base *= weekend_factor
            # Summer AC peak (z2 > 0 = more summer)
            summer_peak = 0.2 * (1 / (1 + np.exp(-z2))) * np.exp(-0.5 * ((t - 15) / 3) ** 2)
            # Winter heating (z2 < 0)
            winter_peak = 0.15 * (1 / (1 + np.exp(z2))) * np.exp(-0.5 * ((t - 8) / 2) ** 2)
            pattern = base + summer_peak + winter_peak

            # Normalize to [0, 1] for display
            pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min() + 1e-8)

            # Plot mini time series in grid cell
            x_off = j * cell_w + 0.02 * cell_w
            y_off = i * cell_h + 0.02 * cell_h
            mini_x = x_off + np.linspace(0, cell_w * 0.9, len(t))
            mini_y = y_off + pattern * cell_h * 0.8

            # Color based on position
            r = 0.5 + 0.5 * z1 / 2.5
            b = 0.5 + 0.5 * z2 / 2.5
            color = plt.cm.coolwarm(0.5 + 0.4 * (z1 + z2) / 5.0)

            ax2.plot(mini_x, mini_y, color=color, linewidth=0.8, alpha=0.9)
            ax2.add_patch(plt.Rectangle((x_off, y_off), cell_w * 0.96, cell_h * 0.96,
                                         fill=False, edgecolor='#dddddd', linewidth=0.5))

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('$z_1$ (평일 \u2192 주말)', fontsize=10)
    ax2.set_ylabel('$z_2$ (겨울 \u2192 여름)', fontsize=10)
    ax2.set_title('(b) 잠재 공간 그리드 디코딩', fontsize=12, fontweight='bold')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.grid(False)

    plt.tight_layout()
    plt.savefig(OUT + 'fig_2_6_vae_latent_space.png', dpi=DPI, bbox_inches='tight',
                facecolor='white')
    plt.close()


# ============================================================
# Fig 2.9: AE Reconstruction of Power Demand
# ============================================================
def fig_2_9_ae_reconstruction():
    print("Generating fig_2_9_ae_reconstruction.png ...")

    hours, load = generate_load_profile(n_days=7, noise_level=0.04)

    # Simulated AE reconstruction (smoothed version)
    from scipy.ndimage import gaussian_filter1d
    recon = gaussian_filter1d(load, sigma=1.5) + np.random.randn(len(load)) * 5
    error = load - recon

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), dpi=DPI,
                              sharex=True, gridspec_kw={'height_ratios': [3, 3, 1.5]})

    # Day labels
    day_labels = ['월', '화', '수', '목', '금', '토', '일']
    day_ticks = [12 + i * 24 for i in range(7)]

    # Top: Original
    ax = axes[0]
    ax.plot(hours, load, color=NAVY, linewidth=1.2)
    ax.fill_between(hours, load.min() - 20, load, alpha=0.08, color=NAVY)
    ax.set_ylabel('전력 수요 (MW)')
    ax.set_title('원본 시계열', fontsize=12, fontweight='bold', loc='left')
    ax.set_xlim(0, 168)
    ax.set_xticks(day_ticks)
    ax.set_xticklabels([])

    # Middle: Reconstruction
    ax = axes[1]
    ax.plot(hours, recon, color=TEAL, linewidth=1.2)
    ax.fill_between(hours, recon.min() - 20, recon, alpha=0.08, color=TEAL)
    ax.set_ylabel('전력 수요 (MW)')
    ax.set_title('오토인코더 재구성', fontsize=12, fontweight='bold', loc='left')
    ax.set_xticks(day_ticks)
    ax.set_xticklabels([])

    # Bottom: Error
    ax = axes[2]
    ax.bar(hours, error, width=0.8, color=CORAL, alpha=0.6, edgecolor='none')
    ax.axhline(0, color=NAVY, linewidth=0.8)
    ax.set_ylabel('오차 (MW)')
    ax.set_xlabel('시각')
    ax.set_title('재구성 오차', fontsize=11, fontweight='bold', loc='left')
    ax.set_xticks(day_ticks)
    ax.set_xticklabels(day_labels)

    # Add vertical day separators
    for a in axes:
        for d in range(1, 7):
            a.axvline(d * 24, color='#cccccc', linewidth=0.6, linestyle='-', zorder=0)

    plt.tight_layout()
    plt.savefig(OUT + 'fig_2_9_ae_reconstruction.png', dpi=DPI, bbox_inches='tight',
                facecolor='white')
    plt.close()


# ============================================================
# Fig 2.10: Anomaly Detection
# ============================================================
def fig_2_10_anomaly_detection():
    print("Generating fig_2_10_anomaly_detection.png ...")

    hours, load = generate_load_profile(n_days=14, noise_level=0.03)

    # Inject anomalies
    # Holiday (day 5): unusually low
    load[5*24:6*24] -= 120
    # Equipment failure spike (day 10, hour 14-16)
    load[10*24+14:10*24+17] += 250
    # Gradual drift (day 12-13)
    load[12*24:14*24] += np.linspace(0, 80, 48)

    # Compute anomaly score (rolling window deviation)
    from scipy.ndimage import gaussian_filter1d
    smoothed = gaussian_filter1d(load, sigma=3)
    anomaly_score = np.abs(load - smoothed)
    # Enhance at known anomaly locations
    anomaly_score[5*24:6*24] += 40
    anomaly_score[10*24+13:10*24+18] += 80
    anomaly_score[12*24:14*24] += 20
    # Normalize
    anomaly_score = anomaly_score / anomaly_score.max()

    threshold = 0.35

    fig, axes = plt.subplots(2, 1, figsize=(13, 6), dpi=DPI,
                              sharex=True, gridspec_kw={'height_ratios': [2.5, 1.5]})

    # Top: Load with anomaly highlights
    ax = axes[0]
    ax.plot(hours, load, color=NAVY, linewidth=1.0, label='전력 수요')

    # Highlight anomaly regions
    anomaly_mask = anomaly_score > threshold
    for i in range(len(hours)):
        if anomaly_mask[i]:
            ax.axvspan(hours[i] - 0.5, hours[i] + 0.5, alpha=0.15, color=CORAL, zorder=0)

    # Annotate specific anomalies
    ax.annotate('공휴일\n(수요 급감)',
                xy=(5*24+12, load[5*24+12]), xytext=(5*24+12, load[5*24+12] + 180),
                arrowprops=dict(arrowstyle='->', color=CORAL, lw=1.2),
                fontsize=9, color=CORAL, ha='center', fontweight='bold')
    ax.annotate('설비 이상\n(스파이크)',
                xy=(10*24+15, load[10*24+15]), xytext=(10*24+15, load[10*24+15] + 120),
                arrowprops=dict(arrowstyle='->', color=CORAL, lw=1.2),
                fontsize=9, color=CORAL, ha='center', fontweight='bold')
    ax.annotate('점진적 드리프트',
                xy=(13*24, load[13*24]), xytext=(13*24, load[13*24] + 150),
                arrowprops=dict(arrowstyle='->', color=CORAL, lw=1.2),
                fontsize=9, color=CORAL, ha='center', fontweight='bold')

    ax.set_ylabel('전력 수요 (MW)')
    ax.set_title('오토인코더 기반 이상 탐지 결과', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)

    # Bottom: Anomaly score
    ax = axes[1]
    ax.fill_between(hours, anomaly_score, alpha=0.4, color=TEAL)
    ax.plot(hours, anomaly_score, color=TEAL, linewidth=1.0)
    ax.axhline(threshold, color=CORAL, linewidth=1.5, linestyle='--', label=f'임계값 ({threshold})')
    ax.set_ylabel('이상 점수')
    ax.set_xlabel('시각 (h)')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim(0, 1.1)
    ax.set_xlim(0, 14 * 24)

    # Day ticks
    day_ticks = [12 + i * 24 for i in range(14)]
    day_labels = [f'D{i+1}' for i in range(14)]
    for a in axes:
        a.set_xticks(day_ticks)
        for d in range(1, 14):
            a.axvline(d * 24, color='#eeeeee', linewidth=0.5, zorder=0)
    axes[1].set_xticklabels(day_labels)

    plt.tight_layout()
    plt.savefig(OUT + 'fig_2_10_anomaly_detection.png', dpi=DPI, bbox_inches='tight',
                facecolor='white')
    plt.close()


# ============================================================
# Fig 2.11: Real vs Synthetic Load Profiles
# ============================================================
def fig_2_11_synthetic_load():
    print("Generating fig_2_11_synthetic_load.png ...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=DPI)
    t = np.arange(24)

    # Generate "real" daily profiles (30 days)
    n_profiles = 30
    real_profiles = []
    for _ in range(n_profiles):
        base = 0.5 + 0.3 * np.sin(2 * np.pi * (t - 6) / 24)
        base += 0.15 * np.exp(-0.5 * ((t - 9) / 2) ** 2)
        base += 0.2 * np.exp(-0.5 * ((t - 19) / 2.5) ** 2)
        noise = np.random.randn(24) * 0.04
        weekend = np.random.random() > 0.7
        if weekend:
            base *= 0.85
        real_profiles.append(800 + 400 * (base + noise))
    real_profiles = np.array(real_profiles)

    # Generate "synthetic" VAE-generated profiles (slightly smoother)
    synth_profiles = []
    for _ in range(n_profiles):
        base = 0.5 + 0.28 * np.sin(2 * np.pi * (t - 6.2) / 24)
        base += 0.14 * np.exp(-0.5 * ((t - 9.1) / 2.1) ** 2)
        base += 0.19 * np.exp(-0.5 * ((t - 18.8) / 2.6) ** 2)
        noise = np.random.randn(24) * 0.035
        weekend = np.random.random() > 0.7
        if weekend:
            base *= 0.86
        synth_profiles.append(800 + 400 * (base + noise))
    synth_profiles = np.array(synth_profiles)

    # Left: Real
    ax = axes[0]
    for i in range(n_profiles):
        ax.plot(t, real_profiles[i], color='#aaaaaa', linewidth=0.5, alpha=0.5)
    ax.plot(t, real_profiles.mean(axis=0), color=TEAL, linewidth=2.5, label='평균', zorder=5)
    ax.fill_between(t,
                     real_profiles.mean(axis=0) - real_profiles.std(axis=0),
                     real_profiles.mean(axis=0) + real_profiles.std(axis=0),
                     alpha=0.15, color=TEAL)
    ax.set_xlabel('시각 (h)')
    ax.set_ylabel('전력 수요 (MW)')
    ax.set_title('(a) 실제 부하 프로파일', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 23)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_ylim(750, 1250)

    # Right: Synthetic
    ax = axes[1]
    for i in range(n_profiles):
        ax.plot(t, synth_profiles[i], color='#aaaaaa', linewidth=0.5, alpha=0.5)
    ax.plot(t, synth_profiles.mean(axis=0), color=AMBER, linewidth=2.5, label='평균', zorder=5)
    ax.fill_between(t,
                     synth_profiles.mean(axis=0) - synth_profiles.std(axis=0),
                     synth_profiles.mean(axis=0) + synth_profiles.std(axis=0),
                     alpha=0.15, color=AMBER)
    ax.set_xlabel('시각 (h)')
    ax.set_ylabel('전력 수요 (MW)')
    ax.set_title('(b) VAE 생성 합성 부하 프로파일', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 23)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_ylim(750, 1250)

    plt.tight_layout()
    plt.savefig(OUT + 'fig_2_11_synthetic_load.png', dpi=DPI, bbox_inches='tight',
                facecolor='white')
    plt.close()


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    fig_2_3_embedding_space()
    fig_2_5_pca_vs_ae()
    fig_2_6_vae_latent_space()
    fig_2_9_ae_reconstruction()
    fig_2_10_anomaly_detection()
    fig_2_11_synthetic_load()
    print("\nAll Chapter 2 figures generated successfully.")
