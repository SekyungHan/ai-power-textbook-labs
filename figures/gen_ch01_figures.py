"""
Chapter 1 figures for AI Power Systems textbook.
Generates 5 matplotlib figures with a professional academic style.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import os

# === Style Setup ===
NAVY = '#2C3E50'
TEAL = '#1B7A8A'
AMBER = '#D4984A'
SAGE = '#5A7D6A'
CORAL = '#C75C3A'

from matplotlib.font_manager import FontProperties, fontManager

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


# ============================================================
# Fig 1.3: Activation Functions
# ============================================================
def fig_1_3():
    x = np.linspace(-5, 5, 500)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_d(x):
        s = sigmoid(x)
        return s * (1 - s)

    def tanh_f(x):
        return np.tanh(x)

    def tanh_d(x):
        return 1 - np.tanh(x)**2

    def relu(x):
        return np.maximum(0, x)

    def relu_d(x):
        return (x > 0).astype(float)

    def gelu(x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

    def gelu_d(x):
        # numerical derivative
        h = 1e-5
        return (gelu(x + h) - gelu(x - h)) / (2 * h)

    funcs = [
        ('Sigmoid', sigmoid, sigmoid_d),
        ('Tanh', tanh_f, tanh_d),
        ('ReLU', relu, relu_d),
        ('GELU', gelu, gelu_d),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    colors_func = [NAVY, TEAL, CORAL, SAGE]

    for idx, (name, f, fd) in enumerate(funcs):
        ax = axes[idx // 2][idx % 2]
        c = colors_func[idx]
        ax.plot(x, f(x), color=c, linewidth=2, label=f'{name}')
        ax.plot(x, fd(x), color=c, linewidth=1.5, linestyle='--', alpha=0.7, label=f'{name} 도함수')
        ax.set_title(name, fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend(loc='best', framealpha=0.9)
        ax.axhline(y=0, color='grey', linewidth=0.5)
        ax.axvline(x=0, color='grey', linewidth=0.5)
        ax.set_xlim(-5, 5)

    fig.suptitle('활성화 함수와 도함수', fontsize=15, fontweight='bold', y=1.0)
    plt.tight_layout()
    fig.savefig(OUT + 'fig_1_3_activation_functions.png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print('[OK] fig_1_3_activation_functions.png')


# ============================================================
# Fig 1.5: Optimizer Comparison on Rosenbrock-like contour
# ============================================================
def fig_1_5():
    # Rosenbrock: f(x,y) = (1-x)^2 + 100*(y - x^2)^2
    def rosenbrock(x, y):
        return (1 - x)**2 + 10 * (y - x**2)**2

    def rosenbrock_grad(x, y):
        dx = -2*(1-x) + 10*2*(y - x**2)*(-2*x)
        dy = 10*2*(y - x**2)
        return np.array([dx, dy])

    # Optimizers
    def run_sgd(start, lr=0.002, steps=300):
        path = [start.copy()]
        p = start.copy()
        for _ in range(steps):
            g = rosenbrock_grad(p[0], p[1])
            p = p - lr * g
            path.append(p.copy())
        return np.array(path)

    def run_momentum(start, lr=0.002, mu=0.9, steps=300):
        path = [start.copy()]
        p = start.copy()
        v = np.zeros(2)
        for _ in range(steps):
            g = rosenbrock_grad(p[0], p[1])
            v = mu * v - lr * g
            p = p + v
            path.append(p.copy())
        return np.array(path)

    def run_adam(start, lr=0.05, steps=300):
        path = [start.copy()]
        p = start.copy()
        m = np.zeros(2)
        v = np.zeros(2)
        b1, b2, eps = 0.9, 0.999, 1e-8
        for t in range(1, steps+1):
            g = rosenbrock_grad(p[0], p[1])
            m = b1*m + (1-b1)*g
            v = b2*v + (1-b2)*g**2
            mh = m / (1 - b1**t)
            vh = v / (1 - b2**t)
            p = p - lr * mh / (np.sqrt(vh) + eps)
            path.append(p.copy())
        return np.array(path)

    start = np.array([-1.5, 1.5])
    path_sgd = run_sgd(start)
    path_mom = run_momentum(start)
    path_adam = run_adam(start)

    # Contour
    xx = np.linspace(-2, 2, 300)
    yy = np.linspace(-1, 3, 300)
    X, Y = np.meshgrid(xx, yy)
    Z = rosenbrock(X, Y)

    fig, ax = plt.subplots(figsize=(8, 6))
    levels = np.logspace(-1, 3.5, 30)
    cs = ax.contour(X, Y, Z, levels=levels, colors=NAVY, linewidths=0.4, alpha=0.5)
    ax.contourf(X, Y, Z, levels=levels, cmap='bone_r', alpha=0.3)

    for path, color, label in [
        (path_sgd, TEAL, 'SGD'),
        (path_mom, AMBER, 'Momentum'),
        (path_adam, CORAL, 'Adam'),
    ]:
        ax.plot(path[:, 0], path[:, 1], '-', color=color, linewidth=1.5, alpha=0.8, label=label)
        ax.plot(path[0, 0], path[0, 1], 'o', color=color, markersize=6)
        ax.plot(path[-1, 0], path[-1, 1], 's', color=color, markersize=5)

    ax.plot(1, 1, '*', color='black', markersize=12, zorder=5, label='최적점 (1,1)')
    ax.set_xlabel('$w_1$')
    ax.set_ylabel('$w_2$')
    ax.set_title('옵티마이저 경로 비교 (Rosenbrock 함수)', fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1, 3)

    plt.tight_layout()
    fig.savefig(OUT + 'fig_1_5_optimizer_comparison.png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print('[OK] fig_1_5_optimizer_comparison.png')


# ============================================================
# Fig 1.7: Universal Approximation Theorem
# ============================================================
def fig_1_7():
    np.random.seed(42)

    # Target function
    x = np.linspace(0, 2*np.pi, 500)
    y_target = np.sin(x) + 0.4*np.sin(3*x) + 0.2*np.cos(5*x)

    # Simple single-hidden-layer NN approximation (random features, least squares)
    def approx_nn(x, y_target, n_neurons):
        np.random.seed(7)
        W = np.random.randn(n_neurons) * 1.5
        b = np.random.randn(n_neurons) * 2
        # Hidden layer: tanh activations
        H = np.tanh(np.outer(x, W) + b)  # (N, n_neurons)
        # Least squares fit
        coeffs, _, _, _ = np.linalg.lstsq(H, y_target, rcond=None)
        return H @ coeffs

    neurons = [2, 5, 10, 50]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    colors = [CORAL, AMBER, TEAL, SAGE]

    for idx, (n, c) in enumerate(zip(neurons, colors)):
        ax = axes[idx // 2][idx % 2]
        y_approx = approx_nn(x, y_target, n)
        ax.plot(x, y_target, color='black', linewidth=2, label='목표 함수')
        ax.plot(x, y_approx, color=c, linewidth=1.8, linestyle='--', label=f'근사 (뉴런 {n}개)')
        ax.set_title(f'은닉 뉴런 = {n}', fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend(loc='best', framealpha=0.9)
        ax.set_xlim(0, 2*np.pi)

        # Compute R^2
        ss_res = np.sum((y_target - y_approx)**2)
        ss_tot = np.sum((y_target - np.mean(y_target))**2)
        r2 = 1 - ss_res / ss_tot
        ax.text(0.97, 0.05, f'$R^2 = {r2:.3f}$', transform=ax.transAxes,
                ha='right', fontsize=10, color=c,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=c, alpha=0.8))

    fig.suptitle('범용 근사 정리: 뉴런 수에 따른 함수 근사', fontsize=14, fontweight='bold', y=1.0)
    plt.tight_layout()
    fig.savefig(OUT + 'fig_1_7_universal_approximation.png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print('[OK] fig_1_7_universal_approximation.png')


# ============================================================
# Fig 1.10: Voltage Prediction (Scatter + Error Histogram)
# ============================================================
def fig_1_10():
    np.random.seed(2024)
    n = 200
    actual = np.random.normal(1.0, 0.025, n)
    actual = np.clip(actual, 0.94, 1.06)
    noise = np.random.normal(0, 0.005, n)
    predicted = actual + noise + 0.001 * (actual - 1.0)**2  # slight nonlinearity
    errors = predicted - actual

    # R^2
    ss_res = np.sum((actual - predicted)**2)
    ss_tot = np.sum((actual - np.mean(actual))**2)
    r2 = 1 - ss_res / ss_tot

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Scatter
    ax1.scatter(actual, predicted, color=TEAL, s=15, alpha=0.6, edgecolors='none')
    lims = [0.94, 1.06]
    ax1.plot(lims, lims, '--', color=CORAL, linewidth=1.5, label='y = x (이상적)')
    ax1.set_xlabel('실제 전압 (p.u.)')
    ax1.set_ylabel('예측 전압 (p.u.)')
    ax1.set_title('전압 예측 정확도', fontweight='bold')
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.set_aspect('equal')
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.text(0.97, 0.05, f'$R^2 = {r2:.4f}$', transform=ax1.transAxes,
             ha='right', fontsize=11, color=NAVY,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=NAVY, alpha=0.8))

    # Histogram
    ax2.hist(errors * 1000, bins=25, color=TEAL, edgecolor='white', alpha=0.85)
    ax2.axvline(x=0, color=CORAL, linewidth=1.5, linestyle='--')
    ax2.set_xlabel('예측 오차 (mV, p.u.\u00d71000)')
    ax2.set_ylabel('빈도')
    ax2.set_title('예측 오차 분포', fontweight='bold')

    mean_err = np.mean(errors) * 1000
    std_err = np.std(errors) * 1000
    ax2.text(0.97, 0.92, f'평균: {mean_err:.2f} mV\n표준편차: {std_err:.2f} mV',
             transform=ax2.transAxes, ha='right', va='top', fontsize=10, color=NAVY,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=NAVY, alpha=0.8))

    plt.tight_layout()
    fig.savefig(OUT + 'fig_1_10_voltage_prediction.png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print('[OK] fig_1_10_voltage_prediction.png')


# ============================================================
# Fig 1.11: Loss Landscape with Training Trajectory
# ============================================================
def fig_1_11():
    # Modified Rosenbrock for a nice valley
    def loss_fn(x, y):
        return 0.5*(1 - x)**2 + 2*(y - x**2)**2 + 0.1*(x**2 + y**2)

    xx = np.linspace(-2, 2, 400)
    yy = np.linspace(-1, 3, 400)
    X, Y = np.meshgrid(xx, yy)
    Z = loss_fn(X, Y)

    # Simulate a training trajectory (Adam-like path toward minimum)
    def loss_grad(x, y):
        dx = -1*(1-x) + 2*2*(y - x**2)*(-2*x) + 0.2*x
        dy = 2*2*(y - x**2) + 0.2*y
        return np.array([dx, dy])

    path = [np.array([-1.5, 2.5])]
    p = path[0].copy()
    m, v = np.zeros(2), np.zeros(2)
    lr, b1, b2, eps = 0.08, 0.9, 0.999, 1e-8
    for t in range(1, 120):
        g = loss_grad(p[0], p[1])
        m = b1*m + (1-b1)*g
        v = b2*v + (1-b2)*g**2
        mh = m / (1 - b1**t)
        vh = v / (1 - b2**t)
        p = p - lr * mh / (np.sqrt(vh) + eps)
        path.append(p.copy())
    path = np.array(path)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Filled contour
    levels = np.linspace(0, np.percentile(Z, 95), 40)
    cf = ax.contourf(X, Y, Z, levels=levels, cmap='YlGnBu', alpha=0.85)
    ax.contour(X, Y, Z, levels=levels, colors=NAVY, linewidths=0.3, alpha=0.4)

    cbar = plt.colorbar(cf, ax=ax, shrink=0.85)
    cbar.set_label('손실 값 (Loss)', fontsize=11)

    # Training trajectory
    ax.plot(path[:, 0], path[:, 1], '-', color=CORAL, linewidth=1.2, alpha=0.7)
    # Plot dots at intervals
    step = 5
    ax.plot(path[::step, 0], path[::step, 1], 'o', color=CORAL, markersize=4, alpha=0.9)
    ax.plot(path[0, 0], path[0, 1], 'D', color=CORAL, markersize=8, label='시작점', zorder=5)
    ax.plot(path[-1, 0], path[-1, 1], '*', color=AMBER, markersize=12, label='수렴점', zorder=5)

    ax.set_xlabel('$\\theta_1$')
    ax.set_ylabel('$\\theta_2$')
    ax.set_title('손실 지형과 학습 경로', fontweight='bold', fontsize=14)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1, 3)

    plt.tight_layout()
    fig.savefig(OUT + 'fig_1_11_loss_landscape.png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print('[OK] fig_1_11_loss_landscape.png')


# ============================================================
# Run all
# ============================================================
if __name__ == '__main__':
    fig_1_3()
    fig_1_5()
    fig_1_7()
    fig_1_10()
    fig_1_11()
    print('\n=== All 5 figures generated. ===')
