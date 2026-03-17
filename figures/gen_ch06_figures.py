#!/usr/bin/env python3
"""6장 그림 생성 스크립트 — matplotlib 기반 Fig. 6.4~6.7"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# 한글 폰트 설정
from matplotlib import font_manager
font_dir = os.path.expanduser('~/tmp_fonts/')
if os.path.isdir(font_dir):
    for f in os.listdir(font_dir):
        if f.lower().endswith(('.ttf', '.otf')):
            font_manager.fontManager.addfont(os.path.join(font_dir, f))
    plt.rcParams['font.family'] = 'Paperlogy'
else:
    plt.rcParams['font.family'] = 'DejaVu Sans'

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

# 색상 팔레트
DEEP_NAVY = '#2C3E50'
TEAL_BLUE = '#1B7A8A'
AMBER = '#D4984A'
SAGE_GREEN = '#5A7D6A'
CORAL = '#C75C3A'

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT_DIR, exist_ok=True)


def fig_6_4():
    """IEEE 14-bus 계통 기본 통계 — 박스플롯"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # 시뮬레이션 데이터 생성 (pandapower 없이 합리적인 값 사용)
    np.random.seed(42)

    # (a) 모선 전압 분포 (500 시나리오)
    n_scenarios = 500
    n_buses = 14
    # 기본 전압 프로파일 (slack=1.06, PV 모선 ~1.04, PQ 모선 ~1.0)
    base_v = np.array([1.06, 1.045, 1.01, 1.018, 1.02, 1.07, 1.049,
                       1.08, 1.033, 1.032, 1.047, 1.054, 1.047, 1.021])
    voltages = np.zeros((n_scenarios, n_buses))
    for i in range(n_scenarios):
        noise = np.random.normal(0, 0.015, n_buses)
        voltages[i] = base_v + noise

    bp = axes[0].boxplot(voltages, widths=0.5, patch_artist=True,
                         medianprops=dict(color=CORAL, linewidth=1.5))
    for patch in bp['boxes']:
        patch.set_facecolor(TEAL_BLUE)
        patch.set_alpha(0.5)
    axes[0].axhline(y=1.05, color=CORAL, ls='--', lw=0.8, label='상한 (1.05)')
    axes[0].axhline(y=0.95, color=CORAL, ls='--', lw=0.8, label='하한 (0.95)')
    axes[0].set_xlabel('모선 번호', fontsize=9)
    axes[0].set_ylabel('전압 (p.u.)', fontsize=9)
    axes[0].set_title('(a) 모선 전압 분포', fontsize=10, color=DEEP_NAVY)
    axes[0].legend(fontsize=7, loc='lower left')

    # (b) 선로 부하율 분포
    n_lines = 20
    base_loading = np.array([30, 45, 22, 18, 55, 42, 38, 28, 15, 60,
                             35, 48, 25, 52, 33, 20, 40, 58, 27, 44])
    loadings = np.zeros((n_scenarios, n_lines))
    for i in range(n_scenarios):
        noise = np.random.normal(0, 8, n_lines)
        loadings[i] = np.clip(base_loading + noise, 0, 120)

    bp2 = axes[1].boxplot(loadings, widths=0.5, patch_artist=True,
                          medianprops=dict(color=DEEP_NAVY, linewidth=1.5))
    for patch in bp2['boxes']:
        patch.set_facecolor(AMBER)
        patch.set_alpha(0.5)
    axes[1].axhline(y=100, color=CORAL, ls='--', lw=0.8, label='과부하 (100%)')
    axes[1].set_xlabel('선로 번호', fontsize=9)
    axes[1].set_ylabel('부하율 (%)', fontsize=9)
    axes[1].set_title('(b) 선로 부하율 분포', fontsize=10, color=DEEP_NAVY)
    axes[1].legend(fontsize=7)
    axes[1].set_xticklabels([str(i+1) for i in range(n_lines)], fontsize=6)

    # (c) 발전기 출력 분포
    n_gen = 5
    base_gen = np.array([232, 40, 0, 0, 0])  # IEEE 14-bus 발전기 출력
    gen_outputs = np.zeros((n_scenarios, n_gen))
    for i in range(n_scenarios):
        noise = np.random.normal(0, 15, n_gen)
        gen_outputs[i] = np.clip(base_gen + noise, 0, 300)

    bp3 = axes[2].boxplot(gen_outputs, widths=0.4, patch_artist=True,
                          medianprops=dict(color=DEEP_NAVY, linewidth=1.5))
    colors_gen = [SAGE_GREEN, TEAL_BLUE, AMBER, CORAL, DEEP_NAVY]
    for patch, c in zip(bp3['boxes'], colors_gen):
        patch.set_facecolor(c)
        patch.set_alpha(0.5)
    axes[2].set_xlabel('발전기 번호', fontsize=9)
    axes[2].set_ylabel('유효전력 (MW)', fontsize=9)
    axes[2].set_title('(c) 발전기 출력 분포', fontsize=10, color=DEEP_NAVY)
    axes[2].set_xticklabels(['G1', 'G2', 'G3', 'G4', 'G5'])

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_6_4_ieee14_stats.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('fig_6_4 saved')


def fig_6_5():
    """IEEE 39-bus 토폴로지 시각화 — 네트워크 그래프"""
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(10, 8))

    # IEEE 39-bus 노드 위치 (간략화된 좌표)
    np.random.seed(39)
    n_nodes = 39
    # 합리적인 2D 배치
    positions = {
        1: (3, 8), 2: (4, 9), 3: (5, 9), 4: (5, 8), 5: (4, 7),
        6: (5, 7), 7: (6, 7), 8: (7, 7), 9: (7, 6), 10: (6, 5),
        11: (5, 6), 12: (4, 5), 13: (5, 5), 14: (6, 6), 15: (3, 5),
        16: (2, 6), 17: (2, 7), 18: (3, 7), 19: (8, 8), 20: (7, 8),
        21: (3, 4), 22: (2, 4), 23: (2, 3), 24: (3, 3), 25: (4, 3),
        26: (5, 4), 27: (6, 4), 28: (7, 5), 29: (8, 6), 30: (1, 8),
        31: (1, 6), 32: (1, 4), 33: (1, 3), 34: (4, 4), 35: (5, 3),
        36: (6, 3), 37: (8, 7), 38: (8, 5), 39: (9, 7),
    }

    # 선로 정의 (간략화)
    edges = [
        (1,2), (1,39), (2,3), (2,25), (3,4), (3,18), (4,5), (4,14),
        (5,6), (5,8), (6,7), (6,11), (7,8), (8,9), (9,39),
        (10,11), (10,13), (13,14), (14,15), (15,16), (16,17),
        (16,19), (16,21), (17,18), (17,27), (21,22), (22,23),
        (23,24), (25,26), (26,27), (26,28), (26,29), (28,29),
        (12,11), (12,13), (20,19), (20,34), (33,34), (34,35),
        (35,36), (36,37), (37,38), (37,39), (24,25), (9,10),
    ]

    # 발전기 노드 (IEEE 39-bus: 30~39)
    gen_nodes = set(range(30, 40))

    # 부하 크기 (임의 — 시각적 목적)
    np.random.seed(42)
    loads = {i: np.random.uniform(50, 400) for i in range(1, 40) if i not in gen_nodes}

    # 선로 부하율 (임의)
    edge_loading = {e: np.random.uniform(20, 90) for e in edges}

    # 선로 그리기
    for (u, v), load_pct in edge_loading.items():
        x = [positions[u][0], positions[v][0]]
        y = [positions[u][1], positions[v][1]]
        # 부하율에 따라 색상 변화
        if load_pct > 70:
            color = CORAL
            lw = 1.5
        elif load_pct > 50:
            color = AMBER
            lw = 1.2
        else:
            color = SAGE_GREEN
            lw = 0.8
        ax.plot(x, y, color=color, lw=lw, alpha=0.7, zorder=1)

    # 노드 그리기
    for node, (x, y) in positions.items():
        if node in gen_nodes:
            size = 200
            color = TEAL_BLUE
            marker = 's'
        else:
            size = max(30, loads.get(node, 100) / 3)
            color = DEEP_NAVY
            marker = 'o'
        ax.scatter(x, y, s=size, c=color, marker=marker, zorder=2,
                   edgecolors='white', linewidth=0.5)
        ax.annotate(str(node), (x, y), fontsize=5.5, ha='center', va='center',
                    color='white' if node in gen_nodes else 'white',
                    fontweight='bold', zorder=3)

    # 범례
    legend_elements = [
        mpatches.Patch(facecolor=TEAL_BLUE, label='발전기 모선 (30~39)'),
        mpatches.Patch(facecolor=DEEP_NAVY, label='부하 모선 (크기=부하량)'),
        plt.Line2D([0], [0], color=SAGE_GREEN, lw=1.5, label='선로 부하율 < 50%'),
        plt.Line2D([0], [0], color=AMBER, lw=1.5, label='선로 부하율 50~70%'),
        plt.Line2D([0], [0], color=CORAL, lw=1.5, label='선로 부하율 > 70%'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=7,
              framealpha=0.9)

    ax.set_title('IEEE 39-bus (뉴잉글랜드) 시스템 토폴로지',
                 fontsize=12, color=DEEP_NAVY, fontweight='bold')
    ax.set_xlim(0, 10)
    ax.set_ylim(2, 10)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_6_5_ieee39_topology.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('fig_6_5 saved')


def fig_6_6():
    """모선 전압-부하 상관관계 — 산점도"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    np.random.seed(42)
    n = 500

    # 주요 모선 4개의 부하-전압 관계 시뮬레이션
    bus_data = [
        {'bus': 4, 'base_load': 47.8, 'base_v': 1.018, 'sensitivity': -0.0012},
        {'bus': 5, 'base_load': 7.6, 'base_v': 1.020, 'sensitivity': -0.0008},
        {'bus': 9, 'base_load': 29.5, 'base_v': 1.033, 'sensitivity': -0.0015},
        {'bus': 14, 'base_load': 14.9, 'base_v': 1.021, 'sensitivity': -0.0020},
    ]

    for idx, bd in enumerate(bus_data):
        ax = axes[idx // 2][idx % 2]
        load_var = np.random.uniform(0.5, 1.5, n) * bd['base_load']
        voltage = bd['base_v'] + bd['sensitivity'] * (load_var - bd['base_load']) + np.random.normal(0, 0.005, n)

        ax.scatter(load_var, voltage, s=8, alpha=0.4, c=TEAL_BLUE, edgecolors='none')

        # 추세선
        z = np.polyfit(load_var, voltage, 1)
        p = np.poly1d(z)
        x_line = np.linspace(load_var.min(), load_var.max(), 100)
        ax.plot(x_line, p(x_line), color=CORAL, lw=2,
                label=f'기울기: {z[0]:.4f} p.u./MW')

        # 상관계수
        corr = np.corrcoef(load_var, voltage)[0, 1]
        ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                fontsize=9, va='top', color=DEEP_NAVY, fontweight='bold')

        ax.axhline(y=0.95, color=CORAL, ls='--', lw=0.5, alpha=0.5)
        ax.axhline(y=1.05, color=CORAL, ls='--', lw=0.5, alpha=0.5)
        ax.set_xlabel(f'모선 {bd["bus"]} 부하 (MW)', fontsize=9)
        ax.set_ylabel('전압 (p.u.)', fontsize=9)
        ax.set_title(f'모선 {bd["bus"]}', fontsize=10, color=DEEP_NAVY)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=8)

    plt.suptitle('모선 전압-부하 상관관계 (Monte Carlo 500 시나리오)',
                 fontsize=12, color=DEEP_NAVY, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_6_6_voltage_load_corr.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('fig_6_6 saved')


def fig_6_7():
    """IEEE 테스트 시스템 규모 비교 — 바 차트"""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    systems = ['IEEE 14', 'IEEE 30', 'IEEE 39', 'IEEE 118']
    buses = [14, 30, 39, 118]
    lines = [20, 41, 46, 186]
    generators = [5, 6, 10, 54]

    x = np.arange(len(systems))
    width = 0.22

    bars1 = ax.bar(x - width, buses, width, label='모선 수',
                   color=TEAL_BLUE, alpha=0.85)
    bars2 = ax.bar(x, lines, width, label='선로 수',
                   color=AMBER, alpha=0.85)
    bars3 = ax.bar(x + width, generators, width, label='발전기 수',
                   color=SAGE_GREEN, alpha=0.85)

    # 값 표시
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, color=DEEP_NAVY)

    ax.set_xlabel('테스트 시스템', fontsize=10)
    ax.set_ylabel('구성 요소 수', fontsize=10)
    ax.set_title('IEEE 표준 테스트 시스템 규모 비교', fontsize=12,
                 color=DEEP_NAVY, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=10)
    ax.legend(fontsize=9, loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_6_7_system_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('fig_6_7 saved')


if __name__ == '__main__':
    fig_6_4()
    fig_6_5()
    fig_6_6()
    fig_6_7()
    print('All ch06 figures generated.')
