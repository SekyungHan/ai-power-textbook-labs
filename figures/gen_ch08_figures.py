#!/usr/bin/env python3
"""8장 그림 생성: Fig 8.3, 8.4, 8.5, 8.6"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# 한글 폰트 설정
font_path = os.path.expanduser('~/tmp_fonts/PAPERLOGY-5MEDIUM.TTF')
if os.path.exists(font_path):
    from matplotlib import font_manager
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = prop.get_name()
else:
    plt.rcParams['font.family'] = 'DejaVu Sans'

plt.rcParams['axes.unicode_minus'] = False

OUT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)


def fig_8_3():
    """Fig 8.3: 논문 품질 그래프 Before/After"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    np.random.seed(42)
    buses = np.arange(1, 15)
    voltage_base = 1.0 + 0.02 * np.random.randn(14)
    voltage_load = voltage_base - 0.01 * np.random.rand(14) * buses / 14

    # --- Before (기본 matplotlib) ---
    ax = axes[0]
    ax.plot(buses, voltage_base, 'o-')
    ax.plot(buses, voltage_load, 's-')
    ax.set_title('Before: 기본 설정')
    ax.set_xlabel('bus')
    ax.set_ylabel('voltage (p.u.)')
    ax.legend(['기저 부하', '중부하'])
    ax.grid(True)

    # --- After (논문 품질) ---
    ax = axes[1]
    ax.plot(buses, voltage_base, 'o-', color='#2C3E50', linewidth=2,
            markersize=6, markerfacecolor='white', markeredgewidth=1.5,
            label='기저 부하 시나리오')
    ax.plot(buses, voltage_load, 's-', color='#C75C3A', linewidth=2,
            markersize=6, markerfacecolor='white', markeredgewidth=1.5,
            label='중부하 시나리오 (1.5배)')

    ax.axhline(y=0.95, color='#D4984A', linestyle='--', linewidth=1, alpha=0.7, label='하한 (0.95 p.u.)')
    ax.axhline(y=1.05, color='#D4984A', linestyle='--', linewidth=1, alpha=0.7, label='상한 (1.05 p.u.)')
    ax.fill_between(buses, 0.95, 1.05, alpha=0.05, color='#5A7D6A')

    ax.set_title('After: 논문 품질 스타일', fontsize=12, fontweight='bold', color='#2C3E50')
    ax.set_xlabel('모선 번호', fontsize=10)
    ax.set_ylabel('전압 크기 (p.u.)', fontsize=10)
    ax.legend(fontsize=8, framealpha=0.9, edgecolor='#E0E0E0')
    ax.set_xlim(0.5, 14.5)
    ax.set_ylim(0.93, 1.07)
    ax.set_xticks(buses)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.suptitle('그림 8.3 — 논문 품질 그래프: Before vs After', fontsize=13, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, 'fig_8_3.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print('fig_8_3 saved')


def fig_8_4():
    """Fig 8.4: IEEE 14-bus 계통도 시각화 (전압 편차 + 선로 부하율)"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # IEEE 14-bus 좌표 (대략적 배치)
    pos = {
        1: (0.1, 0.9), 2: (0.3, 0.8), 3: (0.5, 0.5), 4: (0.4, 0.65),
        5: (0.2, 0.65), 6: (0.6, 0.7), 7: (0.7, 0.55), 8: (0.85, 0.6),
        9: (0.65, 0.4), 10: (0.55, 0.3), 11: (0.5, 0.4), 12: (0.4, 0.35),
        13: (0.45, 0.2), 14: (0.6, 0.15)
    }

    # 전압 (p.u.)
    voltages = {
        1: 1.060, 2: 1.045, 3: 1.010, 4: 1.018, 5: 1.020,
        6: 1.070, 7: 1.049, 8: 1.080, 9: 1.033, 10: 1.032,
        11: 1.047, 12: 1.055, 13: 1.050, 14: 1.021
    }

    # 선로 연결 및 부하율
    lines = [
        (1, 2, 65), (1, 5, 72), (2, 3, 45), (2, 4, 38), (2, 5, 41),
        (3, 4, 22), (4, 5, 55), (4, 7, 30), (4, 9, 25), (5, 6, 48),
        (6, 11, 18), (6, 12, 15), (6, 13, 20), (7, 8, 12), (7, 9, 28),
        (9, 10, 22), (9, 14, 35), (10, 11, 14), (12, 13, 10), (13, 14, 25)
    ]

    # 선로 그리기 (부하율에 따른 색상)
    for i, j, loading in lines:
        x = [pos[i][0], pos[j][0]]
        y = [pos[i][1], pos[j][1]]
        if loading > 60:
            color = '#C75C3A'
            lw = 2.5
        elif loading > 40:
            color = '#D4984A'
            lw = 2.0
        else:
            color = '#5A7D6A'
            lw = 1.5
        ax.plot(x, y, '-', color=color, linewidth=lw, alpha=0.8, zorder=1)
        # 부하율 표기
        mx, my = (x[0] + x[1]) / 2, (y[0] + y[1]) / 2
        ax.annotate(f'{loading}%', (mx, my), fontsize=6, color=color,
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.8))

    # 노드 그리기 (전압 편차에 따른 크기/색상)
    generators = {1, 2, 3, 6, 8}
    for bus, (x, y) in pos.items():
        v_dev = abs(voltages[bus] - 1.0) * 100  # 전압 편차 %
        size = 200 + v_dev * 80

        if bus in generators:
            marker = 's'
            color = '#1B7A8A'
        else:
            marker = 'o'
            color = '#2C3E50'

        ax.scatter(x, y, s=size, c=color, marker=marker, zorder=3,
                   edgecolors='white', linewidth=1.5, alpha=0.9)
        ax.annotate(f'{bus}\n{voltages[bus]:.3f}', (x, y),
                    fontsize=7, ha='center', va='center', color='white',
                    fontweight='bold', zorder=4)

    # 범례
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#1B7A8A', markersize=10, label='발전기 모선'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2C3E50', markersize=10, label='부하 모선'),
        Line2D([0], [0], color='#C75C3A', linewidth=2.5, label='부하율 > 60%'),
        Line2D([0], [0], color='#D4984A', linewidth=2, label='부하율 40~60%'),
        Line2D([0], [0], color='#5A7D6A', linewidth=1.5, label='부하율 < 40%'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=8, framealpha=0.9)

    ax.set_title('IEEE 14-Bus 계통 시각화\n노드 크기 = 전압 편차, 선 색상 = 선로 부하율', fontsize=12, fontweight='bold', color='#2C3E50')
    ax.set_xlim(-0.05, 1.0)
    ax.set_ylim(0.05, 1.0)
    ax.axis('off')

    fig.savefig(os.path.join(OUT, 'fig_8_4.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print('fig_8_4 saved')


def fig_8_5():
    """Fig 8.5: 조류 흐름 히트맵"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    np.random.seed(123)
    n_lines = 20
    n_buses = 14

    line_labels = [f'L{i+1}' for i in range(n_lines)]
    from_bus = [1, 1, 2, 2, 2, 3, 4, 4, 4, 5, 6, 6, 6, 7, 7, 9, 9, 10, 12, 13]
    to_bus =   [2, 5, 3, 4, 5, 4, 5, 7, 9, 6, 11,12,13, 8, 9, 10,14, 11, 13, 14]

    # 유효 전력 (MW)
    p_flow = np.array([156.9, 75.5, 73.2, 56.1, 41.6, -23.3, -61.8, 28.1, 16.1,
                        44.1, 7.4, 7.8, 17.8, 0.0, 28.1, 5.2, 9.4, -3.8, 1.6, 5.6])

    # 무효 전력 (MVar)
    q_flow = np.array([20.1, 5.3, 3.5, -1.6, 3.2, 4.4, -16.7, -10.9, 1.7,
                        12.8, 3.6, 2.5, 7.5, -17.4, 5.7, 4.2, 3.4, -1.6, 0.8, 2.0])

    # 유효 전력 히트맵
    ax = axes[0]
    p_data = p_flow.reshape(n_lines, 1)
    im = ax.imshow(p_data, cmap='RdBu_r', aspect=0.3, vmin=-80, vmax=180)
    ax.set_yticks(range(n_lines))
    line_descs = [f'{from_bus[i]}\u2192{to_bus[i]}' for i in range(n_lines)]
    ax.set_yticklabels(line_descs, fontsize=8)
    ax.set_xticks([0])
    ax.set_xticklabels(['P (MW)'], fontsize=9)
    ax.set_title('유효 전력 조류', fontsize=11, fontweight='bold', color='#2C3E50')

    for i in range(n_lines):
        ax.text(0, i, f'{p_flow[i]:.1f}', ha='center', va='center', fontsize=7,
                color='white' if abs(p_flow[i]) > 60 else 'black', fontweight='bold')

    plt.colorbar(im, ax=ax, shrink=0.6, label='MW')

    # 무효 전력 히트맵
    ax = axes[1]
    q_data = q_flow.reshape(n_lines, 1)
    im2 = ax.imshow(q_data, cmap='PiYG_r', aspect=0.3, vmin=-20, vmax=25)
    ax.set_yticks(range(n_lines))
    ax.set_yticklabels(line_descs, fontsize=8)
    ax.set_xticks([0])
    ax.set_xticklabels(['Q (MVar)'], fontsize=9)
    ax.set_title('무효 전력 조류', fontsize=11, fontweight='bold', color='#2C3E50')

    for i in range(n_lines):
        ax.text(0, i, f'{q_flow[i]:.1f}', ha='center', va='center', fontsize=7,
                color='white' if abs(q_flow[i]) > 12 else 'black', fontweight='bold')

    plt.colorbar(im2, ax=ax, shrink=0.6, label='MVar')

    fig.suptitle('IEEE 14-Bus 조류 흐름 히트맵', fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_8_5.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print('fig_8_5 saved')


def fig_8_6():
    """Fig 8.6: 인터랙티브 플롯 스타일 예시 (static으로 Plotly 스타일 재현)"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    np.random.seed(77)
    hours = np.arange(0, 168)  # 1주일
    base_load = 800 + 200 * np.sin(2 * np.pi * hours / 24) + 50 * np.sin(2 * np.pi * hours / 168)
    load = base_load + 30 * np.random.randn(168)

    # 재생에너지 출력
    solar = np.maximum(0, 150 * np.sin(np.pi * (hours % 24 - 6) / 12)) * (hours % 24 >= 6) * (hours % 24 <= 18)
    solar += 5 * np.random.randn(168)
    solar = np.maximum(solar, 0)

    wind = 100 + 60 * np.sin(2 * np.pi * hours / 36) + 30 * np.random.randn(168)
    wind = np.maximum(wind, 10)

    net_load = load - solar - wind

    ax.fill_between(hours, 0, load, alpha=0.15, color='#2C3E50', label='총 수요')
    ax.plot(hours, load, color='#2C3E50', linewidth=1.5)
    ax.plot(hours, solar, color='#D4984A', linewidth=1.5, label='태양광 출력')
    ax.plot(hours, wind, color='#1B7A8A', linewidth=1.5, label='풍력 출력')
    ax.plot(hours, net_load, color='#C75C3A', linewidth=2, label='순수요 (수요-RE)')

    # 주석
    peak_idx = np.argmax(net_load)
    ax.annotate(f'피크 순수요\n{net_load[peak_idx]:.0f} MW',
                xy=(hours[peak_idx], net_load[peak_idx]),
                xytext=(hours[peak_idx] + 15, net_load[peak_idx] + 80),
                fontsize=8, color='#C75C3A',
                arrowprops=dict(arrowstyle='->', color='#C75C3A', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#C75C3A', alpha=0.9))

    # 일 구분 세로선
    for d in range(1, 7):
        ax.axvline(x=d * 24, color='#E0E0E0', linestyle=':', linewidth=0.8)

    days = ['월', '화', '수', '목', '금', '토', '일']
    for d in range(7):
        ax.text(d * 24 + 12, ax.get_ylim()[0] + 20, days[d], ha='center', fontsize=8, color='gray')

    ax.set_xlabel('시간 (h)', fontsize=10)
    ax.set_ylabel('전력 (MW)', fontsize=10)
    ax.set_title('1주일 전력 수급 시계열 — 인터랙티브 스타일 시각화', fontsize=12, fontweight='bold', color='#2C3E50')
    ax.legend(fontsize=9, loc='upper right', framealpha=0.9, edgecolor='#E0E0E0')
    ax.set_xlim(0, 167)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, linestyle=':')

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_8_6.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print('fig_8_6 saved')


if __name__ == '__main__':
    fig_8_3()
    fig_8_4()
    fig_8_5()
    fig_8_6()
    print('All ch08 figures generated.')
