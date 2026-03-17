#!/usr/bin/env python3
"""7장 matplotlib 그림 생성 스크립트 (Fig 7.5~7.10)"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os

# 한글 폰트 설정
font_path = os.path.expanduser('~/tmp_fonts/PAPERLOGY-5MEDIUM.TTF')
if os.path.exists(font_path):
    import matplotlib.font_manager as fm
    fe = fm.FontEntry(fname=font_path, name='Paperlogy')
    fm.fontManager.ttflist.insert(0, fe)
    plt.rcParams['font.family'] = 'Paperlogy'
else:
    plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

OUT = os.path.dirname(os.path.abspath(__file__))


def fig_7_5():
    """N-1 contingency 분석 결과 히트맵"""
    np.random.seed(42)
    n_lines = 20
    n_buses = 14

    line_names = [f'L{i+1}' for i in range(n_lines)]
    bus_names = [f'Bus {i+1}' for i in range(n_buses)]

    # N-1 분석 결과 시뮬레이션: 전압 편차 (정상에서의 변화)
    # 대부분 작은 영향, 일부 선로 제거 시 큰 영향
    voltage_deviation = np.random.exponential(0.005, (n_lines, n_buses))
    # 특정 선로-모선 조합에서 큰 영향
    voltage_deviation[3, 4] = 0.06   # L4 제거 → Bus 5 전압 큰 하락
    voltage_deviation[3, 5] = 0.045
    voltage_deviation[7, 9] = 0.055  # L8 제거 → Bus 10 전압 큰 하락
    voltage_deviation[7, 10] = 0.04
    voltage_deviation[12, 13] = 0.05 # L13 제거 → Bus 14 전압 하락
    voltage_deviation[15, 7] = 0.035
    voltage_deviation[15, 8] = 0.038

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(voltage_deviation, cmap='YlOrRd', aspect='auto',
                   vmin=0, vmax=0.06)

    ax.set_xticks(range(n_buses))
    ax.set_xticklabels(bus_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(n_lines))
    ax.set_yticklabels(line_names, fontsize=8)
    ax.set_xlabel('영향받는 모선', fontsize=11)
    ax.set_ylabel('제거된 선로', fontsize=11)
    ax.set_title('N-1 Contingency 분석: 전압 편차 히트맵 (p.u.)', fontsize=13, pad=12)

    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label('전압 편차 (p.u.)', fontsize=10)

    # 위반 기준선 표시 (0.05 p.u. 이상 = 위반)
    for i in range(n_lines):
        for j in range(n_buses):
            if voltage_deviation[i, j] > 0.04:
                ax.text(j, i, f'{voltage_deviation[i,j]:.3f}',
                       ha='center', va='center', fontsize=6.5, color='white', weight='bold')

    fig.savefig(os.path.join(OUT, 'fig_7_5.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('fig_7_5 saved')


def fig_7_6():
    """교차 검증: pandapower vs MATPOWER 산점도"""
    np.random.seed(7)

    # IEEE 14-bus 전압 (pandapower 결과 시뮬레이션)
    pp_voltages = np.array([1.060, 1.045, 1.010, 1.018, 1.020,
                            1.070, 1.062, 1.090, 1.056, 1.051,
                            1.057, 1.055, 1.050, 1.036])
    # MATPOWER 결과 (미세한 차이)
    mp_voltages = pp_voltages + np.random.normal(0, 0.0003, len(pp_voltages))

    # 선로 조류 (MW)
    pp_flows = np.array([156.9, 75.5, 73.2, 56.1, 41.5, 28.1, 60.0,
                         44.2, 16.1, 9.3, 5.4, 7.8, 17.5, 6.1, 1.8,
                         3.6, 5.2, 6.7, 4.2, 2.8])
    mp_flows = pp_flows + np.random.normal(0, 0.15, len(pp_flows))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # (a) 전압 비교
    ax = axes[0]
    ax.scatter(mp_voltages, pp_voltages, c='#1B7A8A', s=80, zorder=3,
               edgecolors='white', linewidth=0.5)
    lims = [0.99, 1.10]
    ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1, label='y = x')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('MATPOWER 전압 (p.u.)', fontsize=11)
    ax.set_ylabel('pandapower 전압 (p.u.)', fontsize=11)
    ax.set_title('(a) 모선 전압 비교', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 오차 주석
    rel_err = np.abs(pp_voltages - mp_voltages) / mp_voltages * 100
    ax.text(0.05, 0.92, f'최대 상대 오차: {rel_err.max():.4f}%',
            transform=ax.transAxes, fontsize=9, color='#C75C3A')

    for i in range(len(pp_voltages)):
        ax.annotate(f'{i+1}', (mp_voltages[i], pp_voltages[i]),
                   fontsize=6.5, ha='center', va='bottom',
                   textcoords='offset points', xytext=(0, 5))

    # (b) 선로 조류 비교
    ax = axes[1]
    ax.scatter(mp_flows, pp_flows, c='#D4984A', s=60, zorder=3,
               edgecolors='white', linewidth=0.5)
    lims2 = [0, 170]
    ax.plot(lims2, lims2, 'k--', alpha=0.5, linewidth=1, label='y = x')
    ax.set_xlim(lims2)
    ax.set_ylim(lims2)
    ax.set_xlabel('MATPOWER 선로 조류 (MW)', fontsize=11)
    ax.set_ylabel('pandapower 선로 조류 (MW)', fontsize=11)
    ax.set_title('(b) 선로 유효전력 조류 비교', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    flow_err = np.abs(pp_flows - mp_flows) / np.maximum(mp_flows, 1) * 100
    ax.text(0.05, 0.92, f'최대 상대 오차: {flow_err.max():.2f}%',
            transform=ax.transAxes, fontsize=9, color='#C75C3A')

    fig.savefig(os.path.join(OUT, 'fig_7_6.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('fig_7_6 saved')


def fig_7_7():
    """IEEE 39-bus + 재생에너지 계통도 (networkx)"""
    import networkx as nx

    # IEEE 39-bus 계통 정보 (간략화)
    edges = [
        (1,2),(1,39),(2,3),(2,25),(3,4),(3,18),(4,5),(4,14),
        (5,6),(5,8),(6,7),(6,11),(7,8),(8,9),(9,39),(10,11),
        (10,13),(13,14),(14,15),(15,16),(16,17),(16,19),(16,21),
        (16,24),(17,18),(17,27),(21,22),(22,23),(23,24),(25,26),
        (26,27),(26,28),(26,29),(28,29),
    ]

    # 발전기 모선
    gen_buses = {30:2, 31:6, 32:10, 33:19, 34:20, 35:22, 36:23, 37:25, 38:29, 39:1}
    gen_edges = [(v, k) for k, v in gen_buses.items()]

    G = nx.Graph()
    G.add_edges_from(edges)
    G.add_edges_from(gen_edges)

    # 풍력/태양광 추가 모선
    wind_buses = [25, 26, 29]
    solar_buses = [20, 23]

    pos = nx.spring_layout(G, seed=42, k=2.5, iterations=80)

    fig, ax = plt.subplots(figsize=(14, 10))

    # 선로 그리기
    bus_nodes = [n for n in G.nodes if n <= 39]
    gen_nodes = [n for n in G.nodes if n > 29]
    load_buses = [n for n in range(1, 40) if n not in [b for b in gen_buses.values()]]

    # 일반 모선
    normal_buses = [b for b in bus_nodes if b not in wind_buses and b not in solar_buses]

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4, width=1.5, edge_color='#888888')

    # 일반 모선
    nx.draw_networkx_nodes(G, pos, nodelist=normal_buses, ax=ax,
                          node_color='#2C3E50', node_size=250, alpha=0.8)
    # 발전기 모선
    nx.draw_networkx_nodes(G, pos, nodelist=gen_nodes, ax=ax,
                          node_color='#D4984A', node_size=400, node_shape='s', alpha=0.9)
    # 풍력 모선
    nx.draw_networkx_nodes(G, pos, nodelist=wind_buses, ax=ax,
                          node_color='#1B7A8A', node_size=500, node_shape='^', alpha=0.9)
    # 태양광 모선
    nx.draw_networkx_nodes(G, pos, nodelist=solar_buses, ax=ax,
                          node_color='#C75C3A', node_size=500, node_shape='D', alpha=0.9)

    # 라벨
    labels = {n: str(n) for n in bus_nodes}
    gen_labels = {n: f'G{n-29}' for n in gen_nodes}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=7, font_color='white')
    nx.draw_networkx_labels(G, pos, gen_labels, ax=ax, font_size=6.5, font_color='white')

    # 범례
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2C3E50',
               markersize=12, label='일반 모선'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#D4984A',
               markersize=12, label='기존 발전기'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#1B7A8A',
               markersize=14, label='풍력 발전 (추가)'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#C75C3A',
               markersize=12, label='태양광 발전 (추가)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10,
             framealpha=0.9, edgecolor='#cccccc')

    ax.set_title('IEEE 39-bus 시스템 + 재생에너지 배치', fontsize=14, pad=15)
    ax.axis('off')

    fig.savefig(os.path.join(OUT, 'fig_7_7.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('fig_7_7 saved')


def fig_7_8():
    """재생에너지 침투율별 총비용 + curtailment"""
    penetrations = [10, 20, 30, 40, 50, 60]
    costs = [285400, 261200, 238600, 219800, 208100, 201500]
    curtailments = [0, 12, 48, 156, 384, 720]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 비용 (막대)
    bars = ax1.bar(penetrations, [c/1000 for c in costs], width=7,
                   color='#1B7A8A', alpha=0.7, label='일일 운영 비용', zorder=2)
    ax1.set_xlabel('재생에너지 침투율 (%)', fontsize=12)
    ax1.set_ylabel('일일 운영 비용 (천 $/day)', fontsize=12, color='#1B7A8A')
    ax1.tick_params(axis='y', labelcolor='#1B7A8A')
    ax1.set_ylim(150, 320)
    ax1.grid(axis='y', alpha=0.3)

    # 막대 위에 값 표시
    for bar, cost in zip(bars, costs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f'{cost/1000:.0f}', ha='center', va='bottom', fontsize=9, color='#1B7A8A')

    # Curtailment (선)
    ax2 = ax1.twinx()
    ax2.plot(penetrations, curtailments, 'o-', color='#C75C3A', linewidth=2.5,
            markersize=8, label='Curtailment', zorder=3)
    ax2.set_ylabel('Curtailment (MWh/day)', fontsize=12, color='#C75C3A')
    ax2.tick_params(axis='y', labelcolor='#C75C3A')
    ax2.set_ylim(-30, 800)

    # 위반 영역 표시
    ax1.axvspan(37, 65, alpha=0.08, color='#C75C3A')
    ax1.text(48, 310, '전압 위반 영역', fontsize=9, color='#C75C3A',
            ha='center', style='italic')

    # 범례 통합
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

    ax1.set_title('재생에너지 침투율에 따른 운영 비용 및 Curtailment', fontsize=13, pad=12)

    fig.savefig(os.path.join(OUT, 'fig_7_8.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('fig_7_8 saved')


def fig_7_9():
    """민감도 분석 토네이도 차트"""
    params = ['부하 수준\n(80%→120%)', '재생에너지 침투율\n(10%→60%)',
              '가스 연료비\n(20→40 $/MWh)', '선로 용량\n(80%→150%)',
              'ESS 용량\n(0→500 MWh)']

    # 기준 비용: 250,000 $/day
    base_cost = 250
    # 파라미터 변동 시 비용 변화 범위 [low, high] (천 $/day)
    low_vals = [198, 201, 228, 240, 232]   # 파라미터 최솟값일 때 비용
    high_vals = [312, 285, 272, 258, 248]  # 파라미터 최댓값일 때 비용

    # 영향도 순으로 정렬
    ranges = [h - l for h, l in zip(high_vals, low_vals)]
    sorted_idx = np.argsort(ranges)

    params_s = [params[i] for i in sorted_idx]
    low_s = [low_vals[i] for i in sorted_idx]
    high_s = [high_vals[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = range(len(params_s))
    colors_low = '#1B7A8A'
    colors_high = '#C75C3A'

    for i, (lo, hi) in enumerate(zip(low_s, high_s)):
        # Low side (left of base)
        ax.barh(i, lo - base_cost, left=base_cost, height=0.6,
               color=colors_low, alpha=0.8)
        # High side (right of base)
        ax.barh(i, hi - base_cost, left=base_cost, height=0.6,
               color=colors_high, alpha=0.8)
        # 값 표시
        ax.text(lo - 2, i, f'{lo}', ha='right', va='center', fontsize=9)
        ax.text(hi + 2, i, f'{hi}', ha='left', va='center', fontsize=9)

    ax.axvline(x=base_cost, color='#2C3E50', linewidth=1.5, linestyle='-')
    ax.text(base_cost, len(params_s) + 0.3, f'기준: {base_cost}천 $/day',
           ha='center', fontsize=10, color='#2C3E50', weight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(params_s, fontsize=10)
    ax.set_xlabel('일일 운영 비용 (천 $/day)', fontsize=12)
    ax.set_title('민감도 분석 토네이도 차트', fontsize=13, pad=12)
    ax.grid(axis='x', alpha=0.3)

    # 범례
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors_low, alpha=0.8, label='파라미터 최솟값'),
        Patch(facecolor=colors_high, alpha=0.8, label='파라미터 최댓값'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    fig.savefig(os.path.join(OUT, 'fig_7_9.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('fig_7_9 saved')


def fig_7_10():
    """전압 프로파일 비교 (침투율 10%, 30%, 60%)"""
    np.random.seed(10)
    buses = np.arange(1, 40)

    # 기본 전압 프로파일 (IEEE 39-bus 근사)
    base_v = np.ones(39) * 1.02
    # 외곽 모선은 약간 낮은 전압
    base_v[13:20] = 1.01
    base_v[20:30] = 1.015
    base_v[30:] = 1.04  # 발전기 모선은 높은 전압

    # 침투율별 전압 변화
    v_10 = base_v + np.random.normal(0, 0.005, 39)
    v_10 = np.clip(v_10, 0.95, 1.08)

    v_30 = base_v - 0.02 + np.random.normal(0, 0.008, 39)
    v_30[24:29] -= 0.015  # 풍력 모선 근처 전압 하락
    v_30 = np.clip(v_30, 0.93, 1.08)

    v_60 = base_v - 0.05 + np.random.normal(0, 0.012, 39)
    v_60[24:29] -= 0.035  # 풍력 모선 근처 큰 전압 하락
    v_60[19:23] -= 0.02   # 태양광 모선 근처
    v_60 = np.clip(v_60, 0.90, 1.08)

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(buses, v_10, 'o-', color='#1B7A8A', linewidth=2, markersize=5,
            label='침투율 10%', alpha=0.9)
    ax.plot(buses, v_30, 's-', color='#D4984A', linewidth=2, markersize=5,
            label='침투율 30%', alpha=0.9)
    ax.plot(buses, v_60, '^-', color='#C75C3A', linewidth=2, markersize=5,
            label='침투율 60%', alpha=0.9)

    # 전압 제한선
    ax.axhline(y=1.05, color='#888888', linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(y=0.95, color='#888888', linestyle='--', linewidth=1, alpha=0.7)
    ax.text(40, 1.052, '상한 1.05', fontsize=8, color='#888888', va='bottom')
    ax.text(40, 0.948, '하한 0.95', fontsize=8, color='#888888', va='top')

    # 위반 영역 음영
    ax.axhspan(0.90, 0.95, alpha=0.06, color='#C75C3A')
    ax.axhspan(1.05, 1.10, alpha=0.06, color='#C75C3A')

    # RE 모선 표시
    for b in [25, 26, 29]:
        ax.axvline(x=b, color='#1B7A8A', alpha=0.15, linewidth=8)
    for b in [20, 23]:
        ax.axvline(x=b, color='#C75C3A', alpha=0.15, linewidth=8)

    ax.text(27, 0.915, '풍력 모선', fontsize=8, color='#1B7A8A',
            ha='center', style='italic')
    ax.text(21.5, 0.915, '태양광', fontsize=8, color='#C75C3A',
            ha='center', style='italic')

    ax.set_xlabel('모선 번호', fontsize=12)
    ax.set_ylabel('전압 (p.u.)', fontsize=12)
    ax.set_title('침투율별 모선 전압 프로파일 비교', fontsize=13, pad=12)
    ax.set_xlim(0.5, 39.5)
    ax.set_ylim(0.90, 1.10)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, 40, 2))

    fig.savefig(os.path.join(OUT, 'fig_7_10.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('fig_7_10 saved')


if __name__ == '__main__':
    fig_7_5()
    fig_7_6()
    fig_7_7()
    fig_7_8()
    fig_7_9()
    fig_7_10()
    print('All ch07 figures generated.')
