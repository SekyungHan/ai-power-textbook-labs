"""Generate matplotlib figures for Chapter 4: LLM 활용 기초"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

# Try to set Korean font
try:
    matplotlib.rcParams['font.family'] = 'Paperlogy'
except:
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

# Color palette matching the textbook
deep_navy = '#2C3E50'
teal_blue = '#1B7A8A'
amber = '#D4984A'
sage_green = '#5A7D6A'
coral = '#C75C3A'

output_dir = os.path.dirname(os.path.abspath(__file__))

# --- Fig 4.2: 영어 vs 한국어 토큰 수 비교 ---
fig, ax = plt.subplots(figsize=(8, 5))

sentences = [
    'Optimal Power\nFlow',
    'Voltage\nStability',
    'Contingency\nAnalysis',
    'Economic\nDispatch',
    'Transient Stability\nAssessment',
]
en_tokens = [3, 3, 3, 3, 4]
ko_tokens = [7, 6, 7, 6, 9]

x = np.arange(len(sentences))
width = 0.35

bars_en = ax.bar(x - width/2, en_tokens, width, label='영어 (English)',
                  color=teal_blue, alpha=0.85, edgecolor='white', linewidth=0.5)
bars_ko = ax.bar(x + width/2, ko_tokens, width, label='한국어 (Korean)',
                  color=coral, alpha=0.85, edgecolor='white', linewidth=0.5)

# 비율 표시
for i, (en, ko) in enumerate(zip(en_tokens, ko_tokens)):
    ax.text(i + width/2, ko + 0.3, f'x{ko/en:.1f}', ha='center', va='bottom',
            fontsize=9, color=coral, fontweight='bold')

# 한국어 라벨 추가
ko_labels = ['최적조류계산', '전압안정도', '상정사고분석', '경제급전', '과도안정도평가']
for i, label in enumerate(ko_labels):
    ax.text(i, -1.2, label, ha='center', va='top', fontsize=8, color=deep_navy)

ax.set_xticks(x)
ax.set_xticklabels(sentences, fontsize=9)
ax.set_ylabel('토큰 수', fontsize=11, color=deep_navy)
ax.set_title('영어 vs 한국어 토큰 수 비교', fontsize=13, color=deep_navy, fontweight='bold')
ax.legend(fontsize=10, framealpha=0.9)
ax.set_ylim(0, 12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.axhline(y=0, color=deep_navy, linewidth=0.5)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig_4_2_token_comparison.png'), dpi=200, bbox_inches='tight',
            facecolor='white')
plt.close()
print("Fig 4.2 saved")

# --- Fig 4.4: Lost in the Middle ---
fig, ax = plt.subplots(figsize=(8, 5))

positions = np.arange(1, 21)
# U-shaped curve for retrieval accuracy
accuracy = np.array([
    92, 88, 82, 75, 68, 62, 58, 55, 53, 52,
    53, 55, 58, 62, 66, 72, 78, 84, 89, 94
])

ax.plot(positions, accuracy, 'o-', color=teal_blue, linewidth=2.5, markersize=7,
        markerfacecolor='white', markeredgewidth=2, markeredgecolor=teal_blue)

# 영역 표시
ax.fill_between(positions[:5], accuracy[:5], alpha=0.1, color=sage_green)
ax.fill_between(positions[6:14], accuracy[6:14], alpha=0.1, color=coral)
ax.fill_between(positions[15:], accuracy[15:], alpha=0.1, color=sage_green)

ax.annotate('시작 부분\n(잘 활용)', xy=(2, 88), xytext=(3, 98),
            fontsize=9, color=sage_green, ha='center',
            arrowprops=dict(arrowstyle='->', color=sage_green, lw=1.5))
ax.annotate('중간 부분\n(Lost in the Middle)', xy=(10, 52), xytext=(10, 42),
            fontsize=9, color=coral, ha='center', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=coral, lw=1.5))
ax.annotate('끝 부분\n(잘 활용)', xy=(19, 89), xytext=(17, 98),
            fontsize=9, color=sage_green, ha='center',
            arrowprops=dict(arrowstyle='->', color=sage_green, lw=1.5))

ax.set_xlabel('정보의 위치 (컨텍스트 내)', fontsize=11, color=deep_navy)
ax.set_ylabel('검색 정확도 (%)', fontsize=11, color=deep_navy)
ax.set_title('Lost in the Middle 현상', fontsize=13, color=deep_navy, fontweight='bold')
ax.set_xticks([1, 5, 10, 15, 20])
ax.set_ylim(35, 105)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.axhline(y=70, color='#CCCCCC', linewidth=0.8, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig_4_4_lost_in_middle.png'), dpi=200, bbox_inches='tight',
            facecolor='white')
plt.close()
print("Fig 4.4 saved")

# --- Fig 4.8: Temperature에 따른 출력 분포 ---
fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=False)

tokens = ['전력', '에너지', '전기', '시스템', '네트워크', '구조', '방식', '기타']
logits = np.array([3.5, 2.8, 2.5, 1.8, 1.2, 0.8, 0.4, 0.1])

temperatures = [0.1, 0.7, 1.5]
titles = ['T = 0.1 (결정론적)', 'T = 0.7 (균형)', 'T = 1.5 (창의적)']
colors = [teal_blue, amber, coral]

for ax, T, title, color in zip(axes, temperatures, titles, colors):
    probs = np.exp(logits / T)
    probs = probs / probs.sum()

    bars = ax.barh(range(len(tokens)), probs, color=color, alpha=0.8,
                   edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels(tokens, fontsize=9)
    ax.set_xlabel('확률', fontsize=10, color=deep_navy)
    ax.set_title(title, fontsize=11, color=deep_navy, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 확률값 표시
    for i, (bar, p) in enumerate(zip(bars, probs)):
        if p > 0.02:
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{p:.1%}', va='center', fontsize=8, color=deep_navy)

    ax.invert_yaxis()

fig.suptitle('"전력 계통의 ___" -- 다음 토큰 확률 분포', fontsize=13,
             color=deep_navy, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig_4_8_temperature.png'), dpi=200, bbox_inches='tight',
            facecolor='white')
plt.close()
print("Fig 4.8 saved")

print("\nAll Ch04 figures generated successfully!")
