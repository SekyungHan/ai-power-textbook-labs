# ai_power_textbook_labs 교차 검수 보고서

**검수일**: 2026-03-17
**검수 대상**: labs/ 8개, solutions/ 8개 노트북 vs ~/ai_power_textbook/chapters/ch01~ch08.typ

---

## 1. 교재 원본 vs 노트북 코드 대조

### ch01 — PASS
- 교재 코드블록 2개 모두 lab/solution에 존재
- lab에서 `self.net = None` TODO 처리 (의도적), solution에서 완성
- 누락 없음, 유의미한 차이 없음

### ch04 — PASS
- 교재 코드블록 2개 (Demo 4.1, 4.2) 모두 존재
- `anthropic` 직접 호출 → `call_llm()` wrapper + pre-recorded fallback 변환 (적절)
- Demo 4.2 프롬프트: lab에서 TODO, solution에서 완성

### ch07 — FAIL (누락 2건)
- 교재 코드블록 7개 중 5개 존재, **2개 누락**:
  1. **`validate_results(net)` 함수** (lines 533-564): 3단계 시뮬레이션 결과 검증 (수렴, 전력균형, 전압범위/과부하)
  2. **Demo 7.2b IEEE 39-bus + 재생에너지 모델** (lines 973-1022): `add_renewables()` 함수, 풍력/태양광 시나리오 — 챕터 최종 워크숍 코드

### ch08 — FAIL (누락 1건)
- 교재 코드블록 6개 중 5개 존재, **1개 누락**:
  1. **Fig 8.6 Plotly 인터랙티브 주간 수급 시계열** (lines 536-547): `plotly.graph_objects`로 수요/태양광/풍력/순수요 시계열 생성
- Streamlit→Gradio 변환: **정상** (슬라이더, 출력, launch 모두 적절)
- FastAPI→in-notebook 변환: **정상** (dataclass 기반, 핵심 로직 동일)
- anthropic fallback: ch08에 anthropic 코드 없어 해당 없음

---

## 2. 실행 검증

| 노트북 | 결과 |
|--------|------|
| solutions/ch02_autoencoder_anomaly.ipynb | **PASS** — 정상 실행 (261KB 출력) |
| solutions/ch05_agent_architecture.ipynb | **PASS** — 정상 실행 (15KB 출력) |
| solutions/ch06_eda_tools.ipynb | **PASS** — 정상 실행 (334KB 출력) |

---

## 3. Lab/Solution 분리 확인 — PASS

### Lab TODO 현황 (전수 검사)

| 챕터 | TODO 수 | Placeholder 방식 | 정상 여부 |
|-------|---------|------------------|-----------|
| ch01 | 1 | `self.net = None` | OK |
| ch02 | 2 | `self.encoder = None`, `self.decoder = None` | OK |
| ch03 | 3 | `prompt_a/b/c = None` | OK |
| ch04 | 1 | `prompt = None` | OK |
| ch05 | 1 | `pass` | OK |
| ch06 | 1 | `scale = None` | OK |
| ch07 | 2 | `'v_violation': None`, `'loading_violation': None` | OK |
| ch08 | 1 | `pass` | OK |

- **총 12개 TODO**, 모두 `None` 또는 `pass` 사용
- Fallback trick 없음 — 학생이 채우지 않으면 에러 발생
- **Solution 8개 모두 잔존 TODO 없음**

---

## 4. 특수 패키지 처리 확인

| 항목 | 상태 | 비고 |
|------|------|------|
| Streamlit → Gradio (ch08) | **PASS** | `gr.Slider`, `gr.Interface`, `demo.launch(inline=True)` |
| FastAPI → in-notebook (ch08) | **PASS** | `@dataclass` 기반, 동일 로직 |
| anthropic fallback (ch04, ch05) | **PASS** | `call_llm()` + pre-recorded response, API 키 없이 동작 |

---

## 5. torch seed + requirements.txt

### Seed 일관성 — PASS (경미한 참고사항 있음)
- torch 사용 노트북 (ch01, ch02): `torch.manual_seed(42)` + `np.random.seed(42)` 설정됨
- numpy random 실제 사용 노트북 (ch01, ch02, ch06, ch07, ch08): 모두 seed 설정됨
- ch03, ch05: numpy import하지만 random 함수 미사용 → seed 미설정 (기능상 무해)

### requirements.txt 완전성 — PASS (참고사항 1건)
- 16개 노트북의 모든 third-party import가 requirements.txt에 포함됨
- **참고**: `scikit-learn`이 requirements.txt에 있으나 어떤 노트북에서도 import하지 않음 (불필요한 의존성)

---

## 종합 판정: FAIL

### 이유: 교재 코드 누락 3건

| # | 챕터 | 누락 항목 | 심각도 |
|---|------|-----------|--------|
| 1 | ch07 | `validate_results()` 3단계 검증 함수 | Medium |
| 2 | ch07 | Demo 7.2b IEEE 39-bus + 재생에너지 워크숍 | **High** (캡스톤 실습) |
| 3 | ch08 | Fig 8.6 Plotly 인터랙티브 시계열 | Medium |

### 기타 참고사항 (판정에 영향 없음)
- scikit-learn 불필요 의존성 (requirements.txt)
- ch03/ch05 numpy seed 미설정 (random 미사용이므로 무해)
