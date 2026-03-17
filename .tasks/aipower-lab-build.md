# AI Power Textbook Labs — Build Record

## Date: 2026-03-17

## Summary
교재 「도구에서 동료로」(ch01–ch08) 기반 Google Colab 실습 노트북 프레임워크 구축 완료.

## Deliverables

### utils/ (공용 유틸리티)
- `style.py` — matplotlib 한글 폰트 자동 감지 + 논문 스타일 설정

### labs/ (학생용 TODO 포함)
| File | Topic | TODO |
|------|-------|------|
| ch01_neural_network_basics.ipynb | MLP 전압 예측 | VoltagePredictor 모델 구현 |
| ch02_autoencoder_anomaly.ipynb | 오토인코더 이상 탐지 | DemandAE encoder/decoder 구현 |
| ch03_transformer_opf.ipynb | Transformer OPF | 3가지 프롬프트 작성 |
| ch04_llm_prompting.ipynb | LLM 프롬프팅 | MATPOWER 데이터 해석 프롬프트 |
| ch05_agent_architecture.ipynb | 에이전트 아키텍처 | get_voltage_violations 구현 |
| ch06_eda_tools.ipynb | 탐색적 데이터 분석 | Monte Carlo 부하 스케일링 |
| ch07_research_pipeline.ipynb | 연구 파이프라인 | N-1 위반 플래그 |
| ch08_service_visualization.ipynb | 서비스/시각화 | matplotlib rcParams 설정 |

### solutions/ (완성 코드)
- 위 8개 챕터 모두 완성 코드 포함

### figures/ (시각화 스크립트)
- gen_ch01_figures.py ~ gen_ch08_figures.py
- gen_cover.py, gen_fig_1_2.py, gen_fig_3_causal_mask.py (10개)

### Project files
- requirements.txt — 14 패키지
- README.md — 한국어 사용 가이드

## Special Handling
- streamlit → Gradio 변환 (Colab 호환)
- FastAPI → in-notebook 함수 시뮬레이션
- anthropic API → USE_API 플래그 + pre-recorded fallback
- TODO 패턴: `= None` / `pass` (fallback trick 없음)
- torch seed 고정 (`torch.manual_seed(42)`)

## Execution Verification
| Notebook | Status | Notes |
|----------|--------|-------|
| ch01_neural_network_basics | PASS | 500 epochs, pandapower IEEE 14-bus |
| ch04_llm_prompting | PASS | Pre-recorded fallback, no API needed |
| ch07_research_pipeline | PASS | PyPSA OPF optimal, HiGHS solver |
| ch08_service_visualization | PASS | Gradio, NetworkX, seaborn |
