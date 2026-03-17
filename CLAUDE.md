# ai_power_textbook_labs — 도구에서 동료로 실습 프레임워크

## 소스 교재
- 위치: ~/ai_power_textbook/
- 구조: chapters/ch01.typ ~ ch08.typ (8장)
- Python 코드블록: 28개 (Typst 인라인)
- Figure 생성 스크립트: 10개 (gen_*.py)

## 특수 패키지
- pandapower, pypsa (전력계통)
- streamlit → Gradio로 변환 (Colab 호환)
- FastAPI → in-notebook 서버로 변환
- anthropic → pre-recorded fallback 제공
