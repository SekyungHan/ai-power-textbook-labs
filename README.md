# 도구에서 동료로: AI 시대의 전력 시스템 — 실습 노트북

교재 **「도구에서 동료로」**의 Google Colab 실습 프레임워크입니다.

## 디렉터리 구조

```
ai_power_textbook_labs/
├── labs/          # TODO 포함 실습 노트북 (학생용)
├── solutions/     # 완성 코드
├── utils/         # 공용 유틸리티
├── figures/       # 시각화 스크립트
└── requirements.txt
```

## 챕터 목록

| 챕터 | 주제 |
|------|------|
| ch01 | 신경망 기초 (전압 예측) |
| ch02 | 오토인코더 이상 탐지 |
| ch03 | Transformer와 OPF |
| ch04 | LLM 프롬프팅 |
| ch05 | 에이전트 아키텍처 |
| ch06 | 탐색적 데이터 분석 |
| ch07 | 연구 파이프라인 |
| ch08 | 서비스 및 시각화 |

## 빠른 시작

1. 노트북을 Google Colab에서 엽니다.
2. 첫 셀에서 패키지를 설치합니다:

```python
!pip install -r requirements.txt
```

3. 이후 셀을 순서대로 실행합니다.

## API 없이 실행하기

ch03, ch04 노트북은 Anthropic API를 사용하는 셀이 포함되어 있습니다.
API 키가 없어도 실행할 수 있도록 각 노트북 상단에 `USE_API` 플래그가 있습니다.

```python
USE_API = False  # True로 변경하면 실제 API 호출
```

`USE_API = False`(기본값)이면 사전 녹화된 응답(pre-recorded fallback)을 사용하므로
Anthropic API 키 없이도 모든 실습을 완료할 수 있습니다.
