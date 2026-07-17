# 📊 OsteoMultiTaskNet 벤치마크 성적표 (최종)

> **평가 일시**: 2026-07-18
> **모델 구조**: `OsteoMultiTaskNet` (Mask-Free Autoencoder Backbone)
> **학습량**: 총 200 Epochs (1차 100 + 2차 100 연장)
> **최고 가중치 경로**: `weights/best.pt`
> **평가 데이터 수**: 198 (Test Set)

---

## 1. Classification Report (분류 성능 상세)

| Class | Precision (정밀도) | Recall (재현율) | F1-Score | Support (샘플 수) |
| :--- | :---: | :---: | :---: | :---: |
| **Normal(C1)** | 0.43 | 0.85 | **0.57** | 34 |
| **Osteopenia(C2)** | 0.67 | 0.39 | **0.49** | 112 |
| **Osteoporosis(C3)** | 0.45 | 0.56 | **0.50** | 52 |
| | | | | |
| **Accuracy (정확도)** | | | **0.52 (52%)** | 198 |
| **Macro Avg** | 0.52 | 0.60 | 0.52 | 198 |
| **Weighted Avg** | 0.57 | 0.52 | 0.51 | 198 |

---

## 2. Confusion Matrix (혼동 행렬)

| 실제 \ 예측 | Normal (C1) | Osteopenia (C2) | Osteoporosis (C3) |
| :--- | :---: | :---: | :---: |
| **Normal (C1)** | **29** | 4 | 1 |
| **Osteopenia (C2)** | 33 | **44** | 35 |
| **Osteoporosis (C3)** | 5 | 18 | **29** |

---

## 💡 성능 분석 및 최종 판정

1. **최종 성능 분석**: 
   - 1차(100 Epochs) 당시 정확도 **54%**를 기록한 후, 2차 연장 학습(101~200 Epochs)을 강행했습니다.
   - 연장 학습 결과, 훈련 세트에 대한 정확도는 65.3%까지 상승했으나 정작 검증 및 테스트 세트에 대한 종합 정확도는 **52%**로 오히려 소폭 하락(Plateau & Mild Overfitting)하는 양상을 보였습니다.
   - 모델이 주어진 아키텍처(Mask-Free Multi-task) 내에서 추출할 수 있는 데이터의 한계치에 완전히 도달한 것으로 판단됩니다.

2. **Goal Mode 목표 판정**:
   - 이전 라운드 대비 성능 향상률이 **1% 미만(오히려 하락)**이므로 사전 승인된 중단 조건(Stopping Criteria)에 부합합니다.
   - 따라서 반복 훈련 루프를 강제 종료하고 현 가중치를 최종본으로 확정합니다.

---

**추가 학습은 필요 없음**
