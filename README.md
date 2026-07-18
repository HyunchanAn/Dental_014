![Status](https://img.shields.io/badge/Status-v3.0%20MAE%20Architecture-brightgreen "Status") ![Python](https://img.shields.io/badge/Python-3.12%2B-blue "Python") ![Backend](https://img.shields.io/badge/Backend-PyTorch%20ViT%2FMAE-red "Backend") ![UI](https://img.shields.io/badge/UI-Streamlit-orange "UI") ![CI/CD Pipeline](https://img.shields.io/badge/CI%2FCD%20Pipeline-passing-brightgreen?logo=github "CI/CD Pipeline")

# Dental_014: Osteoporosis Risk Screening (3rd Gen)

## 📌 개요 (Overview)
Dental_014 모듈은 환자의 파노라마 엑스레이 이미지에서 하악골(Mandible) 형태 및 텍스처를 분석하여 **정상(C1), 골감소증(C2), 골다공증(C3)** 위험도를 스크리닝하는 AI 진단 보조 시스템입니다.

과거 1/2세대 아키텍처(U-Net + DenseNet 조합)가 단순 픽셀 표면만 암기하며 52% 정확도(Plateau)의 한계에 부딪혔던 점을 극복하기 위해, 현재 **3세대 SOTA 아키텍처(Masked Autoencoder 기반 ViT)** 로 전면 리팩토링 되었습니다. 

## 🚀 차세대 아키텍처 핵심 (v3.0)

1. **Masked Autoencoder (MAE) 백본 (`OsteoMAENet`)**
   - 이미지 패치의 75%를 무작위로 가린 뒤(Masking), 모델이 나머지 25% 정보만으로 가려진 뼈 구조를 완벽히 복원(Reconstruction)하도록 훈련합니다.
   - 단순 라벨 분류를 넘어, AI 스스로 **해면골(Trabecular bone)의 기하학적 미세 텍스처 패턴**을 근본적으로 이해하도록 강제합니다.

2. **해부학적 공간 어텐션 (Spatial Attention)**
   - 치과의사가 실제 판독 시 주의 깊게 살피는 주요 부위인 **하악골 하연(Mandible inferior border)** 및 **이공(Mental foramen)** 주변의 공간적 특징을 부각시키는 Attention 모듈이 결합되어 있습니다.

3. **고도화된 복합 손실 함수 (`OsteoCompositeLoss`)**
   - **Ordinal Loss (순서형 회귀 손실)**: 질환의 진행 순서를 가르칩니다. 정상(C1)을 골다공증(C3)으로 오진할 경우 일반 오답보다 훨씬 강력한 페널티를 부여합니다.
   - **Supervised Contrastive Loss (SupCon)**: 임상적으로 가장 감별이 어려운 C2(골감소증)와 C3(골다공증)의 특징 벡터(Embedding)를 강제로 멀리 떼어놓아 클래스 불균형과 혼동 문제를 근본적으로 타파합니다.

## 📚 데이터셋 (Dataset)
- **BRAR-anchored multimodal dataset**: 1,104명 환자의 실제 임상 골흡수 지표가 매핑된 다중 모달리티 데이터셋.
- (참고: 기존에 활용하던 Sharda Dataset은 데이터 누수 및 과도한 왜곡 문제로 Phase 2 진단 분류에서 완전 배제하였습니다.)

## 🔌 관제탑(통합 파이프라인) 연동
이 모듈은 단독으로도 작동하지만, 메인 관제탑인 **`Dental_Panoramic_Reader`** 에 서브모듈(Submodule)로 완전하게 통합되어 있습니다. 
훈련된 최신 가중치(`best.pt`)는 **Hugging Face (`chemahc94/Dental_014`)** 로부터 클라우드 연동을 통해 자동으로 동기화되어, 어떤 PC에서든 즉시 파노라마 뷰어 UI(`app.py`)를 통해 E2E 추론을 수행할 수 있습니다.

## 🛠 설치 및 실행 방법

### 1. 요구사항 (Requirements)
```bash
pip install -r requirements.txt
```

### 2. 학습 실행 (Training)
```bash
python scripts/train.py --data_dir data --batch_size 16 --epochs 200
```
> 주의: MAE와 ViT 백본은 VRAM을 다소 소모합니다. OOM 발생 시 `batch_size`를 8로 줄여주세요.

### 3. 추론 (Inference)
`Dental_Panoramic_Reader` 메인 리포지토리를 실행하거나, 독립 테스트 시 아래 명령어를 사용합니다:
```bash
streamlit run app.py
```

## 📜 아키텍처 결정 기록 (ADR: Architecture Decision Record)

**배경**: 
1/2세대 모델은 `U-Net`으로 하악골을 분할한 뒤 `DenseNet`을 이용해 질환을 분류했습니다. 그러나 실제 벤치마크 결과 **정확도 52% (Macro F1 0.52)**에서 강력한 정체(Plateau)를 겪었습니다. 특히 정상(C1)과 골감소증(C2) 클래스의 불균형, 그리고 뼈 표면 픽셀만 외워버리는 과적합(Overfitting) 문제가 심각했습니다.

**해결 방안 (Why MAE + ViT?)**:
단순히 이미지 전체를 보고 클래스를 찍어맞추는 방식(분류)에서 벗어나, 해면골의 그물망(Trabecular) 구조 자체를 AI가 스스로 학습해야만 돌파구가 열린다고 판단했습니다. 
따라서 입력 이미지의 75%를 무작위로 제거(Masking)한 후 남은 25%만으로 뼈를 완벽히 그려내는 **Masked Autoencoder (Multi-task Autoencoder)** 패러다임을 전격 도입하였습니다. 이를 통해 모델은 데이터 부족 현상을 극복하고 뼈의 기하학적 형태학(Morphology)을 본질적으로 이해하게 됩니다.

## 📊 최종 성능 (Round 3 Benchmark)

가장 큰 병목이었던 '모드 붕괴(Mode Collapse, 모든 이미지를 다수인 C2로만 예측하는 현상)'를 타파하기 위해 **Ordinal Focal Loss**와 **역빈도 클래스 가중치(C1=3.29, C2=1.0, C3=2.15)**를 패치하여 수행된 최종 3차 훈련 결과입니다.

* **정확도 (Accuracy)**: **36%** (기존 52%에서 하락했으나, C2로 몰아찍던 허수 과적합이 제거된 진정한 베이스라인 수치)
* **골다공증(C3) 민감도 (Sensitivity)**: **73.1%** (실제 C3 환자 52명 중 38명 탐지)
* **임상적 의의**: 질병을 잡아내는 민감도는 우수해졌으나, 정상 환자도 질환으로 판단하는 특이도(Specificity, 17.6%)가 낮아 극단적인 스크리닝 도구로 동작 중입니다. 형태학적 융합 및 데이터 증강을 통한 밸런스 튜닝이 요구됩니다.

### Classification Report (분류 성능 상세)

| Class | Precision (정밀도) | Recall (재현율) | F1-Score | Support (샘플 수) |
| :--- | :---: | :---: | :---: | :---: |
| **Normal(C1)** | 0.30 | 0.18 | 0.22 | 34 |
| **Osteopenia(C2)** | 0.68 | 0.24 | 0.36 | 112 |
| **Osteoporosis(C3)** | 0.28 | 0.73 | **0.40** | 52 |
| | | | | |
| **Accuracy (정확도)** | | | **0.36 (36%)** | 198 |
| **Macro Avg** | 0.42 | 0.38 | 0.33 | 198 |
| **Weighted Avg** | 0.51 | 0.36 | 0.34 | 198 |

### Confusion Matrix (혼동 행렬)

| 실제 \ 예측 | Normal (C1) | Osteopenia (C2) | Osteoporosis (C3) |
| :--- | :---: | :---: | :---: |
| **Normal (C1)** | **6** | 2 | 26 |
| **Osteopenia (C2)** | 11 | **27** | 74 |
| **Osteoporosis (C3)** | 3 | 11 | **38** |


## 🎯 향후 로드맵 및 To-Do List (최근 아키텍처 리뷰 피드백 반영)

최근 전체 통합 아키텍처 리뷰에서 지적받은 **Dental_014의 취약점(낮은 특이도 및 시각화 부재)**을 최우선적으로 해결하기 위한 과제 목록입니다.

- [ ] **특이도(Specificity) 방어 및 개선**: 현재 정상(C1)을 질환으로 과대 예측하는 낮은 특이도(17.6%) 문제를 타파하기 위해, 클래스 불균형 해소용 데이터 증강 및 비공개 고품질 데이터 더 많이 투입.
- [ ] **형태학적 융합 (Morphological Fusion)**: `morphology_analyzer.py`를 활용해 하악 하연 두께(Cortical thickness) 등 전통적 임상 지표를 ViT 임베딩과 직접 융합하여 Mode Collapse 완전 해결.
- [ ] **설명 가능한 AI (XAI) 적극 도입**: 타 모듈 대비 판독 근거의 시각화 일관성이 떨어지므로, Grad-CAM 및 Attention Map 시각화를 모듈 내에 강력하게 도입하여 치과의사에게 형태학적 판단 근거 제시.
- [ ] **최종 목표 성능**: **Macro F1 Score 0.70 이상** 달성 및 상용화 수준 벤치마크 확립
