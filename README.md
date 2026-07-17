![Status](https://img.shields.io/badge/Status-Phase%202%20Training-brightgreen "Status") ![Python](https://img.shields.io/badge/Python-3.12%2B-blue "Python") ![Backend](https://img.shields.io/badge/Backend-PyTorch-red "Backend") ![Architecture](https://img.shields.io/badge/Architecture-DenseNet-orange "Architecture") ![CI/CD Pipeline](https://img.shields.io/badge/CI%2FCD%20Pipeline-passing-brightgreen?logo=github "CI/CD Pipeline")

# Dental_014: Osteoporosis Risk Screening

## 개요 (Overview)
Dental_014 모듈은 환자의 파노라마 엑스레이 이미지에서 하악골(Mandible) 형태 및 ROI를 분석하여 **골감소증(Osteopenia)** 및 **골다공증(Osteoporosis)** 위험도를 스크리닝하는 딥러닝 기반 진단 보조 시스템입니다. 
파이프라인은 두 단계(Phase)로 나뉘어 설계되었습니다:
1. **Phase 1**: U-Net을 이용한 하악골 관심 영역(ROI) 정밀 분할 및 크롭.
2. **Phase 2**: 크롭된 ROI 이미지를 바탕으로 DenseNet 분류기를 통해 3단계(정상, 골감소증, 골다공증) 위험도 판별.

## 📚 데이터셋 출처 (Datasets)
본 모듈의 단계별 학습 및 검증을 위해 사용된 데이터셋 내역은 다음과 같습니다:

1. **Sharda Dataverse Dataset**
   - **사용처**: Phase 1 (Mandible ROI U-Net Extraction)
   - **설명**: 하악골 형태 및 경계(ROI)를 학습하기 위해 활용된 초기 파노라마 데이터셋입니다. 
   - **비고**: 과도한 이미지 왜곡 증강(Augmentation)과 환자 식별자(`_0_`) 손실로 인한 데이터 누수 현상이 발견되어, 진단 신뢰성이 중요한 Phase 2 분류기 학습에서는 전면 배제되었습니다.

2. **BRAR-anchored multimodal dataset** (현재 활성 데이터셋)
   - **사용처**: Phase 2 (Osteopenia / Osteoporosis Classifier)
   - **설명**: 1,104명 환자의 실제 임상 골흡수 지표가 신뢰성 있게 매핑(Anchored)된 다중 모달리티 데이터셋입니다. `정상(C1)`, `골감소증(C2)`, `골다공증(C3)` 분류를 위한 Single Source of Truth로 훈련에 도입되었습니다.

## 설치 및 실행 방법

### 요구사항 (Requirements)
```bash
pip install -r requirements.txt
```

### 데이터셋 준비 (Dataset Preparation)
1. 다운로드한 BRAR 데이터셋을 `data/BRAR_dataset` 경로에 배치합니다.
2. 분할 스크립트를 실행하여 Train(80) / Test(20) 세트를 자동 구축합니다.
```bash
python scripts/prepare_brar_dataset.py --source data/BRAR_dataset --dest data/clean_cropped_ds
```

### 모델 학습 (Training)
```bash
python scripts/train.py --data-dir data --batch-size 8 --epochs 50
```

### 추론 (Inference / Testing)
- `app.py` 또는 `Dental_Panoramic_Reader` 메인 통합 파이프라인(`registry.py`)을 통해 추론을 수행합니다.
