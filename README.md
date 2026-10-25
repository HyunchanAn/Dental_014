![Status](https://img.shields.io/badge/Status-v3.0%20MAE%20Architecture-brightgreen "Status") ![Python](https://img.shields.io/badge/Python-3.12%2B-blue "Python") ![Backend](https://img.shields.io/badge/Backend-PyTorch%20ViT%2FMAE-red "Backend") ![UI](https://img.shields.io/badge/UI-Streamlit-orange "UI") ![License](https://img.shields.io/badge/License-MIT-green "License") ![CI/CD Pipeline](https://img.shields.io/badge/CI%2FCD%20Pipeline-passing-brightgreen?logo=github "CI/CD Pipeline")

# Dental_014: Osteoporosis Risk Screening (하악 피질골 기반 골다공증 위험도 스크리닝)

## 개요

`Dental_014` 모듈은 환자의 파노라마 방사선 이미지에서 하악골(Mandible) 형태 및 해면골·피질골 텍스처를 분석하여 정상(`C1`), 골감소증(`C2`), 골다공증(`C3`) 위험도를 스크리닝하는 AI 진단 보조 시스템입니다. 과거 1/2세대 아키텍처(`U-Net` + `DenseNet` 조합)가 단순 픽셀 표면만 암기하며 52% 정확도(Plateau)의 한계에 부딪혔던 점을 극복하기 위해, 3세대 아키텍처(Masked Autoencoder 기반 ViT, `OsteoMAENet`)와 형태학적 특징 융합(Morphological Fusion)으로 구축되었습니다.

- [학습 및 추론 하드웨어 사양]:
  - Training Env (메인 워크스테이션): AMD Ryzen 9 9900X, NVIDIA GeForce RTX 5080 (16GB VRAM), 64GB RAM
  - Development / Inference Env (랩탑 워크스테이션): Intel Core i5-14450HX, NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM), 16GB RAM

---

## 주요 기능 및 아키텍처

### 1. 차세대 아키텍처 핵심 (v3.0 `OsteoMAENet`)
1. [Masked Autoencoder (MAE) 백본 (`OsteoMAENet`)]:
   - 이미지 패치의 75%를 무작위로 가린 뒤(Masking), 모델이 나머지 25% 정보만으로 가려진 뼈 구조를 복원(Reconstruction)하도록 훈련하여 해면골(Trabecular bone)의 기하학적 미세 텍스처 패턴을 학습합니다.
2. [해부학적 공간 어텐션 (Spatial Attention)]:
   - 치과의사가 실제 판독 시 주의 깊게 살피는 하악골 하연(Mandible inferior border) 및 이공(Mental foramen) 주변의 공간적 특징을 부각시키는 Attention 모듈이 결합되어 있습니다.
3. [고도화된 복합 손실 함수 (`OsteoCompositeLoss`)]:
   - [Ordinal Loss (순서형 회귀 손실)]: 질환의 진행 순서를 학습하여 정상(`C1`)을 골다공증(`C3`)으로 오진할 경우 강한 페널티를 부여합니다.
   - [Supervised Contrastive Loss (SupCon)]: 임상적으로 감별이 어려운 `C2`(골감소증)와 `C3`(골다공증)의 특징 벡터(Embedding)를 분리하여 클래스 불균형과 혼동 문제를 완화합니다.
4. [형태학적 융합 (Morphological Fusion) 및 XAI 시각화 (`src/dental_014/inference.py`)]:
   - 하악 하연 영역에서 적응형 이진화 마스크를 추출해 `MorphologyAnalyzer`(`mean_thickness_mm`, `min_thickness_mm`, `porosity_index`, `solidity`)를 산출하고, 이를 `OsteoMAENet` 분류 확률과 결합(`_fuse_morphology_and_logits`)하여 정상군(`C1`) 과대 양성(낮은 특이도)을 억제합니다.
   - `SpatialAttention` 가중치와 입력 그래디언트 Saliency를 결합한 XAI 히트맵 오버레이(`_compute_xai_heatmap`)를 추론 결과와 함께 출력합니다.

### 2. 아키텍처 결정 기록 (ADR) 및 통합 파이프라인 연동
- [배경]: 1/2세대 모델은 `U-Net` 분할 후 `DenseNet` 분류 시 정확도 `52%`에서 정체(Plateau)와 과적합 문제를 겪었습니다.
- [해결 방안]: 입력 이미지의 75%를 마스킹 후 복원하는 MAE 패러다임과 하악 피질골 기하학 지표를 결합하였으며, 메인 관제탑인 `Dental_Panoramic_Reader`에 서브모듈로 통합되어 가중치(`best.pt`, Hugging Face `chemahc94/Dental_014`)를 통해 E2E 추론을 수행합니다.

---

## 벤치마크 및 검증 성능

### 1. Holdout Test 벤치마크 (프로덕션 배포 가중치 `models/best.pt`)
- 평가 데이터셋: BRAR-anchored Holdout Test Split (`seed=42`, 환자 단위 격리)
- [Holdout Test Accuracy]: `54.79%`
- [Holdout Test Macro-F1]: `0.4832`

### 2. Round 3 민감도 강화 실험 벤치마크 (Ordinal Focal Loss + 역빈도 가중치 `C1=3.29, C2=1.0, C3=2.15`)
다수 클래스(`C2`) 편향(Mode Collapse)을 억제하고 중증 골다공증(`C3`) 검출 민감도를 극대화하기 위해 수행한 Round 3 실험(`n=198`) 결과는 다음과 같습니다:
- 전체 정확도 (Accuracy): `36%` (`C2` 다수 클래스 편향 제거 후 고민감도 스크리닝 세팅)
- 골다공증(`C3`) 민감도 (Sensitivity): `73.1%` (실제 `C3` 환자 52명 중 38명 탐지)
- 임상적 의의 및 한계: 중증 골다공증(`C3`) 선별 민감도는 높으나 정상 환자를 질환군으로 과대 분류하는 특이도(Specificity `17.6%`) 한계가 존재하므로, 단독 확진 도구가 아닌 1차 고위험군 스크리닝 보조 지표로만 해석해야 합니다.

#### Round 3 Classification Report (`n=198`)
| Class | Precision (정밀도) | Recall (재현율) | F1-Score | Support (샘플 수) |
| :--- | :---: | :---: | :---: | :---: |
| Normal (`C1`) | 0.30 | 0.18 | 0.22 | 34 |
| Osteopenia (`C2`) | 0.68 | 0.24 | 0.36 | 112 |
| Osteoporosis (`C3`) | 0.28 | 0.73 | 0.40 | 52 |
| Accuracy (정확도) | | | 0.36 (36%) | 198 |
| Macro Avg | 0.42 | 0.38 | 0.33 | 198 |
| Weighted Avg | 0.51 | 0.36 | 0.34 | 198 |

#### Round 3 Confusion Matrix (`n=198`)
| 실제 \ 예측 | Normal (`C1`) | Osteopenia (`C2`) | Osteoporosis (`C3`) |
| :--- | :---: | :---: | :---: |
| Normal (`C1`) | 6 | 2 | 26 |
| Osteopenia (`C2`) | 11 | 27 | 74 |
| Osteoporosis (`C3`) | 3 | 11 | 38 |

### 3. 향후 로드맵
- [ ] [특이도(Specificity) 방어 및 개선]: 정상(`C1`) 과대 예측 억제를 위한 추가 코호트 확보 및 Macro F1 Score `0.70` 이상 달성
- [x] [형태학적 융합 (Morphological Fusion)]: `morphology_analyzer.py` 기반 하악 하연 두께·공극률·충실도 지표 융합 완료
- [x] [설명 가능한 AI (XAI) 도입]: `SpatialAttention` 및 입력 그래디언트 Saliency 결합 히트맵 오버레이 연동 완료

---

## 데이터셋 출처 및 라이선스

### 1. 데이터셋 출처 및 접근 제한 (`BRAR-anchored Multimodal Dataset`)
- 본 모듈 학습에 사용된 `BRAR-anchored multimodal dataset`은 1,104명 피험자(유효 파노라마 988장)의 골흡수 지표(Bone Resorption Age Ratio, `level_1` -> `C1`, `level_2` -> `C2`, `level_3` -> `C3`)와 임상 메타데이터(`Age`, `Gender`, `Bone resorption`, `Number of missing teeth`, `Implant`, `Residual root`, `Functional tooth logarithm`)가 매핑된 제한 접근(Restricted/Private) 학술 연구용 데이터셋입니다.
- [원본 데이터 비재배포 원칙]: 환자 개인정보 보호 및 의료 데이터 윤리·보안 규정 준수를 위해 원본 영상 및 메타데이터(`data/` 폴더 전체)는 Git 저장소에 절대 포함하거나 재배포하지 않습니다(`.gitignore` 차단).
- [가중치 이용 조건]: 학습 완료된 모델 가중치(`best.pt`, Hugging Face `chemahc94/Dental_014`)는 비영리 학술 연구 및 통합 파이프라인 기술 검증 목적으로만 제한적으로 제공됩니다.
- [소스 코드 라이선스]: 본 레포지토리의 소프트웨어 소스 코드는 `MIT License` ([LICENSE](LICENSE), `Copyright (c) 2026 Hyunchan An`) 하에 배포됩니다.

### 2. 기존 Sharda 데이터셋 배제 사유 및 환자 그룹 인식 분할 정책
- 초기 검토했던 외부 `Sharda Dataset`은 동일 환자의 좌·우 크롭 패치 및 증강 사본이 환자 고유 식별자 없이 혼재되어 데이터 누수(Data Leakage)를 유발함이 확인되어 전면 배제하였습니다.
- `scripts/prepare_brar_dataset.py`는 파일명 접두사(`patient_image_000001_...` 등)로부터 결정론적 환자 식별 키(`extract_patient_group_key`)를 추출하여 환자 단위(Patient-level Group-aware) 80/20 분할(`seed=42`)과 `assert train_group_set.isdisjoint(test_group_set)` 검증을 강제합니다.

---

## 설치 및 실행 방법

1. 패키지 설치:
   ```bash
   pip install -r requirements.txt
   ```
2. 환자 단위 데이터셋 분할 준비 (Patient-Level 80/20 Split):
   ```bash
   python scripts/prepare_brar_dataset.py --source data/BRAR_dataset --dest data/clean_cropped_ds --seed 42
   ```
3. 학습 실행 (VRAM 부족 시 `--batch_size 8` 권장):
   ```bash
   python scripts/train.py --data_dir data --batch_size 16 --epochs 200
   ```
4. 추론 및 단위 테스트 실행:
   ```bash
   streamlit run app.py
   pytest tests/test_inference_and_xai.py -v
   ```

---

## 참고 문헌 및 연구용 실행 주의사항

- Klemetti, E., Kolmakov, S., & Kroger, H. (1994). Pantomography in assessment of the osteoporosis risk group. Scandinavian Journal of Dental Research, 102(1), 68-72.
- [가중치 백업 안내]: 이전 체크포인트 가중치 파일(Legacy Backups)은 `Emergency_Weights_Vault/Dental_014/Backup_Checkpoints` 폴더에 별도로 보존되어 있습니다.
- [연구용 실행 주의사항]:
  - 기본 실행 주소: `127.0.0.1` (로컬 전용)
  - 외부 공개/상용 배포 금지 및 오픈 데이터셋 외 민감 의료데이터 사용 금지
  - API 키·토큰은 환경변수로만 관리하며, 외부 접근이 필요하면 `ALLOW_EXTERNAL_ACCESS=true`로만 명시적 허용
