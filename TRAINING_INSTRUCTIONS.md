# Dental_014 (Phase 2 Classifier) Training Instructions

이 문서는 Dental_014 모듈의 Phase 2 분류기(Osteoporosis Risk Screening) 훈련을 담당할 **Main Workstation** 측 에이전트(혹은 작업자)에게 전달하는 정식 인수인계 및 실행 지침서입니다.

## 1. 훈련 목표 및 배경 (Context)
- **목표**: 하악골 관심 영역(ROI) 이미지를 입력받아 `정상(C1)`, `골감소증(C2)`, `골다공증(C3)`의 3단계 위험도로 판별하는 DenseNet 분류기 학습.
- **데이터셋 상태**: 심각한 데이터 누수를 일으켰던 기존 Sharda 데이터셋은 폐기되었으며, 임상 골흡수 지표가 신뢰성 있게 맵핑된 **BRAR-anchored multimodal dataset** (총 988장, 8:2 비율 분할 완료)이 `data/clean_cropped_ds`에 세팅되어 있습니다.

## 2. 하드웨어 리소스 최적화 지침 (Hardware Specs)
본 훈련은 다음의 Main Workstation 사양에 맞추어 최적화되어야 합니다.
- **CPU**: AMD Ryzen 9 9900X
- **GPU**: NVIDIA GeForce RTX 5080 (16GB VRAM)
- **RAM**: 64GB
- **권장 Batch Size**: 16 ~ 32 (RTX 5080의 16GB VRAM을 최대한 활용하여 학습 속도 극대화)

## 3. 실행 명령어 (Execution)
데이터 전처리 및 분할 스크립트는 이미 랩탑 환경에서 완료되었으므로, 바로 훈련을 시작하시면 됩니다.

```bash
python scripts/train.py --data-dir data --batch-size 16 --epochs 50 --lr 0.001
```

## 4. 실시간 보고 및 모니터링 원칙 (Mandatory Reporting Protocol)
훈련을 관장하는 에이전트는 다음의 SDAD 규정을 엄수하여 실시간 보고를 수행해야 합니다.

1. **학습 시작 직후 (Initialization Report)**: 
   - 총 에포크 수, 할당된 메모리 크기(VRAM), 설정된 Batch Size 등 학습 환경 전반에 대한 요약을 즉각 보고할 것.
   - 예상 학습 시간이 30분 이상으로 판단될 경우, 첫 보고 시점에 "현재 메인 워크스테이션 환경에서 훈련을 이대로 지속하는 것이 적합한지"에 대한 판단을 포함하여 제안할 것.
2. **5분 주기 실시간 보고 (Interval Report)**:
   - 5분마다 한 번씩 학습 진행 상황(Current Epoch/Loss)을 보고할 것.
   - 예상 종료 시각을 **24시간 체계(예: 17:45)**로 명확하게 제시할 것.
   - Train Loss와 Validation Loss의 간극을 모니터링하여 **과적합(Overfitting) 위험성 평가 결과**를 반드시 포함할 것.

> **작업 인수인계자 (Laptop Environment Agent)**:
> 모델 훈련이 무사히 마무리되면, 훈련된 최종 가중치(`weights/best_model.pth`)를 `Dental_Panoramic_Reader` 파이프라인 관제탑으로 전달해주시기 바랍니다. 수고하세요!
