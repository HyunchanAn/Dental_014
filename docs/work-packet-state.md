---
active_packet_id: "pkt-014-round3-loss-patch"
status: "READY_FOR_EXECUTION"
created_at: "2026-07-18T14:30:00+09:00"
---

# Dental_014 모델 3차 훈련 (Round 3) 작업 지시서

## 1. 개요
* **작업 대상**: `Dental_014` (골다공증 조기 진단 모델)
* **목표**: 2차 연장 학습(Round 2)에서 발생한 심각한 모드 붕괴(Mode Collapse) 문제를 해결하기 위해, 새로 패치된 **Ordinal Focal Loss**와 **Class Weights(역빈도 가중치)**가 적용된 최신 코드로 완전히 새로운 훈련(Round 3)을 처음부터 시작합니다.

## 2. 작업 환경 동기화 (사전 준비)
현재 워크스테이션이 SMB(공유기 네트워크 폴더)를 통해 직접 원장님 랩탑 파일 시스템에 접근하여 작업 중입니다.
* **필수 확인 사항**: `src/dental_014/loss.py` 내부가 `OrdinalFocalLoss` 및 가중치 `[3.29, 1.0, 2.15]`로 성공적으로 갱신되어 있는지 육안으로 확인하세요.

## 3. 학습 실행 (Round 3)
기존 가중치나 Optimizer 상태를 모두 초기화하고(Scratch), 아래 명령어로 백그라운드 훈련 루프를 다시 가동합니다.
```bash
python scripts/train.py --batch_size 16 --epochs 200
```
> [!NOTE]
> VRAM OOM 발생 시 `--batch_size 8`로 조정해 주세요. 이전과 동일한 구조이나 Loss 계산식만 변경되었습니다.

## 4. 학습 모니터링 (Checklist)
이번 Round 3 훈련의 핵심 관건은 **모드 붕괴(Mode Collapse)의 탈출**입니다. 
다음 사항을 집중적으로 모니터링하세요:
1. **클래스 다양성 확보 여부**: 초기 20~30 에포크 내에 모델이 C2만 맹목적으로 찍지 않고, `C1(정상)`과 `C3(골다공증)` 샘플에 대해서도 예측을 시도(Recall 증가)하는지 확인.
2. **Loss 하강 곡선**: 새로 도입된 Focal Loss에 의해 잘 맞추는 C2의 Loss는 0에 가깝게 깎이고, C1/C3의 Loss가 모델 업데이트를 주도하는지 확인.
3. **SupCon 임베딩 공간**: C2와 C3의 임베딩 거리가 유의미하게 멀어지는지 점검.

## 5. 훈련 완료 조건 및 후속 조치
* **훈련 완료 판정**: 200 에포크 완주 후, Macro F1 Score가 유의미하게 개선되고(예: 0.70 접근), 모드 붕괴가 완벽히 치료되었다고 판단되면 훈련을 조기 종료하거나 완료합니다.
* **벤치마크 갱신**: `BENCHMARK_REPORT.md`를 신규 갱신하여 3차 훈련의 성적을 기록합니다.
* **원격 동기화**: `weights/best.pt`를 Hugging Face(`chemahc94/Dental_014`)에 덮어써서 파노라마 통합 관제탑 환경과 싱크를 맞춥니다.

> [!IMPORTANT]
> **Owner Gate**
> 이번 학습 종료 후, `BENCHMARK_REPORT.md` 작성이 끝나면 즉시 보고하여 다음 지시(예: 통합 파이프라인 E2E 검사)를 대기하세요.
