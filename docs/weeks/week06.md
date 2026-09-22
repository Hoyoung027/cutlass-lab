# 6주차 — 비동기 복사와 Pipeline

[전체 로드맵](../../README.md#주차별-로드맵) · [← 5주차](week05.md) · [7주차 →](week07.md)

**구현 목표:** 다음 tile의 데이터 이동과 현재 tile의 연산이 겹치는 구조를 분석합니다.

## 구현 과제

1. GPU와 CUTLASS 버전에 맞는 pipelined GEMM 예제를 선택하고, 사용한 복사·MMA·동기화 방식을 기록합니다.
2. producer/consumer, buffer 소유권, wait/barrier, buffer 재사용 시점을 타임라인으로 그립니다.
3. 지원 범위 안에서 stage 수를 최소 두 가지로 변경하고 tile 크기도 비교합니다. 먼저 tile을 고정한 stage 실험, 다음으로 stage를 고정한 tile 실험을 합니다.
4. shared memory/block, registers/thread, occupancy와 실행 시간을 수집합니다. 컴파일러 보고서나 Nsight Compute를 활용하고 측정 불가 항목은 명시합니다.

명시적 stage 설정을 지원하지 않는 예제라면 지원 예제로 바꾸거나 실제 선택된 stage를 확인할 수 있는 설정을 사용합니다.

## 산출물 및 완료 기준

- [ ] GPU에 맞는 pipelined GEMM 소스와 복사·MMA·동기화 방식 기록을 남겼다.
- [ ] buffer 소유권과 wait/barrier·재사용 시점을 포함한 pipeline 타임라인을 작성했다.
- [ ] 지원되는 stage 최소 두 가지와 tile 크기를 비교한 결과 표를 남겼다.
- [ ] shared memory/block, registers/thread, occupancy와 실행 시간을 기록하거나 측정 불가 이유를 명시했다.
- [ ] K tile 수가 적은 경우와 많은 경우를 비교했다.
- [ ] stage 증가가 항상 이득이 아닌 이유를 자원 사용량과 함께 설명했다.
- [ ] 소스 코드, 빌드·실행 명령, 검증 로그와 해당 주차의 결과·해석을 남겼다.

검증과 측정은 [공통 규칙](../../README.md#공통-검증-및-측정-규칙)을 따릅니다. 완료한 항목은 `- [ ]`를 `- [x]`로 바꿉니다.
