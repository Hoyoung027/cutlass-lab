# 4주차 — Tiling과 Partitioning

[전체 로드맵](../../README.md#주차별-로드맵) · [← 3주차](week03.md) · [5주차 →](week05.md)

**구현 목표:** CuTe로 block tile과 스레드별 부분 영역을 표현하는 GEMM을 만듭니다.

## 구현 과제

1. [공식 CuTe 튜토리얼](https://github.com/NVIDIA/cutlass/tree/main/examples/cute/tutorial)의 기본 GEMM을 따라 구현하고, 각 Tensor의 shape와 stride를 기록합니다.
2. `local_tile`로 block 영역을 선택하고, `TiledCopy`의 thread/value layout으로 복사 작업을 분배합니다.
3. block tile `(BM,BN,BK)`을 최소 두 가지로 바꾸며 스레드 수와 shared memory 요구량을 확인합니다.
4. M/N 경계의 load/store와 마지막 K tile을 처리합니다. 원본 예제가 tile 배수만 지원한다면 predication을 추가하거나 padding 경로를 구현하고 그 비용을 따로 기록합니다.

## 산출물 및 완료 기준

- [ ] CuTe GEMM 소스와 각 Tensor의 shape·stride 기록을 남겼다.
- [ ] block→tile→thread→원소 대응 그림을 작성했다.
- [ ] 최소 두 가지 tile 구성의 정확성·실행 시간 표를 남겼다.
- [ ] tile보다 작은 행렬의 정확성을 검증했다.
- [ ] M/N/K 각각에 나머지가 생기는 입력을 검증하고, padding을 사용했다면 추가 비용을 기록했다.
- [ ] 소스 코드, 빌드·실행 명령, 검증 로그와 해당 주차의 결과·해석을 남겼다.

검증과 측정은 [공통 규칙](../../README.md#공통-검증-및-측정-규칙)을 따릅니다. 완료한 항목은 `- [ ]`를 `- [x]`로 바꿉니다.
