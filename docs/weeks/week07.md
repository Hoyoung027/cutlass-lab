# 7주차 — Epilogue와 커널 융합

[전체 로드맵](../../README.md#주차별-로드맵) · [← 6주차](week06.md) · [8주차 →](week08.md)

**구현 목표:** `D[i,j] = max(Σ_k A[i,k]B[k,j] + bias[j], 0)`을 하나의 GEMM 커널에서 계산합니다.

## 구현 과제

1. 기준 구현은 GEMM, Bias, ReLU를 각각 별도 커널로 실행합니다. bias는 길이 N의 벡터로 고정합니다.
2. CUTLASS epilogue를 이용해 bias broadcast와 ReLU를 GEMM 출력 경로에 결합합니다. mainloop와 epilogue의 역할은 [CUTLASS GEMM API](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html)를 참고합니다.
3. 음수·0·양수 출력이 생기는 입력으로 검증하고, 중간 저장 시 rounding 차이가 생길 수 있으므로 기준값과 허용 오차를 명시합니다.
4. 분리 구현 전체의 GPU 실행 시간과 융합 커널의 시간을 같은 stream에서 비교합니다. 가능하면 동일한 GEMM mainloop 조건을 유지합니다.

## 산출물 및 완료 기준

- [ ] GEMM·Bias·ReLU 분리 구현과 단일 커널 융합 구현을 작성했다.
- [ ] 음수·0·양수 출력에 대해 정확성을 검증하고 기준값·허용 오차를 기록했다.
- [ ] 커널 수와 중간 버퍼 크기를 기록했다.
- [ ] 동일 stream에서 전체 GPU 구간의 실행 시간과 speedup을 비교했다.
- [ ] 작은 shape와 큰 shape의 이득 차이를 launch 비용과 중간 메모리 접근 관점에서 설명했다.
- [ ] 소스 코드, 빌드·실행 명령, 검증 로그와 해당 주차의 결과·해석을 남겼다.

검증과 측정은 [공통 규칙](../../README.md#공통-검증-및-측정-규칙)을 따릅니다. 완료한 항목은 `- [ ]`를 `- [x]`로 바꿉니다.
