# 3주차 — CuTe Layout과 Tensor

[전체 로드맵](../../README.md#주차별-로드맵) · [← 2주차](week02.md) · [4주차 →](week04.md)

**구현 목표:** 논리 좌표, 물리 주소, 스레드별 데이터 소유 관계를 확인합니다.

## 구현 과제

1. `3×4` 행렬에 대해 row-major stride `(4,1)`, column-major stride `(1,3)`, padding이 있는 stride `(6,1)`의 좌표→원소 offset을 출력합니다.
2. 같은 데이터 포인터에 다른 layout을 적용했을 때 해석이 어떻게 달라지는지 비교합니다.
3. CuTe Tensor를 사용한 GPU copy와 `B[j,i] = A[i,j]` transpose를 구현합니다. transpose는 별도 출력 버퍼에 데이터를 실제로 저장합니다.
4. 작은 입력에서 `thread_id, src_coord, src_offset, dst_coord, dst_offset` 대응표를 출력합니다. 성능 측정 빌드에서는 디버그 출력을 제거합니다.

개념과 출력 도구는 [CuTe 시작 가이드](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/00_quickstart.html)를 참고합니다.

## 산출물 및 완료 기준

- [ ] 세 가지 stride의 좌표→offset 표를 작성하고 손계산 결과와 일치함을 확인했다.
- [ ] CuTe copy와 실제 데이터 transpose 소스를 작성했다.
- [ ] 직사각형과 경계 입력에서 copy·transpose의 정확성을 확인했다.
- [ ] 스레드별 읽기·쓰기 좌표와 offset 대응표를 남겼다.
- [ ] layout 변경에 의한 view와 물리적인 데이터 이동의 차이를 설명했다.
- [ ] 소스 코드, 빌드·실행 명령, 검증 로그와 해당 주차의 결과·해석을 남겼다.

검증과 측정은 [공통 규칙](../../README.md#공통-검증-및-측정-규칙)을 따릅니다. 완료한 항목은 `- [ ]`를 `- [x]`로 바꿉니다.
