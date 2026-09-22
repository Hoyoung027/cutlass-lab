# cutlass-lab

CUDA C++로 GEMM을 직접 구현하고, CUTLASS와 CuTe를 활용해 GPU 커널의 데이터 배치·연산·파이프라인·융합을 익히는 8주 학습 프로젝트입니다.

최종 목표는 **정확한 커널을 구현하고, 특정 행렬 크기에서 성능이 달라지는 이유를 측정 결과로 설명하는 것**입니다. cuBLAS보다 빠른 결과를 필수 완료 조건으로 삼지는 않습니다.

## 학습 범위와 준비

제안한 `CUDA 기초 → CUTLASS 실행 → CuTe → tiling → Tensor Core → pipeline → fusion → tuning` 순서로 진행합니다. 2주차에는 라이브러리 사용과 측정에 집중하고, 내부 구조는 3주차부터 단계적으로 분석합니다. 기본 트랙은 **CUTLASS C++ / CuTe C++**이며 Python CuTe DSL은 별도 확장으로 둡니다.

- C++의 포인터, 템플릿, `auto`와 행렬 곱의 기본 개념을 사전 학습합니다.
- 실습용 NVIDIA GPU, 호환되는 드라이버·CUDA Toolkit·C++ 컴파일러·CMake를 준비합니다. 로컬에 NVIDIA GPU가 없다면 원격 GPU 서버에서 빌드와 측정을 수행합니다.
- GPU 모델과 compute capability, 드라이버·CUDA·컴파일러 버전, CUTLASS tag 또는 commit을 기록합니다. 선택한 CUTLASS 버전의 빌드 요구사항을 확인하고 실습 중에는 버전을 고정합니다.
- 5~6주차 예제는 실제 GPU가 지원하는 MMA·비동기 복사 명령과 해당 예제의 요구 아키텍처를 확인해 선택합니다. Hopper 전용 예제를 다른 GPU에 그대로 적용하지 않습니다. 공식 [예제 목록](https://github.com/NVIDIA/cutlass/blob/main/examples/README.md)에서 대상 GPU를 확인합니다.

현재 문서는 학습 및 구현 계획입니다. 구현이나 벤치마크가 완료되었다는 뜻은 아닙니다.

## 주차별 로드맵

주차 링크를 열면 구현 과제와 완료 체크리스트를 볼 수 있습니다. 완료한 항목은 각 파일에서 `- [ ]`를 `- [x]`로 바꿉니다.

| 주차 | 학습 내용 | 구현 목표 및 핵심 과제 | 완료 기준 |
| --- | --- | --- | --- |
| [1주차](docs/weeks/week01.md) | thread/block/warp, global/shared memory, 동기화 | FP32 naive GEMM과 shared-memory tiled GEMM 구현 | 두 구현의 정확성·경계 처리를 확인하고 데이터 재사용 차이 설명 |
| [2주차](docs/weeks/week02.md) | CUTLASS 기본 GEMM, shape·layout·dtype | 공식 예제를 수정하고 여러 shape에서 cuBLAS와 벤치마크 | 같은 연산 조건의 실행 시간·TFLOPS·오차 기록 |
| [3주차](docs/weeks/week03.md) | CuTe `Layout`, `Shape`, `Stride`, `Tensor` | 좌표→offset 출력, copy와 실제 데이터 transpose 구현 | 스레드별 읽기·쓰기 원소와 주소 매핑 설명 |
| [4주차](docs/weeks/week04.md) | tiling, partitioning, `local_tile`, `TiledCopy` | CuTe GEMM 구현 및 tile 변경 | block/thread 담당 영역과 경계 처리 검증 |
| [5주차](docs/weeks/week05.md) | Tensor Core, MMA atom, `TiledMma` | GPU에 맞는 FP16 또는 BF16 GEMM 수정 | 입력·누산·출력 dtype을 구분하고 tile별 성능 비교 |
| [6주차](docs/weeks/week06.md) | 비동기 복사, synchronization, pipeline | pipelined GEMM의 stage 수와 tile 변경 | 복사·연산 중첩 구조와 자원 사용량의 관계 설명 |
| [7주차](docs/weeks/week07.md) | mainloop, epilogue, 커널 융합 | 단일 커널의 `GEMM → Bias → ReLU` 구현 | 분리 구현과 정확성·전체 GPU 실행 시간 비교 |
| [8주차](docs/weeks/week08.md) | 병목 분석, 튜닝 | 특정 shape군 최적화 또는 확장 과제 하나 수행 | 빨라진 조건과 느려진 조건을 포함한 재현 가능한 보고서 작성 |

## 공통 검증 및 측정 규칙

### 정확성

- 작은 입력은 CPU 기준 구현, 큰 입력은 cuBLAS 등 검증된 구현과 비교합니다. 작은 행렬의 기준 계산은 가능하면 FP64로 누산합니다.
- 기본 GEMM 테스트는 `alpha=1`, `beta=0`으로 통일하고, 다른 설정을 추가하면 별도 케이스로 관리합니다.
- 원소별 `abs(actual - ref) <= atol + rtol * abs(ref)`를 검사합니다. 입력 분포·K·dtype에 따라 허용 오차를 정해 기록하고 최대 절대 오차와 실패 원소 수도 남깁니다. NaN/Inf는 별도 확인합니다.
- 고정 seed의 난수, 0, 단위행렬과 양수/음수 혼합 입력을 사용합니다. copy/transpose는 유한 입력의 값이 정확히 보존되는지 검사합니다.
- 직접 작성한 shared-memory·비동기 커널은 Compute Sanitizer로 범위 밖 접근과 경쟁 조건을 확인합니다. CUDA 및 라이브러리 반환 상태도 검사합니다.

아래는 시작용 shape 세트이며 GPU 메모리에 맞게 조정합니다. shape는 `(M,N,K)` 순서입니다.

| 목적 | 예시 |
| --- | --- |
| 손계산·최소 입력 | `(1,1,1)`, `(3,5,7)` |
| 경계 처리 | `(31,33,29)`, `(127,129,65)` |
| 정방형 성능 | `(256,256,256)`, `(1024,1024,1024)`, `(4096,4096,4096)` |
| 직사각형 성능 | `(128,4096,1024)`, `(4096,128,1024)` |
| 작은 M·긴 K | `(16,4096,4096)` |

직접 구현한 범용 커널은 경계 shape를 통과해야 합니다. CUTLASS Tensor Core 예제 등에서 지원하지 않는 shape/alignment는 **미지원**으로 기록하거나 padding/fallback으로 처리하며, 추가 비용을 측정 결과에 명시합니다.

### 성능

- 기본값은 warm-up 10회 후 반복 실행 100회입니다. 짧은 커널은 측정 구간이 충분히 길어지도록 반복 수를 늘립니다.
- 같은 CUDA stream에 기록한 CUDA event로 GPU 시간을 측정하고 종료 event를 동기화한 뒤 읽습니다. 반복 구간의 총 시간을 반복 횟수로 나눕니다.
- 동일 측정 묶음을 최소 5회 수행하고 평균 호출 시간들의 중앙값과 최소/최대값을 보고합니다.
- 메모리 할당, 초기화, host-device 전송, 검증, 디버그 출력은 커널 시간에서 제외합니다. 전처리·padding이 필요하면 이를 포함한 시간도 별도 보고합니다.
- 7주차는 GEMM만 재지 않고 Bias·ReLU까지 포함한 전체 GPU 구간을 잽니다. CPU 호출 비용을 포함한 end-to-end 시간은 별도 항목입니다.
- GEMM 처리량은 `TFLOPS = 2*M*N*K / (time_ms * 1e9)`로 계산합니다. 융합 결과에 이 식을 사용하면 GEMM 연산량 기준임을 명시합니다.
- cuBLAS와 layout, 입력·누산·출력 dtype, alpha/beta, math mode, stream 및 측정 범위를 맞춥니다. 프로파일링 실행과 시간 측정 실행을 분리합니다.
- 같은 GPU에서 비교하고 다른 작업에 의한 간섭 여부를 기록합니다. 지원하지 않는 설정이나 실패한 검증 결과를 정상 성능 결과에 섞지 않습니다.

### 결과 기록

각 주차에 소스 코드, 빌드·실행 명령, 검증 로그, 원본 결과 CSV, 짧은 해석을 남깁니다. CUDA/CUTLASS 환경과 commit은 결과 파일에 함께 저장하거나 별도 환경 파일로 연결합니다.

```text
week,implementation,M,N,K,input_dtype,accum_dtype,output_dtype,layout,tile,stages,time_ms,tflops,max_abs_error,passed
```

해당하지 않는 tile/stage 항목은 `NA`로 기록합니다. 3주차 copy/transpose는 TFLOPS 대신 전송 바이트 수와 유효 대역폭을 사용합니다. 6주차에는 자원 사용량을, 7주차에는 전체 구간 시간과 baseline 대비 speedup을 추가합니다.

## 문서 및 산출물 구조

주차별 학습 문서는 `docs/weeks/`에 파일 하나씩 둡니다. 구현 폴더는 해당 주차의 코드를 작성할 때 생성합니다.

```text
cutlass-lab/
├── README.md                 # 전체 로드맵, 공통 검증·측정 규칙
└── docs/
    └── weeks/
        ├── week01.md         # CUDA GEMM 기초
        ├── week02.md         # CUTLASS 실행과 기준 성능
        ├── week03.md         # CuTe Layout과 Tensor
        ├── week04.md         # Tiling과 Partitioning
        ├── week05.md         # Tensor Core와 MMA
        ├── week06.md         # 비동기 복사와 Pipeline
        ├── week07.md         # Epilogue와 커널 융합
        └── week08.md         # 병목 분석과 최종 과제
```

실습을 시작하면 `week01_cuda_gemm/` 등의 구현 폴더와 `CMakeLists.txt`를 추가합니다. 공통 유틸리티는 `common/`, 측정 CSV와 환경 정보는 `results/`, 분석 보고서는 `reports/`에 모으는 구성을 권장합니다. 산출물을 만든 뒤 해당 주차 문서에 상대 경로 링크를 추가합니다.

## 공식 참고 자료

- [CUTLASS 저장소 및 빌드 안내](https://github.com/NVIDIA/cutlass): 사용할 버전과 설치·빌드 조건 확인
- [CUTLASS 공식 예제 목록](https://github.com/NVIDIA/cutlass/blob/main/examples/README.md): GPU에 맞는 기본 GEMM, pipeline, fusion 예제 선택
- [CuTe 시작 가이드](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/00_quickstart.html): Layout, Tensor 및 출력 도구
- [CuTe GEMM 튜토리얼](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html): tiling과 partitioning을 이용한 GEMM 구성
- [CuTe Predication](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0y_predication.html): tile 경계 처리
- [CUTLASS 3.x GEMM API](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html): mainloop, epilogue, 계층별 구성 요소

`main`과 `latest` 링크는 변경될 수 있으므로 실제 실습에서는 고정한 CUTLASS tag/commit에 해당하는 예제와 문서를 기준으로 합니다.
