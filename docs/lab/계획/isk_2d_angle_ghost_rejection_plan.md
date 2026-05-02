# ISK 2D Angle 및 Ghost Rejection 개선 계획

## 배경

현재 ISK 보드 기준 left/right diagonal 측정에서 raw-like angle 후보에는 실제 대각선 궤적이 비교적 잘 보이지만, 레이더 특성으로 보이는 대칭형 사다리꼴 ghost 후보가 함께 나타난다. 이 ghost가 DBSCAN과 tracker 입력에 섞이면서 최종 display output이 실제 대각선보다 가운데로 끌리는 문제가 있다.

현재 파이프라인은 `tx_num * rx_num` 가상 안테나를 단일 angle 축으로 취급하는 1D angle FFT 성격이 강하다. ISK/ODS의 실제 안테나 배열 차이, elevation TX 포함 여부, multipath 및 side lobe를 충분히 설명하지 못하면 좌우 대칭 ghost와 angle bias가 남을 수 있다.

## 목표

- ISK 보드를 기본 시연 보드로 확정한다.
- 레이더 정면 기준 left diagonal과 right diagonal이 각각 약 45도 방향에서 자연스럽게 나타나도록 한다.
- 실제 측정 궤적은 보존하고, 사다리꼴/대칭형 ghost 후보는 tracker에 들어가기 전에 약화하거나 제거한다.
- Streamlit Tuning Loop는 개선 후보를 고르는 심판 역할로 사용하고, ghost 자체 해결은 detection/tracking 및 angle 처리 개선으로 접근한다.

## 현재까지 반영한 완화책

- `config/profile_isk_3d_150ms.cfg`
  - `aoaFovCfg -1 -90 90 -90 90`에서 `aoaFovCfg -1 -55 55 -30 30`으로 제한했다.
  - 좌우 45도 동선에 guard band를 둔 값이다.
  - 단, DCA1000 raw ADC 자체 처리 경로에서는 이 설정만으로 후단 후보가 완전히 제한되지 않을 수 있다.
- `tools/tuning_loop/run_loop.py`
  - `baseline_safety_v2` 정책을 추가해, x span만 커지고 trajectory fidelity가 나빠지는 tuning 후보를 best로 선택하지 않도록 했다.

## 수정 방향

### 1단계: 측정 및 설정 검증

- ISK profile이 실제 측정에 쓰이고 있는지 session summary에서 확인한다.
- `compRangeBiasAndRxChanPhase`가 현재 identity 값이므로, 실제 보드/설치 상태에서 calibration 필요 여부를 확인한다.
- `aoaFovCfg` 제한 전후로 left/right diagonal을 다시 측정해 Stage Debug에서 wide-angle ghost가 줄었는지 본다.
- frame period 100ms와 150ms를 같은 동선으로 비교한다.

### 2단계: 2D angle 처리 설계

- ISK와 ODS의 virtual antenna geometry를 별도 정의한다.
- `chirpCfg` TX 순서와 RX channel 순서를 실제 virtual channel mapping으로 정리한다.
- azimuth/elevation virtual channel을 분리할 수 있는지 확인한다.
- 현재 `angle_axis_rad` 단일 축을 `azimuth_axis_rad`, `elevation_axis_rad` 또는 board-specific projection으로 확장하는 설계를 만든다.
- 우선 시연은 x/y 평면이 중요하므로, full 3D 표시보다 azimuth 정확도 개선을 1차 목표로 둔다.

### 3단계: Detection ghost rejection

- 같은 range/y 근처에서 좌우 대칭 후보가 동시에 나타날 때, temporal consistency가 높은 쪽을 우선한다.
- single-target mode에서 이전 primary track과 반대 방향으로 순간 이동하는 후보는 ghost 후보로 감점한다.
- DBSCAN 이전 후보 pool에서 사다리꼴 lateral outlier를 약화하는 옵션을 추가한다.
- DBSCAN 대표점을 단순 중심으로 잡지 않고, main trajectory density 또는 primary-consistent candidate 중심으로 잡는다.

### 4단계: Tracking 보호 장치

- primary track 근처 후보를 우선하고, 대칭 ghost birth를 억제한다.
- lateral jump가 큰 후보는 detection score가 높아도 soft gate로 낮춘다.
- 단, 실제 사람이 좌우로 움직이는 horizontal motion까지 죽이지 않도록 scenario별 기준을 분리한다.

### 5단계: 검증 루프

- ISK 4개 raw 세트를 고정 benchmark로 사용한다.
  - left diagonal
  - center
  - right diagonal
  - horizontal
- Streamlit Tuning Loop에서 `baseline_safety_v2` 정책으로 후보를 평가한다.
- 주요 KPI:
  - `trajectory_distance_p95_m`
  - `policy_overall_pass`
  - `policy_smooths_jumpy_raw`
  - `path_cleanliness_score_10`
  - `output_max_step_m`
  - `output_vs_tracking_x_span_ratio`
- 교수님 시연 기준에서는 KPI뿐 아니라 Stage Debug 전/후 이미지도 함께 저장한다.

## frame period / chirp loop 실험 계획

### frame period

- 현재 ISK 150ms profile은 UDP burst pressure를 줄이기 위한 안정성 우선 설정이다.
- 100ms:
  - 장점: 움직임을 더 촘촘히 샘플링하고 tracker jump를 줄일 수 있다.
  - 단점: 데이터 전송량과 처리 부하가 늘어 packet gap이 커질 수 있다.
- 150ms:
  - 장점: transport 안정성이 좋다.
  - 단점: 프레임 사이 이동량이 커져 diagonal 궤적이 덜 촘촘해질 수 있다.
- 같은 동선을 100ms/150ms로 각각 측정해 `transport_quality`, `output_max_step_m`, 궤적 부드러움을 비교한다.

### chirp loop

- loop 증가:
  - 장점: Doppler 안정성과 속도 추정 품질이 좋아질 수 있다.
  - 단점: 프레임당 데이터량과 처리량이 증가한다.
- loop 감소:
  - 장점: 데이터량이 줄고 frame period를 줄이기 쉬워진다.
  - 단점: Doppler 해상도와 속도 기반 gating 품질이 떨어질 수 있다.
- 현재는 loop 32를 유지하고, frame period와 ghost rejection 개선 후 필요할 때 조정한다.

## 권장 작업 순서

1. `aoaFovCfg` 반영 후 ISK left/right diagonal을 새로 측정한다.
2. Stage Debug에서 ghost 후보가 줄었는지 확인한다.
3. 줄지 않으면 detection 단계 ghost rejection부터 구현한다.
4. 동시에 ISK virtual antenna geometry 조사 및 2D angle 설계 문서를 구체화한다.
5. 2D angle 구현은 별도 브랜치/별도 커밋으로 진행한다.
6. Streamlit Tuning Loop는 알고리즘 후보의 합격/불합격 판정 장치로 유지한다.

## 결론

측정 config로 ghost를 완화할 수는 있지만, left/right diagonal에서 보이는 대칭 사다리꼴 ghost를 안정적으로 제거하려면 2D angle 또는 board-specific azimuth 처리와 single-target ghost rejection이 필요하다. 다음 구현의 핵심은 `config` 튜닝이 아니라 `radar_runtime.py`, `detection.py`, `tracking.py`의 구조적 개선이다.
