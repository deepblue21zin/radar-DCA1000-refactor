# Python 기반 mmWave Radar 실시간 캡처/처리 도구

이 프로젝트는 `XWR1843 EVM`과 `DCA1000 EVM`을 사용해 raw ADC 데이터를 실시간으로 수집하고, Python에서 바로 `RDI(Range-Doppler Image)`와 `RAI(Range-Angle Image)`를 생성하며, 그 위에 검출과 추적까지 이어서 확인할 수 있게 만든 도구입니다.

현재 기준 주요 실행 진입점은 `real-time/live_motion_viewer.py`입니다.

## 주요 기능

- DCA1000 UDP 스트림으로 raw ADC 샘플 수신
- Radar cube 생성
- Range-Doppler / Range-Angle 처리
- 움직이는 타깃 검출
- 다중 타깃 추적
- 세션별 로그 저장
- raw frame capture 저장
- 저장한 raw capture replay
- 세션 요약 및 전/후 비교 리포트 생성

## 현재 폴더 구조

- `real-time/`: 실시간 GUI 실행 진입점과 DCA1000 제어 관련 코드
- `tools/runtime_core/`: DSP, detection, tracking, runtime settings, UI layout 등 런타임 핵심 로직
- `tools/diagnostics/`: 세션 리포트, 세션 비교, 문서 생성, 오프라인 분석 도구
- `tools/lab/`: 로컬 전용 실험 관리 앱과 SQLite 레지스트리
- `config/`: 실행 설정과 튜닝 설정
- `docs/`: 프로젝트 분석 문서와 REQ 문서
- `logs/live_motion_viewer/`: 실험 세션 로그 저장 위치
- `logs/raw/`: raw frame capture 저장 위치

## 권장 환경

- 운영체제: Windows
- Python: `3.8` 권장
- 장비:
  - `XWR1843 EVM`
  - `DCA1000 EVM`

`requirements.txt`는 오래된 버전 기준이라 최신 Python에서는 바로 안 맞을 수 있습니다. 가능하면 별도 가상환경에서 맞춰서 사용하는 것을 권장합니다.

## 설치

아래 예시는 **Git Bash 기준**입니다.

```bash
cd /d/capstone_radar/ti_toolbox/radar-DCA1000-refactor
py -3.8 -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

PowerShell을 쓴다면 activation만 아래처럼 바꾸면 됩니다.

```powershell
.\.venv\Scripts\Activate.ps1
```

이미 프롬프트 앞에 `(radar-tracking)`처럼 conda 환경명이 보인다면, 그 환경이 이미 활성화된 상태입니다. 이 경우에는 `.venv`를 다시 활성화할 필요가 없습니다.

## 설정 파일 구조

설정은 지금 3개 파일로 나뉘어 있습니다.

### 1. 거의 안 바꾸는 설정

`config/live_motion_static_settings.json`

이 파일에는 장비/네트워크처럼 자주 안 바뀌는 값이 들어 있습니다.

- `cli_baudrate`
- `network.host_ip`
- `network.data_port`
- `network.config_port`
- `network.fpga_ip`
- `network.fpga_port`
- `dca.*`
- `spatial_view.*`

### 2. 실행할 때 바꾸는 설정

`config/live_motion_runtime_settings.json`

이 파일에는 PC나 세션에 따라 바뀌는 값이 들어 있습니다.

- `config_path`
- `cli_port`
- `tuning_path`
- `logging.variant`
- `logging.scenario_id`
- `logging.input_mode`
- `logging.source_capture`
- `logging.capture_duration_s`
- `logging.write_raw_capture`
- `logging.raw_capture_root`
- `logging.notes`

### 3. 튜닝 설정

`config/live_motion_tuning.json`

이 파일에는 최적화하면서 계속 조정하는 값이 들어 있습니다.

- `processing.*`
- `roi.*`
- `detection.*`
- `tracking.*`
- `pipeline.*`
- `visualization.*`

### 설정 로드 순서

`tools/runtime_core/runtime_settings.py`가 아래 순서로 병합합니다.

1. static settings
2. runtime settings
3. tuning settings

나중에 읽은 값이 앞의 값을 덮어씁니다.

## `CLI COM`, `4096`, `4098`의 의미

여기가 가장 많이 헷갈리는 부분입니다.

### `cli_port`

- 예: `COM11`
- 의미: 레이더 보드의 CLI UART 포트
- 용도: cfg 명령 전송, `sensorStart`, `sensorStop` 같은 CLI 제어
- 수정 위치: `config/live_motion_runtime_settings.json`

즉, 사용 중인 노트북에서 `CLI가 COM11`이면 이 파일의 `cli_port`를 `COM11`로 바꾸면 됩니다.

### `config_port = 4096`

- 의미: DCA1000 제어용 UDP 포트
- 용도: PC가 DCA1000에 설정 명령을 보낼 때 사용
- COM 포트가 아니라 네트워크 포트입니다

### `data_port = 4098`

- 의미: DCA1000 raw ADC 데이터 수신용 UDP 포트
- 용도: DCA1000이 PC로 ADC 샘플 스트리밍할 때 사용
- 이것도 COM 포트가 아니라 네트워크 포트입니다

### 그러면 `data COM10`은?

현재 이 프로젝트는 raw ADC 데이터를 `UART data COM`으로 받지 않습니다.  
이 프로젝트는 **DCA1000 Ethernet UDP**로 raw 데이터를 받기 때문에, Windows에 보이는 `data COM10`은 현재 메인 경로에서는 사용하지 않습니다.

정리하면:

- `CLI COM11` -> 사용함
- `DATA COM10` -> 현재 구조에서는 사용 안 함
- `4096 / 4098` -> Ethernet UDP 포트

즉, `COM11`과 `4096/4098`은 서로 대체 관계가 아니라 **완전히 다른 종류의 연결**입니다.

## 노트북에서 실시간 테스트 전 준비

### 1. CLI 포트 확인

Windows 장치 관리자에서 레이더 CLI 포트를 확인한 뒤, `config/live_motion_runtime_settings.json`의 `cli_port`를 수정합니다.

예:

```json
{
  "cli_port": "COM11"
}
```

### 2. Ethernet IPv4 설정

DCA1000이 연결된 유선 랜카드의 IPv4를 수동으로 설정합니다.

- `IP 주소`: `192.168.33.30`
- `서브넷 마스크`: `255.255.255.0`
- `게이트웨이`: 빈칸
- `DNS`: 빈칸

이 값은 `config/live_motion_static_settings.json`의 `network.host_ip`와 맞아야 합니다.

### 3. 네트워크 프로필

DCA1000이 연결된 유선 어댑터는 가능하면 `개인 네트워크`로 두는 것을 권장합니다. 공용 네트워크에서는 방화벽 때문에 UDP 수신이 더 쉽게 막힐 수 있습니다.

### 4. 방화벽 확인

필요하면 Python 또는 해당 앱이 UDP `4096`, `4098`을 사용할 수 있게 허용해야 합니다.

## 실행 방법

아래 예시는 **Git Bash 기준**입니다.

```bash
cd /d/capstone_radar/ti_toolbox/radar-DCA1000-refactor
python real-time/live_motion_viewer.py
```

가상환경을 사용하는 경우:

```bash
cd /d/capstone_radar/ti_toolbox/radar-DCA1000-refactor
source .venv/Scripts/activate
python real-time/live_motion_viewer.py
```

이미 conda 환경이 활성화되어 있다면 위 `source .venv/Scripts/activate` 단계는 건너뛰고 바로 실행하면 됩니다.

PowerShell에서는 아래처럼 실행하면 됩니다.

```powershell
cd d:\capstone_radar\ti_toolbox\radar-DCA1000-refactor
.\.venv\Scripts\Activate.ps1
python real-time\live_motion_viewer.py
```

`config/live_motion_runtime_settings.json`의 `logging.capture_duration_s`를 설정하면
live 측정이 해당 시간(초) 뒤에 자동 종료됩니다. 예를 들어 `10.0`이면 시작 후 약 10초 뒤
세션이 멈추고 로그/raw capture가 정리됩니다. `null` 또는 `0` 이하로 두면 자동 종료를 끕니다.

정상 실행 시 콘솔에서 아래와 비슷한 흐름이 보입니다.

- `Runtime config: ...`
- `Create socket successfully`
- `Now start data streaming`
- `Received first UDP packet`
- `Received first complete radar frame`

## 세션 로그

실행 후 각 세션은 `logs/live_motion_viewer/<timestamp>/` 아래에 저장됩니다.

- `session_meta.json`
- `runtime_config.json`
- `processed_frames.jsonl`
- `render_frames.jsonl`
- `event_log.jsonl`
- `status_log.jsonl`

live 실행에서는 같은 세션 id 기준으로 `logs/raw/<timestamp>/` 아래에도 raw frame capture가 저장됩니다.

- `capture_manifest.json`
- `raw_frames_index.jsonl`
- `raw_frames.i16`

UDP 수신 중 raw 파일 저장 때문에 순간적으로 수신 thread가 멈추지 않도록, raw frame 저장은 내부 writer thread로 분리되어 있습니다.
그래도 특정 프레임에서만 큰 `udp_gap_count`가 튀면 Windows 전원 모드, USB Ethernet 어댑터, 백신 실시간 검사, 디스크 지연을 먼저 의심하세요.

세션 종료 시 아래 HTML 리포트도 자동 생성됩니다.

- `logs/live_motion_viewer/<timestamp>/index.html`
- `logs/live_motion_viewer/<timestamp>/processed_report.html`
- `logs/live_motion_viewer/<timestamp>/render_report.html`
- `logs/live_motion_viewer/<timestamp>/event_report.html`
- `logs/live_motion_viewer/<timestamp>/ops_report.html`
- `logs/live_motion_viewer/<timestamp>/performance_report.html`
- `logs/live_motion_viewer/<timestamp>/trajectory_replay.html`
- replay 세션일 때: `logs/live_motion_viewer/<timestamp>/replay_report.html`

`performance_report.html`에는 이제 latency/throughput뿐 아니라 `Path Cleanliness`, `Path Max Gap Frames`,
`Path Local Residual RMS`, `Path Jump Ratio`가 같이 들어갑니다. 즉 "점수는 괜찮은데 궤적이 왜 지저분한가"를
경로 기하 품질 관점에서 따로 읽을 수 있습니다.

최근 tracking에는 `local_remeasurement_*` 설정이 추가되어, 매칭된 track 주변의 RAI patch에서 대표점을 한 번 더
재측정한 뒤 기존 measurement와 완만하게 섞습니다. render 쪽에는 `display_hysteresis_*` 설정이 추가되어
짧은 confidence drop이나 missing frame에서 화면상의 track이 바로 사라지지 않게 했고, 개입량은
`render_frames.jsonl`의 `display_held_track_count`로 확인할 수 있습니다.

이번에는 여기에 더해 `measurement_soft_gate_*` 설정이 추가되어, 예측 위치에서 너무 멀리 튄 measurement를
바로 버리지 않고 update 신뢰도를 낮추는 방식으로 반영합니다. 즉 raw에 약한 흔들림이 있어도 tracker가 한 프레임
오차에 과하게 끌려가지 않게 하려는 정책이며, 각 track payload에는 `measurement_quality`,
`measurement_residual_m`가 함께 남아 "이번 프레임 측정값을 얼마나 믿었는지"를 나중에 replay와 로그에서 다시 볼 수 있습니다.

그리고 전체 세션을 선택해서 비교할 수 있는 루트 대시보드도 자동 갱신됩니다.

- `logs/live_motion_viewer/index.html`

## Raw Capture Replay

같은 입력으로 알고리즘 수정 전후를 비교하려면 live 로그만으로는 부족합니다.  
이제 live 실행 시 저장된 raw capture를 다시 태우는 replay 진입점이 추가되어, `rosbag`처럼 동일 입력으로 회귀 검증을 반복할 수 있습니다.

기본 replay 실행:

```bash
python real-time/live_motion_replay.py --capture logs/raw/<session_id>
```

여기서 `<session_id>`는 문자 그대로 입력하는 값이 아니라 실제 raw 세션 폴더명으로 바꿔 넣어야 합니다.  
예를 들어 `logs/raw/20260409_164523/`가 있으면 아래처럼 실행합니다.

```bash
python real-time/live_motion_replay.py --capture logs/raw/20260409_164523
```

옵션:

- `--speed 2.0`: 녹화보다 2배 빠르게 재생
- `--loop`: 반복 재생
- `--wait`: 창만 열고 `Start Replay` 버튼을 눌렀을 때 시작

예:

```bash
python real-time/live_motion_replay.py --capture logs/raw/20260409_164523 --speed 1.5 --loop
```

replay도 live와 같은 `MotionViewer` 파이프라인을 다시 태우므로,

- detection / tracking 수정 전후를 동일 입력으로 비교할 수 있고
- `processed_report.html`, `render_report.html`, `performance_report.html`, `trajectory_replay.html`
- replay 세션이라면 `replay_report.html`
- `before / after` 비교 리포트

를 더 공정하게 해석할 수 있습니다.

replay가 끝나면 세션 리포트를 inline으로 생성하고, `trajectory_replay.html`을 자동으로 열어 시간축 기준 궤적 재생을 바로 확인할 수 있습니다.
이제 세션 개요, `ops_report.html`, `performance_report.html`, 루트 대시보드에는
`Transport Quality = clean / noisy / unusable` 배지도 함께 표시되어,
이번 raw capture를 baseline 튜닝용으로 써도 되는지 바로 판단할 수 있습니다.

## 권장 Stage-Wise Replay 모델

raw replay를 단순히 "다시 틀어 본다" 수준에서 끝내지 않고, 같은 raw 입력을 단계별로 다시 처리해
어느 구간에서 정보가 줄거나 튀는지 확인하는 방식이 가장 좋습니다.

실시간 측정 루프에는 상세 trace를 켜지 않습니다. `process_frame_packet(..., capture_trace=False)`가 기본값이라
live 측정은 기존처럼 처리하고, 아래 상세 분석은 `Stage Debug`에서 raw capture를 오프라인으로 다시 태울 때만 생성됩니다.

오프라인 stage trace는 다음 흐름을 한 프레임씩 기록합니다.

1. `raw_udp_packets`: raw packet/frame health
2. `frame_parsing`: packet assembly와 frame validity
3. `radar_cube`: ADC frame 재배열 결과
4. `static_removal`: static clutter 제거 전후 요약
5. `shared_fft`: 공통 FFT 계산 요약
6. `RDI`: range-doppler 입력 heatmap
7. `RAI`: range-angle 입력 heatmap
8. `CFAR candidates`: local peak 후보 수와 상위 후보
9. `angle validation`: angle contrast/ROI/local peak 검증 통과/탈락
10. `body-center refinement`: 강한 반사점에서 몸 중심에 가까운 대표점으로 재측정한 결과
11. `candidate merge`: 가까운 후보 병합 전후
12. `DBSCAN clustering`: 최종 detection 후보 축약
13. `tracker input filter`: invalid policy와 birth block 이후 tracker 입력
14. `Kalman prediction`: 예측된 track 위치
15. `association`: detection과 track 매칭
16. `Kalman update`: 업데이트 residual과 quality
17. `track birth / miss / delete`: track lifecycle 이벤트
18. `display output`: 화면에 표시되는 confirmed/tentative 결과

핵심은 raw 입력은 그대로 보존하고, 같은 세션을 대상으로 알고리즘 버전을 바꿔도 동일 입력에서 단계별 결과를 다시 만들 수 있다는 점입니다.
이제 `Stage Debug`에서는 기존 `cube preview / RDI / RAI / detections / tracker state`에 더해
`Detailed Stage Trace` 카드, `Stage Count Funnel`, `Candidate Spatial Evolution`, 단계별 표로 위 흐름을 한눈에 볼 수 있습니다.

만약 과거 replay 세션에서 `processed_frames.jsonl`, `render_frames.jsonl`만 있고 `summary.json`이나 HTML이 안 보이면, 아래 명령으로 해당 세션 리포트를 다시 생성할 수 있습니다.

```bash
python -m tools.diagnostics.log_html_reports logs/live_motion_viewer/<replay_session_id>
```
또한 replay 창 자체도 heatmap 중심이 아니라 `x/y` 좌표 기반 trajectory preview를 크게 보여주므로, raw 입력이 실제로 어떤 경로로 추적되는지 즉시 확인하기 쉽습니다.

## 세션 요약 / 비교

세션 요약:

```bash
python -m tools.diagnostics.session_report logs/live_motion_viewer/20260401_210101
```

두 세션 비교:

```bash
python -m tools.diagnostics.session_compare logs/live_motion_viewer/20260401_210101 logs/live_motion_viewer/20260401_211530
```

문서 재생성:

```bash
python -m tools.diagnostics.generate_project_docs
```

## 자동 HTML 로그 리포트

가장 편한 사용 방식은 명령어보다 아래 HTML을 직접 여는 것입니다.

- 전체 비교 대시보드: `logs/live_motion_viewer/index.html`
- 개별 세션 개요: `logs/live_motion_viewer/<timestamp>/index.html`
- 개별 로그 리포트
  - `processed_report.html`
  - `render_report.html`
  - `event_report.html`
  - `ops_report.html`
  - `performance_report.html`
  - `trajectory_replay.html`

루트 대시보드에서는 두 세션을 선택해서 before / after 비교를 할 수 있고, 주요 메트릭 차이가 그래프 형태로 표시됩니다.

`trajectory_replay.html`에서는 정적 발자취만 보는 대신 시간축에 따라 track이 어떻게 이어지고 끊기는지 재생해서 볼 수 있습니다.
같은 세션의 `performance_report.html`과 함께 보면, 눈으로 본 궤적 품질과 KPI를 같이 해석하기 좋습니다.
이때 `Transport Quality`가 `clean`이면 알고리즘 baseline 평가에 더 적합하고,
`noisy` 또는 `unusable`이면 raw/UDP 불연속이 섞였을 수 있으므로 해석을 보수적으로 하는 편이 좋습니다.

## Radar Lab 로컬 실험 관리 앱

raw capture와 replay run이 늘어나면 날짜 폴더만으로는 관리가 빠르게 어려워집니다.
그래서 이 저장소에는 `localhost`에서만 열리는 개인용 실험 관리 앱 `Radar Lab`을 추가했습니다.

핵심 원칙은 아래와 같습니다.

- 외부 서버 없음
- 클라우드 비용 없음
- `logs/raw` 원본을 덮어쓰지 않음
- 메타데이터만 `SQLite`로 인덱싱
- `127.0.0.1`에서만 열리는 로컬 앱

추가 의존성 설치:

```bash
pip install -r requirements-lab.txt
```

레지스트리 갱신만 먼저 하고 싶다면:

```bash
python -m tools.lab.registry
```

앱 실행:

```bash
streamlit run tools/lab/app.py --server.address 127.0.0.1

python -m streamlit run tools/lab/app.py --server.address 127.0.0.1

```

앱이 열리면 아래 기능을 바로 쓸 수 있습니다.

- `Dashboard`: 최근 run/capture, clean/noisy/unusable 분포, benchmark 태그 현황
- `Runs`: run 라이브러리, KPI 확인, 세션 HTML 바로 열기, motion/description annotation 저장
- `Captures`: raw capture 라이브러리, invalid rate와 linked run 확인, capture 단위 annotation 저장
- `Compare`: 같은 raw capture 기준 before / after KPI, 실행 context, tuning parameter diff 비교
- `Analytics/Triage`: 여러 run을 PMF/ECDF 관점으로 모아 보고, 병목별 추천 tuning parameter와 parameter impact 확인
- `Stage Timeline`: raw replay 기반 frame feature를 시간축으로 보고, 어느 프레임에서 병목이 시작되는지 확인
- `Stage Debug`: raw 기준 stage cache 생성, 프레임별 `cube preview / RDI / RAI / detections / tracker state` 확인
  - 참고: `Capture Link`가 있는 run에서만 stage cache를 만들 수 있다. raw 저장 기능 이전의 오래된 live 세션은 HTML 리포트 열람만 가능하다.

로컬 DB는 아래에 생성됩니다.

- `lab_data/radar_lab_registry.db`

세션 ID만으로 측정 내용을 구분하기 어려우면 `Runs`에서 `Detail Session`을 선택한 뒤
`Annotation`에 `Motion / Scenario`와 `Description / Notes`를 저장하세요.
예를 들어 `right-diagonal`, `center-round-trip`, `straight`처럼 달아 두면
Run/Capture 표와 filter에서 대각선, 중앙, 직선 왕복 데이터를 바로 나눠 볼 수 있습니다.

Stage Debug / Stage Timeline의 현재 구현은 아래 흐름으로 동작합니다.

1. 선택한 run의 `runtime_config.json`과 연결된 raw capture를 찾는다.
2. 같은 처리 경로로 raw를 다시 태워 `cube preview`, `RDI`, `RAI`, detection, tracker output을 frame별 cache로 저장한다.
3. 동시에 `frame_features.jsonl`, `feature_summary.json`, `frame_trace.jsonl`, `trace_summary.json`을 생성한다.
4. `Stage Timeline`에서 compute time, detection count, confirmed track count, lead step, residual, RAI contrast를 시간축으로 확인한다.
5. `Stage Debug`에서 선택한 frame의 compact heatmap, `Detailed Stage Trace`, 후보 수 funnel, x/y 후보 변화, `Processing Loop Outputs`를 한 화면에서 확인한다.

터미널에서 직접 stage cache만 만들고 싶다면:

```bash
python -m tools.lab.stage_cache --session 20260409_193546 --limit 60 --force
```

이 cache는 원본 raw를 건드리지 않고, 같은 입력을 여러 알고리즘 버전으로 다시 보면서
"어느 단계에서부터 틀어졌는지"를 재현성 있게 확인하기 위한 오프라인 artifact 계층입니다.
현재는 `full radar cube`와 detection 내부 pre-merge seed까지 저장하지는 않고,
`cube preview + RDI + RAI + serialized detections/tracks + frame feature timeline + detailed frame trace`를 저장합니다.

생성되는 주요 파일:

- `lab_data/stage_cache/<session_id>/frames.jsonl`
- `lab_data/stage_cache/<session_id>/frame_features.jsonl`
- `lab_data/stage_cache/<session_id>/feature_summary.json`
- `lab_data/stage_cache/<session_id>/frame_trace.jsonl`
- `lab_data/stage_cache/<session_id>/trace_summary.json`
- `lab_data/stage_cache/<session_id>/artifacts/frame_XXXXXX.npz`

`Stage Debug`의 `Detailed Stage Trace`는 아래 흐름을 frame snapshot 표와 그래프로 보여줍니다.

- `raw UDP packets -> frame parsing / assembly -> radar cube -> static removal -> shared FFT -> RDI -> RAI`
- `CFAR candidates -> angle validation -> body-center refinement -> candidate merge -> DBSCAN clustering`
- `tracker input filter -> Kalman prediction -> association -> Kalman update -> track birth / miss / delete -> display output`

추가 시각화:

- `Stage Count Funnel`: `CFAR -> angle -> merge -> DBSCAN -> tracker -> display`로 가며 후보 수가 어디서 줄어드는지 막대 형태로 표시합니다.
- `Candidate Spatial Evolution`: angle/body-center/merge/DBSCAN/tracker/display 후보를 같은 `x/y` 좌표계에 겹쳐 보여 줍니다.
- `Whole Sequence Stage View`: 프레임 하나가 아니라 전체 raw replay sequence에서 stage별 대표 궤적을 같은 축의 패널로 나란히 비교합니다.
- `Stage Count Timeline`: 전체 시간축에서 stage별 후보/track 수가 어디서 급락하는지 확인합니다.

권장 해석 순서는 `Whole Sequence Stage View -> Stage Count Timeline -> Cached Frame Index slider -> Detailed Stage Trace`입니다.
현업에서도 전체 sequence로 먼저 "어느 stage부터 경로가 찌그러졌는지"를 보고, 그 다음 문제가 시작된 frame만 잘라서 보는 방식을 많이 씁니다.
현재 `Whole Sequence Stage View`는 raw 자체가 아니라, raw replay를 각 처리 단계에 통과시킨 뒤 나온 stage output trajectory입니다.

`Stage Debug`의 `Processing Loop Outputs`는 아래 네 묶음으로 표시됩니다.

- `Stage Timings`: `cube`, `shared_fft2`, `RDI/RAI projection`, `detect`, `track` 등 단계별 시간
- `Detection Output`: `detect_targets()`가 최종 선택한 detection 후보
- `Tracker Input`: invalid policy와 birth block을 거친 tracker 입력 후보
- `Tracker Output`: `tracker.update()` 이후 confirmed/tentative track

Analytics/Triage 결과를 JSON으로 저장하고 싶다면:

```bash
python -m tools.lab.analytics --out lab_data/analytics/run_triage_snapshot.json
```

이 snapshot은 나중에 AI 병목 분류 모델, 파라미터 sweep 비교, 논문용 통계 표를 만들 때 입력 feature table로 사용할 수 있습니다.

Streamlit 페이지별 역할과 W&B 실험 추적 구조를 분리해서 설계하려면 아래 운영 문서를 같이 보는 것이 좋습니다.

- `docs/lab/AI/streamlit_wandb_operating_model.md`
- `docs/lab/AI/wandb_run_contract.example.json`

## Streamlit에서 W&B로 reviewed run 내보내기

W&B는 아직 raw 측정 직후 자동 업로드되지는 않습니다. 대신 지금은 `Runs` 상세 화면에서
annotation이 끝난 run을 골라 수동으로 W&B에 내보내는 흐름을 추가했습니다.

권장 순서:

1. live 측정 또는 raw replay 실행
2. `python -m tools.lab.registry`
3. `python -m streamlit run tools/lab/app.py --server.address 127.0.0.1`
4. `Runs`에서 `baseline / good / usable / interesting` 같은 annotation 저장
5. 같은 화면의 `W&B Export` 섹션에서 contract preview 확인
6. offline 또는 online 모드로 sync

추가 설치:

```bash
pip install wandb
```

offline 모드는 login 없이 로컬에 run 기록만 남깁니다.
online 모드는 먼저 아래처럼 로그인해야 합니다.

```bash
python -m wandb login
```

CLI로 직접 내보내고 싶다면:

```bash
python -m tools.lab.wandb_sync --session 20260409_193546 --mode offline
```

옵션:

- `--mode online`: 바로 W&B 프로젝트로 업로드
- `--phase benchmark|debug|paper`: phase tag 지정
- `--include-frame-features`: `frame_features.jsonl`도 artifact에 포함
- `--log-frame-metrics`: 대표 frame metric을 W&B step chart로 기록
- `--contract-only`: 실제 sync 없이 `wandb_run_contract.json`만 생성

생성되는 로컬 파일:

- `logs/live_motion_viewer/<session_id>/wandb_run_contract.json`
- `logs/live_motion_viewer/<session_id>/wandb_sync_result.json`
- offline 모드일 때: `lab_data/wandb/` 아래 local W&B run 디렉터리

기본 업로드 대상:

- `summary.json`
- `performance_report.html`
- `trajectory_replay.html`
- `feature_summary.json`
- 선택 시 `frame_features.jsonl`

기본 제외:

- `raw_frames.i16`
- `raw_frames_index.jsonl`
- `frame_trace.jsonl`
- stage `artifacts/frame_XXXXXX.npz`

일부 Windows 환경에서는 보안 정책 때문에 `pandas` DLL 로딩이 막혀 `Streamlit` 표가 바로 깨질 수 있습니다.
현재 `Radar Lab`은 이런 경우 HTML/SVG가 그대로 노출되지 않도록 `st.dataframe` 대신 plain-text 표 fallback으로 표시합니다.

## 참고 문서

- 프로젝트 분석 메인: `docs/project_analysis/index.html`
- 코드별 상세 문서: `docs/project_analysis/code/index.html`
- 요구사항 문서: `docs/REQ/index.html`
- Radar Lab 사용 가이드: `docs/lab/index.html`
- Streamlit/W&B 운영 설계: `docs/lab/AI/streamlit_wandb_operating_model.md`
- W&B run 계약 예시: `docs/lab/AI/wandb_run_contract.example.json`

## 데모

![](Demo.PNG)

- Demo Video: https://youtu.be/Z6rTQDMe6a4

## 연락처

- Jih-Tsun Yu: t108368020@ntut.org.tw
- Jyun-Jhih Lin: t109368038@ntut.org.tw

## 감사의 말

TI, TI e2e forum, 그리고 mmWave Radar 관련 오픈 자료들 덕분에 이 도구를 발전시킬 수 있었습니다. 또한 Chieh-Hsun Hsieh 님의 도움에 감사드립니다.

## 논문 인용

이 도구는 아래 논문의 실시간 프로토타이핑과 관련이 있습니다.

J. Yu, L. Yen and P. Tseng, "mmWave Radar-based Hand Gesture Recognition using Range-Angle Image," 2020 IEEE 91st Vehicular Technology Conference (VTC2020-Spring), Antwerp, Belgium, 2020, pp. 1-5, doi: 10.1109/VTC2020-Spring48590.2020.9128573.

```bibtex
@INPROCEEDINGS{9128573,
  author={J. {Yu} and L. {Yen} and P. {Tseng}},
  booktitle={2020 IEEE 91st Vehicular Technology Conference (VTC2020-Spring)},
  title={mmWave Radar-based Hand Gesture Recognition using Range-Angle Image},
  year={2020},
  pages={1-5},
  doi={10.1109/VTC2020-Spring48590.2020.9128573}
}
```
