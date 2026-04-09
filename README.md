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

## 참고 문서

- 프로젝트 분석 메인: `docs/project_analysis/index.html`
- 코드별 상세 문서: `docs/project_analysis/code/index.html`
- 요구사항 문서: `docs/REQ/index.html`

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
