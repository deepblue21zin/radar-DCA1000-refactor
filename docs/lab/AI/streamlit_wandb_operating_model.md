# Radar Lab Streamlit / W&B Operating Model

작성일: 2026-04-24

대상:

- `tools/lab/app.py`
- `tools/lab/registry.py`
- `tools/lab/analytics.py`
- `tools/lab/stage_cache.py`
- `real-time/session_logging.py`
- `tools/diagnostics/session_report.py`

목표:

- 이 저장소의 `Radar Lab`을 기준으로 Streamlit과 W&B의 역할을 겹치지 않게 분리한다.
- 로컬 `raw -> replay -> compare -> triage` 흐름을 정본으로 유지한다.
- 이후 W&B 연동 구현에서 흔들리지 않을 run naming, grouping, tagging, config, summary, artifact 계약을 먼저 고정한다.

## 1. Local-first 원칙

이 저장소의 실험 정본은 계속 로컬 파일 시스템과 SQLite에 둔다.

정본 경로:

- raw capture: `logs/raw/<capture_id>/`
- run output: `logs/live_motion_viewer/<session_id>/`
- 로컬 레지스트리: `lab_data/radar_lab_registry.db`
- stage cache: `lab_data/stage_cache/<session_id>/`
- analytics snapshot: `lab_data/analytics/run_triage_snapshot.json`

핵심 파일:

- capture 메타데이터: `logs/raw/<capture_id>/capture_manifest.json`
- raw 인덱스: `logs/raw/<capture_id>/raw_frames_index.jsonl`
- raw payload: `logs/raw/<capture_id>/raw_frames.i16`
- run 메타데이터: `logs/live_motion_viewer/<session_id>/session_meta.json`
- 실행 설정: `logs/live_motion_viewer/<session_id>/runtime_config.json`
- 요약 리포트: `logs/live_motion_viewer/<session_id>/summary.json`
- stage feature timeline: `lab_data/stage_cache/<session_id>/frame_features.jsonl`
- stage trace: `lab_data/stage_cache/<session_id>/frame_trace.jsonl`

W&B는 정본이 아니라 로컬 산출물의 요약, 비교 기록, sweep 결과를 저장하는 보조 계층으로만 사용한다.

이 원칙을 지켜야 하는 이유는 분명하다.

1. raw payload와 stage `.npz`는 용량이 크고 보안/정책 영향을 받기 쉽다.
2. 실험 해석의 기준은 여전히 same-capture replay와 local stage debug다.
3. 네 현재 목표는 "실험실 운영 + 논문용 비교 + 병목/튜닝 추적"이지 "클라우드 기반 raw 저장소"가 아니다.

## 2. Source of Truth와 W&B의 경계

Source of Truth는 아래다.

1. `logs/raw/<capture_id>/`: 동일 입력을 재현하기 위한 원본
2. `logs/live_motion_viewer/<session_id>/`: 한 번의 live/replay 실행 결과
3. `lab_data/radar_lab_registry.db`: run, capture, annotation, `run_parameters` 인덱스
4. `lab_data/stage_cache/<session_id>/`: frame-level 원인 분석 증거

W&B에 올리는 것은 아래 두 계층뿐이다.

1. session-level 실험 조건
2. session-level KPI / 병목 요약 / 필요한 report artifact

즉 "원본과 증거"는 로컬에 남기고, "비교와 기록"만 W&B로 보낸다.

## 3. 현재 코드 구조에서 역할이 나뉘는 지점

현재 흐름은 아래 함수들이 만든다.

- raw 및 run 생성:
  - `real-time/session_logging.py`
  - `tools/diagnostics/session_report.py`
- 레지스트리 인덱싱:
  - `tools/lab/registry.py`
- Streamlit UI:
  - `tools/lab/app.py`
- 병목/통계 요약:
  - `tools/lab/analytics.py`
- stage replay cache:
  - `tools/lab/stage_cache.py`

이 구조에서 Streamlit과 W&B를 아래처럼 나눈다.

- Streamlit: "왜 문제가 생겼는가"를 로컬에서 해석하는 화면
- W&B: "어떤 변경이 더 나았는가"를 run 히스토리와 sweep 결과로 누적 관리하는 화면

## 4. Streamlit 페이지별 역할

현재 페이지는 `tools/lab/app.py`에서 아래 순서로 노출된다.

- `Dashboard`
- `Runs`
- `Captures`
- `Compare`
- `Analytics/Triage`
- `Stage Timeline`
- `Stage Debug`

이 페이지들은 그대로 유지하되, 각 페이지의 책임을 더 분명하게 본다.

### 4.1 Dashboard

| 항목 | 내용 |
| --- | --- |
| 목적 | 현재 로컬 실험셋의 상태를 빠르게 파악한다. |
| 핵심 질문 | 최근 run/capture가 얼마나 쌓였는가, `clean/noisy/unusable` 분포가 어떤가, benchmark로 볼 만한 자산이 얼마나 남았는가 |
| 입력 데이터 | `registry.get_registry_overview()`, `fetch_runs()`, `fetch_captures()` |
| 출력 데이터 | 최근 run/capture 목록, transport 분포, annotation 분포 |
| Streamlit에 남겨야 하는 이유 | 실험실 운영 상태 점검 화면이기 때문이다. W&B 실험 비교 화면으로 대체하면 로컬 자산 관리 흐름이 끊긴다. |
| W&B로 넘길 결과 | 없음. Dashboard 자체는 로컬 운영 전용으로 유지한다. |

### 4.2 Runs

| 항목 | 내용 |
| --- | --- |
| 목적 | run 라이브러리를 큐레이션하고 baseline/good/discard를 정한다. |
| 핵심 질문 | 이 run을 기준 run으로 써도 되는가, 어떤 git/tuning/context에서 나온 결과인가 |
| 입력 데이터 | `fetch_runs()`, `fetch_run_detail()`, `summary.json`, `runtime_config.json`, `session_meta.json`, annotation |
| 출력 데이터 | KPI 요약, HTML report 링크, annotation 저장 결과 |
| Streamlit에 남겨야 하는 이유 | 사람이 report를 직접 열고 annotation을 다는 행위는 로컬 검토 단계에 속한다. |
| W&B로 넘길 결과 | `session_id`, `variant`, `scenario_id`, `git_*`, annotation, `summary` KPI |

운영 규칙:

- W&B sync는 이 페이지에서 run의 annotation이 정리된 뒤에만 진행한다.
- annotation의 `notes`는 W&B tag가 아니라 run notes 혹은 config metadata로만 보낸다.

### 4.3 Captures

| 항목 | 내용 |
| --- | --- |
| 목적 | raw asset 라이브러리를 관리하고 baseline 후보 raw를 고른다. |
| 핵심 질문 | 어떤 raw가 transport 영향이 적고 benchmark set에 넣을 가치가 있는가 |
| 입력 데이터 | `fetch_captures()`, `fetch_capture_detail()`, `capture_manifest.json`, linked runs, annotation |
| 출력 데이터 | capture별 invalid rate, transport quality, linked run 목록 |
| Streamlit에 남겨야 하는 이유 | raw suitability를 판단하는 화면은 로컬 정본을 기준으로 봐야 한다. |
| W&B로 넘길 결과 | `capture_id`, `transport_category`, `transport_suitability`, capture annotation 요약 |

운영 규칙:

- benchmark set 판단은 계속 `capture_id` 기준으로 내린다.
- W&B `group`도 이 판단을 따라 `capture_id`를 기본값으로 삼는다.

### 4.4 Compare

| 항목 | 내용 |
| --- | --- |
| 목적 | 같은 raw capture 기준으로 before/after와 baseline을 비교한다. |
| 핵심 질문 | 동일 입력에서 실제로 나아졌는가, 좋아졌다면 어떤 parameter/context 차이 때문인가 |
| 입력 데이터 | `fetch_runs()`, `fetch_run_parameters()`, `summary.json`, `runtime_config.json`, `capture_id` |
| 출력 데이터 | KPI delta, 실행 context 비교, tuning diff |
| Streamlit에 남겨야 하는 이유 | 논문/보고용 결론은 same-capture 비교를 로컬에서 먼저 확인해야 한다. |
| W&B로 넘길 결과 | 같은 `group` 내 run 비교 기준, baseline run reference, parameter diff snapshot |

운영 규칙:

- `Compare`는 계속 최종 해석 화면이다.
- W&B chart가 있더라도 결론 채택 전에는 이 페이지에서 same-capture 여부를 확인한다.

### 4.5 Analytics/Triage

| 항목 | 내용 |
| --- | --- |
| 목적 | 여러 run을 묶어 병목 분포와 parameter impact를 본다. |
| 핵심 질문 | 어떤 병목 label이 자주 나오는가, 어떤 parameter가 KPI 변화와 같이 움직이는가 |
| 입력 데이터 | `tools/lab/analytics.py`, `build_snapshot()`, `parameter_impact_rows()`, `run_parameters`, annotations |
| 출력 데이터 | PMF/ECDF 스타일 분포, primary bottleneck, 추천 tuning parameter, impact 테이블, `run_triage_snapshot.json` |
| Streamlit에 남겨야 하는 이유 | W&B에 올릴 요약을 만들기 전에 로컬 필터와 해석 맥락을 유지해야 한다. |
| W&B로 넘길 결과 | session-level summary metric, primary bottleneck, triage snapshot export |

운영 규칙:

- `lab_data/analytics/run_triage_snapshot.json`은 W&B sync의 입력 후보로 본다.
- 여기서 만든 병목 label과 metric 정의를 W&B summary 이름에도 그대로 쓴다.

### 4.6 Stage Timeline

| 항목 | 내용 |
| --- | --- |
| 목적 | frame-level feature를 시간축으로 보고 이상 구간을 찾는다. |
| 핵심 질문 | 문제는 몇 번째 frame부터 시작되는가, 어떤 metric이 먼저 무너지는가 |
| 입력 데이터 | `lab_data/stage_cache/<session_id>/frame_features.jsonl`, `feature_summary.json`, raw-linked run context |
| 출력 데이터 | frame timeline, threshold crossing, anomaly 구간 |
| Streamlit에 남겨야 하는 이유 | 세밀한 frame 탐색은 로컬 재생과 같이 봐야 의미가 있다. |
| W&B로 넘길 결과 | 필요 시 downsampled frame metric 또는 요약 통계만 선택적으로 전송 |

운영 규칙:

- 처음 도입 단계에서는 frame-level raw timeline 전체를 W&B에 올리지 않는다.
- W&B에는 `compute_total_ms`, `detection_count`, `confirmed_track_count`, `lead_step_m`, `lead_measurement_residual_m`, `frame_severity_10` 정도만 선택적으로 보낸다.

### 4.7 Stage Debug

| 항목 | 내용 |
| --- | --- |
| 목적 | raw replay를 다시 태워 어느 stage에서 처음 깨졌는지 찾는다. |
| 핵심 질문 | candidate가 어디서 사라졌는가, tracker association은 어느 순간부터 틀어졌는가 |
| 입력 데이터 | `lab_data/stage_cache/<session_id>/frame_trace.jsonl`, `trace_summary.json`, `artifacts/frame_XXXXXX.npz` |
| 출력 데이터 | cube preview, RDI/RAI, detections, tracker state, detailed stage trace, count funnel |
| Streamlit에 남겨야 하는 이유 | 이 페이지는 사실상 로컬 디버거이며, 무거운 증거 파일과 긴밀하게 묶여 있다. |
| W&B로 넘길 결과 | 요약 링크나 대표 이미지 정도만 가능. stage artifact 자체는 기본적으로 로컬 전용이다. |

운영 규칙:

- `Stage Debug`는 W&B로 대체하지 않는다.
- `frame_trace.jsonl`과 `.npz`는 기본 업로드 대상에서 제외한다.

## 5. W&B 프로젝트 구조

W&B 구조는 이 저장소의 필드와 일치해야 한다.

### 5.1 Project

권장 기본값:

- `project = "radar-lab"`

분리 운영이 필요할 때만 아래를 추가한다.

- `radar-lab-dev`: 일상 튜닝과 재현성 점검
- `radar-lab-paper`: 논문/발표용으로 확정한 결과만 수동 sync

처음에는 프로젝트를 하나로 유지하는 편이 낫다.

### 5.2 Name

기본 규칙:

- `name = <session_id>`

이유:

- 로컬 경로와 바로 대응된다.
- `logs/live_motion_viewer/<session_id>/`와 1:1로 맞는다.

sweep child run 예외:

- `name = <session_id>__sweep_<nnn>`

### 5.3 Group

기본 규칙:

- `group = <capture_id>`

fallback:

- raw link가 없는 오래된 run은 `group = legacy:<session_id>`

이유:

- 네 실험의 공정 비교 기준은 계속 same-capture다.
- `Compare` 페이지와 W&B 그룹이 같은 기준을 써야 해석이 흔들리지 않는다.

### 5.4 job_type

권장 값:

- `live`
- `replay`
- `sweep`
- `backfill`

매핑 기준:

- `registry.fetch_runs()`의 `input_mode`가 `live` 또는 `replay`
- 동일 capture에 대해 자동 조합을 돌리면 `sweep`
- 과거 실험을 나중에 옮기면 `backfill`

### 5.5 Tags

tag는 검색과 필터링용으로만 사용한다. 자유도가 높은 값을 전부 tag에 넣지 않는다.

권장 tag 집합:

- transport: `transport:<clean|noisy|unusable|insufficient>`
- label: `label:<baseline|good|usable|interesting|discard|unlabeled>`
- motion: `motion:<annotation_motion_pattern>`
- people: `people:<annotation_people_count>`
- phase: `phase:<debug|benchmark|paper>`

태그에 넣지 않는 값:

- `variant`
- `scenario_id`
- `git_commit`
- 세부 tuning parameter

이 값들은 모두 `config`에 넣는다.

### 5.6 Config

`config`는 결과가 아니라 "실험 조건"을 담는다.

권장 구조:

```json
{
  "session": {
    "session_id": "20260424_101530",
    "input_mode": "replay",
    "variant": "body_center_refine_v2",
    "scenario_id": "single_person_square"
  },
  "capture": {
    "capture_id": "20260423_220101",
    "transport_category": "clean",
    "transport_suitability": "baseline_ok"
  },
  "annotation": {
    "label": "baseline",
    "keep_flag": true,
    "people_count": 1,
    "motion_pattern": "square"
  },
  "git": {
    "commit": "abc1234",
    "branch": "main",
    "dirty": false
  },
  "tuning": {
    "tracking": {
      "association_gate": 4.8,
      "measurement_var": 0.04,
      "doppler_cost_weight": 0.35
    },
    "detection": {
      "max_targets": 6,
      "cluster_min_samples": 3
    }
  }
}
```

config source 우선순위:

1. `tools/lab/registry.py`의 run/capture 필드
2. annotation 필드
3. `runtime_config.json`
4. `summary.json.runtime_config.tuning_snapshot`
5. `run_parameters` 테이블

규칙:

- dotted key 대신 nested object를 사용한다.
- 논문용 비교 축은 `capture_id`, `variant`, `scenario_id`, annotation, tuning이다.
- `git_commit`은 짧게 잘라서 tag로 넣지 말고 config에 둔다.

### 5.7 Summary

`summary`에는 결과 KPI와 진단 라벨만 둔다.

권장 필드:

- `performance_score`
- `path_cleanliness_score_10`
- `path_local_residual_rms_m`
- `path_jump_ratio`
- `lead_confirmed_switch_rate`
- `candidate_to_confirmed_ratio`
- `display_to_confirmed_ratio`
- `render_latency_p95_ms`
- `compute_utilization_p95`
- `primary_bottleneck`
- `severity_10`

추가로 보관하면 좋은 값:

- `transport_suitability`
- `same_capture_group_size`
- `linked_capture_present`

metric 이름은 가능하면 `tools/lab/analytics.py`가 쓰는 용어를 그대로 유지한다.

### 5.8 Artifacts

기본 업로드 대상:

- `logs/live_motion_viewer/<session_id>/summary.json`
- `logs/live_motion_viewer/<session_id>/performance_report.html`
- `logs/live_motion_viewer/<session_id>/trajectory_replay.html`
- `lab_data/stage_cache/<session_id>/feature_summary.json`

선택 업로드:

- `lab_data/stage_cache/<session_id>/frame_features.jsonl`
- `lab_data/analytics/run_triage_snapshot.json`에서 해당 run slice를 따로 저장한 JSON

기본 제외:

- `logs/raw/<capture_id>/raw_frames.i16`
- `logs/raw/<capture_id>/raw_frames_index.jsonl`
- `lab_data/stage_cache/<session_id>/frame_trace.jsonl`
- `lab_data/stage_cache/<session_id>/artifacts/frame_XXXXXX.npz`

이유:

- raw와 stage artifact는 로컬 디버깅 증거물이다.
- W&B는 실험 비교와 결과 보관 계층이지 raw 저장소가 아니다.

## 6. 추천 naming / tagging 규칙

이 레포 필드를 아래처럼 연결한다.

| 레포 필드 | 원본 위치 | W&B 필드 | 규칙 |
| --- | --- | --- | --- |
| `session_id` | `session_meta.json`, `runs.session_id` | `name`, `config.session.session_id` | 항상 run 식별자 |
| `capture_id` | `capture_manifest.json`, `runs.capture_id` | `group`, `config.capture.capture_id` | same-capture 비교의 기본 축 |
| `variant` | `session_meta.json`, `runs.variant` | `config.session.variant` | tag 대신 config에 유지 |
| `scenario_id` | `session_meta.json`, `runs.scenario_id` | `config.session.scenario_id` | tag 대신 config에 유지 |
| `transport_category` | `summary.json.transport_quality`, `runs.transport_category` | `tags`, `config.capture.transport_category` | `transport:<value>` |
| `annotation_label` | `annotations.label` | `tags`, `config.annotation.label` | `label:<value>` |
| `annotation_keep_flag` | `annotations.keep_flag` | `tags`, `config.annotation.keep_flag` | `true`면 `phase:benchmark` 후보 |
| `annotation_motion_pattern` | `annotations.motion_pattern` | `tags`, `config.annotation.motion_pattern` | `motion:<value>` |
| `annotation_people_count` | `annotations.people_count` | `tags`, `config.annotation.people_count` | `people:<n>` |
| `git_commit` / `git_branch` / `git_dirty` | `session_meta.json`, `runs.*` | `config.git.*` | tag로는 넣지 않음 |
| tuning parameter | `runtime_config.json`, `tuning_snapshot`, `run_parameters` | `config.tuning.*` | nested object로 저장 |

추가 규칙:

1. 자유서술형 `notes`는 tag로 쓰지 않는다.
2. `variant`와 `scenario_id`는 차트 축에서 filterable해야 하므로 config로 넣는다.
3. `transport_category`는 tag와 config 둘 다에 둬도 된다. 빠른 필터링 가치가 높기 때문이다.

## 7. 무엇을 W&B에 올리고 무엇을 로컬에만 둘 것인가

### W&B에 올릴 것

- run 식별 정보: `session_id`, `capture_id`, `variant`, `scenario_id`
- annotation 요약: `label`, `keep_flag`, `people_count`, `motion_pattern`
- KPI summary: `performance_score`, path/latency/continuity 관련 metric
- 병목 요약: `primary_bottleneck`, `severity_10`
- artifact:
  - `summary.json`
  - `performance_report.html`
  - `trajectory_replay.html`
  - `feature_summary.json`
  - 필요 시 `frame_features.jsonl`

### 로컬에만 둘 것

- `raw_frames.i16`
- `raw_frames_index.jsonl`
- `frame_trace.jsonl`
- `artifacts/frame_XXXXXX.npz`
- stage debug용 대형 중간 산출물

### 경계가 애매한 것

- `frame_features.jsonl`: 처음엔 선택 업로드, 장기적으로는 sweep이나 timeline 비교가 많아질 때만 확대
- `run_triage_snapshot.json`: 전체 파일을 올리기보다 해당 run 관련 slice를 별도 요약으로 만드는 편이 안전

## 8. 운영 절차

권장 순서:

1. live 또는 replay 실행
2. `logs/live_motion_viewer/<session_id>/summary.json` 생성 확인
3. `python -m tools.lab.registry`로 로컬 인덱스 갱신
4. Streamlit `Runs`, `Captures`, `Compare`에서 annotation과 same-capture 비교 완료
5. 필요 시 `python -m tools.lab.analytics --out lab_data/analytics/run_triage_snapshot.json`
6. 그 다음에만 W&B sync 실행

즉 W&B sync는 실험 끝의 "정리 단계"에 둔다.

## 9. 단계별 도입 계획

### Stage 1. MVP: session-level sync

목표:

- run 1개를 W&B run 1개로 올린다.
- local-first 원칙을 깨지 않으면서 비교 대시보드를 만든다.

업로드 범위:

- `summary.json`
- `performance_report.html`
- `trajectory_replay.html`
- annotation + config + summary metric

하지 않는 것:

- raw 업로드
- frame-level log
- sweep 자동화

### Stage 2. Frame-level logging

목표:

- `Stage Timeline`에서 보는 대표 metric만 W&B step chart로 보낸다.

권장 metric:

- `compute_total_ms`
- `detection_count`
- `confirmed_track_count`
- `lead_step_m`
- `lead_measurement_residual_m`
- `frame_severity_10`

규칙:

- raw 전체를 보내지 않는다.
- 꼭 필요한 timeline만 downsample 혹은 full step log로 보낸다.

### Stage 3. Replay sweep

목표:

- 같은 `capture_id` raw에 대해 파라미터 조합을 체계적으로 비교한다.

원칙:

- live 장비에 바로 sweep하지 않는다.
- 반드시 replay raw 기준으로만 sweep한다.

우선 파라미터 후보:

- `tracking.association_gate`
- `tracking.measurement_var`
- `tracking.doppler_cost_weight`
- `detection.max_targets`
- `detection.cluster_min_samples`
- `detection.dbscan_adaptive_eps_bands`

주요 목표 metric:

- maximize: `path_cleanliness_score_10`
- minimize: `path_jump_ratio`, `lead_confirmed_switch_rate`
- constraint: `compute_utilization_p95`, `render_latency_p95_ms`

### Stage 4. Paper / benchmark packaging

목표:

- benchmark tag가 붙은 capture/run만 따로 정리한다.
- 논문용 표와 그림의 근거 run을 W&B에서 추적 가능하게 만든다.

규칙:

- `phase:paper` tag는 검토가 끝난 run에만 준다.
- same-capture compare를 통과한 run만 paper 후보로 올린다.

## 10. 이번 문서와 함께 보는 계약 예시

이 문서의 canonical example은 아래 파일이다.

- `docs/lab/AI/wandb_run_contract.example.json`

이 JSON은 구현 코드가 아니라 운영 계약 예시다. 실제 W&B sync 코드가 만들어질 때는 이 구조를 기본값으로 따른다.
