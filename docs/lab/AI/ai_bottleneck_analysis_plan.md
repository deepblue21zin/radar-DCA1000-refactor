# AI 기반 병목 분석 및 데이터 정제 계획서

작성일: 2026-04-20  
대상: `XWR1843 + DCA1000 + Python` 기반 radar tracking 프로젝트  
목표: 알고리즘을 바로 더 복잡하게 만들기 전에, raw 데이터와 처리 단계별 결과를 정량화하고 AI/통계 기반으로 병목 원인을 찾는 분석 체계를 만든다.

## 1. 결론

지금은 `성능을 직접 높이는 알고리즘`보다 `병목을 찾는 분석 도구`가 먼저다.

이유는 단순하다. 지금 경로가 지저분할 때 원인이 하나가 아니다.

- UDP/raw transport 문제일 수 있다.
- RDI/RAI에서 사람 반사 에너지가 불안정할 수 있다.
- detection이 몸 중심이 아니라 강한 반사점을 따라갈 수 있다.
- candidate merge가 너무 강하거나 약할 수 있다.
- tracker association이 코너/원운동에서 다른 후보로 갈아탈 수 있다.
- display hysteresis가 실제 tracking 문제를 화면에서만 가릴 수 있다.

따라서 먼저 해야 할 일은 `raw -> cube -> RDI/RAI -> detection -> tracking -> display`를 단계별로 분해하고, 각 단계에서 무엇이 깨졌는지 수치와 시각화로 볼 수 있게 만드는 것이다. AI는 이 위에 얹어야 한다. AI가 raw를 바로 고쳐주는 방향보다, 먼저 `병목 진단 보조`, `세션 품질 자동 분류`, `파라미터 sweep 결과 해석`, `논문용 데이터 정제`를 돕는 쪽이 현실적이다.

## 2. 현재 만들어진 기반

이미 어느 정도 기반은 만들어져 있다.

- `logs/raw/<session_id>`에 raw frame capture 저장
- `logs/live_motion_viewer/<session_id>`에 processed/render/event/status 로그 저장
- `trajectory_replay.html`로 시간축 기준 경로 확인
- `performance_report.html`에 path geometry KPI 추가
- `Transport Quality = clean / noisy / unusable` 분류
- `Radar Lab` Streamlit 앱
- `tools/lab/registry.py`로 capture/run 인덱싱
- `tools/lab/stage_cache.py`로 1차 stage cache 생성
- Stage Debug에서 `cube preview / RDI / RAI / detections / tracker state` 확인
- raw replay 전용 `frame_trace.jsonl` / `trace_summary.json`으로 상세 processing loop trace 확인

즉 지금부터는 완전히 새로 만드는 것이 아니라, 이미 있는 `Radar Lab + stage cache`를 분석 플랫폼으로 확장하면 된다.

### 2026-04-20 추가 구현: Analytics/Triage

이번 확장으로 `Radar Lab Statistics + Bottleneck Triage` 1차 버전을 추가했다.

- `tools/lab/analytics.py`를 추가해 run별 병목 원인을 rule-based로 진단한다.
- Streamlit 앱에 `Analytics/Triage` 페이지를 추가했다.
- 여러 세션의 `primary bottleneck` 분포를 PMF처럼 확인할 수 있다.
- KPI target을 기준으로 ECDF 관점의 pass rate를 확인할 수 있다.
- `candidate/confirmed`, `path cleanliness`, `lead switch rate`를 구간화해서 분포를 볼 수 있다.
- severity가 높은 세션부터 `transport / compute / detection / tracking / path quality` 중 어디가 의심되는지 triage board로 확인할 수 있다.
- `lab_data/analytics/run_triage_snapshot.json`으로 분석 snapshot을 export할 수 있다.

현재 진단 label은 다음과 같다.

- `transport_issue`: raw/UDP 수신 품질 문제 가능성이 큼
- `compute_latency`: 처리 계산량이나 stage hotspot 문제 가능성이 큼
- `render_latency`: 화면 표시 지연 문제 가능성이 큼
- `detection_over_split`: 한 사람을 여러 detection 후보로 쪼갤 가능성이 큼
- `display_or_confirmation_loss`: 내부 track은 있으나 화면까지 유지되지 않는 문제 가능성이 큼
- `tracking_association_failure`: 대표 ID가 자주 바뀌는 문제 가능성이 큼
- `representative_point_jump`: 몸 중심 대표점이 프레임마다 흔들리는 문제 가능성이 큼
- `path_jump`: 사람 이동으로 보기 어려운 위치 점프가 섞이는 문제 가능성이 큼
- `path_quality_low`: 발표/논문용 경로 품질 기준에 부족한 세션
- `baseline_candidate`: 알고리즘 비교 기준으로 쓰기 좋은 세션 후보

### 2026-04-20 추가 구현: Stage feature export + Stage Timeline

최종 목표에 더 가깝게 가기 위해 frame 단위 feature export와 timeline 시각화도 추가했다.

- `tools/lab/stage_cache.py`의 cache schema를 v2로 올렸다.
- raw replay stage cache 생성 시 `frame_features.jsonl`을 함께 저장한다.
- 세션별 요약으로 `feature_summary.json`을 저장한다.
- 각 frame에는 raw health, RDI/RAI 품질, detection 수, tracker 수, lead track 위치, lead step, measurement residual, stage timing, frame bottleneck label이 기록된다.
- Streamlit 앱에 `Stage Timeline` 페이지를 추가했다.
- timeline에서 `compute_total_ms`, `detection_count`, `confirmed_track_count`, `lead_step_m`, `lead_measurement_residual_m`, `RAI peak/median` 등을 시간축으로 확인할 수 있다.
- severity가 높은 프레임을 먼저 보고, 해당 frame ordinal을 Stage Debug에서 열어 RDI/RAI와 detection/tracking 상태를 확인하는 흐름으로 설계했다.
- `Analytics/Triage` snapshot에는 stage feature summary도 포함되어, 나중에 AI 병목 분류 모델의 입력 feature로 사용할 수 있다.

이로써 현재 구현은 `세션 단위 진단 -> frame 단위 timeline -> frame 산출물 디버그`까지 이어지는 최소 end-to-end 분석 루프가 된다.

### 2026-04-21 추가 구현: Raw Replay Detailed Stage Trace

실시간 처리에는 영향을 주지 않으면서, raw capture를 기준으로 전체 processing loop를 프레임별로 다시 분석하는 trace를 추가했다.

- live 측정 경로의 기본값은 `capture_trace=False`로 유지한다.
- `Stage Debug`에서 stage cache를 만들 때만 `capture_trace=True`로 raw replay를 실행한다.
- `lab_data/stage_cache/<session_id>/frame_trace.jsonl`에 frame별 상세 trace를 저장한다.
- `lab_data/stage_cache/<session_id>/trace_summary.json`에 세션 단위 stage count 요약을 저장한다.
- Streamlit Stage Debug 화면에 `Detailed Stage Trace` 카드/표를 추가했다.

기록되는 단계는 다음과 같다.

- `raw UDP packets`
- `frame parsing / assembly`
- `radar cube`
- `static removal`
- `shared FFT`
- `RDI`
- `RAI`
- `CFAR candidates`
- `angle validation`
- `body-center refinement`
- `candidate merge`
- `DBSCAN clustering`
- `tracker input filter`
- `Kalman prediction`
- `association`
- `Kalman update`
- `track birth / miss / delete`
- `display output`

이제 “결과가 이상하다”에서 멈추지 않고, 예를 들어 `CFAR 후보는 충분한데 angle validation에서 급감했는지`, `DBSCAN 이후 한 명으로 합쳐졌는지`, `association은 되었지만 display confirmed까지 못 올라갔는지`를 같은 raw 기준으로 확인할 수 있다.

## 3. 최종 목표

최종 목표는 네 가지다.

1. 좋은 raw와 나쁜 raw를 자동으로 구분한다.
2. 경로가 이상할 때 어느 단계에서 처음 문제가 생겼는지 찾는다.
3. 알고리즘 수정 전후를 같은 raw 기준으로 정량 비교한다.
4. 논문에 쓸 수 있도록 데이터셋, 지표, 실험 조건, 실패 사례를 정제해 둔다.

이 목표를 달성하면 “결과가 이상하다”가 아니라 “이 세션은 transport는 clean인데 detection 대표점이 흔들렸고, candidate merge 이후 local residual이 커져 tracker association이 3번 바뀌었다”처럼 설명할 수 있다.

## 4. 전체 시스템 구조

권장 구조는 5계층이다.

### Layer 1. Data Registry

역할: 세션을 찾고 분류한다.

저장 대상:

- capture id
- run id
- raw path
- run result path
- scenario type
- people count
- motion pattern
- transport quality
- notes
- benchmark tag
- paper usable flag

구현 위치:

- `tools/lab/registry.py`
- `lab_data/radar_lab_registry.db`
- `tools/lab/app.py`

### Layer 2. Stage Artifact Cache

역할: 같은 raw를 단계별로 다시 처리하고 중간 결과를 저장한다.

저장 대상:

- raw frame health
- cube preview
- RDI
- RAI
- detection raw candidates
- detection merged candidates
- tracker input
- tracker state
- display tracks
- per-stage timings

현재는 1차 cache가 있으므로 다음 확장은 `detection raw/merged 분리`, `tracker input quality`, `pre/post remeasurement`를 추가하는 것이다.

구현 위치:

- `tools/lab/stage_cache.py`
- `tools/runtime_core/real_time_process.py`
- `tools/lab/app.py`

### Layer 3. Statistical Quality Layer

역할: 확률/통계 기반으로 정상 범위와 이상치를 찾는다.

사용할 수 있는 개념:

- PMF: discrete count 지표 분포
- PDF 또는 KDE: continuous 지표 분포
- CDF/ECDF: 기준 이하/이상 비율
- percentile: p50, p90, p95, p99
- correlation: 어떤 지표가 경로 품질과 같이 움직이는지
- divergence: before/after 분포가 얼마나 바뀌었는지
- control chart: 시간이 지나면서 품질이 drift 되는지

이 계층은 AI보다 먼저 필요하다. AI 모델도 결국 이 feature들을 먹고 판단한다.

### Layer 4. AI Diagnosis Layer

역할: 수많은 지표를 보고 병목 원인을 자동 추정한다.

초기에는 딥러닝보다 아래 방식이 현실적이다.

- rule-based diagnosis
- anomaly detection
- clustering
- random forest / gradient boosting 기반 원인 분류
- feature importance
- weak label 기반 classifier

나중에 데이터가 충분히 쌓이면 아래도 가능하다.

- sequence model
- autoencoder
- temporal convolution
- small transformer
- Bayesian optimization 또는 Optuna 기반 파라미터 추천

### Layer 5. Experiment Tracking / Report Layer

역할: 실험을 추적하고 비교한다.

후보:

- 현재 Radar Lab + SQLite
- local parquet/csv export
- W&B offline mode
- MLflow local
- HTML report

현업 관점에서는 처음부터 W&B에 의존하기보다 `local-first`가 안전하다. W&B는 나중에 실험 수가 많아지고 차트/아티팩트 비교가 불편해질 때 붙이는 것이 좋다.

## 5. 방향성별 계획과 장단점

### 방향 A. Radar Lab을 분석 플랫폼으로 확장

내용:

- 현재 `Runs / Captures / Compare / Stage Debug`를 유지한다.
- 세션별 quality feature를 자동 계산한다.
- good / bad / baseline / discard 태그를 붙인다.
- 같은 raw 기준 before/after를 비교한다.

장점:

- 지금 코드와 가장 잘 맞다.
- 비용이 없다.
- 데이터가 외부로 나가지 않는다.
- 프론트엔드/백엔드 지식이 없어도 유지 가능하다.

단점:

- 큰 규모의 실험 추적에는 W&B 같은 전문 도구보다 약하다.
- Streamlit 성능 한계가 있을 수 있다.
- 처음에는 수동 태깅이 필요하다.

현업 관점:

- 가장 먼저 해야 하는 방향이다.
- 내부 실험 도구로는 충분히 실용적이다.
- 논문용 데이터 정제에도 가장 직접적이다.

우선순위: 1순위

### 방향 B. 단계별 artifact를 더 세밀하게 저장

내용:

- 현재 stage cache를 확장한다.
- detection 내부를 `raw candidate`, `merged candidate`, `body-center refined candidate`, `tracker input`으로 나눈다.
- 각 frame에서 처음 문제가 생긴 stage를 추적한다.

장점:

- 병목 원인을 가장 직접적으로 찾을 수 있다.
- 알고리즘 수정 전후 비교가 명확해진다.
- AI 모델 학습용 feature가 생긴다.

단점:

- 저장 용량이 늘어난다.
- full cube까지 저장하면 너무 무거워질 수 있다.
- stage schema를 잘못 잡으면 나중에 다시 바꿔야 한다.

현업 관점:

- “디버깅 가능한 시스템”을 만들려면 필수다.
- 단, 모든 세션에 full artifact를 저장하지 말고 baseline/debug 세션만 저장하는 게 좋다.

우선순위: 1순위

### 방향 C. PMF/PDF/CDF 기반 품질 분석

내용:

- 각 KPI를 random variable로 본다.
- 세션별 분포를 만들고 clean baseline과 비교한다.
- CDF로 “이 값 이하가 몇 %인지”를 본다.

예시 random variable:

- `udp_gap_count`
- `invalid_frame`
- `compute_total_ms`
- `rdi_ms`
- `rai_ms`
- `candidate_count`
- `confirmed_track_count`
- `lead_switch_interval`
- `measurement_residual_m`
- `measurement_quality`
- `jump_distance_m`
- `path_gap_frames`
- `local_residual_m`

장점:

- AI 없이도 강력하다.
- 논문에 설명하기 좋다.
- baseline 대비 개선 여부를 분포로 보여줄 수 있다.

단점:

- 원인 추정은 간접적이다.
- feature 설계가 필요하다.
- 데이터가 적으면 분포가 불안정하다.

현업 관점:

- ML 전에 반드시 하는 분석이다.
- 특히 p95, CDF, tail probability는 실시간 시스템 평가에 매우 유용하다.

우선순위: 1순위

### 방향 D. AI 기반 병목 원인 분류

내용:

세션 또는 frame window를 입력으로 넣고, 아래 원인 중 하나를 예측한다.

- transport issue
- compute/latency bottleneck
- weak RAI evidence
- detection over-split
- detection over-merge
- representative point jump
- tracker association failure
- display-only hold
- clean success

초기 방식:

- 사람이 일부 세션에 원인 라벨을 붙인다.
- rule-based weak label을 먼저 만든다.
- feature vector를 만들고 classifier를 훈련한다.
- feature importance로 어떤 지표가 원인 판단에 중요한지 본다.

장점:

- 많은 세션을 빠르게 triage할 수 있다.
- 사람이 놓치는 패턴을 찾을 수 있다.
- 나중에 논문에서 “failure taxonomy”로 쓰기 좋다.

단점:

- 라벨이 부족하면 모델이 그럴듯한 헛판단을 할 수 있다.
- 데이터가 편향되면 특정 실험 조건만 외울 수 있다.
- AI 결과를 그대로 믿으면 위험하다.

현업 관점:

- 처음에는 “판정 모델”이 아니라 “진단 보조 모델”로 써야 한다.
- 모델이 말한 원인은 반드시 stage visualization과 함께 검증해야 한다.

우선순위: 2순위

### 방향 E. W&B 또는 MLflow 기반 실험 추적

내용:

- parameter set, git commit, run result, KPI, artifacts를 실험 단위로 기록한다.
- sweep 결과를 대시보드에서 비교한다.

W&B 장점:

- 차트와 실험 비교가 매우 편하다.
- parameter sweep 관리가 좋다.
- artifact/version 관리가 강하다.

W&B 단점:

- 계정/설정이 필요하다.
- 외부 서비스 의존이 생긴다.
- 데이터를 올리면 보안/공개 범위를 신경 써야 한다.
- 지금 단계에서는 과할 수 있다.

추천:

- 지금은 Radar Lab + SQLite를 기본으로 한다.
- 나중에 실험 수가 많아지면 W&B offline mode 또는 MLflow local을 검토한다.
- raw 데이터는 절대 외부로 올리지 않는다.
- 올리더라도 summary KPI와 작은 plot만 올린다.

우선순위: 3순위

### 방향 F. AI로 알고리즘 자체를 최적화

내용:

- 같은 raw benchmark set을 기준으로 파라미터 sweep을 돌린다.
- objective score를 정의한다.
- Bayesian optimization 또는 Optuna로 추천 파라미터를 찾는다.

objective 예시:

- path cleanliness는 높게
- lead switch는 낮게
- max gap은 낮게
- jump ratio는 낮게
- compute p95는 기준 이하
- transport noisy 세션은 objective에서 제외

장점:

- 수동 튜닝보다 빠를 수 있다.
- 파라미터 trade-off를 수치로 볼 수 있다.
- 논문에서 ablation study로 쓰기 좋다.

단점:

- objective를 잘못 만들면 눈으로 보기 나쁜 결과를 좋다고 판단할 수 있다.
- benchmark raw가 적으면 과적합된다.
- 계산 시간이 늘어난다.

현업 관점:

- raw benchmark set이 충분히 쌓인 뒤 해야 한다.
- 지금 바로 하기보다는 분석/진단 체계가 먼저다.

우선순위: 3순위

## 6. PMF/PDF/CDF를 어떻게 쓸 것인가

확률론에서 배우는 PMF, PDF, CDF는 이 프로젝트에 꽤 잘 맞는다.

### PMF

count 값에 사용한다.

예시:

- 한 프레임의 candidate 수
- confirmed track 수
- frame gap 수
- lead switch 횟수
- invalid frame 개수

활용:

- single-person 세션에서 `candidate_count = 1`인 프레임 비율을 본다.
- `confirmed_track_count >= 2`가 얼마나 자주 나오는지 본다.
- lead switch가 특정 구간에 몰리는지 본다.

### PDF 또는 KDE

continuous 값에 사용한다.

예시:

- compute time
- measurement residual
- jump distance
- path local residual
- confidence
- range
- lateral position

활용:

- clean baseline의 residual 분포와 새로운 알고리즘의 residual 분포를 비교한다.
- far range에서 lateral jump 분포가 커지는지 본다.
- compute time tail이 알고리즘 변경 후 두꺼워졌는지 본다.

### CDF / ECDF

기준 이하 비율을 볼 때 사용한다.

예시:

- `measurement_residual_m <= 0.2m`인 프레임이 몇 %인가
- `compute_total_ms <= 50ms`인 프레임이 몇 %인가
- `path_gap_frames <= 2`인 세션이 몇 %인가

장점:

- 논문 그래프로 쓰기 좋다.
- 평균보다 tail을 잘 보여준다.
- 실시간 시스템의 안정성을 설명하기 좋다.

## 7. AI 병목 모델을 위한 feature 설계

AI 모델은 raw IQ 전체를 바로 먹이는 것보다, 먼저 stage feature를 먹이는 것이 현실적이다.

### Frame-level features

- frame index
- invalid flag
- udp gap count
- byte mismatch count
- compute timings
- RDI peak count
- RAI peak spread
- candidate count before merge
- candidate count after merge
- selected candidate confidence
- candidate range
- candidate lateral position
- measurement residual
- measurement quality
- track age
- miss count
- association cost
- display held flag

### Window-level features

10~30 frame 단위로 묶는다.

- candidate count mean/std
- residual mean/std/p95
- jump distance p95
- lead switch count
- max consecutive misses
- compute p95
- invalid rate
- path curvature statistics
- lateral oscillation score

### Session-level features

- transport quality
- total duration
- scenario type
- people count
- path cleanliness
- jump ratio
- max gap
- unique track ids
- display/confirmed ratio
- operational score

## 8. 병목 원인 taxonomy

AI가 맞혀야 할 원인 라벨은 처음부터 너무 많으면 안 된다. 우선 8개 정도가 적당하다.

1. `transport_issue`: UDP gap, invalid frame, byte mismatch가 큼
2. `compute_latency`: compute p95, queue delay, dropped frame이 큼
3. `weak_signal_or_rai`: RAI evidence가 약하거나 angle spread가 큼
4. `detection_over_split`: 한 사람인데 candidate가 여러 개로 나뉨
5. `detection_over_merge`: 두 사람 또는 다른 반사점이 하나로 합쳐짐
6. `representative_jump`: candidate는 있으나 대표점이 프레임마다 튐
7. `tracking_association_failure`: detection은 괜찮은데 track ID가 바뀜
8. `display_policy_only`: 내부 tracking은 괜찮은데 화면 표시 정책 때문에 다르게 보임

이 taxonomy는 논문에도 유용하다. 실패 사례를 체계적으로 분류할 수 있기 때문이다.

## 9. 구현 로드맵

### Phase 0. 데이터 원칙 확정

기간: 1~2일

작업:

- raw는 수정하지 않는다.
- capture와 run을 분리한다.
- benchmark raw는 따로 태그한다.
- paper usable flag를 둔다.
- scenario type을 고정한다.

산출물:

- Radar Lab 태그 규칙
- 데이터셋 필드 정의

### Phase 1. Stage feature export

기간: 3~5일

작업:

- stage cache에 detection pre/post merge를 추가한다.
- frame-level feature jsonl/parquet/csv export를 만든다.
- session-level summary table을 만든다.
- Radar Lab에서 feature table을 볼 수 있게 한다.

산출물:

- `stage_features.jsonl`
- `session_features.csv`
- Stage Debug 확장 화면

### Phase 2. 통계 분석 대시보드

기간: 3~5일

작업:

- PMF/PDF/CDF plot을 추가한다.
- clean/noisy/unusable 분포 비교를 추가한다.
- before/after ECDF 비교를 추가한다.
- p95 tail regression을 표시한다.

산출물:

- Radar Lab `Statistics` 페이지
- 세션별 distribution report

### Phase 3. Rule-based 병목 진단

기간: 3~4일

작업:

- transport issue rule
- detection over-split rule
- representative jump rule
- association failure rule
- display-only issue rule
- compute latency rule

산출물:

- 세션별 `suspected_bottleneck`
- frame window별 원인 후보
- Radar Lab 진단 카드

### Phase 4. AI assisted diagnosis

기간: 1~2주

작업:

- 사람이 일부 세션/구간에 라벨을 붙인다.
- weak label을 만든다.
- classifier를 훈련한다.
- feature importance를 보여준다.
- AI 판단과 rule 판단을 같이 보여준다.

산출물:

- `bottleneck_classifier.pkl`
- confusion matrix
- feature importance plot
- Radar Lab AI Diagnosis 페이지

### Phase 5. 파라미터 최적화

기간: 1~2주

작업:

- benchmark raw set을 고정한다.
- objective score를 정의한다.
- parameter sweep을 돌린다.
- Optuna 또는 간단한 Bayesian optimization을 붙인다.
- 결과를 Radar Lab 또는 W&B/MLflow로 추적한다.

산출물:

- parameter sweep report
- best preset 후보
- ablation table

## 10. 현업 관점에서의 추천 순서

현업 기준으로는 이 순서가 가장 안전하다.

1. Stage feature export
2. PMF/PDF/CDF 통계 대시보드
3. Rule-based bottleneck diagnosis
4. 사람 라벨링 UI
5. AI assisted diagnosis
6. 파라미터 최적화
7. W&B 또는 MLflow 연동

이 순서를 추천하는 이유:

- AI보다 데이터 schema가 먼저다.
- 라벨 없는 AI는 신뢰하기 어렵다.
- 통계 분석만으로도 많은 병목은 잡힌다.
- 논문에는 AI 모델보다 정량 지표와 재현 가능한 실험 설계가 더 중요하다.
- W&B는 편하지만, 지금은 local-first가 비용/보안/복잡도 면에서 더 맞다.

## 11. 바로 다음에 구현하면 좋은 MVP

가장 현실적인 1차 MVP는 이것이다.

### MVP 이름

`Radar Lab Statistics + Bottleneck Triage`

### 기능

- selected run의 stage feature 생성
- frame-level feature table 저장
- session-level feature summary 저장
- PMF: candidate count, confirmed count
- ECDF: compute time, measurement residual, jump distance
- rule-based bottleneck badge
- raw quality badge와 알고리즘 bottleneck badge 분리
- Radar Lab에서 `Statistics` 또는 `AI Diagnosis` 페이지로 확인

### 판단 예시

- `transport clean`인데 `candidate_count >= 3` 비율이 높으면 detection over-split 의심
- `candidate_count`는 안정적인데 `lead switch`가 높으면 tracking association failure 의심
- `measurement_residual` tail이 커지면 representative jump 의심
- `compute p95`가 커지고 queue delay가 증가하면 compute bottleneck 의심
- `display_held_track_count`만 높고 internal track은 안정적이면 display policy issue 의심

## 12. 논문 관점에서의 정리 방식

나중에 학사 논문까지 생각하면 아래 표를 만들 수 있어야 한다.

- 데이터셋 구성표
- scenario별 raw capture 수
- clean/noisy/unusable 비율
- single-person continuity 성능표
- two-person separation 성능표
- algorithm ablation table
- failure taxonomy table
- PMF/CDF 기반 품질 분포 그래프
- representative failure case visualization

논문에서 중요한 것은 “AI를 썼다”보다 “같은 raw 입력에서 정량적으로 비교했고, 실패 원인을 체계적으로 분류했다”이다.

## 13. 최종 제안

지금 방향은 다음처럼 잡는 것이 좋다.

1. 알고리즘 개선은 잠시 보류하지는 않되, 큰 수정은 분석 도구 이후로 미룬다.
2. Radar Lab을 데이터 관리 앱에서 분석 플랫폼으로 확장한다.
3. Stage cache를 더 세밀하게 쪼개고 feature export를 만든다.
4. PMF/PDF/CDF 기반 통계 대시보드를 먼저 만든다.
5. 그 다음 rule-based 진단을 붙인다.
6. 마지막으로 AI 모델을 진단 보조 도구로 붙인다.

한 줄 결론:

`AI로 바로 추적을 고치는 것`보다 `AI가 추적 실패 원인을 설명하도록 만드는 것`이 지금 프로젝트에는 더 가치가 크다.
