from __future__ import annotations

import html
import re
from collections import Counter
from pathlib import Path

from tools.diagnostics.doc_explanations import FILE_EXPLAINERS


ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = ROOT / "docs"
PROJECT_DIR = DOCS_DIR / "project_analysis"
CODE_DIR = PROJECT_DIR / "code"
REQ_DIR = DOCS_DIR / "REQ"

CURRENT_STATE = {
    "score_overall": "82/100",
    "score_structure": "80/100",
    "score_logging": "93/100",
    "code_files": 18,
    "log_artifacts": 9,
}


FILES = [
    {
        "slug": "live_motion_viewer",
        "source": "real-time/live_motion_viewer.py",
        "title": "live_motion_viewer.py",
        "category": "UI / Orchestration",
        "stages": ["Settings", "Control", "UI", "Logging"],
        "summary": "실시간 데모를 실제로 구동하는 메인 앱이다. 장비 연결, 워커 시작, PyQt 창 구성, 최신 프레임 렌더, SessionLogger 연계를 한 곳에서 조율한다.",
        "why": "이 프로젝트를 실행하면 가장 먼저 통과하는 진입점이다. 지금도 가장 많은 책임이 모여 있지만, 최근에는 SessionLogger와 stage timing 연계까지 포함해 운영 허브 역할이 더 분명해졌다.",
        "mentor": "초보자가 볼 때는 화면 파일 같지만 실제로는 앱 컨트롤러에 가깝다. 장비 제어, 워커 연결, 최신 프레임 선택, render 로그 기록, 종료 시 HTML 리포트 트리거까지 모두 이 파일에서 시작한다.",
        "before": "runtime_settings.py, radar_config.py",
        "after": "real_time_process.py, app_layout.py",
        "inputs": "런타임 설정, DCA 제어 명령, 처리 큐의 FramePacket, UI 이벤트",
        "outputs": "렌더링된 화면, render_frames.jsonl, event_log.jsonl, runtime_config.json, 세션 종료 후 HTML 리포트",
        "debug": "최신 프레임만 소비하는 구조라 render 큐 스킵, OpenGL fallback, stage_timings_ms 전달 여부를 먼저 본다.",
        "ops": "현업에서는 AppController, SessionLogger, ViewRenderer로 더 나누는 편이 안전하지만, 현재도 세션 기록 책임 일부는 SessionLogger로 빠져 이전보다 낫다.",
        "read_order": [
            "앱 시작 시 어떤 객체를 만들고 어떤 워커를 붙이는지 본다.",
            "__init__와 start_workers를 읽어 설정, tracker, 처리 스레드 wiring을 이해한다.",
            "log_render_snapshot을 읽어 render 기준 latency와 stage timing이 어떻게 남는지 본다.",
            "shutdown을 읽으면 세션 종료 후 report 생성 흐름이 보인다.",
        ],
        "roles": [
            ("앱 진입점", "PyQt 창과 장비 제어 흐름을 묶어 실제 데모를 시작한다."),
            ("실시간 조정자", "처리 스레드 결과를 UI가 소비할 수 있는 속도로 맞춘다."),
            ("렌더 계측 지점", "사용자가 실제로 본 프레임 기준 latency와 표시 결과를 남긴다."),
        ],
        "landmarks": [
            ("MotionViewer", r"^class MotionViewer:", "앱의 메인 컨트롤러 클래스다."),
            ("__init__", r"^\s*def __init__", "설정, SessionLogger, UI 상태를 준비한다."),
            ("log_event", r"^\s*def log_event", "이벤트 로그를 남긴다."),
            ("log_render_snapshot", r"^\s*def log_render_snapshot", "실제로 그린 프레임 기준 로그를 기록한다."),
            ("configure_dca1000", r"^\s*def configure_dca1000", "DCA1000 네트워크 설정을 적용한다."),
            ("start_workers", r"^\s*def start_workers", "UDP 수신/처리 워커를 시작한다."),
            ("pull_latest_processed_frame", r"^\s*def pull_latest_processed_frame", "렌더할 최신 FramePacket 하나를 고른다."),
            ("build_window", r"^\s*def build_window", "PyQt 창을 조립한다."),
            ("shutdown", r"^\s*def shutdown", "워커와 로그 핸들을 정리한다."),
        ],
        "related": ["real_time_process", "app_layout", "runtime_settings", "session_logging"],
    },
    {
        "slug": "real_time_process",
        "source": "tools/runtime_core/real_time_process.py",
        "title": "real_time_process.py",
        "category": "Capture / Processing",
        "stages": ["Capture", "DSP", "Detection", "Tracking", "Logging"],
        "summary": "UDP 패킷을 프레임으로 조립하고 DSP, detection, tracking까지 이어 붙이는 실시간 처리 파이프라인이다. 최근에는 stage timing과 shared FFT 기반 최적화가 함께 들어갔다.",
        "why": "invalid rate, latency, track 품질, stage별 병목이 모두 여기에 직접 연결된다.",
        "mentor": "프로젝트의 심장이라고 보면 된다. 입력은 raw packet이고 출력은 FramePacket이며, 그 안에 health, detection, tracking 결과와 stage timing이 함께 들어간다.",
        "before": "radar_runtime.py, detection.py, tracking.py",
        "after": "live_motion_viewer.py",
        "inputs": "DCA1000 UDP payload, runtime config, tracker 파라미터",
        "outputs": "FramePacket, processed_frames.jsonl, 처리 큐",
        "debug": "packet health, capture_to_process_ms, shared_fft2_ms, detect_ms, candidate와 confirmed track 사이 손실을 함께 본다.",
        "ops": "assembly, DSP, detection, tracking이 아직 한 hot path에 직렬로 묶여 있어서, 계측을 본 뒤 단계별 경량화를 계속하는 것이 맞다.",
        "read_order": [
            "FramePacket 구조를 먼저 읽고 어떤 데이터가 흘러가는지 본다.",
            "UdpListener가 패킷을 어떻게 모으는지 본다.",
            "_round_stage_timings와 log_processed_frame을 읽어 processed 로그 schema를 본다.",
            "DataProcessor.run을 읽으면 shared FFT 이후 detection, tracking까지 한 프레임 처리 흐름이 보인다.",
        ],
        "roles": [
            ("프레임 계약", "한 프레임의 입력, 중간 결과, 추적 결과를 한 구조로 묶는다."),
            ("실시간 처리", "RDI, RAI, detection, tracking을 한 루프에서 계산한다."),
            ("처리 계측", "stage_timings_ms와 packet health를 함께 남겨 병목을 추적한다."),
        ],
        "landmarks": [
            ("FramePacket", r"^class FramePacket:", "프레임 단위 데이터 컨테이너다."),
            ("_round_stage_timings", r"^def _round_stage_timings", "stage timing dict를 로그용으로 정리한다."),
            ("_serialize_detection", r"^def _serialize_detection", "detection 결과를 로그용 dict로 바꾼다."),
            ("UdpListener", r"^class UdpListener", "UDP payload를 수집하는 스레드다."),
            ("DataProcessor", r"^class DataProcessor", "DSP부터 tracker까지 잇는 처리 스레드다."),
            ("log_processed_frame", r"^\s*def log_processed_frame", "모든 처리 프레임을 JSONL로 남긴다."),
            ("run", r"^\s*def run", "실시간 처리 루프 본체다."),
        ],
        "related": ["radar_runtime", "detection", "tracking", "live_motion_viewer"],
    },
    {
        "slug": "radar_runtime",
        "source": "tools/runtime_core/radar_runtime.py",
        "title": "radar_runtime.py",
        "category": "DSP / Geometry",
        "stages": ["Settings", "DSP", "Detection"],
        "summary": "raw IQ를 radar cube와 시각화용 맵으로 바꾸는 수학 계층이다. 축 정의, clutter 제거, ROI 적용, 벡터화된 shape 변환을 담당한다.",
        "why": "여기서 shape와 axis가 틀리면 뒤 단계가 모두 어긋나고, 여기서 copy가 많으면 DSP 전체가 느려진다.",
        "mentor": "이 파일은 파이프라인의 공통 언어를 만든다. detection과 tracking은 이 파일이 만든 좌표계와 배열 구조를 믿고 움직이며, 최근에는 reshape와 motion collapse도 더 가볍게 정리됐다.",
        "before": "runtime_settings.py, DSP.py",
        "after": "detection.py, real_time_process.py",
        "inputs": "runtime config, raw complex IQ frame",
        "outputs": "radar cube, integrated RDI, motion RAI, ROI mask 정보",
        "debug": "shape mismatch, axis 길이, dtype, clutter 제거 전후 에너지 분포를 확인한다.",
        "ops": "Capon 같은 angle estimator를 넣을 때도 이 파일의 axis model 이해가 먼저고, 실시간성 개선은 보통 여기 reshape/collapse 비용부터 줄인다.",
        "read_order": [
            "RadarRuntimeConfig로 축과 해상도 정의를 먼저 본다.",
            "frame_to_radar_cube와 remove_static_clutter로 raw 데이터를 처리용 cube로 바꾸는 순서를 본다.",
            "integrate_rdi_channels와 collapse_motion_rai로 detection 입력 맵을 어떻게 줄이는지 본다.",
            "apply_cartesian_roi_to_rai로 관심 영역만 남기는 과정을 본다.",
        ],
        "roles": [
            ("형태 변환", "수신 버퍼를 처리용 cube 구조로 바꾼다."),
            ("신호 정제", "static clutter 제거와 적분으로 움직임을 강조한다."),
            ("공간 제한", "ROI를 적용해 관심 영역만 detection으로 넘긴다."),
        ],
        "landmarks": [
            ("RadarRuntimeConfig", r"^class RadarRuntimeConfig:", "런타임 해상도와 축 계산을 담는다."),
            ("parse_runtime_config", r"^def parse_runtime_config", "설정 dict를 runtime config 객체로 변환한다."),
            ("frame_to_radar_cube", r"^def frame_to_radar_cube", "raw IQ를 처리용 cube로 바꾼다."),
            ("remove_static_clutter", r"^def remove_static_clutter", "정지 clutter를 제거한다."),
            ("integrate_rdi_channels", r"^def integrate_rdi_channels", "채널 통합 RDI를 만든다."),
            ("collapse_motion_rai", r"^def collapse_motion_rai", "motion angle map을 만든다."),
            ("apply_cartesian_roi_to_rai", r"^def apply_cartesian_roi_to_rai", "Cartesian ROI를 적용한다."),
        ],
        "related": ["DSP", "runtime_settings", "detection", "real_time_process"],
    },
    {
        "slug": "detection",
        "source": "tools/runtime_core/detection.py",
        "title": "detection.py",
        "category": "Detection",
        "stages": ["Detection"],
        "summary": "RDI와 RAI에서 실제 타깃 후보를 고르는 검출 계층이다. CFAR, angle ROI, clustering 결과가 여기서 합쳐진다.",
        "why": "tracker가 아무리 좋아도 detection이 흔들리면 전체 성능이 무너진다.",
        "mentor": "화면의 점이 왜 여기 찍혔는지 알고 싶을 때 가장 먼저 봐야 하는 파일이다. 후보를 만들고 점수를 주고 최종 detection candidate로 정리한다.",
        "before": "radar_runtime.py, dbscan_cluster.py",
        "after": "tracking.py, live_motion_viewer.py",
        "inputs": "RDI magnitude, RAI magnitude, ROI 설정, clustering 결과",
        "outputs": "DetectionCandidate 리스트",
        "debug": "CFAR threshold, angle ROI, cluster 기준이 과도하지 않은지 본다.",
        "ops": "Capon을 붙일 경우 가장 자연스러운 주입 지점은 detect_targets 내부의 angle refinement 단계다.",
        "read_order": [
            "DetectionRegion과 DetectionCandidate를 먼저 본다.",
            "cfar_threshold_2d로 픽셀 수준 gating을 이해한다.",
            "detect_targets에서 최종 후보 생성 흐름을 확인한다.",
        ],
        "roles": [
            ("후보 생성", "에너지 기반 후보를 range-doppler 공간에서 찾는다."),
            ("공간 제약", "angle ROI와 score 조건으로 후보를 정제한다."),
            ("tracker 입력 정리", "tracker가 쓰기 쉬운 후보 리스트를 만든다."),
        ],
        "landmarks": [
            ("DetectionRegion", r"^class DetectionRegion", "검출 영역 구조체다."),
            ("DetectionCandidate", r"^class DetectionCandidate", "tracker로 넘어가는 후보 데이터다."),
            ("cfar_threshold_2d", r"^def cfar_threshold_2d", "2D CFAR 임계값 계산 함수다."),
            ("_angle_roi_mask", r"^def _angle_roi_mask", "허용 각도 영역 마스크를 만든다."),
            ("_cluster_detection_candidates", r"^def _cluster_detection_candidates", "클러스터 결과를 detection 후보로 묶는다."),
            ("detect_targets", r"^def detect_targets", "검출 메인 함수다."),
        ],
        "related": ["radar_runtime", "dbscan_cluster", "tracking", "real_time_process"],
    },
    {
        "slug": "dbscan_cluster",
        "source": "tools/runtime_core/dbscan_cluster.py",
        "title": "dbscan_cluster.py",
        "category": "Clustering",
        "stages": ["Detection"],
        "summary": "밀집된 점들을 사람 단위 후보로 묶기 위한 DBSCAN 기반 후처리다. adaptive eps band가 핵심이다.",
        "why": "한 사람에서 여러 포인트가 나오더라도 detection 수가 불필요하게 폭증하지 않게 만든다.",
        "mentor": "detection이 점을 찾는 단계라면 clustering은 점 무리를 해석하는 단계다. 멀고 가까운 거리에서 포인트 분포가 달라 adaptive eps가 중요하다.",
        "before": "detection.py",
        "after": "detection.py, tracking.py",
        "inputs": "point cloud 후보, distance band 설정, DBSCAN 파라미터",
        "outputs": "cluster labels, cluster summary",
        "debug": "eps가 과도하면 사람이 쪼개지고 느슨하면 두 사람이 합쳐진다.",
        "ops": "multi-target 장면에서는 이 파일 튜닝만으로도 큰 개선이 나올 수 있다.",
        "read_order": [
            "adaptive eps 설정과 normalize_adaptive_eps_bands를 먼저 본다.",
            "cluster_points가 어떤 입력을 받아 어떤 라벨을 돌려주는지 확인한다.",
            "detection.py에서 이 결과를 어떻게 해석하는지 이어서 본다.",
        ],
        "roles": [
            ("밀집도 기반 묶기", "가까운 점들을 한 군집으로 묶어 사람 후보를 정리한다."),
            ("거리 적응", "거리별 포인트 분포 차이를 감안해 eps를 조절한다."),
            ("검출 안정화", "불필요한 detection 폭증을 줄인다."),
        ],
        "landmarks": [
            ("normalize_adaptive_eps_bands", r"^def normalize_adaptive_eps_bands", "adaptive eps 설정을 정규화한다."),
            ("cluster_points", r"^def cluster_points", "클러스터링 메인 함수다."),
        ],
        "related": ["detection", "tracking", "real_time_process"],
    },
    {
        "slug": "tracking",
        "source": "tools/runtime_core/tracking.py",
        "title": "tracking.py",
        "category": "Tracking",
        "stages": ["Tracking"],
        "summary": "프레임 간 detection을 이어 붙여 일관된 track ID를 유지하는 다중 타깃 tracker다.",
        "why": "사용자 입장에서 중요한 것은 같은 사람이 같은 ID로 유지되는가이다.",
        "mentor": "이 파일은 단순히 점을 따라가는 것이 아니라, 어떤 가설을 유지하고 버리고 확정할지 결정하는 정책 묶음이다.",
        "before": "detection.py",
        "after": "live_motion_viewer.py, session_report.py",
        "inputs": "DetectionCandidate 리스트, gating과 aging 파라미터",
        "outputs": "confirmed track, tentative track, lifecycle 상태",
        "debug": "track birth가 늦지 않은지, ID switch가 잦지 않은지, association이 깨지지 않는지 본다.",
        "ops": "로그가 좋아진 지금은 이 파일의 파라미터 변화 영향을 수치로 보기 쉬워졌다.",
        "read_order": [
            "TrackState와 TrackEstimate로 상태 구조를 본다.",
            "MultiTargetTracker가 내부 리스트를 어떻게 유지하는지 확인한다.",
            "_associate와 update를 읽어 실제 추적 정책을 이해한다.",
        ],
        "roles": [
            ("상태 유지", "각 detection이 시간축에서 어떤 사람인지 연결한다."),
            ("확정 정책", "tentative를 confirmed로 올릴지, 오래된 track을 지울지 결정한다."),
            ("다중 타깃 정합", "가까운 후보가 여러 개일 때 어떤 detection을 어떤 track에 잇는지 계산한다."),
        ],
        "landmarks": [
            ("TrackState", r"^class TrackState", "트랙 상태 열거형이다."),
            ("TrackEstimate", r"^class TrackEstimate", "개별 track 추정치 구조다."),
            ("MultiTargetTracker", r"^class MultiTargetTracker", "tracker 메인 클래스다."),
            ("_associate", r"^\s*def _associate", "detection과 track 매칭 단계다."),
            ("update", r"^\s*def update", "프레임 단위 tracker 갱신 함수다."),
        ],
        "related": ["detection", "real_time_process", "session_report"],
    },
    {
        "slug": "runtime_settings",
        "source": "tools/runtime_core/runtime_settings.py",
        "title": "runtime_settings.py",
        "category": "Configuration",
        "stages": ["Settings", "Logging"],
        "summary": "프로젝트 전반에서 공통으로 쓰는 런타임 설정을 로드하고 병합하는 설정 계층이다.",
        "why": "환경이 달라도 코드 수정 없이 값을 바꾸게 해 주는 중심점이다.",
        "mentor": "실험을 반복할수록 설정 파일 품질이 중요해진다. ROI, tracker뿐 아니라 logging on/off, payload 포함 여부, system snapshot 수집 여부까지 설정에서 제어하게 된 점이 현업적으로 좋다.",
        "before": "외부 runtime_settings.json",
        "after": "live_motion_viewer.py, radar_runtime.py, real_time_process.py",
        "inputs": "기본값 dict, 사용자 JSON override",
        "outputs": "병합된 runtime settings, 프로젝트 기준 절대 경로",
        "debug": "설정 누락 시 기본값이 안전한지, logging 섹션이 비어 있어도 동작하는지 확인한다.",
        "ops": "logging 토글이 설정 파일로 빠지면 full logging과 minimal logging 재실험을 코드 수정 없이 돌릴 수 있어 운영 실험이 쉬워진다.",
        "read_order": [
            "DEFAULT_RUNTIME_SETTINGS를 훑어 전체 실험 파라미터 범위를 본다.",
            "logging 섹션을 확인해 payload, stage timing, system snapshot 토글이 어떻게 들어가는지 본다.",
            "load_runtime_settings와 resolve_project_path를 읽으면 실제 사용 흐름이 보인다.",
        ],
        "roles": [
            ("기본값 제공", "실행에 필요한 안전한 기본 설정을 제공한다."),
            ("덮어쓰기 병합", "JSON override를 깊게 병합해 실험 단위 조정을 쉽게 만든다."),
            ("경로 해석", "프로젝트 루트 기준 절대 경로를 계산한다."),
        ],
        "landmarks": [
            ("DEFAULT_RUNTIME_SETTINGS", r"^DEFAULT_RUNTIME_SETTINGS\s*=", "프로젝트 기본 설정 dict다."),
            ("logging section", r'"logging"\s*:', "variant와 source_capture뿐 아니라 payload, snapshot, stage timing 토글이 들어간다."),
            ("_deep_merge", r"^def _deep_merge", "중첩 설정 병합 함수다."),
            ("load_runtime_settings", r"^def load_runtime_settings", "설정 파일 로딩 진입점이다."),
            ("resolve_project_path", r"^def resolve_project_path", "프로젝트 기준 절대 경로를 계산한다."),
        ],
        "related": ["live_motion_viewer", "real_time_process", "radar_runtime"],
    },
    {
        "slug": "radar_config",
        "source": "tools/runtime_core/radar_config.py",
        "title": "radar_config.py",
        "category": "Hardware Control",
        "stages": ["Control"],
        "summary": "UART를 통해 레이더 보드에 설정을 전송하고 시작과 정지 명령을 내리는 하드웨어 제어 계층이다.",
        "why": "실시간 앱이 신호를 받으려면 먼저 보드가 올바른 cfg를 먹어야 한다.",
        "mentor": "신호 처리보다 아래쪽 계층이다. 실제 장비가 명령을 잘 받는지, 포트가 맞는지, start와 stop 순서가 맞는지 여기서 결정된다.",
        "before": "cfg 텍스트 파일, COM 포트 정보",
        "after": "live_motion_viewer.py",
        "inputs": "시리얼 포트, cfg 파일 경로, 명령 문자열",
        "outputs": "장비 설정 완료 상태, start와 stop side effect",
        "debug": "포트 권한, baudrate, cfg 전송 순서, 예외 처리 누락을 본다.",
        "ops": "print 대신 structured logging으로 옮기면 현장 장비 이슈 재현이 쉬워진다.",
        "read_order": [
            "SerialConfig로 포트 설정 구조를 본다.",
            "SendConfig가 cfg를 어떻게 보내는지 확인한다.",
            "StopRadar를 보면 종료 시 장비 정리 흐름을 이해할 수 있다.",
        ],
        "roles": [
            ("장비 연결", "레이더 보드와 시리얼 링크를 연다."),
            ("설정 전송", "cfg 파일 명령을 순차적으로 전송한다."),
            ("안전 종료", "테스트가 끝났을 때 레이더를 멈추고 포트를 정리한다."),
        ],
        "landmarks": [
            ("SerialConfig", r"^class SerialConfig", "시리얼 연결 설정 객체다."),
            ("SendConfig", r"^def SendConfig", "cfg 파일을 장비에 보낸다."),
            ("StopRadar", r"^def StopRadar", "레이더를 정지시킨다."),
        ],
        "related": ["live_motion_viewer", "legacy_xwr1843_app"],
    },
    {
        "slug": "DSP",
        "source": "tools/runtime_core/DSP.py",
        "title": "DSP.py",
        "category": "DSP",
        "stages": ["DSP"],
        "summary": "Range-Doppler, Range-Angle FFT를 계산하는 기초 DSP 함수 모음이다. 최근에는 공통 FFT를 한 번만 수행하는 shared path와 cached window가 추가됐다.",
        "why": "지금 프로젝트의 기본 angle estimator는 FFT 기반이라 이 결과가 detection 품질과 처리 지연에 직접 들어간다.",
        "mentor": "고전적인 신호 처리 레이어다. 후단 detection과 tracking이 아무리 좋아도 출발점인 스펙트럼 품질이 낮으면 전체가 흔들리고, 여기 중복 FFT가 있으면 latency가 바로 튄다.",
        "before": "raw radar cube",
        "after": "radar_runtime.py, detection.py",
        "inputs": "complex radar cube",
        "outputs": "RDI, RAI magnitude map",
        "debug": "FFT 축 순서, normalization, channel ordering, shared FFT 결과 재사용이 맞는지 반드시 확인해야 한다.",
        "ops": "Capon을 시험하더라도 baseline으로 이 FFT 구현은 계속 남겨 두는 편이 좋고, 실시간성은 shared FFT와 broadcasting에서 먼저 챙기는 게 맞다.",
        "read_order": [
            "_cached_range_doppler_window와 shared_range_doppler_fft를 읽어 공통 전처리와 FFT를 이해한다.",
            "range_doppler_from_fft와 range_angle_from_fft를 읽어 shared 결과가 어떻게 갈라지는지 본다.",
            "마지막에 Range_Doppler와 Range_Angle wrapper로 기존 호출 호환성을 확인한다.",
        ],
        "roles": [
            ("공통 FFT 생성", "range-doppler 축 FFT를 한 번만 계산해 후단이 공유하게 한다."),
            ("채널 정렬", "안테나 축을 활용해 각도 정보를 드러낸다."),
            ("비교 기준", "향후 고급 기법과 비교할 baseline 역할을 한다."),
        ],
        "landmarks": [
            ("_cached_range_doppler_window", r"^def _cached_range_doppler_window", "window를 캐시해 프레임마다 다시 만들지 않게 한다."),
            ("shared_range_doppler_fft", r"^def shared_range_doppler_fft", "공통 range-doppler FFT 계산 함수다."),
            ("range_doppler_from_fft", r"^def range_doppler_from_fft", "공통 FFT 결과를 RDI로 투영한다."),
            ("range_angle_from_fft", r"^def range_angle_from_fft", "공통 FFT 결과를 RAI로 투영한다."),
            ("Range_Doppler", r"^def Range_Doppler", "range-doppler FFT 계산 함수다."),
            ("Range_Angle", r"^def Range_Angle", "range-angle FFT 계산 함수다."),
        ],
        "related": ["radar_runtime", "detection", "read_binfile"],
    },
    {
        "slug": "app_layout",
        "source": "tools/runtime_core/app_layout.py",
        "title": "app_layout.py",
        "category": "UI Layout",
        "stages": ["UI"],
        "summary": "Qt Designer에서 생성된 UI 뼈대를 코드로 저장한 파일이다.",
        "why": "위젯 배치와 이름이 이 파일에서 결정되므로 live_motion_viewer가 어떤 컴포넌트를 붙잡는지 이해하는 데 필요하다.",
        "mentor": "직접 로직은 많지 않지만 창 구조를 이해할 때 중요한 지도 역할을 한다. 어느 패널이 어디 있는지 여기서 보인다.",
        "before": "Qt Designer UI 설계",
        "after": "live_motion_viewer.py",
        "inputs": "QMainWindow 인스턴스",
        "outputs": "위젯 트리와 이름",
        "debug": "위젯 objectName이 바뀌면 상위 코드가 깨질 수 있다.",
        "ops": "UI 구조는 안정적으로 두고 상위 컨트롤러에서만 로직을 바꾸는 편이 좋다.",
        "read_order": [
            "Ui_MainWindow를 보고 위젯 계층을 훑는다.",
            "setupUi에서 패널과 레이블이 무엇인지 확인한다.",
            "그 후 live_motion_viewer에서 어떤 위젯을 실제로 쓰는지 이어서 본다.",
        ],
        "roles": [
            ("UI 골격", "메인 창과 패널 배치를 정의한다."),
            ("식별자 제공", "상위 코드가 참조할 object name을 제공한다."),
            ("디자인 분리", "레이아웃과 동작 로직을 나누는 출발점 역할을 한다."),
        ],
        "landmarks": [
            ("Ui_MainWindow", r"^class Ui_MainWindow", "디자이너 생성 UI 클래스다."),
            ("setupUi", r"^\s*def setupUi", "실제 위젯을 생성하고 배치한다."),
            ("retranslateUi", r"^\s*def retranslateUi", "UI 텍스트를 세팅한다."),
        ],
        "related": ["live_motion_viewer"],
    },
    {
        "slug": "legacy_xwr1843_app",
        "source": "real-time/Real-time-plot-RAI_RDI_XWR1843_app.py",
        "title": "Real-time-plot-RAI_RDI_XWR1843_app.py",
        "category": "Legacy App",
        "stages": ["Legacy", "Control", "UI"],
        "summary": "예전 실시간 플롯 앱이다. 현재 live_motion_viewer 이전 세대 구조를 보여 주는 레거시 진입점이다.",
        "why": "프로젝트가 어떤 방향으로 진화했는지, 왜 새 구조가 필요한지 비교 기준을 준다.",
        "mentor": "완전히 버릴 파일은 아니지만 현재 메인 경로와 혼용하면 온보딩이 헷갈릴 수 있다. 비교 자료로 읽는 것이 좋다.",
        "before": "DSP.py, radar_config.py",
        "after": "현재는 주로 참고용",
        "inputs": "cfg, raw frame queue, matplotlib figure",
        "outputs": "실시간 RDI와 RAI 플롯",
        "debug": "현재 메인 앱과 섞어 쓰지 않도록 entrypoint를 분리하는 것이 중요하다.",
        "ops": "현업에서는 legacy 폴더로 분리하거나 README에 비권장 경로라고 명시하는 편이 좋다.",
        "read_order": [
            "send_cmd로 장비 제어 방식을 본다.",
            "update_figure와 plot으로 예전 렌더 구조를 파악한다.",
            "그 다음 live_motion_viewer와 비교해 구조 차이를 본다.",
        ],
        "roles": [
            ("역사적 기준점", "기존 데모가 어떤 방식으로 동작했는지 보여 준다."),
            ("단순 렌더 예시", "현재 앱보다 단순한 플롯 중심 구조를 제공한다."),
            ("비교 자료", "새 구조의 개선 포인트를 설명할 때 대비군이 된다."),
        ],
        "landmarks": [
            ("send_cmd", r"^def send_cmd", "시리얼 명령 전송 함수다."),
            ("update_figure", r"^def update_figure", "플롯 갱신 함수다."),
            ("plot", r"^def plot", "실행 루프 성격의 진입 함수다."),
        ],
        "related": ["live_motion_viewer", "radar_config", "DSP"],
    },
    {
        "slug": "session_report",
        "source": "tools/diagnostics/session_report.py",
        "title": "session_report.py",
        "category": "Reporting",
        "stages": ["Logging", "Reports"],
        "summary": "한 세션 폴더를 읽어 summary.json을 생성하는 리포트 빌더다. processed/render 집계뿐 아니라 system snapshot, stage timing, operational assessment까지 묶는다.",
        "why": "로그가 남는 것만으로 끝나지 않고, 사람이 바로 비교 가능한 숫자와 운영 점수로 정리해야 하기 때문이다.",
        "mentor": "processed 로그와 render 로그를 왜 나눴는지 가장 잘 보여 주는 파일이다. 이제는 알고리즘 성능과 UI 체감 성능뿐 아니라 환경 상태와 stage별 병목까지 같은 세션 안에서 묶어 본다.",
        "before": "session_meta.json, processed_frames.jsonl, render_frames.jsonl, system_snapshot.json",
        "after": "summary.json, session_compare.py",
        "inputs": "세션 디렉터리의 JSON과 JSONL 로그",
        "outputs": "summary.json",
        "debug": "legacy status_log만 있는 옛 세션도 읽히는지, stage_timings_ms와 system_snapshot이 빠져도 안전한지 확인한다.",
        "ops": "리포트 스키마를 고정해 두어야 장기 추세 비교가 편하고, 운영 점수와 환경 진단이 summary에 같이 들어가야 현장 판단이 빨라진다.",
        "read_order": [
            "_load_jsonl과 _summarize_numeric로 집계 기본기를 본다.",
            "_summarize_stage_timings와 _build_system_summary로 새 진단 축을 본다.",
            "build_summary에서 processed와 render와 assessment를 어떻게 합치는지 확인한다.",
            "main으로 실제 CLI 사용 흐름을 본다.",
        ],
        "roles": [
            ("세션 요약", "프레임 로그를 사람이 읽기 쉬운 summary.json으로 압축한다."),
            ("지표 표준화", "invalid rate, latency, track count 같은 지표를 일관된 구조로 만든다."),
            ("운영 진단", "system snapshot과 stage timing을 요약해 현업형 판단 재료를 만든다."),
        ],
        "landmarks": [
            ("_load_jsonl", r"^def _load_jsonl", "JSONL 로더다."),
            ("_summarize_numeric", r"^def _summarize_numeric", "평균, p50, p95를 계산한다."),
            ("_summarize_stage_timings", r"^def _summarize_stage_timings", "stage timing 분포를 요약한다."),
            ("_build_system_summary", r"^def _build_system_summary", "system_snapshot.json을 리포트용 구조로 정리한다."),
            ("build_summary", r"^def build_summary", "리포트 핵심 집계 함수다."),
            ("main", r"^def main", "CLI entrypoint다."),
        ],
        "related": ["session_compare", "operational_assessment", "log_html_reports", "real_time_process"],
    },
    {
        "slug": "session_compare",
        "source": "tools/diagnostics/session_compare.py",
        "title": "session_compare.py",
        "category": "Reporting",
        "stages": ["Reports"],
        "summary": "before와 after 두 summary.json을 읽어 개선인지 회귀인지 판단하는 비교기다. 최근에는 operational score도 기본 비교 항목에 들어간다.",
        "why": "실험이 늘어날수록 좋아진 느낌이 아니라 지표상 좋아졌는가를 자동으로 판단해야 하기 때문이다.",
        "mentor": "현업에서는 이런 비교기가 있어야 실험이 누적될수록 팀이 같은 언어로 의사결정을 할 수 있다.",
        "before": "session_report.py",
        "after": "comparison_vs_*.json, HTML 비교 리포트 원본 데이터",
        "inputs": "before와 after summary.json",
        "outputs": "comparison json, 콘솔 비교 결과",
        "debug": "metric direction, 0 division, None 값 처리, legacy session의 missing metric을 안전하게 다루는지 확인해야 한다.",
        "ops": "threshold 기반 fail와 pass를 붙이면 회귀 검증 자동화에 연결하기 좋고, operational score를 같이 보면 현업 설명이 쉬워진다.",
        "read_order": [
            "METRICS 목록으로 operational score와 latency가 어떤 순서로 비교되는지 먼저 본다.",
            "build_comparison에서 delta와 judgement 계산을 본다.",
            "main에서 CLI 출력 형식을 확인한다.",
        ],
        "roles": [
            ("A/B 비교", "baseline과 candidate 실험 결과를 같은 포맷으로 비교한다."),
            ("방향성 판단", "낮을수록 좋은 지표와 높을수록 좋은 지표를 구분한다."),
            ("회귀 감지", "개선과 회귀 여부를 빠르게 요약한다."),
        ],
        "landmarks": [
            ("METRICS", r"^METRICS\s*=", "비교 대상 지표 목록이다."),
            ("build_comparison", r"^def build_comparison", "비교 결과를 생성한다."),
            ("main", r"^def main", "CLI entrypoint다."),
        ],
        "related": ["session_report", "operational_assessment", "log_html_reports"],
    },
    {
        "slug": "session_logging",
        "source": "real-time/session_logging.py",
        "title": "session_logging.py",
        "category": "Logging / Orchestration",
        "stages": ["Logging", "Reports"],
        "summary": "세션 폴더 생성, 메타데이터 기록, event/render 로그 파일 핸들 관리, 종료 시 HTML 리포트 생성을 맡는 로깅 허브다.",
        "why": "live_motion_viewer.py에서 세션 준비와 종료 보고를 분리해 운영 책임 경계를 만든 핵심 파일이다.",
        "mentor": "예전에는 앱 파일이 로그 준비까지 거의 다 들고 있었다. 지금은 SessionLogger가 세션 폴더 구조와 system snapshot, report generation을 묶어 주는 전용 계층이 됐다.",
        "before": "live_motion_viewer.py, runtime_settings.py",
        "after": "session_meta.json, event_log.jsonl, render_frames.jsonl, summary.json, ops_report.html",
        "inputs": "세션 메타데이터, runtime summary, logging 토글",
        "outputs": "세션 디렉터리와 로그/리포트 파일",
        "debug": "enabled 토글, 파일 핸들 생명주기, system snapshot 수집 실패 시 degrade gracefully 되는지 확인한다.",
        "ops": "세션 경계가 파일 단위로 명확해지면 현장 장애 대응과 로그 보존 정책을 다루기 쉬워진다.",
        "read_order": [
            "__init__로 어떤 파일을 여는지 본다.",
            "build_session_metadata와 prepare를 읽어 세션 시작 시점 작업을 이해한다.",
            "close를 읽으면 summary와 HTML 리포트 생성까지 끝 흐름이 보인다.",
        ],
        "roles": [
            ("세션 경계 정의", "한 번 실행한 로그와 메타데이터를 한 폴더에 묶는다."),
            ("이벤트 로깅", "render/event/status 로그 파일 쓰기를 표준화한다."),
            ("종료 후 정리", "summary와 HTML 리포트를 자동 생성한다."),
        ],
        "landmarks": [
            ("SessionLogger", r"^class SessionLogger:", "세션 로깅 메인 클래스다."),
            ("build_session_metadata", r"^\s*def build_session_metadata", "git 정보와 logging 토글을 메타데이터로 정리한다."),
            ("prepare", r"^\s*def prepare", "세션 폴더, 로그 파일, system snapshot을 준비한다."),
            ("log_event", r"^\s*def log_event", "이벤트 로그를 남긴다."),
            ("write_render_record", r"^\s*def write_render_record", "render/status 로그 레코드를 기록한다."),
            ("close", r"^\s*def close", "로그 파일을 닫고 HTML 리포트를 생성한다."),
        ],
        "related": ["live_motion_viewer", "session_report", "log_html_reports", "system_snapshot"],
    },
    {
        "slug": "system_snapshot",
        "source": "tools/diagnostics/system_snapshot.py",
        "title": "system_snapshot.py",
        "category": "Diagnostics",
        "stages": ["Logging", "Reports"],
        "summary": "세션 시작 시 전원 계획, NIC/IP 상태, 방화벽 프로필, 프로세스 우선순위, NumPy/스레드 환경을 저장하는 환경 진단 유틸리티다.",
        "why": "코드 변화가 거의 없는데도 성능이 달라질 때, 실행 환경이 원인인지 확인할 수 있게 해 준다.",
        "mentor": "현업에서는 같은 코드라도 전원 모드, NIC 상태, 방화벽, CPU priority 때문에 결과가 흔들린다. 그걸 세션 단위로 남기는 역할이 이 파일이다.",
        "before": "SessionLogger.prepare",
        "after": "system_snapshot.json, session_report.py",
        "inputs": "expected host IP, 현재 프로세스 환경",
        "outputs": "system_snapshot.json 구조체",
        "debug": "PowerShell 호출 실패 시 None 처리, expected_host_ip 비교, priority class 해석이 안전한지 본다.",
        "ops": "실험 환경을 로그와 함께 묶어 두면 '왜 오늘만 느린가' 같은 질문에 훨씬 빨리 답할 수 있다.",
        "read_order": [
            "_run_command와 _run_powershell_json으로 외부 정보 수집 방식을 본다.",
            "_parse_power_scheme와 _build_windows_runtime_snapshot으로 핵심 필드를 이해한다.",
            "capture_system_snapshot으로 최종 JSON schema를 확인한다.",
        ],
        "roles": [
            ("환경 수집", "전원 계획, NIC, 방화벽, 프로세스 priority를 읽는다."),
            ("운영 진단", "host_ip_present 같은 현업형 점검 항목을 계산한다."),
            ("리포트 원본 제공", "session_report와 ops_report가 쓰는 환경 스냅샷을 만든다."),
        ],
        "landmarks": [
            ("_run_powershell_json", r"^def _run_powershell_json", "PowerShell JSON 결과를 읽는다."),
            ("_parse_power_scheme", r"^def _parse_power_scheme", "활성 전원 계획을 해석한다."),
            ("_build_windows_runtime_snapshot", r"^def _build_windows_runtime_snapshot", "Windows 네트워크와 방화벽 상태를 수집한다."),
            ("capture_system_snapshot", r"^def capture_system_snapshot", "최종 environment snapshot을 만든다."),
        ],
        "related": ["session_logging", "session_report", "log_html_reports"],
    },
    {
        "slug": "operational_assessment",
        "source": "tools/diagnostics/operational_assessment.py",
        "title": "operational_assessment.py",
        "category": "Diagnostics",
        "stages": ["Reports"],
        "summary": "세션 summary와 event 요약을 받아 현업형 점수, 등급, 강점/문제/권고를 계산하는 평가 엔진이다.",
        "why": "평균 latency 하나만으로는 운영 수준을 설명하기 어렵기 때문에, 무결성, 가시성, 준비도를 함께 점수화한다.",
        "mentor": "이 파일 덕분에 '좋아 보인다'가 아니라 '몇 점짜리 시스템인가'를 같은 기준으로 말할 수 있다.",
        "before": "session_report.py",
        "after": "summary.json.assessment, ops_report.html",
        "inputs": "summary dict, event summary",
        "outputs": "assessment dict, strengths, issues, recommendations",
        "debug": "missing metric이 많을 때도 점수가 과도하게 깨지지 않는지, threshold 방향이 맞는지 확인한다.",
        "ops": "팀 내부 루브릭을 코드로 묶으면 세션 비교와 발표 자료가 훨씬 일관돼진다.",
        "read_order": [
            "GRADE_BANDS로 등급 체계를 본다.",
            "build_event_summary로 세션 수명주기 요약을 이해한다.",
            "build_operational_assessment에서 점수 합산과 권고 생성을 읽는다.",
        ],
        "roles": [
            ("점수화", "latency, integrity, visibility, readiness를 한 점수로 합친다."),
            ("등급 부여", "A~F와 현업 해석 라벨을 붙인다."),
            ("설명 생성", "strengths, issues, recommendations를 문장으로 만든다."),
        ],
        "landmarks": [
            ("GRADE_BANDS", r"^GRADE_BANDS\s*=", "등급 컷오프 테이블이다."),
            ("build_event_summary", r"^def build_event_summary", "event 로그를 운영 관점 요약으로 바꾼다."),
            ("build_operational_assessment", r"^def build_operational_assessment", "점수와 권고를 계산한다."),
        ],
        "related": ["session_report", "session_compare", "log_html_reports"],
    },
    {
        "slug": "log_html_reports",
        "source": "tools/diagnostics/log_html_reports.py",
        "title": "log_html_reports.py",
        "category": "Reporting",
        "stages": ["Reports"],
        "summary": "summary.json과 comparison 결과를 읽어 세션 index, ops_report, 루트 대시보드까지 HTML로 생성하는 리포트 렌더러다.",
        "why": "JSON만 있으면 숫자는 남지만, 회의나 현장 설명에서는 한눈에 보이는 HTML 리포트가 훨씬 빠르다.",
        "mentor": "이 파일은 숫자를 전달 가능한 화면으로 바꾸는 마지막 계층이다. stage timing, system snapshot, operational score를 실제 문서로 보여 주는 역할을 한다.",
        "before": "session_report.py, session_compare.py, operational_assessment.py",
        "after": "index.html, ops_report.html, 비교용 루트 dashboard",
        "inputs": "summary.json, comparison json, event summary",
        "outputs": "세션별/루트 HTML 리포트",
        "debug": "legacy session처럼 일부 로그가 비어 있어도 HTML이 깨지지 않는지, 링크가 올바른지 확인한다.",
        "ops": "현업에서는 팀이 동일한 리포트를 보고 이야기할 수 있어야 하므로, HTML 레이어 품질도 꽤 중요하다.",
        "read_order": [
            "COMMON_STYLE과 COMMON_SCRIPT로 리포트 공통 뼈대를 본다.",
            "_build_session_html과 _build_ops_html에서 세션 페이지 구성을 읽는다.",
            "generate_reports로 실제 파일 생성 흐름을 확인한다.",
        ],
        "roles": [
            ("세션 대시보드 생성", "한 세션의 핵심 지표를 index.html로 만든다."),
            ("운영 평가 시각화", "ops_report.html로 점수와 권고를 보여 준다."),
            ("루트 비교 허브", "여러 세션을 한눈에 비교하는 상위 index를 만든다."),
        ],
        "landmarks": [
            ("_collect_session_rows", r"^def _collect_session_rows", "루트 대시보드용 세션 목록을 모은다."),
            ("_build_ops_html", r"^def _build_ops_html", "현업 평가 리포트를 만든다."),
            ("generate_reports", r"^def generate_reports", "세션/루트 HTML 생성 진입점이다."),
            ("main", r"^def main", "CLI entrypoint다."),
        ],
        "related": ["session_report", "session_compare", "operational_assessment", "session_logging"],
    },
    {
        "slug": "read_binfile",
        "source": "tools/diagnostics/read_binfile.py",
        "title": "read_binfile.py",
        "category": "Offline Utility",
        "stages": ["DSP", "Reports"],
        "summary": "mmWave Studio로 저장한 바이너리 파일을 complex radar cube 형태로 읽는 오프라인 유틸리티다.",
        "why": "replay와 offline benchmark, 실험 재현성 확보를 위해 꼭 필요한 입력 계층이다.",
        "mentor": "지금은 live path가 중심이지만 진짜 전후 비교를 하려면 결국 같은 raw 입력을 다시 돌려야 한다. 그 출발점이 이 파일이다.",
        "before": "raw .bin capture",
        "after": "DSP.py, future replay pipeline",
        "inputs": ".bin 파일, config 배열, mode, header 여부",
        "outputs": "frame x chirp x sample x channel complex cube",
        "debug": "mode와 tx와 rx 수가 틀리면 shape가 완전히 어긋난다.",
        "ops": "향후 replay benchmark를 만들 때 가장 먼저 재사용할 파일이다.",
        "read_order": [
            "remove_header로 raw capture 전처리 방식을 본다.",
            "read_bin_file에서 mode별 shape 변환을 확인한다.",
            "그 다음 DSP.py와 연결해서 offline replay 경로를 상상하면 이해가 쉽다.",
        ],
        "roles": [
            ("오프라인 입력 복원", "binary capture를 복소수 cube로 복원한다."),
            ("장비 모드 분기", "XWR 계열별 데이터 구조 차이를 처리한다."),
            ("재현 실험 기반", "같은 입력으로 알고리즘 전후 비교를 가능하게 만든다."),
        ],
        "landmarks": [
            ("remove_header", r"^def remove_header", "Studio header를 제거한다."),
            ("read_bin_file", r"^def read_bin_file", "bin 파일을 complex cube로 변환한다."),
        ],
        "related": ["DSP", "session_report", "session_compare"],
    },
]

for item in FILES:
    item.update(FILE_EXPLAINERS.get(item["slug"], {}))

FILE_LOOKUP = {item["slug"]: item for item in FILES}

FLOW_DETAILS = {
    "runtime_settings": [
        ("build_default_settings", "static, runtime, tuning 기본값을 하나의 settings dict로 합친다.", "기본 settings 구조", "load_runtime_settings"),
        ("load_runtime_settings", "static -> runtime -> tuning 순서로 세 개의 설정 파일을 읽고 병합한다.", "merged settings dict", "live_motion_viewer.py / real_time_process.py"),
        ("build_settings_snapshot", "최종 settings에서 static, runtime, tuning 묶음을 다시 떼어 낸다.", "settings snapshot", "session logging"),
        ("resolve_project_path", "상대 경로를 프로젝트 루트 기준 절대 경로로 바꾼다.", "config/log 경로", "live_motion_viewer.py"),
    ],
    "radar_config": [
        ("SerialConfig", "레이다 보드와 통신할 시리얼 포트 조건을 준비한다.", "시리얼 연결 정보", "SendConfig"),
        ("SendConfig", "cfg 파일 명령을 순서대로 레이다 보드에 전송한다.", "장비가 적용한 센서 설정", "live_motion_viewer.py"),
        ("StopRadar", "실험 종료 시 레이다 구동을 멈추고 포트를 정리한다.", "정지된 장비 상태", "shutdown 흐름"),
    ],
    "live_motion_viewer": [
        ("MotionViewer.__init__", "static/runtime/tuning 설정을 읽고 detection region, spatial view, SessionLogger를 묶는다.", "앱 초기 상태와 세션 로그 경로", "start_workers / build_window"),
        ("start_workers", "runtime settings와 tuning 파라미터를 바탕으로 UdpListener, DataProcessor, MultiTargetTracker를 만든다.", "실시간 처리 큐", "real_time_process.py"),
        ("configure_dca1000", "DCA1000 패킷 크기와 지연 설정을 하드웨어에 적용한다.", "DCA 설정 완료 상태", "open_radar"),
        ("pull_latest_processed_frame", "큐에 쌓인 FramePacket 중 최신 프레임 하나를 꺼낸다.", "렌더 대상 FramePacket", "overlay 업데이트 / log_render_snapshot"),
        ("log_render_snapshot", "실제로 화면에 표시한 프레임 기준 정보와 stage timing을 JSONL로 남긴다.", "render_frames.jsonl", "session_report.py"),
    ],
    "real_time_process": [
        ("UdpListener.run", "UDP packet stream을 받아 패킷 순서와 누락 상태를 포함해 프레임 재료를 모은다.", "packet queue", "DataProcessor.run"),
        ("DataProcessor.run", "packet queue를 소비해 raw IQ -> radar cube -> shared FFT -> RDI/RAI -> detections -> tracks로 처리한다.", "FramePacket + stage_timings_ms", "live_motion_viewer.py"),
        ("log_processed_frame", "처리 완료 직후 FramePacket 핵심 값과 stage timing을 JSONL로 저장한다.", "processed_frames.jsonl", "session_report.py"),
    ],
    "read_binfile": [
        ("remove_header", "mmWave Studio 캡처에 붙은 UDP header를 제거한다.", "헤더가 제거된 int16 stream", "read_bin_file"),
        ("read_bin_file", "raw int16 데이터를 complex IQ cube로 reshape하고 안테나 축을 정렬한다.", "frame x chirp x sample x channel cube", "DSP.py / future replay path"),
    ],
    "DSP": [
        ("shared_range_doppler_fft", "window를 적용하고 range/doppler FFT를 한 번만 계산한다.", "shared FFT cube", "range_doppler_from_fft / range_angle_from_fft"),
        ("range_doppler_from_fft", "공통 FFT 결과를 RDI magnitude map으로 투영한다.", "RDI magnitude map", "radar_runtime.py / detection.py"),
        ("range_angle_from_fft", "공통 FFT 결과를 angle 축으로 다시 변환해 RAI를 만든다.", "RAI magnitude map", "radar_runtime.py / detection.py"),
    ],
    "radar_runtime": [
        ("parse_runtime_config", "settings dict를 축 길이와 ROI 규칙을 포함한 runtime config 객체로 바꾼다.", "RadarRuntimeConfig", "frame_to_radar_cube"),
        ("frame_to_radar_cube", "한 프레임 raw IQ를 range/doppler/antenna 계산이 가능한 배열 구조로 바꾼다.", "radar cube", "integrate_rdi_channels / collapse_motion_rai"),
        ("integrate_rdi_channels", "채널 정보를 모아 detection에 쓸 range-doppler 에너지를 만든다.", "integrated RDI", "detection.py"),
        ("collapse_motion_rai", "움직임 기반 angle 정보를 압축해 detection에 유리한 angle map을 만든다.", "motion RAI", "detection.py"),
        ("apply_cartesian_roi_to_rai", "관심 공간 밖의 angle map을 제거해 불필요한 후보를 줄인다.", "ROI 적용된 RAI", "detection.py"),
    ],
    "dbscan_cluster": [
        ("normalize_adaptive_eps_bands", "거리별 eps 설정을 정규화해 clustering 기준을 준비한다.", "adaptive eps bands", "cluster_points"),
        ("cluster_points", "근접한 point들을 사람 단위 cluster로 묶는다.", "cluster labels / cluster summary", "detection.py"),
    ],
    "detection": [
        ("cfar_threshold_2d", "RDI에서 배경 대비 유의미한 peak 후보를 찾기 위한 threshold를 만든다.", "thresholded peak 후보", "detect_targets"),
        ("_angle_roi_mask", "허용 각도 범위를 마스크로 만들어 detection 범위를 제한한다.", "angle ROI mask", "detect_targets"),
        ("_cluster_detection_candidates", "cluster 결과를 detection candidate 구조로 변환한다.", "candidate 묶음", "detect_targets"),
        ("detect_targets", "RDI, RAI, ROI, clustering 결과를 합쳐 최종 DetectionCandidate 리스트를 만든다.", "DetectionCandidate list", "tracking.py / live_motion_viewer.py"),
    ],
    "tracking": [
        ("_associate", "현재 detections와 기존 tracks 사이 최적 매칭을 계산한다.", "association result", "update"),
        ("update", "association 결과를 반영해 tentative/confirmed/dead 상태를 갱신한다.", "TrackEstimate list", "live_motion_viewer.py / real_time_process.py"),
    ],
    "app_layout": [
        ("setupUi", "메인 윈도우와 각 패널, 위젯 object name을 생성한다.", "Qt widget tree", "live_motion_viewer.py"),
        ("retranslateUi", "레이블과 버튼 텍스트를 채운다.", "초기화된 UI 텍스트", "사용자 화면"),
    ],
    "session_report": [
        ("_load_jsonl", "processed/render/event 로그를 JSON record 배열로 읽는다.", "record list", "build_summary"),
        ("_summarize_numeric", "latency와 count 값에서 mean, p50, p95를 계산한다.", "통계 dict", "build_summary"),
        ("_summarize_stage_timings", "stage timing dict를 stage별 분포로 요약한다.", "stage timing summary", "build_summary"),
        ("build_summary", "processed와 render와 system snapshot을 함께 읽어 세션 요약과 운영 점수를 만든다.", "summary.json 내용", "session_compare.py"),
        ("main", "CLI에서 세션 폴더를 받아 summary.json을 파일로 쓴다.", "summary.json", "실험 비교 단계"),
    ],
    "session_compare": [
        ("_load_summary", "before와 after summary.json을 로드한다.", "두 세션의 summary dict", "build_comparison"),
        ("build_comparison", "핵심 지표와 operational score delta, improved/regressed 판단을 계산한다.", "comparison dict", "main"),
        ("main", "비교 결과를 comparison_vs_*.json으로 저장하고 콘솔에 요약 출력한다.", "comparison json", "실험 회귀 판단"),
    ],
    "session_logging": [
        ("SessionLogger.build_session_metadata", "git 정보와 logging 토글을 세션 메타데이터로 묶는다.", "session metadata", "prepare"),
        ("SessionLogger.prepare", "세션 폴더, runtime_config, system_snapshot, 로그 파일 핸들을 준비한다.", "session directory + file handles", "live_motion_viewer.py"),
        ("SessionLogger.close", "로그 파일을 닫고 derived summary와 HTML 리포트를 생성한다.", "summary.json + index.html + ops_report.html", "사용자 검토"),
    ],
    "system_snapshot": [
        ("_build_windows_runtime_snapshot", "Windows NIC, 방화벽, IP, priority 관련 정보를 수집한다.", "network/process snapshot", "capture_system_snapshot"),
        ("capture_system_snapshot", "전원 계획과 Python 환경까지 합쳐 system_snapshot.json 구조를 만든다.", "system snapshot dict", "session_logging.py / session_report.py"),
    ],
    "log_html_reports": [
        ("generate_reports", "summary.json과 비교 결과를 읽어 세션 HTML과 루트 dashboard를 다시 만든다.", "index.html / ops_report.html", "사용자 검토"),
        ("_collect_session_rows", "루트 dashboard에 들어갈 세션 목록과 점수를 모은다.", "dashboard rows", "index.html"),
    ],
    "legacy_xwr1843_app": [
        ("send_cmd", "예전 구조에서 시리얼 명령을 장비에 보내는 진입점이다.", "장비 설정 상태", "plot"),
        ("update_figure", "구형 matplotlib 기반 플롯을 한 프레임씩 갱신한다.", "업데이트된 figure", "사용자 화면"),
        ("plot", "데이터 수신과 렌더 루프를 묶어 예전 데모를 구동한다.", "RDI/RAI 플롯", "legacy path"),
    ],
}

DATA_CONTRACTS = [
    {
        "name": "Raw UDP Packet",
        "kind": "Transport Unit",
        "produced_by": "DCA1000 hardware / UdpListener",
        "consumed_by": "real_time_process.py",
        "why": "실시간 파이프라인의 가장 아래 입력이다. packet sequence와 byte count가 흔들리면 그 위 모든 품질이 무너진다.",
        "fields": [
            "sequence_id: DCA1000 헤더 기준 패킷 순서",
            "byte_count: 누적 바이트 카운트",
            "payload: IQ 데이터 조각",
            "packet health: gap / mismatch / out-of-sequence 여부",
        ],
    },
    {
        "name": "RadarRuntimeConfig",
        "kind": "Runtime Config",
        "produced_by": "radar_runtime.parse_runtime_config",
        "consumed_by": "real_time_process.py / radar_runtime.py",
        "why": "ADC 샘플 수, chirp loop, FFT 크기, axis 해상도를 정의하는 공통 기준이다.",
        "fields": [
            "adc_sample, chirp_loops, tx_num, rx_num",
            "range_fft_size, doppler_fft_size, angle_fft_size",
            "frame_length, virtual_antennas",
            "range_axis_m, angle_axis_rad",
        ],
    },
    {
        "name": "FramePacket",
        "kind": "Frame Contract",
        "produced_by": "DataProcessor.run",
        "consumed_by": "live_motion_viewer.py / session_report.py",
        "why": "한 프레임의 raw IQ, packet health, detection, tracking 결과를 한 컨테이너로 묶는 핵심 계약이다.",
        "fields": [
            "frame_id, capture_ts, assembled_ts, processed_ts",
            "iq, rdi, rai",
            "udp_gap_count, byte_mismatch_count, invalid, invalid_reason",
            "detections, confirmed_tracks, tentative_tracks",
            "stage_timings_ms",
        ],
    },
    {
        "name": "DetectionCandidate",
        "kind": "Detection Result",
        "produced_by": "detection.detect_targets",
        "consumed_by": "tracking.update / logging",
        "why": "검출 단계가 tracker에 넘기는 최소 단위다. 위치와 peak 세기와 score가 함께 있다.",
        "fields": [
            "range_bin, doppler_bin, angle_bin",
            "range_m, angle_deg, x_m, y_m",
            "rdi_peak, rai_peak",
            "score",
        ],
    },
    {
        "name": "TrackEstimate",
        "kind": "Tracking Result",
        "produced_by": "tracking.update",
        "consumed_by": "live_motion_viewer.py / processed/render logs",
        "why": "사용자가 실제로 보게 되는 사람 단위 결과다. ID, 위치, 속도, confidence가 포함된다.",
        "fields": [
            "track_id",
            "x_m, y_m, vx_m_s, vy_m_s",
            "range_m, angle_deg, doppler_bin",
            "confidence, age, hits, misses",
        ],
    },
    {
        "name": "summary.json",
        "kind": "Session Report",
        "produced_by": "session_report.build_summary",
        "consumed_by": "session_compare.py / trend analysis",
        "why": "한 세션의 processed와 render 성능을 바로 비교 가능한 숫자로 정리한 산출물이다.",
        "fields": [
            "session_meta: variant, scenario_id, source_capture",
            "processed.frame_count, invalid_rate, capture_to_process_ms",
            "render.frame_count, capture_to_render_ms, skipped_render_frames",
            "multi_target success metrics",
        ],
    },
]

PIPELINE_GRAPH = [
    {
        "stage": "Capture",
        "file": "real_time_process.py",
        "function": "UdpListener.run",
        "input": "raw UDP packet",
        "output": "packet queue",
    },
    {
        "stage": "Assembly / DSP",
        "file": "real_time_process.py + radar_runtime.py + DSP.py",
        "function": "DataProcessor.run / frame_to_radar_cube / Range_Doppler / Range_Angle",
        "input": "packet queue + IQ payload",
        "output": "radar cube + RDI + RAI",
    },
    {
        "stage": "Detection",
        "file": "detection.py + dbscan_cluster.py",
        "function": "cfar_threshold_2d / cluster_points / detect_targets",
        "input": "RDI + RAI",
        "output": "DetectionCandidate[]",
    },
    {
        "stage": "Tracking",
        "file": "tracking.py",
        "function": "_associate / update",
        "input": "DetectionCandidate[]",
        "output": "TrackEstimate[]",
    },
    {
        "stage": "Frame Contract",
        "file": "real_time_process.py",
        "function": "FramePacket + log_processed_frame",
        "input": "IQ + maps + detections + tracks",
        "output": "FramePacket + processed_frames.jsonl",
    },
    {
        "stage": "Render / Session Logs",
        "file": "live_motion_viewer.py",
        "function": "pull_latest_processed_frame / log_render_snapshot",
        "input": "FramePacket",
        "output": "render_frames.jsonl + event_log.jsonl",
    },
    {
        "stage": "Report",
        "file": "session_report.py",
        "function": "build_summary",
        "input": "processed + render + meta logs",
        "output": "summary.json",
    },
    {
        "stage": "Compare",
        "file": "session_compare.py",
        "function": "build_comparison",
        "input": "before / after summary.json",
        "output": "comparison_vs_*.json",
    },
]


def find_line_number(source: str, pattern: str) -> int | None:
    match = re.search(pattern, source, re.MULTILINE)
    if not match:
        return None
    return source[: match.start()].count("\n") + 1


def read_source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def render_code_block(source: str) -> str:
    lines = source.splitlines()
    rendered = []
    for index, line in enumerate(lines, start=1):
        rendered.append(
            f'<div class="code-line" id="L{index}"><a class="ln" href="#L{index}">{index}</a>'
            f'<span class="code-text">{html.escape(line)}</span></div>'
        )
    return "\n".join(rendered)


def render_stage_pills(stages: list[str]) -> str:
    return "".join(f'<span class="pill">{html.escape(stage)}</span>' for stage in stages)


def render_role_cards(roles: list[tuple[str, str]]) -> str:
    blocks = []
    for heading, body in roles:
        blocks.append(
            f'<article class="card"><h3>{html.escape(heading)}</h3><p>{html.escape(body)}</p></article>'
        )
    return "\n".join(blocks)


def render_read_order(items: list[str]) -> str:
    return "".join(f"<li>{html.escape(item)}</li>" for item in items)


def render_explainer_cards(steps: list[dict]) -> str:
    cards = []
    for step in steps:
        cards.append(
            f"""
            <article class="contract-card">
              <div class="contract-top">
                <div><h3>{html.escape(step['title'])}</h3></div>
                <span class="contract-tag">Step</span>
              </div>
              <p>{html.escape(step['summary'])}</p>
              <div class="contract-meta">
                <div><span>왜 필요한가</span><strong>{html.escape(step['reason'])}</strong></div>
                <div><span>다음으로 넘기는 것</span><strong>{html.escape(step['handoff'])}</strong></div>
              </div>
            </article>
            """
        )
    return "\n".join(cards)


def render_related_links(slugs: list[str]) -> str:
    cards = []
    for slug in slugs:
        meta = FILE_LOOKUP[slug]
        cards.append(
            "<a class=\"link-card\" href=\"{slug}.html\">"
            "<article class=\"card mini-card\">"
            "<div class=\"signal\">Related</div>"
            "<h3>{title}</h3>"
            "<p>{summary}</p>"
            "</article>"
            "</a>".format(
                slug=html.escape(meta["slug"]),
                title=html.escape(meta["title"]),
                summary=html.escape(meta["summary"]),
            )
        )
    return "\n".join(cards)


def render_landmarks(meta: dict, source: str) -> tuple[str, list[tuple[str, int | None, str]]]:
    blocks = []
    found = []
    for label, pattern, desc in meta["landmarks"]:
        line = find_line_number(source, pattern)
        found.append((label, line, desc))
        line_text = f"Line {line}" if line else "Line n/a"
        href = f"#L{line}" if line else "#code"
        blocks.append(
            '<a class="landmark" href="{href}"><strong>{label}</strong>'
            "<span>{line_text}</span><p>{desc}</p></a>".format(
                href=href,
                label=html.escape(label),
                line_text=html.escape(line_text),
                desc=html.escape(desc),
            )
        )
    return "\n".join(blocks), found


def render_key_table(found: list[tuple[str, int | None, str]]) -> str:
    rows = []
    for label, line, desc in found:
        line_value = f"<a href=\"#L{line}\">L{line}</a>" if line else "-"
        rows.append(
            "<tr><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                html.escape(label),
                line_value,
                html.escape(desc),
            )
        )
    return "\n".join(rows)


def base_style() -> str:
    return """
    :root {
      --bg: #f2f6f8;
      --panel: rgba(255,255,255,.95);
      --ink: #14222d;
      --muted: #536d7b;
      --line: #d8e2e8;
      --brand: #0d6e6e;
      --brand2: #153f68;
      --accent: #ff7d32;
      --shadow: 0 18px 50px rgba(18,39,56,.12);
      --mono: "JetBrains Mono", Consolas, monospace;
      --sans: "Pretendard Variable", "IBM Plex Sans KR", "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      font-family: var(--sans);
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(13,110,110,.18), transparent 32%),
        radial-gradient(circle at top right, rgba(21,63,104,.14), transparent 28%),
        linear-gradient(180deg, #edf4f6 0%, #f8fbfc 28%, #eef3f5 100%);
    }
    a { color: inherit; }
    .wrap { width: min(1320px, calc(100vw - 40px)); margin: 0 auto 60px; }
    .topbar {
      position: sticky; top: 0; z-index: 30; backdrop-filter: blur(18px);
      background: rgba(242,246,248,.84); border-bottom: 1px solid rgba(20,34,45,.08);
    }
    .topbar-inner {
      width: min(1320px, calc(100vw - 40px)); margin: 0 auto; padding: 16px 0;
      display: flex; justify-content: space-between; align-items: center; gap: 16px;
    }
    .brand { display: flex; align-items: center; gap: 14px; font-weight: 700; }
    .mark {
      width: 44px; height: 44px; border-radius: 14px;
      background: linear-gradient(135deg, var(--brand), #19a0a0 60%, var(--brand2));
      box-shadow: var(--shadow); position: relative;
    }
    .mark::before, .mark::after {
      content: ""; position: absolute; border: 2px solid rgba(255,255,255,.9); border-radius: 999px;
    }
    .mark::before { inset: 10px; }
    .mark::after { inset: 18px; }
    .nav { display: flex; gap: 10px; flex-wrap: wrap; }
    .nav a {
      text-decoration: none; color: var(--muted); font-size: 14px;
      padding: 10px 14px; border-radius: 999px; background: rgba(255,255,255,.78);
      border: 1px solid rgba(20,34,45,.06);
    }
    .hero {
      margin-top: 28px; padding: 32px; border-radius: 32px; color: #f7fbfd;
      background: linear-gradient(135deg, rgba(13,110,110,.94), rgba(21,63,104,.93));
      box-shadow: var(--shadow); overflow: hidden; position: relative;
    }
    .hero::before {
      content: ""; position: absolute; right: -80px; top: -60px; width: 240px; height: 240px;
      border-radius: 50%; background: radial-gradient(circle, rgba(255,255,255,.18), transparent 68%);
    }
    .hero-grid { display: grid; grid-template-columns: 1.4fr 1fr; gap: 24px; position: relative; z-index: 1; }
    .eyebrow {
      display: inline-block; margin-bottom: 18px; padding: 8px 12px; border-radius: 999px;
      background: rgba(255,255,255,.12); font-size: 13px;
    }
    h1 { margin: 0 0 16px; font-size: clamp(32px, 4vw, 54px); line-height: 1.04; letter-spacing: -.04em; }
    .hero p { margin: 0; line-height: 1.72; font-size: 16px; color: rgba(247,251,253,.92); }
    .hero-meta { display: grid; grid-template-columns: repeat(2, 1fr); gap: 14px; }
    .meta { padding: 18px; border-radius: 20px; background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.12); }
    .meta span { font-size: 13px; color: rgba(247,251,253,.76); }
    .meta strong { display: block; margin-top: 10px; font-size: 28px; letter-spacing: -.04em; }
    .section {
      margin-top: 26px; padding: 28px; border-radius: 28px; background: var(--panel);
      border: 1px solid rgba(20,34,45,.06); box-shadow: var(--shadow);
    }
    .section h2 { margin: 0 0 10px; font-size: 30px; letter-spacing: -.03em; }
    .intro { margin: 0 0 24px; color: var(--muted); line-height: 1.72; font-size: 15px; }
    .grid2, .grid3, .grid4 { display: grid; gap: 18px; }
    .grid2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .grid3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    .grid4 { grid-template-columns: repeat(4, minmax(0, 1fr)); }
    .card, .flow-step, .table-wrap, .landmark, .metric, .note { background: #fff; border: 1px solid var(--line); border-radius: 20px; }
    .card, .metric, .note, .flow-step { padding: 20px; }
    .card h3, .metric h3, .note h3 { margin: 0 0 10px; font-size: 19px; letter-spacing: -.02em; }
    .card p, .metric p, .note p, li, td, th { color: var(--muted); line-height: 1.72; font-size: 14px; }
    .signal {
      display: inline-flex; align-items: center; gap: 8px; font-size: 12px; font-weight: 700;
      text-transform: uppercase; letter-spacing: .08em; color: var(--brand); margin-bottom: 10px;
    }
    .signal::before {
      content: ""; width: 10px; height: 10px; border-radius: 50%; background: currentColor;
      box-shadow: 0 0 0 6px rgba(13,110,110,.12);
    }
    .pill-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; }
    .pill {
      display: inline-flex; align-items: center; border-radius: 999px; padding: 8px 12px;
      background: rgba(13,110,110,.08); color: var(--brand); font-size: 12px; font-weight: 700;
      letter-spacing: .04em; text-transform: uppercase;
    }
    .flow { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; align-items: stretch; }
    .flow-step strong { display: block; margin-bottom: 10px; font-size: 18px; }
    .flow-step span { display: inline-block; font-size: 12px; color: var(--brand); margin-bottom: 8px; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; }
    .landmark-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }
    .landmark { padding: 18px; text-decoration: none; display: block; }
    .landmark strong { display: block; font-size: 18px; margin-bottom: 8px; }
    .landmark span { display: inline-block; font-size: 12px; color: var(--brand); font-weight: 700; margin-bottom: 10px; }
    .landmark p { margin: 0; color: var(--muted); font-size: 14px; line-height: 1.65; }
    .table-wrap { overflow: hidden; }
    table { width: 100%; border-collapse: collapse; }
    th, td { text-align: left; padding: 14px 16px; border-bottom: 1px solid var(--line); vertical-align: top; }
    th { background: #f7fafb; color: var(--ink); font-size: 13px; text-transform: uppercase; letter-spacing: .06em; }
    .link-card { text-decoration: none; color: inherit; display: block; }
    .mini-card { height: 100%; }
    .contract-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 18px; }
    .contract-card { background: #fff; border: 1px solid var(--line); border-radius: 22px; padding: 20px; }
    .contract-top { display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; margin-bottom: 14px; }
    .contract-top h3 { margin: 0; font-size: 20px; letter-spacing: -.02em; }
    .contract-tag {
      display: inline-flex; align-items: center; border-radius: 999px; padding: 8px 12px;
      background: rgba(21,63,104,.08); color: var(--brand2); font-size: 12px; font-weight: 700;
      text-transform: uppercase; letter-spacing: .05em;
    }
    .contract-meta { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; margin: 14px 0; }
    .contract-meta div { background: #f7fafb; border: 1px solid var(--line); border-radius: 14px; padding: 12px; }
    .contract-meta span { display: block; font-size: 11px; text-transform: uppercase; letter-spacing: .06em; color: var(--brand); margin-bottom: 6px; font-weight: 700; }
    .contract-meta strong { display: block; font-size: 13px; line-height: 1.5; color: var(--ink); }
    .field-list { margin-top: 14px; padding-left: 18px; }
    .field-list li { margin-bottom: 6px; }
    details.accordion {
      background: #fff; border: 1px solid var(--line); border-radius: 20px; overflow: hidden;
    }
    details.accordion + details.accordion { margin-top: 14px; }
    details.accordion summary {
      list-style: none; cursor: pointer; padding: 18px 20px;
      display: flex; justify-content: space-between; align-items: center; gap: 16px;
      font-weight: 700; color: var(--ink);
    }
    details.accordion summary::-webkit-details-marker { display: none; }
    .summary-copy { display: flex; flex-direction: column; gap: 6px; }
    .summary-copy small { color: var(--muted); font-weight: 500; font-size: 13px; }
    .summary-tag {
      display: inline-flex; align-items: center; border-radius: 999px; padding: 8px 12px;
      background: rgba(13,110,110,.08); color: var(--brand); font-size: 12px; font-weight: 700;
      text-transform: uppercase; letter-spacing: .05em;
    }
    .accordion-body { padding: 0 20px 20px; }
    .handoff {
      display: grid; grid-template-columns: 160px 1.4fr 1fr 1fr;
      gap: 0; border: 1px solid var(--line); border-radius: 16px; overflow: hidden; margin-top: 14px;
    }
    .handoff-cell {
      padding: 14px 16px; border-right: 1px solid var(--line); border-bottom: 1px solid var(--line);
      color: var(--muted); font-size: 14px; line-height: 1.68; background: #fff;
    }
    .handoff-cell:nth-child(4n) { border-right: 0; }
    .handoff.head .handoff-cell {
      background: #f7fafb; color: var(--ink); font-size: 12px; font-weight: 700;
      text-transform: uppercase; letter-spacing: .06em;
    }
    .handoff .fn { font-family: var(--mono); color: var(--brand2); font-size: 13px; }
    .pipeline-strip {
      display: grid; grid-template-columns: repeat(8, minmax(0, 1fr)); gap: 12px; align-items: stretch;
    }
    .pipeline-node {
      position: relative; background: #fff; border: 1px solid var(--line); border-radius: 22px; padding: 18px;
      min-height: 200px;
    }
    .pipeline-node:not(:last-child)::after {
      content: "→"; position: absolute; right: -18px; top: calc(50% - 14px); z-index: 2;
      width: 28px; height: 28px; border-radius: 999px; background: var(--brand); color: #fff;
      display: flex; align-items: center; justify-content: center; font-weight: 700;
      box-shadow: 0 8px 20px rgba(13,110,110,.18);
    }
    .pipeline-node .stage { display: inline-flex; margin-bottom: 10px; font-size: 11px; text-transform: uppercase; letter-spacing: .08em; color: var(--brand); font-weight: 800; }
    .pipeline-node h3 { margin: 0 0 10px; font-size: 18px; letter-spacing: -.02em; }
    .pipeline-node p { margin: 0 0 10px; color: var(--muted); line-height: 1.65; font-size: 14px; }
    .pipeline-node .mono { font-family: var(--mono); color: var(--brand2); font-size: 12px; }
    .code-shell { background: #0f1720; color: #ecf6fb; border-radius: 24px; overflow: hidden; border: 1px solid rgba(255,255,255,.06); }
    .code-head {
      display: flex; justify-content: space-between; align-items: center; gap: 16px;
      padding: 16px 20px; background: rgba(255,255,255,.04); border-bottom: 1px solid rgba(255,255,255,.06);
      font-family: var(--mono); font-size: 13px;
    }
    .code-body { max-height: 880px; overflow: auto; font-family: var(--mono); font-size: 13px; }
    .code-line { display: grid; grid-template-columns: 72px 1fr; border-bottom: 1px solid rgba(255,255,255,.03); }
    .ln {
      display: block; text-decoration: none; color: #86b5d3; text-align: right; padding: 0 14px;
      background: rgba(255,255,255,.02); border-right: 1px solid rgba(255,255,255,.05); line-height: 22px;
    }
    .code-text { display: block; white-space: pre; padding: 0 16px; line-height: 22px; }
    ul { padding-left: 18px; margin: 0; }
    .footer { margin-top: 28px; text-align: center; color: var(--muted); font-size: 13px; }
    @media (max-width: 1080px) {
      .hero-grid, .grid4, .grid3, .grid2, .flow, .landmark-grid, .contract-grid { grid-template-columns: 1fr 1fr; }
      .pipeline-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .pipeline-node::after { display: none; }
    }
    @media (max-width: 760px) {
      .wrap, .topbar-inner { width: min(100vw - 24px, 1320px); }
      .hero-grid, .grid4, .grid3, .grid2, .flow, .landmark-grid, .hero-meta, .contract-grid, .contract-meta, .pipeline-strip { grid-template-columns: 1fr; }
      .section { padding: 22px; }
      .code-line { grid-template-columns: 58px 1fr; }
      .handoff { grid-template-columns: 1fr; }
      .handoff-cell { border-right: 0; }
      .nav { display: none; }
    }
    """


def html_page(title: str, body: str) -> str:
    return (
        "<!DOCTYPE html><html lang=\"ko\"><head><meta charset=\"UTF-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">"
        f"<title>{html.escape(title)}</title><style>{base_style()}</style></head><body>{body}</body></html>"
    )


def render_code_page(meta: dict) -> str:
    source_path = ROOT / meta["source"]
    source = read_source(source_path)
    lines = source.count("\n") + 1
    landmarks_html, found = render_landmarks(meta, source)
    table_rows = render_key_table(found)
    related_html = render_related_links(meta["related"])
    roles_html = render_role_cards(meta["roles"])
    code_html = render_code_block(source)
    explainer_cards = render_explainer_cards(meta.get("explainer_steps", []))
    simple_model = html.escape(meta.get("simple_model", meta["mentor"]))
    explainer_intro = html.escape(
        meta.get("explainer_intro", "입력에서 출력까지 실제 처리 순서를 단계별로 읽으면 파일 역할이 더 빨리 잡힌다.")
    )
    blind_spots = render_read_order(
        meta.get("blind_spots", ["기존 입력/출력/디버깅 포인트를 먼저 읽고 실제 코드 흐름을 이어서 보면 된다."])
    )

    body = f"""
    <div class="topbar">
      <div class="topbar-inner">
        <div class="brand"><div class="mark"></div><div>Radar Project Code Explorer</div></div>
        <nav class="nav">
          <a href="../index.html">메인 분석</a>
          <a href="index.html">코드 카탈로그</a>
          <a href="../../REQ/index.html">REQ</a>
          <a href="#code">전체 코드</a>
        </nav>
      </div>
    </div>
    <div class="wrap">
      <section class="hero">
        <div class="hero-grid">
          <div>
            <span class="eyebrow">{html.escape(meta["category"])}</span>
            <h1>{html.escape(meta["title"])}</h1>
            <p>{html.escape(meta["summary"])} {html.escape(meta["mentor"])}</p>
            <div class="pill-row">{render_stage_pills(meta["stages"])}</div>
          </div>
          <div class="hero-meta">
            <div class="meta"><span>Source Path</span><strong>{html.escape(meta["source"])}</strong></div>
            <div class="meta"><span>Line Count</span><strong>{lines}</strong></div>
            <div class="meta"><span>Before</span><strong>{html.escape(meta["before"])}</strong></div>
            <div class="meta"><span>After</span><strong>{html.escape(meta["after"])}</strong></div>
          </div>
        </div>
      </section>

      <section class="section">
        <h2>이 파일이 맡는 역할</h2>
        <p class="intro">{html.escape(meta["why"])}</p>
        <div class="flow">
          <article class="flow-step"><span>Input Side</span><strong>{html.escape(meta["before"])}</strong><p>{html.escape(meta["inputs"])}</p></article>
          <article class="flow-step"><span>This File</span><strong>{html.escape(meta["title"])}</strong><p>{html.escape(meta["mentor"])}</p></article>
          <article class="flow-step"><span>Output Side</span><strong>{html.escape(meta["after"])}</strong><p>{html.escape(meta["outputs"])}</p></article>
        </div>
      </section>

      <section class="section">
        <h2>핵심 책임</h2>
        <p class="intro">처음 보는 사람이 이 파일을 읽을 때 놓치면 안 되는 책임을 세 갈래로 나눴다.</p>
        <div class="grid3">{roles_html}</div>
      </section>

      <section class="section">
        <h2>쉽게 이해하는 모델</h2>
        <p class="intro">처음 보는 사람이 이 파일을 딱 한 문장으로 붙잡아야 할 때의 설명이다.</p>
        <div class="card"><p>{simple_model}</p></div>
      </section>

      <section class="section">
        <h2>입력에서 출력까지 단계별 설명</h2>
        <p class="intro">{explainer_intro}</p>
        <div class="contract-grid">{explainer_cards}</div>
      </section>

      <section class="section">
        <h2>놓치기 쉬운 부분</h2>
        <p class="intro">코드를 읽을 때 자주 헷갈리거나, 설명 없이 넘어가면 오해하기 쉬운 부분을 따로 정리했다.</p>
        <div class="card"><ul>{blind_spots}</ul></div>
      </section>

      <section class="section">
        <h2>핵심 라인 바로가기</h2>
        <p class="intro">실제 코드 기준으로 중요한 클래스와 함수를 직접 눌러 내려갈 수 있다.</p>
        <div class="landmark-grid">{landmarks_html}</div>
      </section>

      <section class="section">
        <h2>해석 가이드</h2>
        <p class="intro">입력, 출력, 디버깅 포인트를 같이 보면 이 파일 역할이 훨씬 빨리 정리된다.</p>
        <div class="grid4">
          <article class="note"><h3>입력</h3><p>{html.escape(meta["inputs"])}</p></article>
          <article class="note"><h3>출력</h3><p>{html.escape(meta["outputs"])}</p></article>
          <article class="note"><h3>디버깅 포인트</h3><p>{html.escape(meta["debug"])}</p></article>
          <article class="note"><h3>현업 메모</h3><p>{html.escape(meta["ops"])}</p></article>
        </div>
      </section>

      <section class="section">
        <h2>읽는 순서</h2>
        <p class="intro">시간을 아끼려면 아래 순서대로 보면 된다.</p>
        <div class="card"><ul>{render_read_order(meta["read_order"])}</ul></div>
      </section>

      <section class="section">
        <h2>중요 심볼 표</h2>
        <p class="intro">코드 리뷰나 발표 자료용으로 바로 참고할 수 있는 함수와 클래스 요약이다.</p>
        <div class="table-wrap">
          <table>
            <thead><tr><th>심볼</th><th>라인</th><th>설명</th></tr></thead>
            <tbody>{table_rows}</tbody>
          </table>
        </div>
      </section>

      <section class="section">
        <h2>연관 파일</h2>
        <p class="intro">이 파일 다음에 이어서 보면 좋은 코드들이다.</p>
        <div class="grid4">{related_html}</div>
      </section>

      <section class="section" id="code">
        <h2>전체 코드</h2>
        <p class="intro">UI는 유지한 채로 실제 코드 전문을 그대로 볼 수 있게 구성했다. 라인 번호를 누르면 해당 줄 링크를 쉽게 잡을 수 있다.</p>
        <div class="code-shell">
          <div class="code-head"><span>{html.escape(meta["source"])}</span><span>{lines} lines</span></div>
          <div class="code-body">{code_html}</div>
        </div>
      </section>

      <div class="footer">Generated from source. 문서와 코드가 어긋나면 이 생성 스크립트를 다시 실행하면 된다.</div>
    </div>
    """
    return html_page(f"{meta['title']} - Radar Project Code Explorer", body)


def render_code_index() -> str:
    category_counts = Counter(item["category"] for item in FILES)
    cards = []
    for meta in FILES:
        cards.append(
            f"""
            <a class="link-card" href="{html.escape(meta['slug'])}.html">
              <article class="card mini-card">
                <div class="signal">{html.escape(meta['category'])}</div>
                <h3>{html.escape(meta['title'])}</h3>
                <p>{html.escape(meta['summary'])}</p>
                <div class="pill-row">{render_stage_pills(meta['stages'])}</div>
              </article>
            </a>
            """
        )
    category_cards = []
    for name, count in sorted(category_counts.items()):
        category_cards.append(
            f'<article class="metric"><h3>{html.escape(name)}</h3><p>{count} files</p></article>'
        )

    body = f"""
    <div class="topbar">
      <div class="topbar-inner">
        <div class="brand"><div class="mark"></div><div>Radar Project Code Catalog</div></div>
        <nav class="nav">
          <a href="../index.html">메인 분석</a>
          <a href="../../REQ/index.html">REQ</a>
          <a href="#catalog">파일 목록</a>
        </nav>
      </div>
    </div>
    <div class="wrap">
      <section class="hero">
        <div class="hero-grid">
          <div>
            <span class="eyebrow">Code Catalog</span>
            <h1>프로젝트 코드 탐색기</h1>
            <p>이제 각 파일 페이지는 단순 코드 덤프가 아니라 역할, 입력과 출력, 핵심 함수, 읽는 순서, 관련 파일, 전체 코드 전문을 한 화면에서 같이 보여 준다.</p>
          </div>
          <div class="hero-meta">
            <div class="meta"><span>Documented Files</span><strong>{CURRENT_STATE['code_files']}</strong></div>
            <div class="meta"><span>Log Artifacts</span><strong>{CURRENT_STATE['log_artifacts']}</strong></div>
            <div class="meta"><span>Structure Score</span><strong>{CURRENT_STATE['score_structure']}</strong></div>
            <div class="meta"><span>Logging Score</span><strong>{CURRENT_STATE['score_logging']}</strong></div>
          </div>
        </div>
      </section>

      <section class="section">
        <h2>카테고리 분포</h2>
        <p class="intro">어떤 파일이 어떤 레이어를 담당하는지 먼저 보면 전체 구조가 훨씬 빨리 잡힌다.</p>
        <div class="grid4">{''.join(category_cards)}</div>
      </section>

      <section class="section" id="catalog">
        <h2>파일 목록</h2>
        <p class="intro">각 카드를 누르면 역할 설명과 전체 코드가 함께 있는 상세 페이지로 이동한다.</p>
        <div class="grid4">{''.join(cards)}</div>
      </section>
    </div>
    """
    return html_page("Project Analysis Code Catalog", body)


def render_flow_details_section() -> str:
    blocks = []
    for meta in FILES:
        steps = FLOW_DETAILS.get(meta["slug"], [])
        if not steps:
            continue
        rows = [
            '<div class="handoff head">'
            '<div class="handoff-cell">함수 / 메서드</div>'
            '<div class="handoff-cell">무슨 처리를 하는가</div>'
            '<div class="handoff-cell">만드는 데이터</div>'
            '<div class="handoff-cell">다음 코드</div>'
            '</div>'
        ]
        cell_rows = []
        for function_name, action, output_data, next_code in steps:
            cell_rows.extend(
                [
                    f'<div class="handoff-cell fn">{html.escape(function_name)}</div>',
                    f'<div class="handoff-cell">{html.escape(action)}</div>',
                    f'<div class="handoff-cell">{html.escape(output_data)}</div>',
                    f'<div class="handoff-cell">{html.escape(next_code)}</div>',
                ]
            )
        rows.append(f'<div class="handoff">{"".join(cell_rows)}</div>')
        blocks.append(
            f"""
            <details class="accordion">
              <summary>
                <span class="summary-copy">
                  <span>{html.escape(meta['title'])}</span>
                  <small>{html.escape(meta['summary'])}</small>
                </span>
                <span class="summary-tag">{html.escape(meta['category'])}</span>
              </summary>
              <div class="accordion-body">
                <p class="intro" style="margin-top:0;">{html.escape(meta['mentor'])}</p>
                {''.join(rows)}
              </div>
            </details>
            """
        )
    return "\n".join(blocks)


def render_data_contracts_section() -> str:
    cards = []
    for contract in DATA_CONTRACTS:
        field_items = "".join(f"<li>{html.escape(item)}</li>" for item in contract["fields"])
        cards.append(
            f"""
            <article class="contract-card">
              <div class="contract-top">
                <div>
                  <h3>{html.escape(contract['name'])}</h3>
                </div>
                <span class="contract-tag">{html.escape(contract['kind'])}</span>
              </div>
              <p>{html.escape(contract['why'])}</p>
              <div class="contract-meta">
                <div><span>Produced By</span><strong>{html.escape(contract['produced_by'])}</strong></div>
                <div><span>Consumed By</span><strong>{html.escape(contract['consumed_by'])}</strong></div>
              </div>
              <ul class="field-list">{field_items}</ul>
            </article>
            """
        )
    return "\n".join(cards)


def render_pipeline_graph() -> str:
    nodes = []
    for item in PIPELINE_GRAPH:
        nodes.append(
            f"""
            <article class="pipeline-node">
              <span class="stage">{html.escape(item['stage'])}</span>
              <h3>{html.escape(item['file'])}</h3>
              <p class="mono">{html.escape(item['function'])}</p>
              <p><strong>입력</strong><br>{html.escape(item['input'])}</p>
              <p><strong>출력</strong><br>{html.escape(item['output'])}</p>
            </article>
            """
        )
    return "\n".join(nodes)


def render_project_index() -> str:
    highlight_cards = "".join(
        f'<article class="metric"><h3>{html.escape(meta["title"])}</h3><p>{html.escape(meta["summary"])}</p></article>'
        for meta in [
            FILE_LOOKUP["live_motion_viewer"],
            FILE_LOOKUP["real_time_process"],
            FILE_LOOKUP["tracking"],
            FILE_LOOKUP["session_report"],
        ]
    )
    file_cards = []
    flow_details = render_flow_details_section()
    data_contracts = render_data_contracts_section()
    pipeline_graph = render_pipeline_graph()
    for meta in FILES:
        file_cards.append(
            f"""
            <a class="link-card" href="code/{html.escape(meta['slug'])}.html">
              <article class="card mini-card">
                <div class="signal">{html.escape(meta['category'])}</div>
                <h3>{html.escape(meta['title'])}</h3>
                <p>{html.escape(meta['mentor'])}</p>
              </article>
            </a>
            """
        )

    body = f"""
    <div class="topbar">
      <div class="topbar-inner">
        <div class="brand"><div class="mark"></div><div>DCA1000 Radar Project Analysis</div></div>
        <nav class="nav">
          <a href="#goal">목표</a>
          <a href="#pipeline">구조</a>
          <a href="code/index.html">코드 카탈로그</a>
          <a href="../REQ/index.html">REQ</a>
        </nav>
      </div>
    </div>
    <div class="wrap">
      <section class="hero">
        <div class="hero-grid">
          <div>
            <span class="eyebrow">Project Analysis</span>
            <h1>실시간 객체 추적 레이더 프로젝트</h1>
            <p>이 프로젝트의 핵심 목표는 DCA1000과 mmWave 레이더로 들어오는 raw IQ를 실시간으로 처리해 움직이는 객체를 검출하고, 프레임 사이에서 같은 객체를 같은 track으로 유지하며, 그 과정을 눈으로 확인하고 나중에 비교 분석할 수 있게 만드는 것이다.</p>
          </div>
          <div class="hero-meta">
            <div class="meta"><span>Overall</span><strong>{CURRENT_STATE['score_overall']}</strong></div>
            <div class="meta"><span>Code Structure</span><strong>{CURRENT_STATE['score_structure']}</strong></div>
            <div class="meta"><span>Logging</span><strong>{CURRENT_STATE['score_logging']}</strong></div>
            <div class="meta"><span>Documented Files</span><strong>{CURRENT_STATE['code_files']}</strong></div>
          </div>
        </div>
      </section>

      <section class="section" id="goal">
        <h2>프로젝트 목표 정리</h2>
        <p class="intro">한 문장으로 줄이면, 이 프로젝트는 실시간 객체 추적 레이더 데모를 만들고 그것을 환경 스냅샷과 운영 점수까지 포함해 재현 가능하게 측정하는 시스템으로 진화하고 있다.</p>
        <div class="grid3">
          <article class="card"><h3>실시간 검출</h3><p>UDP로 들어오는 raw frame에서 움직이는 후보를 안정적으로 찾아야 한다.</p></article>
          <article class="card"><h3>실시간 추적</h3><p>같은 사람을 여러 프레임에서 같은 ID로 유지해야 한다.</p></article>
          <article class="card"><h3>지속 측정</h3><p>processed, render, event, system snapshot, ops report 구조가 들어가면서 전과 후를 숫자와 점수로 비교하는 기반이 생겼다.</p></article>
        </div>
      </section>

      <section class="section" id="pipeline">
        <h2>전체 구조</h2>
        <p class="intro">처리 흐름은 아래와 같이 이해하면 된다.</p>
        <div class="flow">
          <article class="flow-step"><span>1. 설정 / 제어</span><strong>runtime_settings.py + radar_config.py</strong><p>실험 파라미터를 읽고 장비를 켠다.</p></article>
          <article class="flow-step"><span>2. 수신 / 처리</span><strong>real_time_process.py + radar_runtime.py + DSP.py</strong><p>raw IQ를 cube, RDI, RAI로 만든다.</p></article>
          <article class="flow-step"><span>3. 검출 / 추적</span><strong>detection.py + dbscan_cluster.py + tracking.py</strong><p>사람 후보를 찾고 track ID를 유지한다.</p></article>
        </div>
        <div class="flow" style="margin-top:16px;">
          <article class="flow-step"><span>4. 실시간 표시</span><strong>live_motion_viewer.py + app_layout.py</strong><p>최신 프레임만 골라 UI에 올린다.</p></article>
          <article class="flow-step"><span>5. 로그</span><strong>session_meta / processed / render / event / system_snapshot</strong><p>처리 로그와 렌더 로그를 분리해 남기고 실행 환경도 함께 저장한다.</p></article>
          <article class="flow-step"><span>6. 리포트</span><strong>session_report.py + operational_assessment.py + log_html_reports.py</strong><p>summary, 운영 점수, HTML 대시보드를 만든다.</p></article>
        </div>
      </section>

      <section class="section">
        <h2>왜 로그 구조가 중요해졌는가</h2>
        <p class="intro">예전에는 status_log.jsonl이 사실상 render 기준 로그였다. 지금은 처리된 모든 프레임과 실제 화면에 보인 프레임을 분리해서 볼 수 있고, 그 세션을 어떤 전원 계획과 NIC 상태에서 돌렸는지도 함께 남긴다.</p>
        <div class="grid3">
          <article class="card"><h3>이전 상태</h3><p>UI가 최신 프레임만 소비하면 중간 처리 프레임이 스킵되고, 알고리즘 자체 성능이 로그에 충분히 남지 않았다.</p></article>
          <article class="card"><h3>현재 상태</h3><p>processed_frames.jsonl은 처리기 기준 전체 프레임을, render_frames.jsonl은 사용자 체감 기준 프레임을, system_snapshot.json은 실행 환경 상태를 남긴다.</p></article>
          <article class="card"><h3>의미</h3><p>이제 알고리즘 개선, UI 병목, 실행 환경 문제를 분리해서 측정할 수 있다. session_report와 ops_report가 그 차이를 숫자와 점수로 보여 준다.</p></article>
        </div>
      </section>

      <section class="section">
        <h2>파일 간 데이터 전달 맵</h2>
        <p class="intro">raw packet이 어떤 파일과 함수들을 거쳐 최종적으로 비교 리포트까지 가는지 한 줄로 따라갈 수 있게 정리했다. 회의나 발표에서 전체 흐름을 설명할 때 가장 먼저 보여주기 좋은 섹션이다.</p>
        <div class="pipeline-strip">{pipeline_graph}</div>
      </section>

      <section class="section">
        <h2>핵심 데이터 구조</h2>
        <p class="intro">현업 문서에서는 코드만 보는 것보다 데이터 계약을 먼저 보는 편이 훨씬 빠르다. 아래 카드들은 이 프로젝트를 이해할 때 꼭 알아야 하는 데이터 단위를 정리한 것이다.</p>
        <div class="contract-grid">{data_contracts}</div>
      </section>

      <section class="section">
        <h2>핵심 코드 4개</h2>
        <p class="intro">처음 들어온 사람이 가장 먼저 이해하면 좋은 파일들이다.</p>
        <div class="grid4">{highlight_cards}</div>
      </section>

      <section class="section">
        <h2>코드 탐색</h2>
        <p class="intro">각 카드에서 파일 역할 설명과 실제 코드 전문을 함께 볼 수 있다.</p>
        <div class="grid4">{''.join(file_cards)}</div>
      </section>

      <section class="section">
        <h2>코드별 처리와 전달 흐름</h2>
        <p class="intro">현업 문서에서는 단순히 파일 설명만 있는 것보다, 어떤 입력이 어떤 함수에서 어떻게 가공되고 다음 코드로 어떤 데이터가 넘어가는지까지 적혀 있어야 온보딩과 디버깅이 빨라진다. 아래 아코디언은 그 handoff를 파일별로 정리한 것이다.</p>
        {flow_details}
      </section>

      <section class="section">
        <h2>처음 보는 사람용 읽기 순서</h2>
        <p class="intro">시간을 아끼려면 아래 순서가 가장 효율적이다.</p>
        <div class="card">
          <ul>
            <li>1단계: runtime_settings.py와 session_logging.py로 실험 파라미터와 로그 토글을 본다.</li>
            <li>2단계: live_motion_viewer.py로 앱이 어떤 워커와 로그를 엮는지 이해한다.</li>
            <li>3단계: real_time_process.py, radar_runtime.py, DSP.py로 실제 처리 파이프라인과 shared FFT 경로를 따라간다.</li>
            <li>4단계: detection.py, dbscan_cluster.py, tracking.py로 결과가 어떻게 안정화되는지 본다.</li>
            <li>5단계: session_report.py, operational_assessment.py, log_html_reports.py로 진단과 리포트 체계를 이해한다.</li>
          </ul>
        </div>
      </section>

      <section class="section">
        <h2>현업 시점 평가</h2>
        <p class="intro">관측성과 실험 추적 능력은 분명히 좋아졌고, shared FFT와 stage timing처럼 성능 개선 기반도 생겼다. 다만 구조 분리와 재생 기반 회귀 검증은 아직 더 필요하다.</p>
        <div class="grid3">
          <article class="card"><h3>좋아진 점</h3><p>processed, render, event, system snapshot, summary, ops report 구조가 생기면서 개선 판단 근거가 훨씬 강해졌다.</p></article>
          <article class="card"><h3>여전히 아쉬운 점</h3><p>MotionViewer에 책임이 많이 몰려 있고 processing hot path는 아직 직렬 계산 비중이 높다.</p></article>
          <article class="card"><h3>다음 우선순위</h3><p>AppController 분리, replay 입력 경로 정리, CFAR/후보 루프 벡터화, regression threshold 추가가 다음 단계다.</p></article>
        </div>
      </section>
    </div>
    """
    return html_page("DCA1000 Radar Project Analysis", body)


def render_req_index() -> str:
    body = f"""
    <div class="topbar">
      <div class="topbar-inner">
        <div class="brand"><div class="mark"></div><div>Radar Project Improvement REQ</div></div>
        <nav class="nav">
          <a href="../project_analysis/index.html">메인 분석</a>
          <a href="../project_analysis/code/index.html">코드 카탈로그</a>
          <a href="#status">현재 상태</a>
        </nav>
      </div>
    </div>
    <div class="wrap">
      <section class="hero">
        <div class="hero-grid">
          <div>
            <span class="eyebrow">REQ / Improvement Spec</span>
            <h1>실무형 리팩터링 요구사항</h1>
            <p>이 문서는 지금 구조에서 무엇이 부족하고 무엇을 어떤 순서로 고쳐야 하는지 구현 관점에서 정리한 명세서다. 최근에는 로그 분리, system snapshot, stage timing, operational report, shared FFT 최적화가 반영되었고, 그 다음 단계도 함께 업데이트했다.</p>
          </div>
          <div class="hero-meta">
            <div class="meta"><span>Overall</span><strong>{CURRENT_STATE['score_overall']}</strong></div>
            <div class="meta"><span>Logging</span><strong>{CURRENT_STATE['score_logging']}</strong></div>
            <div class="meta"><span>Active REQ</span><strong>6</strong></div>
            <div class="meta"><span>Implemented</span><strong>Phase 1-3</strong></div>
          </div>
        </div>
      </section>

      <section class="section" id="status">
        <h2>현재 반영된 항목</h2>
        <p class="intro">최근 업데이트로 관측성과 비교 자동화 기반은 실제 코드에 반영되었다.</p>
        <div class="grid3">
          <article class="card"><h3>완료 1</h3><p>processed_frames.jsonl, render_frames.jsonl, event_log.jsonl, session_meta.json 구조가 추가되었다.</p></article>
          <article class="card"><h3>완료 2</h3><p>session_report.py가 system_snapshot, stage timing, operational assessment를 포함한 summary.json을 생성한다.</p></article>
          <article class="card"><h3>완료 3</h3><p>log_html_reports.py가 index.html과 ops_report.html을 만들고, DSP.py에는 shared FFT 기반 hot-path 최적화가 반영되었다.</p></article>
        </div>
      </section>

      <section class="section">
        <h2>남은 핵심 요구사항</h2>
        <p class="intro">이제부터는 구조 분리, 재현 실험, 남은 processing 병목 제거를 단계적으로 진행하는 것이 맞다.</p>
        <div class="table-wrap">
          <table>
            <thead><tr><th>REQ</th><th>상태</th><th>목표</th><th>구현 포인트</th></tr></thead>
            <tbody>
              <tr><td>MotionViewer 책임 분리</td><td>In Progress</td><td>UI, 앱 제어, 세션 로그 책임을 더 명확히 나눈다.</td><td>AppController / ViewRenderer / SessionLogger 경계 강화</td></tr>
              <tr><td>Processing hot-path 경량화</td><td>In Progress</td><td>real_time_process.py의 직렬 계산 비용을 더 줄인다.</td><td>CFAR 벡터화, 후보 pruning, logging writer 분리</td></tr>
              <tr><td>print 기반 로그 정리</td><td>Pending</td><td>장비 제어와 예외 로그도 구조화한다.</td><td>logging 모듈 또는 JSON logger 도입</td></tr>
              <tr><td>Replay 입력 경로</td><td>Pending</td><td>같은 raw 입력으로 전후 비교 가능하게 만든다.</td><td>read_binfile.py 기반 session_replay 경로 추가</td></tr>
              <tr><td>Regression Threshold</td><td>Pending</td><td>요약 비교 결과를 pass와 fail로 판정한다.</td><td>session_compare.py에 임계값 규칙 추가</td></tr>
              <tr><td>Hybrid Capon PoC</td><td>Pending</td><td>각도 해상도를 개선하되 실시간성은 유지한다.</td><td>detection 후보 상위 N개에 angle refinement 적용</td></tr>
            </tbody>
          </table>
        </div>
      </section>

      <section class="section">
        <h2>우선순위 상세</h2>
        <p class="intro">현업 기준으로는 아래 순서가 가장 안전하다.</p>
        <div class="grid2">
          <article class="card"><h3>1. processing 병목 제거</h3><p>shared FFT는 들어갔지만 detect_targets, CFAR, logging write 비용은 아직 크다. 여기서 추가 벡터화와 비동기화를 먼저 진행하는 것이 체감 효과가 크다.</p></article>
          <article class="card"><h3>2. 구조 안정화</h3><p>현재는 live_motion_viewer.py에 앱 제어와 렌더가 많이 몰려 있다. SessionLogger는 분리됐으니 다음은 AppController와 ViewRenderer 경계를 더 만드는 단계다.</p></article>
          <article class="card"><h3>3. 재현 실험 가능화</h3><p>실제 전후 비교는 동일 입력이 있어야 의미가 있다. read_binfile.py를 기반으로 replay 경로를 만들고 session_meta에 source_capture를 강제 기록한다.</p></article>
          <article class="card"><h3>4. 비교 자동화 강화</h3><p>session_compare.py는 이미 delta를 내지만 아직 fail와 pass 정책은 없다. invalid_rate, latency p95, operational score 기준을 추가한다.</p></article>
        </div>
      </section>

      <section class="section">
        <h2>Definition of Done</h2>
        <p class="intro">아래 기준이 충족되면 이 프로젝트는 PoC를 넘어 반복 실험 가능한 개발 상태에 들어간다.</p>
        <div class="card">
          <ul>
            <li>live path와 replay path 모두 같은 summary schema를 만든다.</li>
            <li>before와 after 비교가 scenario_id와 source_capture 기준으로 자동 묶인다.</li>
            <li>구조상 UI, 처리, 로깅 책임이 분리되어 한 파일 집중도가 낮아진다.</li>
            <li>tracking과 latency와 operational score 지표에 최소 회귀 기준이 걸린다.</li>
            <li>README와 docs만 읽어도 신규 인원이 실행 경로와 비교 경로를 이해할 수 있다.</li>
          </ul>
        </div>
      </section>
    </div>
    """
    return html_page("Radar Project Improvement REQ", body)


def main() -> None:
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    CODE_DIR.mkdir(parents=True, exist_ok=True)
    REQ_DIR.mkdir(parents=True, exist_ok=True)

    write_text(PROJECT_DIR / "index.html", render_project_index())
    write_text(CODE_DIR / "index.html", render_code_index())
    write_text(REQ_DIR / "index.html", render_req_index())
    for meta in FILES:
        write_text(CODE_DIR / f"{meta['slug']}.html", render_code_page(meta))


if __name__ == "__main__":
    main()
