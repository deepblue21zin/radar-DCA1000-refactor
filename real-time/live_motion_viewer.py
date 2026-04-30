import os
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
import sys
import time

import numpy as np

# Keep pyqtgraph on the same Qt binding as the generated UI module.
os.environ.setdefault('PYQTGRAPH_QT_LIB', 'PyQt5')

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dca1000_control import DcaConfigClient
from tools.runtime_core.app_layout import Ui_MainWindow
from tools.runtime_core.detection import DetectionRegion
from tools.runtime_core.radar_config import SerialConfig
from tools.runtime_core.radar_runtime import (
    apply_cartesian_roi_to_rai,
    parse_runtime_config,
    radial_bin_limit,
)
from tools.runtime_core.real_time_process import (
    DataProcessor,
    RawCaptureReplaySource,
    RawFrameCaptureWriter,
    UdpListener,
)
from tools.runtime_core.runtime_settings import load_runtime_settings, resolve_project_path
from session_logging import SessionLogger
from spatial_view import SpatialViewController, build_heatmap_lookup_table
from tools.runtime_core.tracking import MultiTargetTracker


SETTINGS = load_runtime_settings(PROJECT_ROOT)
STATIC_SETTINGS_PATH = Path(SETTINGS['_static_settings_path'])
RUNTIME_SETTINGS_PATH = Path(SETTINGS['_runtime_settings_path'])
TUNING_PATH = Path(SETTINGS['_tuning_path'])
CONFIG_PATH = Path(SETTINGS['_config_path_resolved'])
STATIC = SETTINGS['static']
RUNTIME = SETTINGS['runtime']
TUNING = SETTINGS['tuning']
CLI_PORT = RUNTIME['cli_port']
CLI_BAUDRATE = int(STATIC['cli_baudrate'])
HOST_IP = STATIC['network']['host_ip']
DATA_PORT = int(STATIC['network']['data_port'])
CONFIG_PORT = int(STATIC['network']['config_port'])
FPGA_IP = STATIC['network']['fpga_ip']
FPGA_PORT = int(STATIC['network']['fpga_port'])
BUFFER_SIZE = int(STATIC['network']['buffer_size'])
REMOVE_STATIC = bool(TUNING['processing']['remove_static'])
DOPPLER_GUARD_BINS = int(TUNING['processing']['doppler_guard_bins'])
INVERT_LATERAL_AXIS = bool(TUNING['processing'].get('invert_lateral_axis', False))
ROI_LATERAL_M = float(TUNING['roi']['lateral_m'])
ROI_FORWARD_M = float(TUNING['roi']['forward_m'])
ROI_MIN_FORWARD_M = float(TUNING['roi']['min_forward_m'])
ALLOW_STRONGEST_FALLBACK = bool(TUNING['detection']['allow_strongest_fallback'])
TRACK_CONFIRM_HITS = int(TUNING['tracking']['confirm_hits'])
TRACK_MAX_MISSES = int(TUNING['tracking']['max_misses'])
TRACK_PROCESS_VAR = float(TUNING['tracking']['process_var'])
TRACK_MEASUREMENT_VAR = float(TUNING['tracking']['measurement_var'])
TRACK_RANGE_MEASUREMENT_SCALE = float(TUNING['tracking']['range_measurement_scale'])
TRACK_CONFIDENCE_MEASUREMENT_SCALE = float(TUNING['tracking']['confidence_measurement_scale'])
TRACK_ASSOCIATION_GATE = float(TUNING['tracking']['association_gate'])
TRACK_DOPPLER_ZERO_GUARD_BINS = int(TUNING['tracking']['doppler_zero_guard_bins'])
TRACK_DOPPLER_GATE_BINS = int(TUNING['tracking']['doppler_gate_bins'])
TRACK_DOPPLER_COST_WEIGHT = float(TUNING['tracking']['doppler_cost_weight'])
TRACK_REPORT_MISS_TOLERANCE = int(TUNING['tracking']['report_miss_tolerance'])
TRACK_LOST_GATE_FACTOR = float(TUNING['tracking']['lost_gate_factor'])
TRACK_TENTATIVE_GATE_FACTOR = float(TUNING['tracking']['tentative_gate_factor'])
TRACK_BIRTH_SUPPRESSION_RADIUS_M = float(TUNING['tracking']['birth_suppression_radius_m'])
TRACK_PRIMARY_TRACK_BIRTH_SCALE = float(TUNING['tracking']['primary_track_birth_scale'])
TRACK_BIRTH_SUPPRESSION_WEAK_RADIUS_SCALE = float(TUNING['tracking'].get('birth_suppression_weak_radius_scale', 1.0))
TRACK_BIRTH_SUPPRESSION_SCORE_RATIO = float(TUNING['tracking'].get('birth_suppression_score_ratio', 0.0))
TRACK_BIRTH_SUPPRESSION_CONFIDENCE_RATIO = float(TUNING['tracking'].get('birth_suppression_confidence_ratio', 0.0))
TRACK_BIRTH_SUPPRESSION_DOPPLER_BINS = int(TUNING['tracking'].get('birth_suppression_doppler_bins', 0))
TRACK_BIRTH_SUPPRESSION_MISS_TOLERANCE = int(TUNING['tracking']['birth_suppression_miss_tolerance'])
TRACK_PRIMARY_TRACK_HOLD_FRAMES = int(TUNING['tracking']['primary_track_hold_frames'])
TRACK_LATERAL_DEADBAND_M = float(TUNING['tracking']['lateral_deadband_m'])
TRACK_LATERAL_DEADBAND_RANGE_SCALE = float(TUNING['tracking']['lateral_deadband_range_scale'])
TRACK_LATERAL_SMOOTHING_ALPHA = float(TUNING['tracking']['lateral_smoothing_alpha'])
TRACK_LATERAL_VELOCITY_DAMPING = float(TUNING['tracking']['lateral_velocity_damping'])
TRACK_LOCAL_REMEASUREMENT_ENABLED = bool(TUNING['tracking'].get('local_remeasurement_enabled', False))
TRACK_LOCAL_REMEASUREMENT_BLEND = float(TUNING['tracking'].get('local_remeasurement_blend', 0.0))
TRACK_LOCAL_REMEASUREMENT_MAX_SHIFT_M = float(TUNING['tracking'].get('local_remeasurement_max_shift_m', 0.0))
TRACK_LOCAL_REMEASUREMENT_TRACK_BIAS = float(TUNING['tracking'].get('local_remeasurement_track_bias', 0.0))
TRACK_LOCAL_REMEASUREMENT_PATCH_BANDS = list(TUNING['tracking'].get('local_remeasurement_patch_bands', []))
TRACK_MEASUREMENT_SOFT_GATE_ENABLED = bool(TUNING['tracking'].get('measurement_soft_gate_enabled', True))
TRACK_MEASUREMENT_SOFT_GATE_FLOOR = float(TUNING['tracking'].get('measurement_soft_gate_floor', 0.35))
TRACK_MEASUREMENT_SOFT_GATE_START_M = float(TUNING['tracking'].get('measurement_soft_gate_start_m', 0.16))
TRACK_MEASUREMENT_SOFT_GATE_FULL_M = float(TUNING['tracking'].get('measurement_soft_gate_full_m', 0.52))
TRACK_MEASUREMENT_SOFT_GATE_RANGE_SCALE = float(TUNING['tracking'].get('measurement_soft_gate_range_scale', 0.05))
TRACK_MEASUREMENT_SOFT_GATE_SPEED_SCALE = float(TUNING['tracking'].get('measurement_soft_gate_speed_scale', 0.06))
DISPLAY_MIN_CONFIDENCE = float(TUNING['detection']['display_min_confidence'])
PIPELINE_QUEUE_SIZE = int(TUNING['pipeline']['queue_size'])
BLOCK_TRACK_BIRTH_ON_INVALID = bool(TUNING['pipeline']['block_track_birth_on_invalid'])
INVALID_POLICY = TUNING['pipeline']['invalid_policy']
DCA_CONFIG_TIMEOUT_S = float(SETTINGS['dca']['config_timeout_s'])
DCA_PACKET_SIZE_BYTES = int(SETTINGS['dca']['packet_size_bytes'])
DCA_PACKET_DELAY_US = int(SETTINGS['dca']['packet_delay_us'])
DCA_PACKET_DELAY_TICKS_PER_US = int(SETTINGS['dca']['packet_delay_ticks_per_us'])
DBSCAN_ADAPTIVE_EPS_BANDS = tuple(TUNING['detection']['dbscan_adaptive_eps_bands'])
DBSCAN_CLUSTER_MIN_SAMPLES = int(TUNING['detection']['cluster_min_samples'])
DBSCAN_CLUSTER_VELOCITY_WEIGHT = float(TUNING['detection']['cluster_velocity_weight'])
DETECTION_MAX_TARGETS = int(TUNING['detection']['max_targets'])
DETECTION_ALGORITHM = TUNING['detection']['algorithm']
DETECTION_TUNING = {
    'cfar_training_cells': tuple(DETECTION_ALGORITHM['cfar_training_cells']),
    'cfar_guard_cells': tuple(DETECTION_ALGORITHM['cfar_guard_cells']),
    'cfar_scale': float(DETECTION_ALGORITHM['cfar_scale']),
    'global_quantile': float(DETECTION_ALGORITHM['global_quantile']),
    'angle_quantile': float(DETECTION_ALGORITHM['angle_quantile']),
    'angle_contrast_scale': float(DETECTION_ALGORITHM['angle_contrast_scale']),
    'min_cartesian_separation_m': float(DETECTION_ALGORITHM['min_cartesian_separation_m']),
    'angle_centroid_radius_bands': list(DETECTION_ALGORITHM.get('angle_centroid_radius_bands', [])),
    'body_center_patch_bands': list(DETECTION_ALGORITHM.get('body_center_patch_bands', [])),
    'candidate_merge_bands': list(DETECTION_ALGORITHM.get('candidate_merge_bands', [])),
    'duplicate_suppression_enabled': bool(DETECTION_ALGORITHM.get('duplicate_suppression_enabled', True)),
    'duplicate_suppression_radius_m': float(DETECTION_ALGORITHM.get('duplicate_suppression_radius_m', 0.55)),
    'duplicate_suppression_range_scale': float(DETECTION_ALGORITHM.get('duplicate_suppression_range_scale', 0.03)),
    'duplicate_suppression_doppler_bins': int(DETECTION_ALGORITHM.get('duplicate_suppression_doppler_bins', 6)),
    'duplicate_suppression_score_ratio': float(DETECTION_ALGORITHM.get('duplicate_suppression_score_ratio', 0.82)),
}
LOG_ROOT = PROJECT_ROOT / 'logs' / 'live_motion_viewer'
SPATIAL_VIEW_HEIGHT = int(STATIC['spatial_view']['height'])
SPATIAL_VIEW_Y = int(STATIC['spatial_view']['y'])
SPATIAL_POINT_BASE_Z_M = float(STATIC['spatial_view']['point_base_z_m'])
SPATIAL_POINT_CONFIDENCE_SCALE_M = float(STATIC['spatial_view']['point_confidence_scale_m'])
SHOW_TENTATIVE_TRACKS = bool(TUNING['visualization']['show_tentative_tracks'])
TENTATIVE_MIN_CONFIDENCE = float(TUNING['visualization']['tentative_min_confidence'])
TENTATIVE_MIN_HITS = int(TUNING['visualization']['tentative_min_hits'])
DISPLAY_HYSTERESIS_FRAMES = int(TUNING['visualization'].get('display_hysteresis_frames', 0))
DISPLAY_HYSTERESIS_CONFIDENCE_FLOOR = float(TUNING['visualization'].get('display_hysteresis_confidence_floor', 0.0))
DISPLAY_PRIMARY_BONUS_FRAMES = int(TUNING['visualization'].get('display_primary_bonus_frames', 0))
LOG_VARIANT = str(RUNTIME['logging']['variant'])
LOG_SCENARIO_ID = str(RUNTIME['logging']['scenario_id'])
LOG_INPUT_MODE = str(RUNTIME['logging']['input_mode'])
LOG_SOURCE_CAPTURE = str(RUNTIME['logging']['source_capture'])
LOG_NOTES = str(RUNTIME['logging']['notes'])
LOG_ENABLED = bool(RUNTIME['logging'].get('enabled', True))
LOG_WRITE_RAW_CAPTURE = bool(RUNTIME['logging'].get('write_raw_capture', True))
LOG_RAW_CAPTURE_ROOT = str(RUNTIME['logging'].get('raw_capture_root', 'logs/raw'))
LOG_WRITE_PROCESSED_FRAMES = bool(RUNTIME['logging'].get('write_processed_frames', True))
LOG_WRITE_RENDER_FRAMES = bool(RUNTIME['logging'].get('write_render_frames', True))
LOG_WRITE_STATUS_LOG = bool(RUNTIME['logging'].get('write_status_log', True))
LOG_WRITE_EVENT_LOG = bool(RUNTIME['logging'].get('write_event_log', True))
LOG_INCLUDE_PAYLOADS = bool(RUNTIME['logging'].get('include_payloads', True))
LOG_CAPTURE_SYSTEM_SNAPSHOT = bool(RUNTIME['logging'].get('capture_system_snapshot', True))
LOG_CAPTURE_STAGE_TIMING = bool(RUNTIME['logging'].get('capture_stage_timing', True))
LOG_REPORT_GENERATION_MODE = str(RUNTIME['logging'].get('report_generation_mode', 'deferred'))


def _normalize_capture_duration_s(value):
    try:
        duration_s = float(value)
    except (TypeError, ValueError):
        return None
    if duration_s <= 0:
        return None
    return duration_s


LOG_CAPTURE_DURATION_S = _normalize_capture_duration_s(
    RUNTIME['logging'].get('capture_duration_s')
)

class MotionViewer:
    def __init__(
        self,
        *,
        input_mode=None,
        source_capture=None,
        replay_speed=1.0,
        replay_loop=False,
        auto_start=False,
        write_raw_capture=None,
    ):
        self.input_mode = str(input_mode if input_mode is not None else LOG_INPUT_MODE).strip().lower()
        self.source_capture = str(source_capture if source_capture is not None else LOG_SOURCE_CAPTURE).strip()
        self.replay_speed = max(float(replay_speed), 0.01)
        self.replay_loop = bool(replay_loop)
        self.auto_start = bool(auto_start)
        self.hardware_enabled = self.input_mode != 'replay'
        self.report_generation_mode = 'inline' if self.input_mode == 'replay' else LOG_REPORT_GENERATION_MODE
        self.replay_completion_requested = False
        self.capture_duration_s = (
            LOG_CAPTURE_DURATION_S if self.hardware_enabled else None
        )
        self.capture_stop_timer = None
        self.write_raw_capture = bool(
            LOG_WRITE_RAW_CAPTURE if write_raw_capture is None else write_raw_capture
        ) and self.hardware_enabled
        self.raw_capture_root = resolve_project_path(PROJECT_ROOT, LOG_RAW_CAPTURE_ROOT)
        self.runtime_config = parse_runtime_config(
            CONFIG_PATH,
            remove_static=REMOVE_STATIC,
            doppler_guard_bins=DOPPLER_GUARD_BINS,
            lateral_axis_sign=(-1.0 if INVERT_LATERAL_AXIS else 1.0),
        )
        self.track_angle_resolution_rad = self.estimate_track_angle_resolution_rad()
        self.raw_frame_queue = Queue(maxsize=PIPELINE_QUEUE_SIZE)
        self.processed_frame_queue = Queue(maxsize=PIPELINE_QUEUE_SIZE)
        self.radar_ctrl = None
        self.dca_client = None
        self.collector = None
        self.processor = None
        self.img_rdi = None
        self.img_rai = None
        self.rdi_scatter = None
        self.rai_scatter = None
        self.rdi_tentative_scatter = None
        self.rai_tentative_scatter = None
        self.spatial_label = None
        self.replay_plot = None
        self.replay_origin_scatter = None
        self.replay_origin_label = None
        self.replay_track_history = {}
        self.replay_track_items = {}
        self.display_hysteresis_state = {}
        self.spatial_view = SpatialViewController(
            roi_lateral_m=ROI_LATERAL_M,
            roi_forward_m=ROI_FORWARD_M,
            roi_min_forward_m=ROI_MIN_FORWARD_M,
            view_y=SPATIAL_VIEW_Y,
            view_height=SPATIAL_VIEW_HEIGHT,
            point_base_z_m=SPATIAL_POINT_BASE_Z_M,
            point_confidence_scale_m=SPATIAL_POINT_CONFIDENCE_SCALE_M,
        )
        self.app = None
        self.main_window = None
        self.ui = None
        self.waiting_for_data_logged = False
        self.stream_started_at = None
        self.first_image_logged = False
        self.frame_index = 0
        self.skipped_render_frames = 0
        self.min_range_bin = radial_bin_limit(
            self.runtime_config,
            ROI_MIN_FORWARD_M,
        ) if ROI_MIN_FORWARD_M > 0 else 0
        self.max_range_bin = radial_bin_limit(
            self.runtime_config,
            np.sqrt((ROI_LATERAL_M ** 2) + (ROI_FORWARD_M ** 2)),
        )
        self.display_range_bins = max(self.max_range_bin - self.min_range_bin, 1)
        self.detection_region = DetectionRegion(
            lateral_limit_m=ROI_LATERAL_M,
            forward_limit_m=ROI_FORWARD_M,
            min_forward_m=ROI_MIN_FORWARD_M,
            max_targets=DETECTION_MAX_TARGETS,
            allow_strongest_fallback=ALLOW_STRONGEST_FALLBACK,
            adaptive_eps_bands=DBSCAN_ADAPTIVE_EPS_BANDS,
            cluster_min_samples=DBSCAN_CLUSTER_MIN_SAMPLES,
            cluster_velocity_weight=DBSCAN_CLUSTER_VELOCITY_WEIGHT,
        )
        self.session_logger = SessionLogger(
            project_root=PROJECT_ROOT,
            log_root=LOG_ROOT,
            raw_capture_root=self.raw_capture_root,
            variant=LOG_VARIANT,
            scenario_id=LOG_SCENARIO_ID,
            input_mode=self.input_mode,
            source_capture=self.source_capture,
            notes=LOG_NOTES,
            enabled=LOG_ENABLED,
            write_raw_capture=self.write_raw_capture,
            write_processed_frames=LOG_WRITE_PROCESSED_FRAMES,
            write_render_frames=LOG_WRITE_RENDER_FRAMES,
            write_status_log=LOG_WRITE_STATUS_LOG,
            write_event_log=LOG_WRITE_EVENT_LOG,
            include_payloads=LOG_INCLUDE_PAYLOADS,
            capture_system_snapshot_enabled=LOG_CAPTURE_SYSTEM_SNAPSHOT,
            report_generation_mode=self.report_generation_mode,
        )
        self.processed_log_path = (
            self.session_logger.processed_log_path
            if LOG_ENABLED and LOG_WRITE_PROCESSED_FRAMES
            else None
        )
        self.session_logger.prepare(self.runtime_summary())

    def runtime_summary(self):
        return {
            'static_settings_path': str(STATIC_SETTINGS_PATH),
            'runtime_settings_path': str(RUNTIME_SETTINGS_PATH),
            'settings_path': str(RUNTIME_SETTINGS_PATH),
            'tuning_path': str(TUNING_PATH),
            'static_snapshot': STATIC,
            'runtime_snapshot': RUNTIME,
            'tuning_snapshot': TUNING,
            'cfg': str(CONFIG_PATH),
            'adc_sample': self.runtime_config.adc_sample,
            'chirp_loops': self.runtime_config.chirp_loops,
            'frame_length': self.runtime_config.frame_length,
            'tx_num': self.runtime_config.tx_num,
            'rx_num': self.runtime_config.rx_num,
            'virtual_antennas': self.runtime_config.virtual_antennas,
            'remove_static': self.runtime_config.remove_static,
            'invert_lateral_axis': INVERT_LATERAL_AXIS,
            'lateral_axis_sign': self.runtime_config.lateral_axis_sign,
            'range_resolution_m': round(self.runtime_config.range_resolution_m, 4),
            'max_range_m': round(self.runtime_config.max_range_m, 2),
            'roi_lateral_m': ROI_LATERAL_M,
            'roi_forward_m': ROI_FORWARD_M,
            'roi_min_forward_m': ROI_MIN_FORWARD_M,
            'allow_strongest_fallback': ALLOW_STRONGEST_FALLBACK,
            'dbscan_adaptive_eps_bands': list(DBSCAN_ADAPTIVE_EPS_BANDS),
            'detection_algorithm': dict(DETECTION_TUNING),
            'track_confirm_hits': TRACK_CONFIRM_HITS,
            'track_max_misses': TRACK_MAX_MISSES,
            'track_process_var': TRACK_PROCESS_VAR,
            'track_measurement_var': TRACK_MEASUREMENT_VAR,
            'track_range_measurement_scale': TRACK_RANGE_MEASUREMENT_SCALE,
            'track_confidence_measurement_scale': TRACK_CONFIDENCE_MEASUREMENT_SCALE,
            'track_angle_resolution_deg': round(float(np.degrees(self.track_angle_resolution_rad)), 3),
            'track_association_gate': TRACK_ASSOCIATION_GATE,
            'track_doppler_zero_guard_bins': TRACK_DOPPLER_ZERO_GUARD_BINS,
            'track_doppler_gate_bins': TRACK_DOPPLER_GATE_BINS,
            'track_doppler_cost_weight': TRACK_DOPPLER_COST_WEIGHT,
            'track_birth_suppression_radius_m': TRACK_BIRTH_SUPPRESSION_RADIUS_M,
            'track_primary_track_birth_scale': TRACK_PRIMARY_TRACK_BIRTH_SCALE,
            'track_birth_suppression_weak_radius_scale': TRACK_BIRTH_SUPPRESSION_WEAK_RADIUS_SCALE,
            'track_birth_suppression_score_ratio': TRACK_BIRTH_SUPPRESSION_SCORE_RATIO,
            'track_birth_suppression_confidence_ratio': TRACK_BIRTH_SUPPRESSION_CONFIDENCE_RATIO,
            'track_birth_suppression_doppler_bins': TRACK_BIRTH_SUPPRESSION_DOPPLER_BINS,
            'track_birth_suppression_miss_tolerance': TRACK_BIRTH_SUPPRESSION_MISS_TOLERANCE,
            'track_primary_track_hold_frames': TRACK_PRIMARY_TRACK_HOLD_FRAMES,
            'track_lateral_deadband_m': TRACK_LATERAL_DEADBAND_M,
            'track_lateral_deadband_range_scale': TRACK_LATERAL_DEADBAND_RANGE_SCALE,
            'track_lateral_smoothing_alpha': TRACK_LATERAL_SMOOTHING_ALPHA,
            'track_lateral_velocity_damping': TRACK_LATERAL_VELOCITY_DAMPING,
            'track_local_remeasurement_enabled': TRACK_LOCAL_REMEASUREMENT_ENABLED,
            'track_local_remeasurement_blend': TRACK_LOCAL_REMEASUREMENT_BLEND,
            'track_local_remeasurement_max_shift_m': TRACK_LOCAL_REMEASUREMENT_MAX_SHIFT_M,
            'track_local_remeasurement_track_bias': TRACK_LOCAL_REMEASUREMENT_TRACK_BIAS,
            'track_local_remeasurement_patch_bands': TRACK_LOCAL_REMEASUREMENT_PATCH_BANDS,
            'track_measurement_soft_gate_enabled': TRACK_MEASUREMENT_SOFT_GATE_ENABLED,
            'track_measurement_soft_gate_floor': TRACK_MEASUREMENT_SOFT_GATE_FLOOR,
            'track_measurement_soft_gate_start_m': TRACK_MEASUREMENT_SOFT_GATE_START_M,
            'track_measurement_soft_gate_full_m': TRACK_MEASUREMENT_SOFT_GATE_FULL_M,
            'track_measurement_soft_gate_range_scale': TRACK_MEASUREMENT_SOFT_GATE_RANGE_SCALE,
            'track_measurement_soft_gate_speed_scale': TRACK_MEASUREMENT_SOFT_GATE_SPEED_SCALE,
            'pipeline_queue_size': PIPELINE_QUEUE_SIZE,
            'block_track_birth_on_invalid': BLOCK_TRACK_BIRTH_ON_INVALID,
            'invalid_policy': dict(INVALID_POLICY),
            'dca_packet_size_bytes': DCA_PACKET_SIZE_BYTES,
            'dca_packet_delay_us': DCA_PACKET_DELAY_US,
            'show_tentative_tracks': SHOW_TENTATIVE_TRACKS,
            'tentative_min_confidence': TENTATIVE_MIN_CONFIDENCE,
            'tentative_min_hits': TENTATIVE_MIN_HITS,
            'display_hysteresis_frames': DISPLAY_HYSTERESIS_FRAMES,
            'display_hysteresis_confidence_floor': DISPLAY_HYSTERESIS_CONFIDENCE_FLOOR,
            'display_primary_bonus_frames': DISPLAY_PRIMARY_BONUS_FRAMES,
            'log_variant': LOG_VARIANT,
            'log_scenario_id': LOG_SCENARIO_ID,
            'log_input_mode': self.input_mode,
            'log_source_capture': self.source_capture,
            'log_replay_speed': self.replay_speed if self.input_mode == 'replay' else None,
            'log_replay_loop': self.replay_loop if self.input_mode == 'replay' else None,
            'log_notes': LOG_NOTES,
            'log_capture_duration_s': self.capture_duration_s,
            'log_enabled': LOG_ENABLED,
            'log_write_raw_capture': self.write_raw_capture,
            'log_raw_capture_root': str(self.raw_capture_root),
            'raw_capture_dir': (
                str(self.session_logger.raw_capture_dir)
                if self.write_raw_capture
                else None
            ),
            'log_write_processed_frames': LOG_WRITE_PROCESSED_FRAMES,
            'log_write_render_frames': LOG_WRITE_RENDER_FRAMES,
            'log_write_status_log': LOG_WRITE_STATUS_LOG,
            'log_write_event_log': LOG_WRITE_EVENT_LOG,
            'log_include_payloads': LOG_INCLUDE_PAYLOADS,
            'log_capture_system_snapshot': LOG_CAPTURE_SYSTEM_SNAPSHOT,
            'log_capture_stage_timing': LOG_CAPTURE_STAGE_TIMING,
            'log_report_generation_mode': self.report_generation_mode,
        }

    def log_event(self, event_type, **payload):
        self.session_logger.log_event(
            event_type,
            frame_index=self.frame_index,
            stream_started_at=self.stream_started_at,
            **payload,
        )

    def build_tracker(self):
        return MultiTargetTracker(
            process_var=TRACK_PROCESS_VAR,
            measurement_var=TRACK_MEASUREMENT_VAR,
            range_measurement_scale=TRACK_RANGE_MEASUREMENT_SCALE,
            confidence_measurement_scale=TRACK_CONFIDENCE_MEASUREMENT_SCALE,
            angle_resolution_rad=self.track_angle_resolution_rad,
            association_gate=TRACK_ASSOCIATION_GATE,
            doppler_center_bin=self.runtime_config.doppler_fft_size // 2,
            doppler_zero_guard_bins=TRACK_DOPPLER_ZERO_GUARD_BINS,
            doppler_gate_bins=TRACK_DOPPLER_GATE_BINS,
            doppler_cost_weight=TRACK_DOPPLER_COST_WEIGHT,
            min_confirmed_hits=TRACK_CONFIRM_HITS,
            max_missed_frames=TRACK_MAX_MISSES,
            report_miss_tolerance=TRACK_REPORT_MISS_TOLERANCE,
            lost_gate_factor=TRACK_LOST_GATE_FACTOR,
            tentative_gate_factor=TRACK_TENTATIVE_GATE_FACTOR,
            birth_suppression_radius_m=TRACK_BIRTH_SUPPRESSION_RADIUS_M,
            primary_track_birth_scale=TRACK_PRIMARY_TRACK_BIRTH_SCALE,
            birth_suppression_weak_radius_scale=TRACK_BIRTH_SUPPRESSION_WEAK_RADIUS_SCALE,
            birth_suppression_score_ratio=TRACK_BIRTH_SUPPRESSION_SCORE_RATIO,
            birth_suppression_confidence_ratio=TRACK_BIRTH_SUPPRESSION_CONFIDENCE_RATIO,
            birth_suppression_doppler_bins=TRACK_BIRTH_SUPPRESSION_DOPPLER_BINS,
            birth_suppression_miss_tolerance=TRACK_BIRTH_SUPPRESSION_MISS_TOLERANCE,
            primary_track_hold_frames=TRACK_PRIMARY_TRACK_HOLD_FRAMES,
            lateral_deadband_m=TRACK_LATERAL_DEADBAND_M,
            lateral_deadband_range_scale=TRACK_LATERAL_DEADBAND_RANGE_SCALE,
            lateral_smoothing_alpha=TRACK_LATERAL_SMOOTHING_ALPHA,
            lateral_velocity_damping=TRACK_LATERAL_VELOCITY_DAMPING,
            local_remeasurement_enabled=TRACK_LOCAL_REMEASUREMENT_ENABLED,
            local_remeasurement_blend=TRACK_LOCAL_REMEASUREMENT_BLEND,
            local_remeasurement_max_shift_m=TRACK_LOCAL_REMEASUREMENT_MAX_SHIFT_M,
            local_remeasurement_track_bias=TRACK_LOCAL_REMEASUREMENT_TRACK_BIAS,
            local_remeasurement_patch_bands=TRACK_LOCAL_REMEASUREMENT_PATCH_BANDS,
            measurement_soft_gate_enabled=TRACK_MEASUREMENT_SOFT_GATE_ENABLED,
            measurement_soft_gate_floor=TRACK_MEASUREMENT_SOFT_GATE_FLOOR,
            measurement_soft_gate_start_m=TRACK_MEASUREMENT_SOFT_GATE_START_M,
            measurement_soft_gate_full_m=TRACK_MEASUREMENT_SOFT_GATE_FULL_M,
            measurement_soft_gate_range_scale=TRACK_MEASUREMENT_SOFT_GATE_RANGE_SCALE,
            measurement_soft_gate_speed_scale=TRACK_MEASUREMENT_SOFT_GATE_SPEED_SCALE,
        )

    def resolve_source_capture_path(self):
        if not self.source_capture:
            raise ValueError("Replay mode requires a source_capture path.")
        candidate = resolve_project_path(PROJECT_ROOT, self.source_capture)
        if candidate.exists():
            return candidate
        shorthand_candidate = PROJECT_ROOT / 'logs' / 'raw' / self.source_capture
        if shorthand_candidate.exists():
            return shorthand_candidate
        raise FileNotFoundError(f"Replay capture directory not found: {candidate}")

    @staticmethod
    def serialize_detection(detection):
        return {
            'range_bin': int(detection.range_bin),
            'doppler_bin': int(detection.doppler_bin),
            'angle_bin': int(detection.angle_bin),
            'range_m': round(float(detection.range_m), 4),
            'angle_deg': round(float(detection.angle_deg), 3),
            'x_m': round(float(detection.x_m), 4),
            'y_m': round(float(detection.y_m), 4),
            'rdi_peak': round(float(detection.rdi_peak), 4),
            'rai_peak': round(float(detection.rai_peak), 4),
            'score': round(float(detection.score), 4),
        }

    def serialize_track(self, track):
        return {
            'track_id': int(track.track_id),
            'range_bin': int(self.range_bin_for_track(track)),
            'angle_bin': int(self.angle_bin_for_track(track)),
            'doppler_bin': int(track.doppler_bin),
            'range_m': round(float(track.range_m), 4),
            'angle_deg': round(float(track.angle_deg), 3),
            'x_m': round(float(track.x_m), 4),
            'y_m': round(float(track.y_m), 4),
            'vx_m_s': round(float(track.vx_m_s), 4),
            'vy_m_s': round(float(track.vy_m_s), 4),
            'rdi_peak': round(float(track.rdi_peak), 4),
            'rai_peak': round(float(track.rai_peak), 4),
            'score': round(float(track.score), 4),
            'confidence': round(float(track.confidence), 4),
            'age': int(track.age),
            'hits': int(track.hits),
            'misses': int(track.misses),
            'measurement_quality': round(float(track.measurement_quality), 4),
            'measurement_residual_m': round(float(track.measurement_residual_m), 4),
        }

    def log_render_snapshot(
        self,
        status_text,
        frame_packet,
        render_ts,
        skipped_frames,
        detections,
        tracker_input_count,
        display_tracks,
        tentative_tracks,
        tentative_display_tracks,
        display_held_track_count=0,
    ):
        elapsed_s = None
        if self.stream_started_at is not None:
            elapsed_s = max(frame_packet.capture_ts - self.stream_started_at, 0.0)

        capture_to_process_ms = None
        process_to_render_ms = None
        capture_to_render_ms = max((render_ts - frame_packet.capture_ts) * 1000.0, 0.0)
        if frame_packet.processed_ts is not None:
            capture_to_process_ms = max(
                (frame_packet.processed_ts - frame_packet.capture_ts) * 1000.0,
                0.0,
            )
            process_to_render_ms = max(
                (render_ts - frame_packet.processed_ts) * 1000.0,
                0.0,
            )

        record = {
            'frame_index': self.frame_index,
            'frame_id': int(frame_packet.frame_id),
            'wall_time': datetime.now().isoformat(timespec='milliseconds'),
            'elapsed_s': None if elapsed_s is None else round(elapsed_s, 4),
            'capture_ts': round(frame_packet.capture_ts, 6),
            'assembled_ts': round(frame_packet.assembled_ts, 6),
            'processed_ts': None if frame_packet.processed_ts is None else round(frame_packet.processed_ts, 6),
            'render_ts': round(render_ts, 6),
            'capture_to_process_ms': None if capture_to_process_ms is None else round(capture_to_process_ms, 3),
            'process_to_render_ms': None if process_to_render_ms is None else round(process_to_render_ms, 3),
            'capture_to_render_ms': round(capture_to_render_ms, 3),
            'packets_in_frame': int(frame_packet.packets_in_frame),
            'sequence_start': frame_packet.sequence_start,
            'sequence_end': frame_packet.sequence_end,
            'byte_count_start': frame_packet.byte_count_start,
            'byte_count_end': frame_packet.byte_count_end,
            'udp_gap_count': int(frame_packet.udp_gap_count),
            'byte_mismatch_count': int(frame_packet.byte_mismatch_count),
            'out_of_sequence_count': int(frame_packet.out_of_sequence_count),
            'invalid': bool(frame_packet.invalid),
            'invalid_reason': frame_packet.invalid_reason,
            'track_birth_blocked': bool(frame_packet.track_birth_blocked),
            'tracker_policy': frame_packet.tracker_policy,
            'skipped_render_frames': int(skipped_frames),
            'status_text': status_text,
            'candidate_count': len(detections),
            'tracker_input_count': int(tracker_input_count),
            'display_track_count': len(display_tracks),
            'display_held_track_count': int(display_held_track_count),
            'tentative_track_count': len(tentative_tracks),
            'tentative_display_track_count': len(tentative_display_tracks),
        }
        if frame_packet.stage_timings_ms:
            record['stage_timings_ms'] = {
                key: round(float(value), 3)
                for key, value in frame_packet.stage_timings_ms.items()
                if value is not None
            }
        if LOG_INCLUDE_PAYLOADS:
            record['detections'] = [self.serialize_detection(detection) for detection in detections]
            record['display_tracks'] = [self.serialize_track(track) for track in display_tracks]
            record['tentative_tracks'] = [self.serialize_track(track) for track in tentative_tracks]
            record['tentative_display_tracks'] = [self.serialize_track(track) for track in tentative_display_tracks]
        self.session_logger.write_render_record(record)

    def log_status_snapshot(
        self,
        status_text,
        frame_packet,
        render_ts,
        skipped_frames,
        detections,
        tracker_input_count,
        display_tracks,
        tentative_tracks,
        tentative_display_tracks,
        display_held_track_count=0,
    ):
        self.log_render_snapshot(
            status_text,
            frame_packet,
            render_ts,
            skipped_frames,
            detections,
            tracker_input_count,
            display_tracks,
            tentative_tracks,
            tentative_display_tracks,
            display_held_track_count=display_held_track_count,
        )

    def configure_dca1000(self):
        if self.dca_client is not None:
            self.dca_client.close()
        self.dca_client = DcaConfigClient(
            host_ip=HOST_IP,
            config_port=CONFIG_PORT,
            fpga_ip=FPGA_IP,
            fpga_port=FPGA_PORT,
            timeout_s=DCA_CONFIG_TIMEOUT_S,
            packet_size_bytes=DCA_PACKET_SIZE_BYTES,
            packet_delay_us=DCA_PACKET_DELAY_US,
            packet_delay_ticks_per_us=DCA_PACKET_DELAY_TICKS_PER_US,
            event_callback=self.log_event,
        )
        self.dca_client.configure()

    def start_workers(self):
        if (
            self.collector is not None and self.collector.is_alive()
            and self.processor is not None and self.processor.is_alive()
        ):
            self.log_event('workers_already_running')
            return
        if self.input_mode == 'replay':
            capture_path = self.resolve_source_capture_path()
            self.collector = RawCaptureReplaySource(
                'ReplaySource',
                self.raw_frame_queue,
                capture_path,
                playback_speed=self.replay_speed,
                loop=self.replay_loop,
                autostart=False,
            )
        else:
            data_address = (HOST_IP, DATA_PORT)
            raw_capture_writer = None
            if self.write_raw_capture:
                raw_capture_writer = RawFrameCaptureWriter(
                    self.session_logger.raw_capture_data_path,
                    self.session_logger.raw_capture_index_path,
                )
            self.collector = UdpListener(
                'Listener',
                self.raw_frame_queue,
                self.runtime_config.frame_length,
                data_address,
                BUFFER_SIZE,
                raw_capture_writer=raw_capture_writer,
            )
        self.processor = DataProcessor(
            'Processor',
            self.runtime_config,
            self.raw_frame_queue,
            self.processed_frame_queue,
            self.detection_region,
            self.min_range_bin,
            self.max_range_bin,
            self.build_tracker(),
            block_track_birth_on_invalid=BLOCK_TRACK_BIRTH_ON_INVALID,
            invalid_policy=INVALID_POLICY,
            processed_frame_log_path=self.processed_log_path,
            detection_params=DETECTION_TUNING,
            write_processed_frames=LOG_ENABLED and LOG_WRITE_PROCESSED_FRAMES,
            include_payloads=LOG_INCLUDE_PAYLOADS,
            capture_stage_timing=LOG_CAPTURE_STAGE_TIMING,
        )
        self.collector.daemon = True
        self.processor.daemon = True
        self.collector.start()
        self.processor.start()
        self.log_event(
            'workers_started',
            queue_size=PIPELINE_QUEUE_SIZE,
            input_mode=self.input_mode,
            source_capture=self.source_capture if self.input_mode == 'replay' else None,
            processed_log=None if self.processed_log_path is None else str(self.processed_log_path.name),
            logging_enabled=LOG_ENABLED,
            raw_capture_enabled=self.write_raw_capture,
            raw_capture_dir=(
                str(self.session_logger.raw_capture_dir)
                if self.write_raw_capture
                else None
            ),
            include_payloads=LOG_INCLUDE_PAYLOADS,
            capture_stage_timing=LOG_CAPTURE_STAGE_TIMING,
        )

    def open_radar(self):
        if self.input_mode == 'replay':
            capture_path = self.resolve_source_capture_path()
            self.stream_started_at = time.perf_counter()
            self.stop_capture_timer()
            self.waiting_for_data_logged = False
            self.first_image_logged = False
            self.frame_index = 0
            self.skipped_render_frames = 0
            self.replay_completion_requested = False
            self.replay_track_history.clear()
            self.display_hysteresis_state.clear()
            for items in self.replay_track_items.values():
                items['trail'].setData([], [])
                items['start'].setData([])
                items['current'].setData([])
            self.log_event(
                'replay_start',
                source_capture=str(capture_path),
                replay_speed=self.replay_speed,
                replay_loop=self.replay_loop,
            )
            if isinstance(self.collector, RawCaptureReplaySource):
                self.collector.start_streaming()
            self.update_figure()
            return
        self.log_event(
            'radar_open_start',
            cli_port=CLI_PORT,
            cfg=str(CONFIG_PATH),
        )
        self.radar_ctrl = SerialConfig(
            name='ConnectRadar',
            CLIPort=CLI_PORT,
            BaudRate=CLI_BAUDRATE,
        )
        self.radar_ctrl.StopRadar()
        self.radar_ctrl.SendConfig(str(CONFIG_PATH))
        self.stream_started_at = time.perf_counter()
        self.waiting_for_data_logged = False
        self.first_image_logged = False
        self.frame_index = 0
        self.skipped_render_frames = 0
        self.display_hysteresis_state.clear()
        self.log_event('radar_open_complete')
        self.arm_capture_timer()
        self.update_figure()

    def stop_capture_timer(self):
        if self.capture_stop_timer is not None and self.capture_stop_timer.isActive():
            self.capture_stop_timer.stop()

    def arm_capture_timer(self):
        self.stop_capture_timer()
        if self.capture_duration_s is None or self.capture_stop_timer is None:
            return
        duration_ms = max(int(round(self.capture_duration_s * 1000.0)), 1)
        self.capture_stop_timer.start(duration_ms)
        self.log_event(
            'capture_duration_armed',
            duration_s=self.capture_duration_s,
            duration_ms=duration_ms,
        )
        if self.ui is not None:
            self.ui.statusbar.showMessage(
                f'Live capture will auto-stop after {self.capture_duration_s:.1f}s.'
            )
        print(f'Live capture auto-stop armed for {self.capture_duration_s:.1f}s')

    def handle_capture_duration_elapsed(self):
        if self.capture_duration_s is None:
            return
        self.log_event('capture_duration_elapsed', duration_s=self.capture_duration_s)
        if self.ui is not None:
            self.ui.statusbar.showMessage(
                f'Capture duration reached ({self.capture_duration_s:.1f}s). Stopping session...'
            )
        print(
            f'Capture duration reached ({self.capture_duration_s:.1f}s). '
            'Stopping session.'
        )
        QtCore.QTimer.singleShot(0, self.app.instance().exit)

    def pull_latest_processed_frame(self):
        frame_packet = self.processed_frame_queue.get_nowait()
        skipped_frames = 0
        while True:
            try:
                frame_packet = self.processed_frame_queue.get_nowait()
                skipped_frames += 1
            except Empty:
                break
        self.skipped_render_frames += skipped_frames
        return frame_packet, skipped_frames

    def update_figure(self):
        try:
            frame_packet, skipped_frames = self.pull_latest_processed_frame()
        except Empty:
            if (
                self.input_mode == 'replay'
                and self.stream_started_at is not None
                and not self.replay_completion_requested
                and self.collector is not None
                and not self.collector.is_alive()
                and self.processor is not None
                and not self.processor.is_alive()
            ):
                self.replay_completion_requested = True
                self.log_event('replay_complete')
                self.ui.statusbar.showMessage(
                    'Replay complete. Generating trajectory replay report...'
                )
                QtCore.QTimer.singleShot(0, self.app.instance().exit)
                return
            if (
                self.stream_started_at is not None
                and not self.first_image_logged
                and not self.waiting_for_data_logged
                and (time.perf_counter() - self.stream_started_at) > 8.0
            ):
                print(
                    "No processed frames yet. If the images stay black, "
                    "check LVDS streaming, DCA1000 link, and COM port settings."
                )
                self.waiting_for_data_logged = True
                self.log_event('waiting_for_processed_frames', threshold_s=8.0)
            QtCore.QTimer.singleShot(20, self.update_figure)
            return

        rdi = frame_packet.rdi
        rai = frame_packet.rai
        detections = list(frame_packet.detections)
        if rdi is None or rai is None:
            QtCore.QTimer.singleShot(5, self.update_figure)
            return

        if not self.first_image_logged:
            print("Displaying first processed frame")
            self.first_image_logged = True
            self.log_event('first_rendered_frame', frame_id=int(frame_packet.frame_id))

        cropped_rdi = rdi[self.min_range_bin:self.max_range_bin, :]
        roi_rai = apply_cartesian_roi_to_rai(
            rai,
            self.runtime_config,
            lateral_limit_m=ROI_LATERAL_M,
            forward_limit_m=ROI_FORWARD_M,
            min_forward_m=ROI_MIN_FORWARD_M,
        )

        if self.img_rdi is not None:
            self.img_rdi.setImage(cropped_rdi.T, axisOrder='row-major')
        if self.img_rai is not None:
            self.img_rai.setImage(np.flipud(roi_rai.T), axisOrder='row-major')
        render_ts = time.perf_counter()
        self.update_detection_overlay(frame_packet, detections, render_ts, skipped_frames)
        QtCore.QTimer.singleShot(1, self.update_figure)

    @staticmethod
    def display_track_sort_key(track):
        return (
            bool(track.is_primary),
            float(track.confidence),
            float(track.score),
            int(track.hits),
        )

    def select_display_tracks(self, confirmed_tracks):
        current_frame = int(self.frame_index)
        selected_tracks = []
        selected_ids = set()
        held_track_count = 0

        for track in confirmed_tracks:
            if (
                track.misses <= TRACK_REPORT_MISS_TOLERANCE
                and track.confidence >= DISPLAY_MIN_CONFIDENCE
            ):
                selected_tracks.append(track)
                selected_ids.add(int(track.track_id))
                self.display_hysteresis_state[int(track.track_id)] = {
                    'track': replace(track),
                    'last_seen_frame': current_frame,
                    'is_primary': bool(track.is_primary),
                }

        for track in confirmed_tracks:
            track_id = int(track.track_id)
            if track_id in selected_ids:
                continue
            state = self.display_hysteresis_state.get(track_id)
            if not state:
                continue
            bonus_frames = DISPLAY_PRIMARY_BONUS_FRAMES if (
                bool(track.is_primary) or bool(state.get('is_primary', False))
            ) else 0
            grace_frames = DISPLAY_HYSTERESIS_FRAMES + bonus_frames
            frames_since = current_frame - int(state.get('last_seen_frame', current_frame))
            if (
                grace_frames > 0
                and 0 < frames_since <= grace_frames
                and track.confidence >= DISPLAY_HYSTERESIS_CONFIDENCE_FLOOR
            ):
                held_track = replace(track)
                selected_tracks.append(held_track)
                selected_ids.add(track_id)
                held_track_count += 1
                self.display_hysteresis_state[track_id] = {
                    'track': held_track,
                    'last_seen_frame': int(state.get('last_seen_frame', current_frame)),
                    'is_primary': bool(track.is_primary),
                }

        current_track_ids = {int(track.track_id) for track in confirmed_tracks}
        for track_id, state in list(self.display_hysteresis_state.items()):
            if track_id in selected_ids or track_id in current_track_ids:
                continue
            bonus_frames = DISPLAY_PRIMARY_BONUS_FRAMES if bool(state.get('is_primary', False)) else 0
            grace_frames = DISPLAY_HYSTERESIS_FRAMES + bonus_frames
            frames_since = current_frame - int(state.get('last_seen_frame', current_frame))
            if grace_frames <= 0 or frames_since <= 0 or frames_since > grace_frames:
                continue
            last_track = state.get('track')
            if last_track is None:
                continue
            held_track = replace(last_track, misses=int(last_track.misses) + frames_since)
            selected_tracks.append(held_track)
            selected_ids.add(track_id)
            held_track_count += 1

        max_keep_frames = DISPLAY_HYSTERESIS_FRAMES + DISPLAY_PRIMARY_BONUS_FRAMES + 2
        self.display_hysteresis_state = {
            track_id: state
            for track_id, state in self.display_hysteresis_state.items()
            if current_frame - int(state.get('last_seen_frame', current_frame)) <= max_keep_frames
        }
        return (
            sorted(selected_tracks, key=self.display_track_sort_key, reverse=True),
            held_track_count,
        )

    def update_detection_overlay(self, frame_packet, detections, render_ts, skipped_frames):
        tracker_input_count = int(frame_packet.tracker_input_count)
        confirmed_tracks = list(frame_packet.confirmed_tracks)
        tentative_tracks = list(frame_packet.tentative_tracks)
        display_tracks, display_held_track_count = self.select_display_tracks(confirmed_tracks)
        rdi_points = [
            {
                'pos': (
                    self.range_bin_for_track(track) - self.min_range_bin,
                    track.doppler_bin,
                ),
            }
            for track in display_tracks
        ]
        rai_points = [
            {
                'pos': (
                    self.range_bin_for_track(track) - self.min_range_bin,
                    self.runtime_config.angle_fft_size - 1 - self.angle_bin_for_track(track),
                ),
            }
            for track in display_tracks
        ]
        tentative_display_tracks = []
        if SHOW_TENTATIVE_TRACKS:
            confirmed_track_ids = {track.track_id for track in display_tracks}
            tentative_display_tracks = [
                track for track in tentative_tracks
                if track.track_id not in confirmed_track_ids
                and track.misses == 0
                and track.hits >= TENTATIVE_MIN_HITS
                and track.confidence >= TENTATIVE_MIN_CONFIDENCE
            ]
            tentative_display_tracks = sorted(
                tentative_display_tracks,
                key=self.display_track_sort_key,
                reverse=True,
            )

        tentative_rdi_points = [
            {
                'pos': (
                    self.range_bin_for_track(track) - self.min_range_bin,
                    track.doppler_bin,
                ),
            }
            for track in tentative_display_tracks
        ]
        tentative_rai_points = [
            {
                'pos': (
                    self.range_bin_for_track(track) - self.min_range_bin,
                    self.runtime_config.angle_fft_size - 1 - self.angle_bin_for_track(track),
                ),
            }
            for track in tentative_display_tracks
        ]

        if self.rdi_scatter is not None:
            self.rdi_scatter.setData(rdi_points)
        if self.rai_scatter is not None:
            self.rai_scatter.setData(rai_points)
        if self.rdi_tentative_scatter is not None:
            self.rdi_tentative_scatter.setData(tentative_rdi_points)
        if self.rai_tentative_scatter is not None:
            self.rai_tentative_scatter.setData(tentative_rai_points)

        if self.replay_plot is not None:
            self.update_replay_trajectory_plot(confirmed_tracks, tentative_tracks)
        else:
            self.spatial_view.update(display_tracks, tentative_display_tracks)

        integrity_suffix = ''
        if frame_packet.invalid:
            integrity_suffix = (
                f' | invalid gaps={frame_packet.udp_gap_count} '
                f'seq={frame_packet.out_of_sequence_count} '
                f'byte={frame_packet.byte_mismatch_count}'
            )
            if frame_packet.track_birth_blocked:
                integrity_suffix += ' births=off'
            if frame_packet.tracker_policy == 'drop':
                integrity_suffix += ' tracker=drop'
        elif skipped_frames:
            integrity_suffix = f' | skipped={skipped_frames}'

        if display_tracks:
            lead_track = display_tracks[0]
            status_text = (
                'Candidates/Tracks: '
                f'{tracker_input_count}/{len(display_tracks)} | lead '
                f'id={lead_track.track_id} '
                f'r={lead_track.range_m:.2f}m '
                f'angle={lead_track.angle_deg:.1f}deg '
                f'x={lead_track.x_m:.2f}m '
                f'y={lead_track.y_m:.2f}m'
                f'{integrity_suffix}'
            )
            if tentative_display_tracks:
                status_text += f' | tentative={len(tentative_display_tracks)}'
            if display_held_track_count:
                status_text += f' | held={display_held_track_count}'
            self.ui.statusbar.showMessage(status_text)
        else:
            status_text = (
                'Candidates/Tracks: '
                f'{tracker_input_count}/0 | tentative={len(tentative_display_tracks)}'
                f'{integrity_suffix}'
            )
            self.ui.statusbar.showMessage(status_text)

        self.frame_index += 1
        self.log_status_snapshot(
            status_text,
            frame_packet,
            render_ts,
            skipped_frames,
            detections,
            tracker_input_count,
            display_tracks,
            tentative_tracks,
            tentative_display_tracks,
            display_held_track_count=display_held_track_count,
        )

    def range_bin_for_track(self, track):
        return int(
            np.clip(
                np.argmin(np.abs(self.runtime_config.range_axis_m - track.range_m)),
                0,
                self.runtime_config.range_fft_size - 1,
            )
        )

    def angle_bin_for_track(self, track):
        angle_rad = np.radians(track.angle_deg)
        return int(
            np.clip(
                np.argmin(np.abs(self.runtime_config.angle_axis_rad - angle_rad)),
                0,
                self.runtime_config.angle_fft_size - 1,
            )
        )

    def estimate_track_angle_resolution_rad(self):
        angle_axis = np.asarray(self.runtime_config.angle_axis_rad, dtype=np.float64)
        if angle_axis.size < 2:
            return 0.0
        diffs = np.abs(np.diff(angle_axis))
        finite_diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
        if finite_diffs.size == 0:
            return 0.0
        center_start = max((finite_diffs.size // 2) - 2, 0)
        center_end = min(center_start + 4, finite_diffs.size)
        center_diffs = finite_diffs[center_start:center_end]
        return float(np.mean(center_diffs))

    def replay_track_color(self, track_id, tentative=False):
        palette = [
            (31, 119, 180),
            (255, 127, 14),
            (44, 160, 44),
            (214, 39, 40),
            (148, 103, 189),
            (140, 86, 75),
            (227, 119, 194),
            (127, 127, 127),
        ]
        r, g, b = palette[(max(int(track_id), 1) - 1) % len(palette)]
        alpha = 140 if tentative else 230
        return r, g, b, alpha

    def setup_replay_trajectory_plot(self):
        self.ui.label.setGeometry(QtCore.QRect(220, 18, 360, 36))
        self.ui.label.setText('Radar XY Trajectory Replay')
        self.ui.label_2.hide()
        self.ui.graphicsView.setGeometry(QtCore.QRect(30, 60, 741, 520))
        self.ui.graphicsView_2.hide()
        self.ui.graphicsView.setBackground('w')
        self.ui.pushButton_start.setGeometry(QtCore.QRect(30, 600, 151, 31))
        self.ui.pushButton_exit.setGeometry(QtCore.QRect(680, 600, 91, 31))

        self.replay_plot = self.ui.graphicsView.addPlot(row=0, col=0)
        self.replay_plot.showGrid(x=True, y=True, alpha=0.25)
        self.replay_plot.setLabel('bottom', 'x: radar-left (-) / radar-right (+) (m)')
        self.replay_plot.setLabel('left', 'y: forward (m)')
        self.replay_plot.setXRange(-ROI_LATERAL_M, ROI_LATERAL_M, padding=0.04)
        self.replay_plot.setYRange(0.0, ROI_FORWARD_M, padding=0.04)
        self.replay_plot.hideButtons()
        self.replay_plot.setMenuEnabled(False)
        self.replay_plot.getAxis('bottom').setPen(pg.mkPen(90, 110, 140))
        self.replay_plot.getAxis('left').setPen(pg.mkPen(90, 110, 140))
        self.replay_plot.getAxis('bottom').setTextPen(pg.mkColor(90, 110, 140))
        self.replay_plot.getAxis('left').setTextPen(pg.mkColor(90, 110, 140))

        center_line = pg.InfiniteLine(pos=0.0, angle=90, pen=pg.mkPen(120, 144, 180, width=2))
        center_line.setZValue(-20)
        self.replay_plot.addItem(center_line)

        self.replay_origin_scatter = pg.ScatterPlotItem(
            x=[0.0],
            y=[0.0],
            pen=pg.mkPen(10, 15, 25, width=2),
            brush=pg.mkBrush(10, 15, 25),
            size=14,
        )
        self.replay_plot.addItem(self.replay_origin_scatter)
        self.replay_origin_label = pg.TextItem(text='radar', color=(40, 45, 55))
        self.replay_origin_label.setPos(0.02, 0.05)
        self.replay_plot.addItem(self.replay_origin_label)

    def ensure_replay_track_items(self, track_id, tentative=False):
        item_key = int(track_id)
        if item_key in self.replay_track_items:
            return self.replay_track_items[item_key]

        color = self.replay_track_color(track_id, tentative=tentative)
        pen = pg.mkPen(color, width=2.0 if not tentative else 1.5)
        trail_item = self.replay_plot.plot([], [], pen=pen)
        start_item = pg.ScatterPlotItem(
            pen=pg.mkPen(color[:3], width=2),
            brush=pg.mkBrush(0, 0, 0, 0),
            size=10,
        )
        current_item = pg.ScatterPlotItem(
            pen=pg.mkPen(color[:3], width=1.5),
            brush=pg.mkBrush(*color[:3], 210 if not tentative else 140),
            size=10 if not tentative else 8,
        )
        self.replay_plot.addItem(start_item)
        self.replay_plot.addItem(current_item)
        self.replay_track_items[item_key] = {
            'trail': trail_item,
            'start': start_item,
            'current': current_item,
        }
        return self.replay_track_items[item_key]

    def update_replay_trajectory_plot(self, confirmed_tracks, tentative_tracks):
        if self.replay_plot is None:
            return

        active_keys = set()
        plotted_tracks = []
        if confirmed_tracks:
            plotted_tracks.extend((track, False) for track in confirmed_tracks)
        else:
            plotted_tracks.extend((track, True) for track in tentative_tracks)

        for track, tentative in plotted_tracks:
            item_key = int(track.track_id)
            history_state = self.replay_track_history.setdefault(
                item_key,
                {'points': [], 'last_frame_index': None},
            )
            last_frame_index = history_state['last_frame_index']
            if (
                last_frame_index is not None
                and self.frame_index - last_frame_index > 1
                and history_state['points']
                and history_state['points'][-1] != (np.nan, np.nan)
            ):
                history_state['points'].append((np.nan, np.nan))
            history_state['points'].append((float(track.x_m), float(track.y_m)))
            history_state['last_frame_index'] = self.frame_index

            if len(history_state['points']) > 480:
                history_state['points'] = history_state['points'][-480:]

            items = self.ensure_replay_track_items(track.track_id, tentative=tentative)
            points = np.asarray(history_state['points'], dtype=float)
            if points.size:
                items['trail'].setData(points[:, 0], points[:, 1])
                valid_points = points[~np.isnan(points).any(axis=1)]
                if valid_points.size:
                    start_x, start_y = valid_points[0]
                    items['start'].setData([{'pos': (float(start_x), float(start_y))}])
                    items['current'].setData([{'pos': (float(track.x_m), float(track.y_m))}])
                else:
                    items['start'].setData([])
                    items['current'].setData([])
            else:
                items['trail'].setData([], [])
                items['start'].setData([])
                items['current'].setData([])
            active_keys.add(item_key)

        for item_key, items in self.replay_track_items.items():
            if item_key in active_keys:
                continue
            items['current'].setData([])

    def build_window(self):
        self.app = QtWidgets.QApplication(sys.argv)
        self.main_window = QtWidgets.QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.main_window)
        self.main_window.resize(802, 680)
        if self.input_mode == 'replay':
            self.main_window.setWindowTitle('Replay Radar')
            self.ui.pushButton_start.setText('Start Replay')
            self.setup_replay_trajectory_plot()
        else:
            self.ui.label.setGeometry(QtCore.QRect(110, 235, 211, 41))
            self.ui.label_2.setGeometry(QtCore.QRect(500, 235, 211, 41))
            self.ui.graphicsView.setGeometry(QtCore.QRect(30, 275, 361, 300))
            self.ui.graphicsView_2.setGeometry(QtCore.QRect(410, 275, 361, 300))
            self.ui.pushButton_start.setGeometry(QtCore.QRect(30, 600, 151, 31))
            self.ui.pushButton_exit.setGeometry(QtCore.QRect(680, 600, 91, 31))

            self.spatial_label = QtWidgets.QLabel(self.ui.centralwidget)
            self.spatial_label.setGeometry(QtCore.QRect(255, 8, 300, 32))
            spatial_font = self.ui.label.font()
            self.spatial_label.setFont(spatial_font)
            self.spatial_view.attach(self.ui.centralwidget, self.spatial_label)

            self.ui.label.setText('Moving Range-Doppler')
            self.ui.label_2.setText('Moving Range-Angle')
            self.ui.pushButton_start.setText('Send Radar Config')
            view_rdi = self.ui.graphicsView.addViewBox()
            view_rai = self.ui.graphicsView_2.addViewBox()
            view_rdi.setAspectLocked(False)
            view_rai.setAspectLocked(False)

            self.img_rdi = pg.ImageItem(border='w')
            self.img_rai = pg.ImageItem(border='w')
            self.rdi_scatter = pg.ScatterPlotItem(
                pen=pg.mkPen(255, 60, 60, width=2),
                brush=pg.mkBrush(0, 0, 0, 0),
                size=14,
            )
            self.rai_scatter = pg.ScatterPlotItem(
                pen=pg.mkPen(255, 60, 60, width=2),
                brush=pg.mkBrush(0, 0, 0, 0),
                size=14,
            )
            self.rdi_tentative_scatter = pg.ScatterPlotItem(
                pen=pg.mkPen(255, 205, 64, width=1.5),
                brush=pg.mkBrush(0, 0, 0, 0),
                size=11,
            )
            self.rai_tentative_scatter = pg.ScatterPlotItem(
                pen=pg.mkPen(255, 205, 64, width=1.5),
                brush=pg.mkBrush(0, 0, 0, 0),
                size=11,
            )
            lookup_table = build_heatmap_lookup_table()
            self.img_rdi.setLookupTable(lookup_table)
            self.img_rai.setLookupTable(lookup_table)
            view_rdi.addItem(self.img_rdi)
            view_rai.addItem(self.img_rai)
            view_rdi.addItem(self.rdi_tentative_scatter)
            view_rai.addItem(self.rai_tentative_scatter)
            view_rdi.addItem(self.rdi_scatter)
            view_rai.addItem(self.rai_scatter)
            view_rdi.setRange(
                QtCore.QRectF(
                    0,
                    0,
                    self.display_range_bins,
                    self.runtime_config.doppler_fft_size,
                )
            )
            view_rai.setRange(
                QtCore.QRectF(
                    0,
                    0,
                    self.display_range_bins,
                    self.runtime_config.angle_fft_size,
                )
            )

        self.ui.pushButton_start.clicked.connect(self.open_radar)
        self.ui.pushButton_exit.clicked.connect(self.app.instance().exit)
        self.capture_stop_timer = QtCore.QTimer(self.main_window)
        self.capture_stop_timer.setSingleShot(True)
        self.capture_stop_timer.timeout.connect(self.handle_capture_duration_elapsed)
        self.main_window.show()
        if (
            self.input_mode != 'replay'
            and not self.spatial_view.available
            and self.spatial_view.import_error is not None
        ):
            self.ui.statusbar.showMessage(
                f'3D view unavailable: {self.spatial_view.import_error}'
            )
            self.log_event('opengl_unavailable', error=str(self.spatial_view.import_error))
        return self.app

    def shutdown(self):
        self.log_event('shutdown_start')
        self.stop_capture_timer()
        if self.collector is not None:
            close_collector = getattr(self.collector, 'close', None)
            if callable(close_collector):
                close_collector()
        if self.dca_client is not None:
            self.dca_client.stop_stream()
            self.dca_client.close()
            self.dca_client = None

        if self.radar_ctrl is not None:
            try:
                self.radar_ctrl.StopRadar()
            except OSError:
                pass
        if self.processor is not None:
            close_processor = getattr(self.processor, 'close', None)
            if callable(close_processor):
                close_processor()
        if self.collector is not None and self.collector.is_alive():
            self.collector.join(timeout=1.0)
        if self.processor is not None and self.processor.is_alive():
            self.processor.join(timeout=1.0)
        self.session_logger.close(
            frame_index=self.frame_index,
            skipped_render_frames_total=self.skipped_render_frames,
        )

    def run(self):
        print('Runtime config:', self.runtime_summary())
        try:
            if self.hardware_enabled:
                self.configure_dca1000()
            self.start_workers()
            app = self.build_window()
            if self.input_mode == 'replay' and self.auto_start:
                QtCore.QTimer.singleShot(0, self.open_radar)
            app.instance().exec_()
        except Exception as exc:
            self.log_event('session_error', error=repr(exc))
            raise
        finally:
            self.shutdown()


if __name__ == '__main__':
    MotionViewer().run()
