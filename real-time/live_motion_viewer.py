import os
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
from tools.runtime_core.real_time_process import DataProcessor, UdpListener
from tools.runtime_core.runtime_settings import load_runtime_settings
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
}
LOG_ROOT = PROJECT_ROOT / 'logs' / 'live_motion_viewer'
SPATIAL_VIEW_HEIGHT = int(STATIC['spatial_view']['height'])
SPATIAL_VIEW_Y = int(STATIC['spatial_view']['y'])
SPATIAL_POINT_BASE_Z_M = float(STATIC['spatial_view']['point_base_z_m'])
SPATIAL_POINT_CONFIDENCE_SCALE_M = float(STATIC['spatial_view']['point_confidence_scale_m'])
SHOW_TENTATIVE_TRACKS = bool(TUNING['visualization']['show_tentative_tracks'])
TENTATIVE_MIN_CONFIDENCE = float(TUNING['visualization']['tentative_min_confidence'])
TENTATIVE_MIN_HITS = int(TUNING['visualization']['tentative_min_hits'])
LOG_VARIANT = str(RUNTIME['logging']['variant'])
LOG_SCENARIO_ID = str(RUNTIME['logging']['scenario_id'])
LOG_INPUT_MODE = str(RUNTIME['logging']['input_mode'])
LOG_SOURCE_CAPTURE = str(RUNTIME['logging']['source_capture'])
LOG_NOTES = str(RUNTIME['logging']['notes'])
LOG_ENABLED = bool(RUNTIME['logging'].get('enabled', True))
LOG_WRITE_PROCESSED_FRAMES = bool(RUNTIME['logging'].get('write_processed_frames', True))
LOG_WRITE_RENDER_FRAMES = bool(RUNTIME['logging'].get('write_render_frames', True))
LOG_WRITE_STATUS_LOG = bool(RUNTIME['logging'].get('write_status_log', True))
LOG_WRITE_EVENT_LOG = bool(RUNTIME['logging'].get('write_event_log', True))
LOG_INCLUDE_PAYLOADS = bool(RUNTIME['logging'].get('include_payloads', True))
LOG_CAPTURE_SYSTEM_SNAPSHOT = bool(RUNTIME['logging'].get('capture_system_snapshot', True))
LOG_CAPTURE_STAGE_TIMING = bool(RUNTIME['logging'].get('capture_stage_timing', True))
LOG_REPORT_GENERATION_MODE = str(RUNTIME['logging'].get('report_generation_mode', 'deferred'))

class MotionViewer:
    def __init__(self):
        self.runtime_config = parse_runtime_config(
            CONFIG_PATH,
            remove_static=REMOVE_STATIC,
            doppler_guard_bins=DOPPLER_GUARD_BINS,
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
            variant=LOG_VARIANT,
            scenario_id=LOG_SCENARIO_ID,
            input_mode=LOG_INPUT_MODE,
            source_capture=LOG_SOURCE_CAPTURE,
            notes=LOG_NOTES,
            enabled=LOG_ENABLED,
            write_processed_frames=LOG_WRITE_PROCESSED_FRAMES,
            write_render_frames=LOG_WRITE_RENDER_FRAMES,
            write_status_log=LOG_WRITE_STATUS_LOG,
            write_event_log=LOG_WRITE_EVENT_LOG,
            include_payloads=LOG_INCLUDE_PAYLOADS,
            capture_system_snapshot_enabled=LOG_CAPTURE_SYSTEM_SNAPSHOT,
            report_generation_mode=LOG_REPORT_GENERATION_MODE,
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
            'tx_num': self.runtime_config.tx_num,
            'rx_num': self.runtime_config.rx_num,
            'virtual_antennas': self.runtime_config.virtual_antennas,
            'remove_static': self.runtime_config.remove_static,
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
            'pipeline_queue_size': PIPELINE_QUEUE_SIZE,
            'block_track_birth_on_invalid': BLOCK_TRACK_BIRTH_ON_INVALID,
            'invalid_policy': dict(INVALID_POLICY),
            'dca_packet_size_bytes': DCA_PACKET_SIZE_BYTES,
            'dca_packet_delay_us': DCA_PACKET_DELAY_US,
            'show_tentative_tracks': SHOW_TENTATIVE_TRACKS,
            'tentative_min_confidence': TENTATIVE_MIN_CONFIDENCE,
            'tentative_min_hits': TENTATIVE_MIN_HITS,
            'log_variant': LOG_VARIANT,
            'log_scenario_id': LOG_SCENARIO_ID,
            'log_input_mode': LOG_INPUT_MODE,
            'log_source_capture': LOG_SOURCE_CAPTURE,
            'log_notes': LOG_NOTES,
            'log_enabled': LOG_ENABLED,
            'log_write_processed_frames': LOG_WRITE_PROCESSED_FRAMES,
            'log_write_render_frames': LOG_WRITE_RENDER_FRAMES,
            'log_write_status_log': LOG_WRITE_STATUS_LOG,
            'log_write_event_log': LOG_WRITE_EVENT_LOG,
            'log_include_payloads': LOG_INCLUDE_PAYLOADS,
            'log_capture_system_snapshot': LOG_CAPTURE_SYSTEM_SNAPSHOT,
            'log_capture_stage_timing': LOG_CAPTURE_STAGE_TIMING,
            'log_report_generation_mode': LOG_REPORT_GENERATION_MODE,
        }

    def log_event(self, event_type, **payload):
        self.session_logger.log_event(
            event_type,
            frame_index=self.frame_index,
            stream_started_at=self.stream_started_at,
            **payload,
        )

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
        data_address = (HOST_IP, DATA_PORT)
        self.collector = UdpListener(
            'Listener',
            self.raw_frame_queue,
            self.runtime_config.frame_length,
            data_address,
            BUFFER_SIZE,
        )
        self.processor = DataProcessor(
            'Processor',
            self.runtime_config,
            self.raw_frame_queue,
            self.processed_frame_queue,
            self.detection_region,
            self.min_range_bin,
            self.max_range_bin,
            MultiTargetTracker(
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
            ),
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
            processed_log=None if self.processed_log_path is None else str(self.processed_log_path.name),
            logging_enabled=LOG_ENABLED,
            include_payloads=LOG_INCLUDE_PAYLOADS,
            capture_stage_timing=LOG_CAPTURE_STAGE_TIMING,
        )

    def open_radar(self):
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
        self.log_event('radar_open_complete')
        self.update_figure()

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

        self.img_rdi.setImage(cropped_rdi.T, axisOrder='row-major')
        self.img_rai.setImage(np.flipud(roi_rai.T), axisOrder='row-major')
        render_ts = time.perf_counter()
        self.update_detection_overlay(frame_packet, detections, render_ts, skipped_frames)
        QtCore.QTimer.singleShot(1, self.update_figure)

    def update_detection_overlay(self, frame_packet, detections, render_ts, skipped_frames):
        tracker_input_count = int(frame_packet.tracker_input_count)
        confirmed_tracks = list(frame_packet.confirmed_tracks)
        tentative_tracks = list(frame_packet.tentative_tracks)
        display_tracks = [
            track for track in confirmed_tracks
            if track.misses <= TRACK_REPORT_MISS_TOLERANCE
            and track.confidence >= DISPLAY_MIN_CONFIDENCE
        ]
        display_tracks = sorted(
            display_tracks,
            key=lambda track: (track.confidence, track.score, track.hits),
            reverse=True,
        )
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
                key=lambda track: (track.confidence, track.score, track.hits),
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

        self.rdi_scatter.setData(rdi_points)
        self.rai_scatter.setData(rai_points)
        self.rdi_tentative_scatter.setData(tentative_rdi_points)
        self.rai_tentative_scatter.setData(tentative_rai_points)
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

    def build_window(self):
        self.app = QtWidgets.QApplication(sys.argv)
        self.main_window = QtWidgets.QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.main_window)
        self.main_window.resize(802, 680)

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
        self.main_window.show()
        if not self.spatial_view.available and self.spatial_view.import_error is not None:
            self.ui.statusbar.showMessage(
                f'3D view unavailable: {self.spatial_view.import_error}'
            )
            self.log_event('opengl_unavailable', error=str(self.spatial_view.import_error))
        return self.app

    def shutdown(self):
        self.log_event('shutdown_start')
        if self.dca_client is not None:
            self.dca_client.stop_stream()
            self.dca_client.close()
            self.dca_client = None

        if self.radar_ctrl is not None:
            try:
                self.radar_ctrl.StopRadar()
            except OSError:
                pass
        self.session_logger.close(
            frame_index=self.frame_index,
            skipped_render_frames_total=self.skipped_render_frames,
        )

    def run(self):
        print('Runtime config:', self.runtime_summary())
        try:
            self.configure_dca1000()
            self.start_workers()
            app = self.build_window()
            app.instance().exec_()
        except Exception as exc:
            self.log_event('session_error', error=repr(exc))
            raise
        finally:
            self.shutdown()


if __name__ == '__main__':
    MotionViewer().run()
