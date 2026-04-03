from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

from tools.diagnostics.system_snapshot import capture_system_snapshot


def _run_git_command(project_root: Path, *args):
    try:
        result = subprocess.run(
            ['git', *args],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


class SessionLogger:
    def __init__(
        self,
        *,
        project_root: str | Path,
        log_root: str | Path,
        variant: str,
        scenario_id: str,
        input_mode: str,
        source_capture: str,
        notes: str,
        enabled: bool = True,
        write_processed_frames: bool = True,
        write_render_frames: bool = True,
        write_status_log: bool = True,
        write_event_log: bool = True,
        include_payloads: bool = True,
        capture_system_snapshot_enabled: bool = True,
        report_generation_mode: str = "deferred",
    ):
        self.project_root = Path(project_root)
        self.log_root = Path(log_root)
        self.created_at = datetime.now()
        self.session_dir = self.log_root / self.created_at.strftime('%Y%m%d_%H%M%S')
        self.session_meta_path = self.session_dir / 'session_meta.json'
        self.processed_log_path = self.session_dir / 'processed_frames.jsonl'
        self.render_log_path = self.session_dir / 'render_frames.jsonl'
        self.event_log_path = self.session_dir / 'event_log.jsonl'
        self.status_log_path = self.session_dir / 'status_log.jsonl'
        self.runtime_config_path = self.session_dir / 'runtime_config.json'
        self.system_snapshot_path = self.session_dir / 'system_snapshot.json'
        self.report_generation_log_path = self.session_dir / 'report_generation.log'

        self.variant = str(variant)
        self.scenario_id = str(scenario_id)
        self.input_mode = str(input_mode)
        self.source_capture = str(source_capture)
        self.notes = str(notes)
        self.enabled = bool(enabled)
        self.write_processed_frames = bool(write_processed_frames)
        self.write_render_frames = bool(write_render_frames)
        self.write_status_log = bool(write_status_log)
        self.write_event_log = bool(write_event_log)
        self.include_payloads = bool(include_payloads)
        self.capture_system_snapshot_enabled = bool(capture_system_snapshot_enabled)
        self.report_generation_mode = str(report_generation_mode or "deferred").strip().lower()

        self.render_log_file = None
        self.status_log_file = None
        self.event_log_file = None
        self.session_metadata = self.build_session_metadata()

    def build_session_metadata(self):
        git_commit = _run_git_command(self.project_root, 'rev-parse', '--short', 'HEAD')
        git_branch = _run_git_command(self.project_root, 'rev-parse', '--abbrev-ref', 'HEAD')
        git_status = _run_git_command(self.project_root, 'status', '--short')
        return {
            'schema_version': 4,
            'session_id': self.session_dir.name,
            'created_at': self.created_at.isoformat(timespec='seconds'),
            'variant': self.variant,
            'scenario_id': self.scenario_id,
            'input_mode': self.input_mode,
            'source_capture': self.source_capture,
            'notes': self.notes,
            'project_root': str(self.project_root),
            'git_commit': git_commit,
            'git_branch': git_branch,
            'git_dirty': bool(git_status),
            'logging': {
                'enabled': self.enabled,
                'write_processed_frames': self.write_processed_frames,
                'write_render_frames': self.write_render_frames,
                'write_status_log': self.write_status_log,
                'write_event_log': self.write_event_log,
                'include_payloads': self.include_payloads,
                'capture_system_snapshot': self.capture_system_snapshot_enabled,
                'report_generation_mode': self.report_generation_mode,
            },
        }

    def _normalized_report_generation_mode(self):
        if self.report_generation_mode in {'inline', 'manual', 'deferred'}:
            return self.report_generation_mode
        return 'deferred'

    def _report_generation_command(self):
        return [
            sys.executable,
            '-m',
            'tools.diagnostics.log_html_reports',
            str(self.session_dir),
        ]

    def _report_generation_command_text(self):
        return subprocess.list2cmdline(self._report_generation_command())

    def _launch_deferred_report_generation(self):
        command = self._report_generation_command()
        creation_flags = 0
        for flag_name in ('CREATE_NO_WINDOW', 'DETACHED_PROCESS', 'CREATE_NEW_PROCESS_GROUP'):
            creation_flags |= getattr(subprocess, flag_name, 0)

        with self.report_generation_log_path.open('w', encoding='utf-8') as log_file:
            subprocess.Popen(
                command,
                cwd=self.project_root,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                creationflags=creation_flags,
            )
        return self.report_generation_log_path

    def _generate_reports_inline(self):
        from tools.diagnostics.log_html_reports import generate_reports

        report_result = generate_reports(self.session_dir)
        dashboard_path = report_result.get('dashboard_path')
        if dashboard_path is not None:
            print(f'Generated HTML log dashboard: {dashboard_path}')

    def prepare(self, runtime_summary: dict):
        if not self.enabled:
            print('Session logging disabled by runtime settings')
            return

        self.session_dir.mkdir(parents=True, exist_ok=True)
        with self.session_meta_path.open('w', encoding='utf-8') as meta_file:
            json.dump(self.session_metadata, meta_file, indent=2, ensure_ascii=False)
        with self.runtime_config_path.open('w', encoding='utf-8') as runtime_file:
            json.dump(runtime_summary, runtime_file, indent=2, ensure_ascii=False)

        if self.write_render_frames:
            self.render_log_file = self.render_log_path.open('a', encoding='utf-8', buffering=1)
        if self.write_status_log:
            self.status_log_file = self.status_log_path.open('a', encoding='utf-8', buffering=1)
        if self.write_event_log:
            self.event_log_file = self.event_log_path.open('a', encoding='utf-8', buffering=1)

        system_snapshot = None
        if self.capture_system_snapshot_enabled:
            network_snapshot = (runtime_summary.get('static_snapshot') or {}).get('network') or {}
            expected_host_ip = network_snapshot.get('host_ip')
            system_snapshot = capture_system_snapshot(expected_host_ip=expected_host_ip)
            with self.system_snapshot_path.open('w', encoding='utf-8') as snapshot_file:
                json.dump(system_snapshot, snapshot_file, indent=2, ensure_ascii=False)

        print(f'Logging session to: {self.session_dir}')
        self.log_event(
            'session_prepared',
            frame_index=0,
            log_files={
                'processed_frames': self.processed_log_path.name if self.write_processed_frames else None,
                'render_frames': self.render_log_path.name if self.write_render_frames else None,
                'event_log': self.event_log_path.name if self.write_event_log else None,
                'legacy_status_log': self.status_log_path.name if self.write_status_log else None,
                'system_snapshot': self.system_snapshot_path.name if system_snapshot is not None else None,
            },
        )
        if system_snapshot is not None:
            power = system_snapshot.get('power') or {}
            process = system_snapshot.get('process') or {}
            network = system_snapshot.get('network') or {}
            self.log_event(
                'system_snapshot_captured',
                frame_index=0,
                power_plan=power.get('active_scheme_name'),
                process_priority_class=process.get('priority_class'),
                expected_host_ip=network.get('expected_host_ip'),
                host_ip_present=network.get('host_ip_present'),
            )

    def log_event(self, event_type: str, *, frame_index: int = 0, stream_started_at=None, **payload):
        if self.event_log_file is None:
            return
        record = {
            'event_type': event_type,
            'wall_time': datetime.now().isoformat(timespec='milliseconds'),
            'session_id': self.session_dir.name,
            'frame_index': int(frame_index),
        }
        if stream_started_at is not None:
            record['elapsed_since_stream_start_s'] = round(
                max(time.perf_counter() - stream_started_at, 0.0),
                4,
            )
        if payload:
            record.update(payload)
        self.event_log_file.write(json.dumps(record, ensure_ascii=False) + '\n')

    def write_render_record(self, record: dict):
        if self.render_log_file is not None:
            self.render_log_file.write(json.dumps(record, ensure_ascii=False) + '\n')
        if self.status_log_file is not None:
            self.status_log_file.write(json.dumps(record, ensure_ascii=False) + '\n')

    def close(self, *, frame_index: int, skipped_render_frames_total: int):
        if not self.enabled:
            return

        report_generation_mode = self._normalized_report_generation_mode()
        report_generation_command = self._report_generation_command_text()

        if self.render_log_file is not None:
            self.render_log_file.close()
            self.render_log_file = None

        if self.status_log_file is not None:
            self.status_log_file.close()
            self.status_log_file = None

        if self.event_log_file is not None:
            self.event_log_file.write(
                json.dumps(
                    {
                        'event_type': 'shutdown_complete',
                        'wall_time': datetime.now().isoformat(timespec='milliseconds'),
                        'session_id': self.session_dir.name,
                        'frame_index': int(frame_index),
                        'skipped_render_frames_total': int(skipped_render_frames_total),
                        'report_generation_mode': report_generation_mode,
                        'report_generation_command': report_generation_command,
                        'report_generation_log': (
                            self.report_generation_log_path.name
                            if report_generation_mode == 'deferred'
                            else None
                        ),
                    },
                    ensure_ascii=False,
                )
                + '\n'
            )
            self.event_log_file.close()
            self.event_log_file = None

        # Give background writers a brief moment to flush their last records
        # before building the derived summaries and HTML reports.
        time.sleep(0.2)
        try:
            if report_generation_mode == 'inline':
                self._generate_reports_inline()
            elif report_generation_mode == 'manual':
                print(
                    'Report generation left in manual mode. '
                    f'Run this after measurement: {report_generation_command}'
                )
            else:
                report_log = self._launch_deferred_report_generation()
                print(
                    'Queued deferred HTML log report generation: '
                    f'{report_log} | command: {report_generation_command}'
                )
        except Exception as exc:
            print(f'Failed to generate HTML log reports: {exc!r}')
