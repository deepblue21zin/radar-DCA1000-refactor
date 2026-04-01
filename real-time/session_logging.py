from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import subprocess
import time

from tools.diagnostics.log_html_reports import generate_reports


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

        self.variant = str(variant)
        self.scenario_id = str(scenario_id)
        self.input_mode = str(input_mode)
        self.source_capture = str(source_capture)
        self.notes = str(notes)

        self.render_log_file = None
        self.status_log_file = None
        self.event_log_file = None
        self.session_metadata = self.build_session_metadata()

    def build_session_metadata(self):
        git_commit = _run_git_command(self.project_root, 'rev-parse', '--short', 'HEAD')
        git_branch = _run_git_command(self.project_root, 'rev-parse', '--abbrev-ref', 'HEAD')
        git_status = _run_git_command(self.project_root, 'status', '--short')
        return {
            'schema_version': 2,
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
        }

    def prepare(self, runtime_summary: dict):
        self.session_dir.mkdir(parents=True, exist_ok=True)
        with self.session_meta_path.open('w', encoding='utf-8') as meta_file:
            json.dump(self.session_metadata, meta_file, indent=2, ensure_ascii=False)
        with self.runtime_config_path.open('w', encoding='utf-8') as runtime_file:
            json.dump(runtime_summary, runtime_file, indent=2, ensure_ascii=False)

        self.render_log_file = self.render_log_path.open('a', encoding='utf-8', buffering=1)
        self.status_log_file = self.status_log_path.open('a', encoding='utf-8', buffering=1)
        self.event_log_file = self.event_log_path.open('a', encoding='utf-8', buffering=1)
        print(f'Logging session to: {self.session_dir}')
        self.log_event(
            'session_prepared',
            frame_index=0,
            log_files={
                'processed_frames': self.processed_log_path.name,
                'render_frames': self.render_log_path.name,
                'event_log': self.event_log_path.name,
                'legacy_status_log': self.status_log_path.name,
            },
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
            report_result = generate_reports(self.session_dir)
            dashboard_path = report_result.get("dashboard_path")
            if dashboard_path is not None:
                print(f'Generated HTML log dashboard: {dashboard_path}')
        except Exception as exc:
            print(f'Failed to generate HTML log reports: {exc!r}')
