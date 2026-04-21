from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


CLEAN = {"invalid": 0.01, "gap": 8, "ooo": 1, "mismatch": 1}
NOISY = {"invalid": 0.05, "gap": 64, "ooo": 2, "mismatch": 2}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def database_path(project_root: Path) -> Path:
    return Path(project_root) / "lab_data" / "radar_lab_registry.db"


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _nested_get(data: dict, dotted_key: str, default: Any = None) -> Any:
    current: Any = data
    for key in dotted_key.split("."):
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def _as_int(value: Any) -> int:
    return 1 if bool(value) else 0


def _quality_from_stats(frame_count: int | None, invalid_rate: float | None, gap: int | None, ooo: int | None, mismatch: int | None) -> dict:
    if not frame_count:
        return {
            "category": "insufficient",
            "label": "insufficient",
            "tone": "brand",
            "suitability": "판단 보류",
            "detail": "raw index나 summary가 부족해 transport 품질을 아직 판단하기 어렵습니다.",
        }
    invalid_rate = float(invalid_rate or 0.0)
    gap = int(gap or 0)
    ooo = int(ooo or 0)
    mismatch = int(mismatch or 0)
    if invalid_rate <= CLEAN["invalid"] and gap <= CLEAN["gap"] and ooo <= CLEAN["ooo"] and mismatch <= CLEAN["mismatch"]:
        return {
            "category": "clean",
            "label": "clean",
            "tone": "good",
            "suitability": "baseline 튜닝 적합",
            "detail": "transport 영향이 작아 detection/tracking 수정 효과를 보기 좋은 세션입니다.",
        }
    if invalid_rate <= NOISY["invalid"] and gap <= NOISY["gap"] and ooo <= NOISY["ooo"] and mismatch <= NOISY["mismatch"]:
        return {
            "category": "noisy",
            "label": "noisy",
            "tone": "warn",
            "suitability": "robustness 확인용",
            "detail": "transport 흔들림이 일부 섞여 있어 알고리즘 해석 시 주의가 필요합니다.",
        }
    return {
        "category": "unusable",
        "label": "unusable",
        "tone": "danger",
        "suitability": "baseline 튜닝 부적합",
        "detail": "transport 영향이 커서 이 세션만 보고 알고리즘 회귀를 판단하면 왜곡될 수 있습니다.",
    }


def _scan_raw_index(index_path: Path) -> dict:
    stats = {
        "frame_count": 0,
        "invalid_frame_count": 0,
        "invalid_rate": None,
        "max_udp_gap_count": 0,
        "max_out_of_sequence_count": 0,
        "max_byte_mismatch_count": 0,
    }
    if not index_path.exists():
        return stats
    try:
        with index_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                stats["frame_count"] += 1
                if record.get("invalid"):
                    stats["invalid_frame_count"] += 1
                stats["max_udp_gap_count"] = max(stats["max_udp_gap_count"], int(record.get("udp_gap_count") or 0))
                stats["max_out_of_sequence_count"] = max(stats["max_out_of_sequence_count"], int(record.get("out_of_sequence_count") or 0))
                stats["max_byte_mismatch_count"] = max(stats["max_byte_mismatch_count"], int(record.get("byte_mismatch_count") or 0))
    except OSError:
        return stats
    if stats["frame_count"] > 0:
        stats["invalid_rate"] = stats["invalid_frame_count"] / stats["frame_count"]
    return stats


def _connect(project_root: Path) -> sqlite3.Connection:
    db_path = database_path(project_root)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    # Local experiment registry does not need crash-safe journaling as strongly as a shared DB.
    # MEMORY temp/journal modes avoid filesystem locking issues on some sandboxed drives.
    connection.execute("PRAGMA journal_mode=MEMORY")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS captures (
            capture_id TEXT PRIMARY KEY,
            capture_dir TEXT NOT NULL,
            created_at TEXT,
            source_session_id TEXT,
            source_session_dir TEXT,
            variant TEXT,
            scenario_id TEXT,
            frame_count INTEGER,
            invalid_rate REAL,
            max_udp_gap_count INTEGER,
            max_out_of_sequence_count INTEGER,
            max_byte_mismatch_count INTEGER,
            transport_category TEXT,
            transport_label TEXT,
            transport_tone TEXT,
            transport_suitability TEXT,
            transport_detail TEXT,
            linked_run_count INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS runs (
            session_id TEXT PRIMARY KEY,
            session_dir TEXT NOT NULL,
            created_at TEXT,
            input_mode TEXT,
            variant TEXT,
            scenario_id TEXT,
            source_capture TEXT,
            capture_id TEXT,
            git_commit TEXT,
            git_branch TEXT,
            git_dirty INTEGER NOT NULL DEFAULT 0,
            transport_category TEXT,
            transport_label TEXT,
            transport_tone TEXT,
            transport_suitability TEXT,
            operational_score REAL,
            operational_grade TEXT,
            performance_score REAL,
            performance_grade TEXT,
            render_latency_p95_ms REAL,
            compute_utilization_p95 REAL,
            candidate_to_confirmed_ratio REAL,
            display_to_confirmed_ratio REAL,
            lead_confirmed_switch_rate REAL,
            path_cleanliness_score_10 REAL,
            path_max_gap_frames REAL,
            path_local_residual_rms_m REAL,
            path_jump_ratio REAL,
            slowest_stage_name TEXT,
            slowest_stage_p95_ms REAL,
            summary_path TEXT,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS annotations (
            target_type TEXT NOT NULL,
            target_id TEXT NOT NULL,
            label TEXT,
            keep_flag INTEGER NOT NULL DEFAULT 0,
            people_count INTEGER,
            motion_pattern TEXT,
            notes TEXT,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (target_type, target_id)
        );
        CREATE TABLE IF NOT EXISTS registry_meta (
            meta_key TEXT PRIMARY KEY,
            meta_value TEXT NOT NULL
        );
        """
    )
    connection.commit()
    return connection


def _capture_id_from_path(path_value: Any, raw_root: Path) -> str | None:
    if not path_value:
        return None
    try:
        candidate = Path(path_value)
    except (TypeError, ValueError):
        return None
    if not candidate.is_absolute():
        candidate = raw_root / candidate
    return candidate.name or None


def _build_capture_record(project_root: Path, capture_dir: Path) -> dict:
    manifest = _load_json(capture_dir / "capture_manifest.json")
    stats = _scan_raw_index(capture_dir / "raw_frames_index.jsonl")
    quality = _quality_from_stats(
        stats["frame_count"],
        stats["invalid_rate"],
        stats["max_udp_gap_count"],
        stats["max_out_of_sequence_count"],
        stats["max_byte_mismatch_count"],
    )
    source_session_dir = manifest.get("source_session_dir")
    return {
        "capture_id": str(manifest.get("session_id") or capture_dir.name),
        "capture_dir": str(capture_dir),
        "created_at": manifest.get("created_at"),
        "source_session_id": Path(source_session_dir).name if source_session_dir else None,
        "source_session_dir": source_session_dir,
        "variant": manifest.get("variant"),
        "scenario_id": manifest.get("scenario_id"),
        "frame_count": stats["frame_count"],
        "invalid_rate": stats["invalid_rate"],
        "max_udp_gap_count": stats["max_udp_gap_count"],
        "max_out_of_sequence_count": stats["max_out_of_sequence_count"],
        "max_byte_mismatch_count": stats["max_byte_mismatch_count"],
        "transport_category": quality["category"],
        "transport_label": quality["label"],
        "transport_tone": quality["tone"],
        "transport_suitability": quality["suitability"],
        "transport_detail": quality["detail"],
        "updated_at": _now(),
    }


def _build_run_record(project_root: Path, session_dir: Path) -> dict | None:
    session_meta = _load_json(session_dir / "session_meta.json")
    summary = _load_json(session_dir / "summary.json")
    if not session_meta and not summary:
        return None
    meta = summary.get("session_meta") or session_meta
    runtime_config = summary.get("runtime_config") or _load_json(session_dir / "runtime_config.json")
    transport = summary.get("transport_quality", {})
    scoring = _nested_get(summary, "performance.scoring", {}) or {}
    kpis = scoring.get("kpis", {}) if isinstance(scoring, dict) else {}
    geometry = _nested_get(summary, "performance.geometry.reference", {}) or {}
    continuity = _nested_get(summary, "performance.continuity", {}) or {}
    slowest = _nested_get(summary, "diagnostics.preferred_stage_timings_ms.slowest_stage", {}) or {}
    raw_root = Path(project_root) / "logs" / "raw"
    capture_id = _capture_id_from_path(
        meta.get("raw_capture_dir")
        or _nested_get(runtime_config, "raw_capture_dir")
        or meta.get("source_capture"),
        raw_root,
    )
    return {
        "session_id": str(meta.get("session_id") or summary.get("session_id") or session_dir.name),
        "session_dir": str(session_dir),
        "created_at": meta.get("created_at") or summary.get("summary_generated_at"),
        "input_mode": meta.get("input_mode"),
        "variant": meta.get("variant"),
        "scenario_id": meta.get("scenario_id"),
        "source_capture": meta.get("source_capture"),
        "capture_id": capture_id,
        "git_commit": meta.get("git_commit"),
        "git_branch": meta.get("git_branch"),
        "git_dirty": _as_int(meta.get("git_dirty")),
        "transport_category": transport.get("category"),
        "transport_label": transport.get("label"),
        "transport_tone": transport.get("tone"),
        "transport_suitability": transport.get("suitability"),
        "operational_score": _nested_get(summary, "assessment.overall.score"),
        "operational_grade": _nested_get(summary, "assessment.overall.grade"),
        "performance_score": scoring.get("overall_score_100"),
        "performance_grade": scoring.get("grade"),
        "render_latency_p95_ms": _nested_get(kpis, "render_latency_p95.value"),
        "compute_utilization_p95": _nested_get(kpis, "compute_utilization_p95.value"),
        "candidate_to_confirmed_ratio": _nested_get(kpis, "candidate_to_confirmed.value") or continuity.get("candidate_to_confirmed_ratio"),
        "display_to_confirmed_ratio": _nested_get(kpis, "display_to_confirmed.value") or continuity.get("display_to_confirmed_ratio"),
        "lead_confirmed_switch_rate": continuity.get("lead_confirmed", {}).get("switch_rate"),
        "path_cleanliness_score_10": _nested_get(kpis, "path_cleanliness.value") or geometry.get("path_cleanliness_score_10"),
        "path_max_gap_frames": _nested_get(kpis, "path_max_gap_frames.value") or geometry.get("max_gap_frames"),
        "path_local_residual_rms_m": _nested_get(kpis, "path_local_residual_rms.value") or geometry.get("local_residual_rms_m"),
        "path_jump_ratio": _nested_get(kpis, "path_jump_ratio.value") or geometry.get("jump_ratio"),
        "slowest_stage_name": slowest.get("name"),
        "slowest_stage_p95_ms": slowest.get("p95_ms"),
        "summary_path": str(session_dir / "summary.json"),
        "updated_at": _now(),
    }


def _upsert(connection: sqlite3.Connection, table: str, record: dict, keys: list[str]) -> None:
    assignments = ", ".join(f"{key}=excluded.{key}" for key in keys if key != keys[0])
    columns = ", ".join(keys)
    placeholders = ", ".join(f":{key}" for key in keys)
    connection.execute(
        f"""
        INSERT INTO {table} ({columns})
        VALUES ({placeholders})
        ON CONFLICT({keys[0]}) DO UPDATE SET {assignments}
        """,
        record,
    )


def _iter_run_session_dirs(live_root: Path):
    if not live_root.exists():
        return
    seen: set[Path] = set()
    for session_dir in sorted(live_root.rglob("*")):
        if not session_dir.is_dir():
            continue
        if session_dir in seen:
            continue
        if any(
            (session_dir / marker).exists()
            for marker in ("session_meta.json", "runtime_config.json", "processed_frames.jsonl", "render_frames.jsonl")
        ):
            seen.add(session_dir)
            yield session_dir


def refresh_registry(project_root: Path) -> dict:
    project_root = Path(project_root)
    raw_root = project_root / "logs" / "raw"
    live_root = project_root / "logs" / "live_motion_viewer"
    capture_count = 0
    run_count = 0
    with _connect(project_root) as connection:
        connection.execute("DELETE FROM captures")
        connection.execute("DELETE FROM runs")
        if raw_root.exists():
            for capture_dir in sorted(raw_root.iterdir()):
                if not capture_dir.is_dir():
                    continue
                record = _build_capture_record(project_root, capture_dir)
                _upsert(connection, "captures", record, list(record.keys()))
                capture_count += 1
        if live_root.exists():
            for session_dir in _iter_run_session_dirs(live_root):
                record = _build_run_record(project_root, session_dir)
                if record is None:
                    continue
                _upsert(connection, "runs", record, list(record.keys()))
                run_count += 1
        connection.execute("UPDATE captures SET linked_run_count = 0")
        connection.execute(
            """
            UPDATE captures
            SET linked_run_count = (
                SELECT COUNT(*)
                FROM runs
                WHERE runs.capture_id = captures.capture_id
            )
            """
        )
        connection.execute(
            """
            INSERT INTO registry_meta(meta_key, meta_value)
            VALUES('last_refresh_at', ?)
            ON CONFLICT(meta_key) DO UPDATE SET meta_value=excluded.meta_value
            """,
            (_now(),),
        )
        connection.commit()
    return {
        "captures_indexed": capture_count,
        "runs_indexed": run_count,
        "db_path": str(database_path(project_root)),
    }


def get_registry_overview(project_root: Path) -> dict:
    with _connect(project_root) as connection:
        capture_counts = {
            row["transport_category"] or "unknown": row["count"]
            for row in connection.execute("SELECT transport_category, COUNT(*) AS count FROM captures GROUP BY transport_category").fetchall()
        }
        run_counts = {
            row["transport_category"] or "unknown": row["count"]
            for row in connection.execute("SELECT transport_category, COUNT(*) AS count FROM runs GROUP BY transport_category").fetchall()
        }
        label_counts = {
            row["label"] or "unlabeled": row["count"]
            for row in connection.execute(
                "SELECT label, COUNT(*) AS count FROM annotations WHERE target_type='run' GROUP BY label"
            ).fetchall()
        }
        refresh_row = connection.execute(
            "SELECT meta_value FROM registry_meta WHERE meta_key='last_refresh_at'"
        ).fetchone()
        return {
            "capture_total": connection.execute("SELECT COUNT(*) FROM captures").fetchone()[0],
            "run_total": connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0],
            "capture_transport_counts": capture_counts,
            "run_transport_counts": run_counts,
            "run_annotation_counts": label_counts,
            "last_refresh_at": refresh_row["meta_value"] if refresh_row else None,
            "db_path": str(database_path(project_root)),
        }


def _rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict]:
    return [dict(row) for row in rows]


def fetch_runs(project_root: Path) -> list[dict]:
    with _connect(project_root) as connection:
        rows = connection.execute(
            """
            SELECT runs.*,
                   annotations.label AS annotation_label,
                   annotations.keep_flag AS annotation_keep_flag,
                   annotations.people_count AS annotation_people_count,
                   annotations.motion_pattern AS annotation_motion_pattern,
                   annotations.notes AS annotation_notes
            FROM runs
            LEFT JOIN annotations
              ON annotations.target_type='run'
             AND annotations.target_id=runs.session_id
            ORDER BY COALESCE(runs.created_at, runs.session_id) DESC
            """
        ).fetchall()
        return _rows_to_dicts(rows)


def fetch_captures(project_root: Path) -> list[dict]:
    with _connect(project_root) as connection:
        rows = connection.execute(
            """
            SELECT captures.*,
                   annotations.label AS annotation_label,
                   annotations.keep_flag AS annotation_keep_flag,
                   annotations.people_count AS annotation_people_count,
                   annotations.motion_pattern AS annotation_motion_pattern,
                   annotations.notes AS annotation_notes
            FROM captures
            LEFT JOIN annotations
              ON annotations.target_type='capture'
             AND annotations.target_id=captures.capture_id
            ORDER BY COALESCE(captures.created_at, captures.capture_id) DESC
            """
        ).fetchall()
        return _rows_to_dicts(rows)


def fetch_run_detail(project_root: Path, session_id: str) -> dict | None:
    row = next((item for item in fetch_runs(project_root) if item["session_id"] == session_id), None)
    if row is None:
        return None
    session_dir = Path(row["session_dir"])
    row["summary"] = _load_json(session_dir / "summary.json")
    row["session_meta"] = _load_json(session_dir / "session_meta.json")
    row["runtime_config"] = _load_json(session_dir / "runtime_config.json")
    return row


def fetch_capture_detail(project_root: Path, capture_id: str) -> dict | None:
    row = next((item for item in fetch_captures(project_root) if item["capture_id"] == capture_id), None)
    if row is None:
        return None
    capture_dir = Path(row["capture_dir"])
    row["manifest"] = _load_json(capture_dir / "capture_manifest.json")
    row["linked_runs"] = [item for item in fetch_runs(project_root) if item.get("capture_id") == capture_id]
    return row


def save_annotation(
    project_root: Path,
    *,
    target_type: str,
    target_id: str,
    label: str | None,
    keep_flag: bool,
    people_count: int | None,
    motion_pattern: str | None,
    notes: str | None,
) -> None:
    label = (label or "").strip() or None
    motion_pattern = (motion_pattern or "").strip() or None
    notes = (notes or "").strip() or None
    people_count = int(people_count) if people_count not in (None, 0) else None
    with _connect(project_root) as connection:
        if not any([label, keep_flag, people_count, motion_pattern, notes]):
            connection.execute(
                "DELETE FROM annotations WHERE target_type=? AND target_id=?",
                (target_type, target_id),
            )
        else:
            connection.execute(
                """
                INSERT INTO annotations (
                    target_type, target_id, label, keep_flag, people_count,
                    motion_pattern, notes, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(target_type, target_id) DO UPDATE SET
                    label=excluded.label,
                    keep_flag=excluded.keep_flag,
                    people_count=excluded.people_count,
                    motion_pattern=excluded.motion_pattern,
                    notes=excluded.notes,
                    updated_at=excluded.updated_at
                """,
                (target_type, target_id, label, _as_int(keep_flag), people_count, motion_pattern, notes, _now()),
            )
        connection.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh local Radar Lab registry.")
    parser.add_argument(
        "--project-root",
        default=Path(__file__).resolve().parents[2],
        help="Project root directory. Defaults to repository root.",
    )
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()
    refresh = refresh_registry(project_root)
    overview = get_registry_overview(project_root)
    print(
        json.dumps(
            {
                "project_root": str(project_root),
                "db_path": refresh["db_path"],
                "captures_indexed": refresh["captures_indexed"],
                "runs_indexed": refresh["runs_indexed"],
                "capture_total": overview["capture_total"],
                "run_total": overview["run_total"],
                "last_refresh_at": overview["last_refresh_at"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
