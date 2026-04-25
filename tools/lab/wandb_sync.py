from __future__ import annotations

import argparse
from datetime import datetime
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

from tools.lab import analytics, registry, stage_cache


DEFAULT_PROJECT = "radar-lab"
REVIEW_LABELS = {"baseline", "good", "usable", "interesting"}
FRAME_METRIC_KEYS = (
    "compute_total_ms",
    "detection_count",
    "confirmed_track_count",
    "lead_step_m",
    "lead_measurement_residual_m",
    "frame_severity_10",
)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _parse_parameter_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, list, dict)):
        return value
    text = str(value).strip()
    if not text:
        return ""
    if text == "true":
        return True
    if text == "false":
        return False
    if text == "null":
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _deep_merge(target: dict, source: dict) -> dict:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value
    return target


def _nested_assign(target: dict, dotted_key: str, value: Any) -> None:
    current = target
    parts = [part for part in str(dotted_key).split(".") if part]
    if not parts:
        return
    for part in parts[:-1]:
        next_value = current.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            current[part] = next_value
        current = next_value
    current[parts[-1]] = value


def _slug(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "").strip())
    text = text.strip("-")
    return text or "unknown"


def _project_relative(project_root: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _artifact_spec(name: str, type_name: str, path: Path, *, required: bool) -> dict:
    return {
        "name": name,
        "type": type_name,
        "local_path": str(path),
        "required": bool(required),
        "present": bool(path.exists()),
    }


def _transport_suitability_code(detail: dict) -> str:
    category = str(detail.get("transport_category") or "").strip().lower()
    if category == "clean":
        return "baseline_ok"
    if category == "noisy":
        return "robustness_only"
    if category == "unusable":
        return "not_for_baseline"
    return "insufficient"


def _recommended_phase(detail: dict) -> str:
    label = str(detail.get("annotation_label") or "").strip().lower()
    keep_flag = bool(detail.get("annotation_keep_flag"))
    if label == "baseline":
        return "benchmark"
    if keep_flag or label in {"good", "usable", "interesting"}:
        return "benchmark"
    if label == "discard":
        return "debug"
    return "debug"


def wandb_available() -> bool:
    return importlib.util.find_spec("wandb") is not None


def local_wandb_root(project_root: Path) -> Path:
    return Path(project_root).resolve() / "lab_data" / "wandb"


def contract_path(project_root: Path, session_id: str) -> Path:
    detail = registry.fetch_run_detail(project_root, session_id)
    if detail is None:
        raise ValueError(f"Run not found: {session_id}")
    return Path(detail["session_dir"]) / "wandb_run_contract.json"


def sync_result_path(project_root: Path, session_id: str) -> Path:
    detail = registry.fetch_run_detail(project_root, session_id)
    if detail is None:
        raise ValueError(f"Run not found: {session_id}")
    return Path(detail["session_dir"]) / "wandb_sync_result.json"


def read_sync_result(project_root: Path, session_id: str) -> dict | None:
    payload = _load_json(sync_result_path(project_root, session_id))
    return payload or None


def sync_readiness(detail: dict) -> dict:
    session_dir = Path(detail["session_dir"])
    label = str(detail.get("annotation_label") or "").strip().lower()
    keep_flag = bool(detail.get("annotation_keep_flag"))
    transport = str(detail.get("transport_category") or "").strip().lower()

    hard_blockers: list[str] = []
    soft_blockers: list[str] = []
    warnings: list[str] = []

    if not (session_dir / "summary.json").exists():
        hard_blockers.append("`summary.json`이 없어 W&B run payload를 만들 수 없습니다.")

    if label == "discard":
        soft_blockers.append("이 run은 `discard`로 표시되어 있습니다.")
    elif not (label in REVIEW_LABELS or keep_flag):
        soft_blockers.append("아직 benchmark/good 계열 annotation이 없어 검토 완료 run으로 보기 어렵습니다.")

    if transport == "unusable":
        soft_blockers.append("transport quality가 `unusable`이라 baseline 결과로 누적하기엔 위험합니다.")
    elif transport == "noisy":
        warnings.append("transport quality가 `noisy`입니다. robustness 비교용으로만 해석하는 편이 좋습니다.")

    if not detail.get("capture_id"):
        warnings.append("linked raw capture가 없습니다. W&B group은 `legacy:<session_id>`로 들어갑니다.")

    stage_dir = stage_cache.stage_cache_dir(Path(detail["session_dir"]).parents[2], detail["session_id"])
    if not stage_dir.exists():
        warnings.append("stage cache가 없어 feature summary/frame timeline artifact는 생략됩니다.")

    return {
        "ready": not hard_blockers and not soft_blockers,
        "hard_blockers": hard_blockers,
        "soft_blockers": soft_blockers,
        "warnings": warnings,
        "recommended_phase": _recommended_phase(detail),
    }


def _build_tuning_config(detail: dict, parameters: list[dict]) -> dict:
    tuning: dict[str, Any] = {}
    runtime_snapshot = (detail.get("runtime_config") or {}).get("tuning_snapshot")
    if isinstance(runtime_snapshot, dict):
        tuning = json.loads(json.dumps(runtime_snapshot, ensure_ascii=False))
    for row in parameters:
        param_key = row.get("param_key")
        if not param_key:
            continue
        _nested_assign(tuning, str(param_key), _parse_parameter_value(row.get("param_value")))
    return tuning


def build_run_contract(
    project_root: Path,
    session_id: str,
    *,
    project: str = DEFAULT_PROJECT,
    mode: str = "offline",
    phase: str | None = None,
    include_frame_features: bool = False,
) -> dict:
    project_root = Path(project_root).resolve()
    detail = registry.fetch_run_detail(project_root, session_id)
    if detail is None:
        raise ValueError(f"Run not found: {session_id}")

    session_dir = Path(detail["session_dir"]).resolve()
    capture_id = detail.get("capture_id")
    capture_dir = (project_root / "logs" / "raw" / str(capture_id)).resolve() if capture_id else None
    stage_dir = stage_cache.stage_cache_dir(project_root, session_id).resolve()
    stage_feature_summary = stage_cache.load_stage_feature_summary(project_root, session_id) or {}
    diagnosis = analytics.diagnose_run(detail)
    all_runs = registry.fetch_runs(project_root)
    same_capture_group_size = (
        sum(1 for row in all_runs if row.get("capture_id") == capture_id)
        if capture_id
        else 1
    )
    phase_value = str(phase or _recommended_phase(detail)).strip().lower() or "debug"
    label = str(detail.get("annotation_label") or "").strip() or "unlabeled"
    people_count = detail.get("annotation_people_count")
    motion_pattern = str(detail.get("annotation_motion_pattern") or "").strip()
    parameters = registry.fetch_run_parameters(project_root, session_id)

    tags = [
        f"transport:{detail.get('transport_category') or 'unknown'}",
        f"label:{label}",
        f"phase:{phase_value}",
    ]
    if motion_pattern:
        tags.append(f"motion:{motion_pattern}")
    if people_count not in (None, ""):
        tags.append(f"people:{people_count}")

    artifacts = [
        _artifact_spec("session-summary", "radar-session-summary", session_dir / "summary.json", required=True),
        _artifact_spec("performance-report", "radar-session-report", session_dir / "performance_report.html", required=False),
        _artifact_spec("trajectory-replay", "radar-session-report", session_dir / "trajectory_replay.html", required=False),
        _artifact_spec("feature-summary", "radar-stage-summary", stage_dir / "feature_summary.json", required=False),
    ]
    if include_frame_features:
        artifacts.append(
            _artifact_spec("frame-features", "radar-stage-features", stage_dir / "frame_features.jsonl", required=False)
        )

    excluded_local_only = []
    if capture_dir is not None:
        excluded_local_only.extend(
            [
                _project_relative(project_root, capture_dir / "raw_frames.i16"),
                _project_relative(project_root, capture_dir / "raw_frames_index.jsonl"),
            ]
        )
    excluded_local_only.extend(
        [
            _project_relative(project_root, stage_dir / "frame_trace.jsonl"),
            _project_relative(project_root, stage_dir / "artifacts"),
        ]
    )

    contract = {
        "project": str(project or DEFAULT_PROJECT).strip() or DEFAULT_PROJECT,
        "name": detail["session_id"],
        "group": str(capture_id) if capture_id else f"legacy:{detail['session_id']}",
        "job_type": detail.get("input_mode") if detail.get("input_mode") in {"live", "replay"} else "backfill",
        "tags": tags,
        "config": {
            "session": {
                "session_id": detail["session_id"],
                "input_mode": detail.get("input_mode"),
                "variant": detail.get("variant"),
                "scenario_id": detail.get("scenario_id"),
            },
            "capture": {
                "capture_id": capture_id,
                "transport_category": detail.get("transport_category"),
                "transport_suitability": _transport_suitability_code(detail),
            },
            "annotation": {
                "label": None if label == "unlabeled" else label,
                "keep_flag": bool(detail.get("annotation_keep_flag")),
                "people_count": people_count,
                "motion_pattern": motion_pattern or None,
                "notes": detail.get("annotation_notes") or None,
            },
            "git": {
                "commit": detail.get("git_commit"),
                "branch": detail.get("git_branch"),
                "dirty": bool(detail.get("git_dirty")),
            },
            "paths": {
                "session_dir": _project_relative(project_root, session_dir),
                "capture_dir": _project_relative(project_root, capture_dir) if capture_dir is not None else None,
                "registry_db": _project_relative(project_root, project_root / "lab_data" / "radar_lab_registry.db"),
                "stage_cache_dir": _project_relative(project_root, stage_dir),
            },
            "tuning": _build_tuning_config(detail, parameters),
        },
        "summary": {
            "performance_score": detail.get("performance_score"),
            "path_cleanliness_score_10": detail.get("path_cleanliness_score_10"),
            "path_local_residual_rms_m": detail.get("path_local_residual_rms_m"),
            "path_jump_ratio": detail.get("path_jump_ratio"),
            "lead_confirmed_switch_rate": detail.get("lead_confirmed_switch_rate"),
            "candidate_to_confirmed_ratio": detail.get("candidate_to_confirmed_ratio"),
            "display_to_confirmed_ratio": detail.get("display_to_confirmed_ratio"),
            "render_latency_p95_ms": detail.get("render_latency_p95_ms"),
            "compute_utilization_p95": detail.get("compute_utilization_p95"),
            "primary_bottleneck": diagnosis["primary_bottleneck"],
            "severity_10": diagnosis["severity_score_10"],
            "transport_suitability": _transport_suitability_code(detail),
            "linked_capture_present": bool(capture_id),
            "same_capture_group_size": same_capture_group_size,
            "stage_cache_present": stage_dir.exists(),
            "stage_top_frame_bottleneck": stage_feature_summary.get("top_frame_bottleneck"),
        },
        "artifacts": artifacts,
        "excluded_local_only": [item for item in excluded_local_only if item],
        "mode": str(mode or "offline").strip().lower() or "offline",
    }
    return contract


def write_run_contract(
    project_root: Path,
    session_id: str,
    *,
    project: str = DEFAULT_PROJECT,
    mode: str = "offline",
    phase: str | None = None,
    include_frame_features: bool = False,
) -> Path:
    payload = build_run_contract(
        project_root,
        session_id,
        project=project,
        mode=mode,
        phase=phase,
        include_frame_features=include_frame_features,
    )
    path = contract_path(project_root, session_id)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _log_frame_metrics(run: Any, features_path: Path) -> int:
    features = _load_jsonl(features_path)
    step_count = 0
    for index, row in enumerate(features):
        step = row.get("ordinal")
        try:
            step_value = int(step)
        except (TypeError, ValueError):
            step_value = index
        payload = {}
        for key in FRAME_METRIC_KEYS:
            value = row.get(key)
            if value is None:
                continue
            try:
                payload[key] = float(value)
            except (TypeError, ValueError):
                continue
        if not payload:
            continue
        run.log(payload, step=step_value)
        step_count += 1
    return step_count


def sync_run(
    project_root: Path,
    session_id: str,
    *,
    project: str = DEFAULT_PROJECT,
    mode: str = "offline",
    phase: str | None = None,
    include_frame_features: bool = False,
    log_frame_metrics: bool = False,
) -> dict:
    if not wandb_available():
        raise RuntimeError("wandb가 설치되어 있지 않습니다. `pip install wandb` 후 다시 시도해 주세요.")

    import wandb  # type: ignore

    project_root = Path(project_root).resolve()
    contract = build_run_contract(
        project_root,
        session_id,
        project=project,
        mode=mode,
        phase=phase,
        include_frame_features=include_frame_features or log_frame_metrics,
    )
    contract_file = write_run_contract(
        project_root,
        session_id,
        project=project,
        mode=mode,
        phase=phase,
        include_frame_features=include_frame_features or log_frame_metrics,
    )

    wandb_root = local_wandb_root(project_root)
    wandb_root.mkdir(parents=True, exist_ok=True)
    mode_value = str(contract.get("mode") or "offline").lower()

    logged_artifacts: list[dict] = []
    logged_frame_steps = 0

    with wandb.init(
        project=contract["project"],
        name=contract["name"],
        group=contract["group"],
        job_type=contract["job_type"],
        tags=list(contract.get("tags") or []),
        config=dict(contract.get("config") or {}),
        mode=mode_value,
        dir=str(wandb_root),
        reinit=True,
    ) as run:
        for key, value in (contract.get("summary") or {}).items():
            if value is None:
                continue
            run.summary[key] = value

        if log_frame_metrics:
            features_path = stage_cache.stage_cache_paths(project_root, session_id)["features_path"]
            if features_path.exists():
                logged_frame_steps = _log_frame_metrics(run, features_path)

        for spec in contract.get("artifacts") or []:
            local_path = Path(spec["local_path"])
            if not local_path.exists():
                if spec.get("required"):
                    raise RuntimeError(f"Required artifact not found: {local_path}")
                continue
            artifact_name = f"{_slug(session_id)}-{_slug(spec.get('name'))}"
            artifact = wandb.Artifact(artifact_name, type=str(spec.get("type") or "radar-artifact"))
            artifact.add_file(str(local_path), name=local_path.name)
            run.log_artifact(artifact)
            logged_artifacts.append(
                {
                    "name": artifact_name,
                    "type": str(spec.get("type") or "radar-artifact"),
                    "local_path": str(local_path),
                }
            )

        result = {
            "synced_at": _now(),
            "session_id": session_id,
            "project": contract["project"],
            "run_name": contract["name"],
            "group": contract["group"],
            "mode": mode_value,
            "phase": phase or _recommended_phase(registry.fetch_run_detail(project_root, session_id) or {}),
            "url": getattr(run, "url", None),
            "run_id": getattr(run, "id", None),
            "local_run_dir": getattr(run, "dir", None),
            "contract_path": str(contract_file),
            "artifact_count": len(logged_artifacts),
            "artifacts": logged_artifacts,
            "frame_metric_steps_logged": logged_frame_steps,
        }

    result_path = sync_result_path(project_root, session_id)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync a reviewed Radar Lab run to W&B.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[2], help="Repository root.")
    parser.add_argument("--session", required=True, help="Radar Lab session_id to export.")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="W&B project name.")
    parser.add_argument("--mode", default="offline", choices=["offline", "online"], help="W&B mode.")
    parser.add_argument("--phase", default=None, help="Optional phase tag such as benchmark/debug/paper.")
    parser.add_argument(
        "--include-frame-features",
        action="store_true",
        help="Attach frame_features.jsonl when present.",
    )
    parser.add_argument(
        "--log-frame-metrics",
        action="store_true",
        help="Log selected frame timeline metrics to the W&B run.",
    )
    parser.add_argument(
        "--contract-only",
        action="store_true",
        help="Only write wandb_run_contract.json without creating a W&B run.",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    if args.contract_only:
        path = write_run_contract(
            project_root,
            args.session,
            project=args.project,
            mode=args.mode,
            phase=args.phase,
            include_frame_features=args.include_frame_features or args.log_frame_metrics,
        )
        print(json.dumps({"contract_path": str(path)}, ensure_ascii=False, indent=2))
        return

    result = sync_run(
        project_root,
        args.session,
        project=args.project,
        mode=args.mode,
        phase=args.phase,
        include_frame_features=args.include_frame_features,
        log_frame_metrics=args.log_frame_metrics,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
