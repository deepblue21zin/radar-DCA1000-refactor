from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.eval_harness.path_shape import build_path_shape_metrics


SESSION_RE = re.compile(r"Session logs:\s*(.+)$")
TRAJECTORY_RE = re.compile(r"Trajectory replay report:\s*(.+trajectory_replay\.html)")
RUN_ROOT = PROJECT_ROOT / "lab_data" / "tuning_runs"

ISK_SCENARIOS = {
    "left-diagonal": {
        "label": "left-diagonal",
        "capture": "logs/raw/20260430_185438",
        "description": "ISK left diagonal round trip",
    },
    "center": {
        "label": "center",
        "capture": "logs/raw/20260430_185634",
        "description": "ISK center forward/backward round trip",
    },
    "right-diagonal": {
        "label": "right-diagonal",
        "capture": "logs/raw/20260430_185749",
        "description": "ISK right diagonal round trip",
    },
    "horizontal": {
        "label": "horizontal",
        "capture": "logs/raw/20260430_185914",
        "description": "ISK horizontal angle-change round trip",
    },
}


PARAMETER_SPECS = [
    {
        "key": "detection.min_cartesian_separation_m",
        "label": "Detection separation",
        "kind": "scalar",
        "path": ["detection", "algorithm", "min_cartesian_separation_m"],
        "factors": [0.9, 1.1, 1.25],
        "min": 0.2,
        "max": 1.3,
    },
    {
        "key": "detection.dbscan_eps",
        "label": "DBSCAN eps bands",
        "kind": "band_field",
        "path": ["detection", "dbscan_adaptive_eps_bands"],
        "field": "eps",
        "factors": [0.9, 1.1, 1.25],
        "min": 0.25,
        "max": 1.4,
    },
    {
        "key": "detection.candidate_merge_radius",
        "label": "Candidate merge radius bands",
        "kind": "band_field",
        "path": ["detection", "algorithm", "candidate_merge_bands"],
        "field": "merge_radius_m",
        "factors": [0.9, 1.1, 1.25],
        "min": 0.25,
        "max": 1.4,
    },
    {
        "key": "detection.body_center_relative_floor",
        "label": "Body-center relative floor bands",
        "kind": "band_field",
        "path": ["detection", "algorithm", "body_center_patch_bands"],
        "field": "relative_floor",
        "factors": [0.9, 1.1],
        "min": 0.25,
        "max": 0.9,
    },
    {
        "key": "detection.angle_contrast_scale",
        "label": "Angle contrast scale",
        "kind": "scalar",
        "path": ["detection", "algorithm", "angle_contrast_scale"],
        "factors": [0.85, 1.15],
        "min": 0.5,
        "max": 2.2,
    },
    {
        "key": "detection.angle_quantile",
        "label": "Angle quantile",
        "kind": "scalar",
        "path": ["detection", "algorithm", "angle_quantile"],
        "factors": [0.95, 1.05],
        "min": 0.5,
        "max": 0.95,
    },
    {
        "key": "tracking.association_gate",
        "label": "Association gate",
        "kind": "scalar",
        "path": ["tracking", "association_gate"],
        "factors": [0.85, 1.15, 1.35],
        "min": 1.5,
        "max": 10.0,
    },
    {
        "key": "tracking.measurement_var",
        "label": "Measurement variance",
        "kind": "scalar",
        "path": ["tracking", "measurement_var"],
        "factors": [0.75, 1.25],
        "min": 0.05,
        "max": 1.5,
    },
    {
        "key": "tracking.lateral_deadband_m",
        "label": "Lateral deadband",
        "kind": "scalar",
        "path": ["tracking", "lateral_deadband_m"],
        "factors": [0.5, 0.75, 1.25],
        "min": 0.0,
        "max": 0.25,
    },
    {
        "key": "tracking.lateral_smoothing_alpha",
        "label": "Lateral smoothing alpha",
        "kind": "scalar",
        "path": ["tracking", "lateral_smoothing_alpha"],
        "factors": [0.75, 1.15],
        "min": 0.05,
        "max": 0.9,
    },
    {
        "key": "tracking.local_remeasurement_blend",
        "label": "Local remeasurement blend",
        "kind": "scalar",
        "path": ["tracking", "local_remeasurement_blend"],
        "factors": [0.65, 1.25],
        "min": 0.0,
        "max": 0.8,
    },
    {
        "key": "tracking.local_remeasurement_max_shift_m",
        "label": "Local remeasurement max shift",
        "kind": "scalar",
        "path": ["tracking", "local_remeasurement_max_shift_m"],
        "factors": [0.75, 1.25],
        "min": 0.02,
        "max": 0.45,
    },
    {
        "key": "tracking.measurement_soft_gate_floor",
        "label": "Measurement soft gate floor",
        "kind": "scalar",
        "path": ["tracking", "measurement_soft_gate_floor"],
        "factors": [0.75, 1.2],
        "min": 0.05,
        "max": 0.9,
    },
    {
        "key": "tracking.primary_track_hold_frames",
        "label": "Primary hold frames",
        "kind": "integer",
        "path": ["tracking", "primary_track_hold_frames"],
        "deltas": [-2, 2],
        "min": 0,
        "max": 12,
    },
    {
        "key": "visualization.display_hysteresis_frames",
        "label": "Display hysteresis frames",
        "kind": "integer",
        "path": ["visualization", "display_hysteresis_frames"],
        "deltas": [-3, 3],
        "min": 0,
        "max": 15,
    },
]


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _nested_get(payload: dict[str, Any] | None, path: list[str], default=None):
    current: Any = payload or {}
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def _nested_set(payload: dict[str, Any], path: list[str], value: Any) -> None:
    current: Any = payload
    for key in path[:-1]:
        current = current.setdefault(key, {})
    current[path[-1]] = value


def _clamp(value: float, low: float | None, high: float | None) -> float:
    if low is not None:
        value = max(float(low), value)
    if high is not None:
        value = min(float(high), value)
    return value


def _spec_by_key() -> dict[str, dict[str, Any]]:
    return {spec["key"]: spec for spec in PARAMETER_SPECS}


def _round_number(value: float) -> float:
    return round(float(value), 5)


def _apply_variant(base_config: dict[str, Any], spec: dict[str, Any], step_value: float | int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = deepcopy(base_config)
    changes: list[dict[str, Any]] = []
    kind = spec["kind"]
    if kind in {"scalar", "integer"}:
        path = list(spec["path"])
        before = _nested_get(config, path)
        if before is None:
            return config, changes
        if kind == "integer":
            after = int(round(float(before) + float(step_value)))
            after = int(_clamp(after, spec.get("min"), spec.get("max")))
        else:
            after = _round_number(_clamp(float(before) * float(step_value), spec.get("min"), spec.get("max")))
        if after == before:
            return config, changes
        _nested_set(config, path, after)
        changes.append({"param": spec["key"], "path": ".".join(path), "before": before, "after": after})
        return config, changes

    if kind == "band_field":
        bands = _nested_get(config, list(spec["path"]), default=[])
        if not isinstance(bands, list):
            return config, changes
        for index, band in enumerate(bands):
            if not isinstance(band, dict) or spec["field"] not in band:
                continue
            before = band[spec["field"]]
            after = _round_number(_clamp(float(before) * float(step_value), spec.get("min"), spec.get("max")))
            if after == before:
                continue
            band[spec["field"]] = after
            changes.append(
                {
                    "param": spec["key"],
                    "path": ".".join([*spec["path"], str(index), spec["field"]]),
                    "before": before,
                    "after": after,
                }
            )
        return config, changes

    return config, changes


def _candidate_steps(spec: dict[str, Any]) -> list[float | int]:
    if spec["kind"] == "integer":
        return list(spec.get("deltas") or [])
    return list(spec.get("factors") or [])


def _session_dirs() -> set[Path]:
    root = PROJECT_ROOT / "logs" / "live_motion_viewer"
    if not root.exists():
        return set()
    return {path.resolve() for path in root.iterdir() if path.is_dir()}


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _command_text(command: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in str(part) else str(part) for part in command)


def _infer_session_dir(stdout_text: str, before_dirs: set[Path]) -> Path:
    for line in stdout_text.splitlines():
        session_match = SESSION_RE.search(line)
        if session_match:
            session_dir = _project_path(session_match.group(1).strip())
            if session_dir.exists():
                return session_dir.resolve()
        trajectory_match = TRAJECTORY_RE.search(line)
        if trajectory_match:
            session_dir = _project_path(trajectory_match.group(1).strip()).parent
            if session_dir.exists():
                return session_dir.resolve()
    after_dirs = _session_dirs()
    new_dirs = sorted(after_dirs - before_dirs, key=lambda item: item.stat().st_mtime)
    if new_dirs:
        return new_dirs[-1].resolve()
    raise RuntimeError("Could not infer replay session directory from replay output.")


def _run_command(
    command: list[str],
    *,
    run_dir: Path,
    label: str,
    timeout_s: float | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    started_at = datetime.now().isoformat(timespec="seconds")
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        env=env,
    )
    stdout_path = run_dir / f"{label}.stdout.log"
    stderr_path = run_dir / f"{label}.stderr.log"
    stdout_path.write_text(completed.stdout or "", encoding="utf-8", errors="replace")
    stderr_path.write_text(completed.stderr or "", encoding="utf-8", errors="replace")
    record = {
        "label": label,
        "command": command,
        "command_text": _command_text(command),
        "returncode": completed.returncode,
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }
    if completed.returncode != 0:
        raise RuntimeError(f"{label} failed with exit {completed.returncode}. See {stdout_path} and {stderr_path}")
    return record


def _run_replay(
    capture: Path,
    tuning: Path,
    *,
    run_dir: Path,
    label: str,
    speed: float,
    timeout_s: float | None,
    runtime_settings: Path | None,
) -> dict[str, Any]:
    before = _session_dirs()
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if runtime_settings is not None:
        env["RADAR_RUNTIME_SETTINGS_PATH"] = str(runtime_settings)
    command = [
        sys.executable,
        "-B",
        str(PROJECT_ROOT / "real-time" / "live_motion_replay.py"),
        "--capture",
        str(capture),
        "--speed",
        str(float(speed)),
        "--tuning",
        str(tuning),
        "--no-open-report",
    ]
    command_record = _run_command(command, run_dir=run_dir, label=label, timeout_s=timeout_s, env=env)
    stdout_text = Path(command_record["stdout_log"]).read_text(encoding="utf-8", errors="replace")
    session_dir = _infer_session_dir(stdout_text, before)
    _run_command(
        [sys.executable, "-B", "-m", "tools.diagnostics.session_report", str(session_dir)],
        run_dir=run_dir,
        label=f"report_{label}",
        timeout_s=timeout_s,
        env=env,
    )
    summary = _load_json(session_dir / "summary.json")
    path_shape = build_path_shape_metrics(session_dir)
    summary["eval"] = {"path_shape": path_shape}
    return {
        "label": label,
        "session_id": session_dir.name,
        "session_dir": str(session_dir),
        "summary_path": str(session_dir / "summary.json"),
        "tuning": str(tuning),
        "command": command_record,
        "summary": _extract_kpis(summary),
        "path_shape": path_shape,
    }


def _num(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:
        return default
    return number


def _metric(summary: dict[str, Any], dotted_path: str, default=None):
    current: Any = summary
    for key in dotted_path.split("."):
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def _extract_kpis(summary: dict[str, Any]) -> dict[str, Any]:
    path_shape = summary.get("eval", {}).get("path_shape", {})
    output = path_shape.get("output", {})
    tracking = path_shape.get("tracking", {})
    output_vs_tracking = path_shape.get("output_vs_tracking", {})
    policy = path_shape.get("policy", {})
    return {
        "transport": _metric(summary, "transport_quality.category"),
        "performance_score": _metric(summary, "performance.score_100"),
        "path_cleanliness_score_10": _metric(summary, "performance.geometry.reference.path_cleanliness_score_10"),
        "candidate_to_confirmed_ratio": _metric(summary, "performance.continuity.candidate_to_confirmed_ratio"),
        "lead_switch_count": _metric(summary, "performance.continuity.lead_confirmed.switch_count"),
        "output_x_span_m": output.get("x_span_m"),
        "output_y_span_m": output.get("y_span_m"),
        "output_major_span_m": output.get("major_span_m"),
        "output_width_ratio": output.get("width_ratio"),
        "output_step_p95_m": output.get("step_p95_m"),
        "output_max_step_m": output.get("max_step_m"),
        "tracking_x_span_m": tracking.get("x_span_m"),
        "tracking_y_span_m": tracking.get("y_span_m"),
        "output_vs_tracking_x_span_ratio": output_vs_tracking.get("x_span_ratio"),
        "output_vs_tracking_major_span_ratio": output_vs_tracking.get("major_span_ratio"),
        "trajectory_distance_p95_m": path_shape.get("trajectory_fidelity", {}).get("distance_p95_m"),
        "policy_overall_pass": policy.get("overall_pass"),
        "policy_preserves_tracking_shape": policy.get("output_preserves_tracking_shape"),
        "policy_smooths_jumpy_raw": policy.get("output_smooths_jumpy_raw"),
    }


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "pass", "passed"}
    return bool(value)


def _band_score(value: float, good: float, poor: float, *, higher_is_better: bool) -> float:
    if higher_is_better:
        if value >= good:
            return 1.0
        if value <= poor:
            return 0.0
        return (value - poor) / max(good - poor, 1e-6)
    if value <= good:
        return 1.0
    if value >= poor:
        return 0.0
    return (poor - value) / max(poor - good, 1e-6)


def _scenario_shape_score(scenario: str, kpis: dict[str, Any]) -> tuple[float, list[str]]:
    x_span = _num(kpis.get("output_x_span_m"))
    y_span = _num(kpis.get("output_y_span_m"))
    x_ratio = _num(kpis.get("output_vs_tracking_x_span_ratio"))
    width_ratio = _num(kpis.get("output_width_ratio"))
    notes: list[str] = []

    if scenario == "center":
        score = 12.0 * _band_score(x_span, 0.25, 0.65, higher_is_better=False)
        score += 8.0 * _band_score(y_span, 1.5, 0.7, higher_is_better=True)
        if x_span > 0.65:
            notes.append("center_x_span_too_wide")
        return score, notes

    if scenario == "horizontal":
        lateral_ratio = x_span / max(y_span, 0.2)
        score = 10.0 * _band_score(x_span, 0.8, 0.25, higher_is_better=True)
        score += 7.0 * _band_score(lateral_ratio, 0.55, 0.2, higher_is_better=True)
        score += 3.0 * _band_score(y_span, 2.4, 4.0, higher_is_better=False)
        if lateral_ratio < 0.35:
            notes.append("horizontal_lateral_span_too_small")
        return score, notes

    score = 8.0 * _band_score(x_span, 0.55, 0.25, higher_is_better=True)
    score += 5.0 * _band_score(y_span, 2.4, 1.2, higher_is_better=True)
    score += 4.0 * _band_score(width_ratio, 0.08, 0.28, higher_is_better=False)
    score += 3.0 * _band_score(x_ratio, 2.4, 3.4, higher_is_better=False)
    if x_span < 0.45:
        notes.append("diagonal_x_span_too_small")
    if x_ratio > 3.0:
        notes.append("output_over_expands_tracking_x_span")
    return score, notes


def _baseline_safety_checks(
    scenario: str,
    kpis: dict[str, Any],
    baseline_kpis: dict[str, Any] | None,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if kpis.get("transport") != "clean":
        reasons.append("transport_not_clean")
    if not _is_true(kpis.get("policy_overall_pass")):
        reasons.append("policy_overall_pass_failed")
    if not _is_true(kpis.get("policy_preserves_tracking_shape")):
        reasons.append("policy_preserves_tracking_shape_failed")
    if scenario in {"left-diagonal", "right-diagonal", "horizontal"} and not _is_true(
        kpis.get("policy_smooths_jumpy_raw")
    ):
        reasons.append("policy_smooths_jumpy_raw_failed")

    clean = _num(kpis.get("path_cleanliness_score_10"))
    if clean < 7.5:
        reasons.append("path_cleanliness_below_minimum")

    if baseline_kpis:
        base_distance = _num(baseline_kpis.get("trajectory_distance_p95_m"), default=0.0)
        distance = _num(kpis.get("trajectory_distance_p95_m"), default=0.0)
        if base_distance > 0 and distance > base_distance * 1.08 + 0.05:
            reasons.append("trajectory_distance_p95_regressed")

        base_step = _num(baseline_kpis.get("output_step_p95_m"), default=0.0)
        step = _num(kpis.get("output_step_p95_m"), default=0.0)
        if base_step > 0 and step > base_step * 1.20 + 0.03:
            reasons.append("output_step_p95_regressed")

        base_max_step = _num(baseline_kpis.get("output_max_step_m"), default=0.0)
        max_step = _num(kpis.get("output_max_step_m"), default=0.0)
        if base_max_step > 0 and max_step > base_max_step * 1.25 + 0.04:
            reasons.append("output_max_step_regressed")

        base_clean = _num(baseline_kpis.get("path_cleanliness_score_10"), default=0.0)
        if base_clean > 0 and clean < base_clean - 0.35:
            reasons.append("path_cleanliness_regressed")

        base_candidate_ratio = _num(baseline_kpis.get("candidate_to_confirmed_ratio"), default=0.0)
        candidate_ratio = _num(kpis.get("candidate_to_confirmed_ratio"), default=0.0)
        if base_candidate_ratio > 0 and candidate_ratio > base_candidate_ratio * 1.15 + 0.05:
            reasons.append("candidate_to_confirmed_ratio_regressed")

        base_x_span = _num(baseline_kpis.get("output_x_span_m"), default=0.0)
        x_span = _num(kpis.get("output_x_span_m"), default=0.0)
        if scenario in {"left-diagonal", "right-diagonal", "horizontal"} and base_x_span > 0:
            if x_span < base_x_span * 0.75:
                reasons.append("lateral_span_collapsed_vs_baseline")
            if x_span > max(base_x_span * 2.4, base_x_span + 0.75):
                reasons.append("lateral_span_over_expanded_vs_baseline")

        x_ratio = _num(kpis.get("output_vs_tracking_x_span_ratio"), default=0.0)
        if scenario in {"left-diagonal", "right-diagonal"} and x_ratio > 3.0:
            reasons.append("output_tracking_x_ratio_too_high")

    return not reasons, reasons


def _score_scenario(
    scenario: str,
    kpis: dict[str, Any],
    baseline_kpis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    score = 0.0
    score += 8.0 if kpis.get("transport") == "clean" else 0.0
    score += 18.0 if _is_true(kpis.get("policy_overall_pass")) else 0.0
    score += 10.0 if _is_true(kpis.get("policy_preserves_tracking_shape")) else 0.0
    score += 10.0 if _is_true(kpis.get("policy_smooths_jumpy_raw")) else 0.0
    score += 14.0 * _band_score(_num(kpis.get("path_cleanliness_score_10")), 9.0, 6.5, higher_is_better=True)
    score += 8.0 * _band_score(_num(kpis.get("output_step_p95_m"), default=1.0), 0.18, 0.45, higher_is_better=False)
    score += 8.0 * _band_score(_num(kpis.get("output_max_step_m"), default=1.0), 0.22, 0.70, higher_is_better=False)
    score += 9.0 * _band_score(_num(kpis.get("trajectory_distance_p95_m"), default=5.0), 1.45, 2.40, higher_is_better=False)
    score += 5.0 * _band_score(_num(kpis.get("candidate_to_confirmed_ratio"), default=9.0), 1.10, 1.80, higher_is_better=False)
    score += 3.0 * _band_score(_num(kpis.get("lead_switch_count"), default=9.0), 1.0, 5.0, higher_is_better=False)
    shape_score, shape_notes = _scenario_shape_score(scenario, kpis)
    score += shape_score

    accepted, reject_reasons = _baseline_safety_checks(scenario, kpis, baseline_kpis)
    if reject_reasons:
        score -= min(35.0, 12.0 + (len(reject_reasons) * 5.0))

    if baseline_kpis:
        base_distance = _num(baseline_kpis.get("trajectory_distance_p95_m"), default=0.0)
        distance = _num(kpis.get("trajectory_distance_p95_m"), default=0.0)
        if base_distance > 0:
            score += max(-14.0, min(10.0, (base_distance - distance) / base_distance * 24.0))

        base_clean = _num(baseline_kpis.get("path_cleanliness_score_10"), default=0.0)
        clean = _num(kpis.get("path_cleanliness_score_10"), default=0.0)
        if base_clean > 0:
            score += max(-8.0, min(6.0, (clean - base_clean) * 3.0))

        base_max_step = _num(baseline_kpis.get("output_max_step_m"), default=0.0)
        max_step = _num(kpis.get("output_max_step_m"), default=0.0)
        if base_max_step > 0:
            score += max(-8.0, min(5.0, (base_max_step - max_step) / base_max_step * 10.0))

    score = round(max(0.0, min(100.0, score)), 3)
    return {
        "score": score,
        "accepted": bool(accepted),
        "passed": bool(accepted),
        "reject_reasons": reject_reasons,
        "diagnostic_notes": shape_notes,
        "scenario_rule": scenario,
        "policy_version": "baseline_safety_v2",
    }


def _summarize_trial(
    label: str,
    replay: dict[str, Any],
    changes: list[dict[str, Any]],
    baseline_kpis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scoring = _score_scenario(label, replay["summary"], baseline_kpis)
    return {
        "label": replay["label"],
        "session_id": replay["session_id"],
        "session_dir": replay["session_dir"],
        "summary_path": replay["summary_path"],
        "tuning": replay["tuning"],
        "changes": changes,
        "score": scoring["score"],
        "accepted": scoring["accepted"],
        "passed": scoring["passed"],
        "target_pass": False,
        "reject_reasons": scoring["reject_reasons"],
        "diagnostic_notes": scoring["diagnostic_notes"],
        "policy_version": scoring["policy_version"],
        "kpis": replay["summary"],
    }


def _build_variant_queue(selected_keys: list[str], max_trials: int) -> list[tuple[dict[str, Any], float | int]]:
    specs = _spec_by_key()
    queue: list[tuple[dict[str, Any], float | int]] = []
    for key in selected_keys:
        spec = specs.get(key)
        if not spec:
            continue
        for step in _candidate_steps(spec):
            queue.append((spec, step))
    return queue[: max(0, int(max_trials))]


def _mark_target_pass(trial: dict[str, Any], target_score: float) -> None:
    trial["target_pass"] = bool(trial.get("accepted")) and float(trial.get("score") or 0.0) >= float(target_score)


def _is_better_trial(candidate: dict[str, Any], current: dict[str, Any]) -> bool:
    candidate_accepted = bool(candidate.get("accepted"))
    current_accepted = bool(current.get("accepted"))
    if candidate_accepted != current_accepted:
        return candidate_accepted
    candidate_score = float(candidate.get("score") or 0.0)
    current_score = float(current.get("score") or 0.0)
    return candidate_score > current_score


def run_loop(
    *,
    scenario: str,
    capture: Path,
    baseline_tuning: Path,
    candidate_tuning: Path,
    selected_params: list[str],
    max_trials: int,
    speed: float,
    timeout_s: float | None,
    target_score: float,
    runtime_settings: Path | None = None,
) -> dict[str, Any]:
    run_id = f"{_now_stamp()}_{scenario}"
    run_dir = RUN_ROOT / run_id
    tuning_dir = run_dir / "tunings"
    run_dir.mkdir(parents=True, exist_ok=True)

    baseline_tuning = _project_path(baseline_tuning)
    candidate_tuning = _project_path(candidate_tuning)
    capture = _project_path(capture)
    runtime_settings = _project_path(runtime_settings) if runtime_settings is not None else None
    base_candidate_config = _load_json(candidate_tuning)

    metadata = {
        "schema_version": 1,
        "run_id": run_id,
        "scenario": scenario,
        "capture": str(capture),
        "baseline_tuning": str(baseline_tuning),
        "candidate_tuning_seed": str(candidate_tuning),
        "runtime_settings": str(runtime_settings) if runtime_settings is not None else None,
        "selected_params": selected_params,
        "max_trials": int(max_trials),
        "speed": float(speed),
        "target_score": float(target_score),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    _write_json(run_dir / "metadata.json", metadata)

    baseline_replay = _run_replay(
        capture,
        baseline_tuning,
        run_dir=run_dir,
        label="baseline",
        speed=speed,
        timeout_s=timeout_s,
        runtime_settings=runtime_settings,
    )
    baseline_trial = _summarize_trial(scenario, baseline_replay, [])
    baseline_trial["accepted"] = True
    baseline_trial["passed"] = True
    baseline_trial["target_pass"] = float(baseline_trial.get("score") or 0.0) >= float(target_score)
    baseline_trial["reject_reasons"] = []

    seed_tuning = tuning_dir / "candidate_seed.json"
    _write_json(seed_tuning, base_candidate_config)
    seed_replay = _run_replay(
        capture,
        seed_tuning,
        run_dir=run_dir,
        label="candidate_seed",
        speed=speed,
        timeout_s=timeout_s,
        runtime_settings=runtime_settings,
    )
    seed_trial = _summarize_trial(scenario, seed_replay, [], baseline_trial["kpis"])
    _mark_target_pass(seed_trial, target_score)
    trials = [seed_trial]
    best_trial = seed_trial
    best_config = base_candidate_config

    for index, (spec, step) in enumerate(_build_variant_queue(selected_params, max_trials), start=1):
        variant_config, changes = _apply_variant(best_config, spec, step)
        if not changes:
            continue
        tuning_path = tuning_dir / f"candidate_iter_{index:02d}_{spec['key'].replace('.', '_')}.json"
        _write_json(tuning_path, variant_config)
        replay = _run_replay(
            capture,
            tuning_path,
            run_dir=run_dir,
            label=f"candidate_iter_{index:02d}",
            speed=speed,
            timeout_s=timeout_s,
            runtime_settings=runtime_settings,
        )
        trial = _summarize_trial(scenario, replay, changes, baseline_trial["kpis"])
        _mark_target_pass(trial, target_score)
        trials.append(trial)
        if _is_better_trial(trial, best_trial):
            best_trial = trial
            best_config = variant_config
        if bool(trial["target_pass"]):
            break

    best_tuning_path = run_dir / "best_tuning.json"
    if best_trial.get("tuning"):
        best_tuning_path.write_text(Path(best_trial["tuning"]).read_text(encoding="utf-8"), encoding="utf-8")

    result = {
        **metadata,
        "selection_policy_version": "baseline_safety_v2",
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "status": "pass" if bool(best_trial["target_pass"]) else "fail",
        "baseline": baseline_trial,
        "best": {**best_trial, "best_tuning_path": str(best_tuning_path)},
        "trials": trials,
        "run_dir": str(run_dir),
    }
    _write_json(run_dir / "result.json", result)
    print(f"Tuning loop result: {run_dir / 'result.json'}")
    print(f"Status: {result['status']} | best={best_trial['session_id']} | score={best_trial['score']}")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run config-only replay tuning loop for Radar Lab.")
    parser.add_argument("--scenario", choices=sorted(ISK_SCENARIOS), required=True)
    parser.add_argument("--capture", help="Raw capture path. Defaults to the selected ISK scenario capture.")
    parser.add_argument("--baseline-tuning", required=True)
    parser.add_argument("--candidate-tuning", required=True)
    parser.add_argument(
        "--runtime-settings",
        default="config/live_motion_runtime_isk.json",
        help="Runtime settings JSON. Defaults to the ISK runtime settings.",
    )
    parser.add_argument("--params", nargs="*", default=[spec["key"] for spec in PARAMETER_SPECS])
    parser.add_argument("--max-trials", type=int, default=6)
    parser.add_argument("--speed", type=float, default=5.0)
    parser.add_argument("--timeout-s", type=float, default=0.0)
    parser.add_argument("--target-score", type=float, default=80.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    scenario_info = ISK_SCENARIOS[args.scenario]
    capture = Path(args.capture or scenario_info["capture"])
    run_loop(
        scenario=args.scenario,
        capture=capture,
        baseline_tuning=Path(args.baseline_tuning),
        candidate_tuning=Path(args.candidate_tuning),
        selected_params=list(args.params),
        max_trials=int(args.max_trials),
        speed=float(args.speed),
        timeout_s=float(args.timeout_s) if args.timeout_s and args.timeout_s > 0 else None,
        target_score=float(args.target_score),
        runtime_settings=Path(args.runtime_settings) if args.runtime_settings else None,
    )


if __name__ == "__main__":
    main()
