from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "docs" / "evals" / "runs"
SESSION_RE = re.compile(r"Logging session to:\s*(.+)")
TRAJECTORY_RE = re.compile(r"Trajectory replay report:\s*(.+trajectory_replay\.html)")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _safe_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    return safe.strip("_") or "radar_eval"


def _project_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _session_root() -> Path:
    return PROJECT_ROOT / "logs" / "live_motion_viewer"


def _session_dirs() -> set[Path]:
    root = _session_root()
    if not root.exists():
        return set()
    return {path.resolve() for path in root.iterdir() if path.is_dir()}


def _resolve_session(value: str) -> Path:
    path = _project_path(value)
    if path.name == "summary.json":
        path = path.parent
    if path.exists() and path.is_dir():
        return path.resolve()

    shorthand = _session_root() / value
    if shorthand.exists() and shorthand.is_dir():
        return shorthand.resolve()
    raise FileNotFoundError(f"Session not found: {value}")


def _nested_get(data: dict[str, Any], dotted_key: str) -> Any:
    current: Any = data
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _compare(actual: Any, op: str, expected: Any) -> bool:
    if op in {"in", "not_in"}:
        expected_values = expected if isinstance(expected, list) else [expected]
        result = actual in expected_values
        return result if op == "in" else not result

    if op in {"==", "eq"}:
        return actual == expected
    if op in {"!=", "ne"}:
        return actual != expected

    actual_number = _as_number(actual)
    expected_number = _as_number(expected)
    if actual_number is None or expected_number is None:
        return False

    if op == "<":
        return actual_number < expected_number
    if op == "<=":
        return actual_number <= expected_number
    if op == ">":
        return actual_number > expected_number
    if op == ">=":
        return actual_number >= expected_number
    raise ValueError(f"Unsupported criterion operator: {op}")


def _command_text(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def _run_command(
    command: list[str],
    *,
    output_dir: Path,
    label: str,
    env: dict[str, str] | None = None,
    dry_run: bool = False,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    command_record = {
        "label": label,
        "command": command,
        "command_text": _command_text(command),
        "started_at": _now(),
        "dry_run": bool(dry_run),
    }
    if dry_run:
        print(f"[dry-run] {label}: {command_record['command_text']}")
        command_record.update({"returncode": None, "finished_at": _now()})
        return command_record

    print(f"[run] {label}: {command_record['command_text']}")
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    stdout_path = output_dir / f"{label}.stdout.log"
    stderr_path = output_dir / f"{label}.stderr.log"
    stdout_path.write_text(completed.stdout or "", encoding="utf-8", errors="replace")
    stderr_path.write_text(completed.stderr or "", encoding="utf-8", errors="replace")
    command_record.update(
        {
            "returncode": completed.returncode,
            "finished_at": _now(),
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
        }
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({label}, exit {completed.returncode}). "
            f"See {stdout_path} and {stderr_path}"
        )
    return command_record


def _infer_session_dir(stdout_text: str, before_dirs: set[Path]) -> Path:
    for line in stdout_text.splitlines():
        match = SESSION_RE.search(line)
        if match:
            path = _project_path(match.group(1).strip())
            if path.exists() and path.is_dir():
                return path.resolve()

        trajectory_match = TRAJECTORY_RE.search(line)
        if trajectory_match:
            path = _project_path(trajectory_match.group(1).strip()).parent
            if path.exists() and path.is_dir():
                return path.resolve()

    after_dirs = _session_dirs()
    new_dirs = sorted(after_dirs - before_dirs, key=lambda path: path.stat().st_mtime)
    if new_dirs:
        return new_dirs[-1].resolve()
    raise RuntimeError("Could not infer replay session directory from command output.")


def _summary_path(session_dir: Path) -> Path:
    return session_dir / "summary.json"


def _load_summary(session_dir: Path) -> dict[str, Any]:
    path = _summary_path(session_dir)
    if not path.exists():
        raise FileNotFoundError(f"summary.json not found: {path}")
    return _load_json(path)


def _load_summary_with_eval(session_dir: Path) -> dict[str, Any]:
    summary = _load_summary(session_dir)
    try:
        from tools.eval_harness.path_shape import build_path_shape_metrics

        summary["eval"] = {
            "path_shape": build_path_shape_metrics(session_dir),
        }
    except Exception as exc:
        summary["eval"] = {
            "path_shape": {
                "error": repr(exc),
            }
        }
    return summary


def _build_summary(session_dir: Path, output_dir: Path, dry_run: bool) -> dict[str, Any]:
    command = [sys.executable, "-m", "tools.diagnostics.session_report", str(session_dir)]
    return _run_command(command, output_dir=output_dir, label=f"report_{session_dir.name}", dry_run=dry_run)


def _refresh_registry(output_dir: Path, dry_run: bool) -> dict[str, Any]:
    command = [sys.executable, "-m", "tools.lab.registry"]
    return _run_command(command, output_dir=output_dir, label="registry_refresh", dry_run=dry_run)


def _build_stage_cache(
    session_id: str,
    task: dict[str, Any],
    output_dir: Path,
    dry_run: bool,
) -> dict[str, Any]:
    stage_cfg = task.get("stage_cache") or {}
    command = [
        sys.executable,
        "-m",
        "tools.lab.stage_cache",
        "--session",
        session_id,
    ]
    limit = int(stage_cfg.get("limit") or 0)
    if limit > 0:
        command.extend(["--limit", str(limit)])
    if stage_cfg.get("force"):
        command.append("--force")
    return _run_command(command, output_dir=output_dir, label=f"stage_cache_{session_id}", dry_run=dry_run)


def _run_replay(
    task: dict[str, Any],
    run_cfg: dict[str, Any],
    *,
    role: str,
    output_dir: Path,
    dry_run: bool,
    timeout_s: float | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    capture = run_cfg.get("capture") or task.get("capture")
    if not capture:
        raise ValueError("Task must define capture or role-specific capture.")

    command = [
        sys.executable,
        str(PROJECT_ROOT / "real-time" / "live_motion_replay.py"),
        "--capture",
        str(capture),
        "--speed",
        str(run_cfg.get("speed", task.get("speed", 1.0))),
    ]
    if task.get("loop") or run_cfg.get("loop"):
        command.append("--loop")
    if task.get("wait") or run_cfg.get("wait"):
        command.append("--wait")
    tuning = run_cfg.get("tuning")
    if tuning:
        command.extend(["--tuning", str(tuning)])

    before_dirs = _session_dirs()
    command_record = _run_command(
        command,
        output_dir=output_dir,
        label=f"{role}_replay",
        dry_run=dry_run,
        timeout_s=timeout_s,
    )
    if dry_run:
        return {
            "role": role,
            "capture": capture,
            "tuning": tuning,
            "session_id": None,
            "session_dir": None,
            "summary_path": None,
        }, command_record

    stdout_text = Path(command_record["stdout_log"]).read_text(encoding="utf-8", errors="replace")
    session_dir = _infer_session_dir(stdout_text, before_dirs)
    report_record = _build_summary(session_dir, output_dir, dry_run=False)
    summary = _load_summary_with_eval(session_dir)
    return {
        "role": role,
        "capture": capture,
        "tuning": tuning,
        "session_id": session_dir.name,
        "session_dir": str(session_dir),
        "summary_path": str(_summary_path(session_dir)),
        "report_command": report_record,
        "summary": {
            "transport_quality": _nested_get(summary, "transport_quality.category"),
            "candidate_to_confirmed_ratio": _nested_get(
                summary,
                "performance.continuity.candidate_to_confirmed_ratio",
            ),
            "lead_switch_count": _nested_get(
                summary,
                "performance.continuity.lead_confirmed.switch_count",
            ),
            "unique_confirmed_track_ids": _nested_get(
                summary,
                "performance.continuity.unique_confirmed_track_ids",
            ),
            "path_cleanliness_score_10": _nested_get(
                summary,
                "performance.geometry.reference.path_cleanliness_score_10",
            ),
            "path_shape_policy_pass": _nested_get(
                summary,
                "eval.path_shape.policy.overall_pass",
            ),
            "path_shape_width_ratio_delta": _nested_get(
                summary,
                "eval.path_shape.output_vs_raw.width_ratio_delta",
            ),
        },
    }, command_record


def _use_existing_session(value: str, role: str) -> dict[str, Any]:
    session_dir = _resolve_session(value)
    summary = _load_summary_with_eval(session_dir)
    return {
        "role": role,
        "session_id": session_dir.name,
        "session_dir": str(session_dir),
        "summary_path": str(_summary_path(session_dir)),
        "summary": {
            "transport_quality": _nested_get(summary, "transport_quality.category"),
            "candidate_to_confirmed_ratio": _nested_get(
                summary,
                "performance.continuity.candidate_to_confirmed_ratio",
            ),
            "lead_switch_count": _nested_get(
                summary,
                "performance.continuity.lead_confirmed.switch_count",
            ),
            "unique_confirmed_track_ids": _nested_get(
                summary,
                "performance.continuity.unique_confirmed_track_ids",
            ),
            "path_cleanliness_score_10": _nested_get(
                summary,
                "performance.geometry.reference.path_cleanliness_score_10",
            ),
            "path_shape_policy_pass": _nested_get(
                summary,
                "eval.path_shape.policy.overall_pass",
            ),
            "path_shape_width_ratio_delta": _nested_get(
                summary,
                "eval.path_shape.output_vs_raw.width_ratio_delta",
            ),
        },
    }


def _criterion_value(
    criterion: dict[str, Any],
    *,
    baseline_summary: dict[str, Any] | None,
    candidate_summary: dict[str, Any],
) -> Any:
    metric = criterion["metric"]
    mode = criterion.get("mode", "candidate")
    if mode == "candidate":
        return _nested_get(candidate_summary, metric)
    if mode == "baseline":
        if baseline_summary is None:
            return None
        return _nested_get(baseline_summary, metric)
    if mode == "delta":
        if baseline_summary is None:
            return None
        before = _as_number(_nested_get(baseline_summary, metric))
        after = _as_number(_nested_get(candidate_summary, metric))
        if before is None or after is None:
            return None
        return round(after - before, 6)
    raise ValueError(f"Unsupported criterion mode: {mode}")


def _evaluate(
    task: dict[str, Any],
    *,
    baseline_session: dict[str, Any] | None,
    candidate_session: dict[str, Any],
) -> list[dict[str, Any]]:
    baseline_summary = _load_summary_with_eval(Path(baseline_session["session_dir"])) if baseline_session else None
    candidate_summary = _load_summary_with_eval(Path(candidate_session["session_dir"]))

    results = []
    for index, criterion in enumerate(task.get("acceptance", []), start=1):
        actual = _criterion_value(
            criterion,
            baseline_summary=baseline_summary,
            candidate_summary=candidate_summary,
        )
        op = criterion.get("op", "<=")
        expected = criterion.get("value")
        passed = _compare(actual, op, expected)
        results.append(
            {
                "index": index,
                "name": criterion.get("name") or criterion["metric"],
                "metric": criterion["metric"],
                "mode": criterion.get("mode", "candidate"),
                "op": op,
                "expected": expected,
                "actual": actual,
                "passed": bool(passed),
            }
        )
    return results


def _print_outcome(outcome: dict[str, Any]) -> None:
    status = outcome["status"].upper()
    print(f"\n[{status}] {outcome['task']['name']}")
    candidate = outcome.get("candidate") or {}
    print(f"candidate session: {candidate.get('session_id')} | {candidate.get('session_dir')}")
    baseline = outcome.get("baseline") or {}
    if baseline:
        print(f"baseline session:  {baseline.get('session_id')} | {baseline.get('session_dir')}")
    for criterion in outcome.get("criteria", []):
        mark = "PASS" if criterion["passed"] else "FAIL"
        print(
            f"- {mark}: {criterion['name']} "
            f"actual={criterion['actual']} {criterion['op']} expected={criterion['expected']}"
        )
    print(f"outcome: {outcome['outcome_path']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one replay-based radar evaluation task and grade the resulting session."
    )
    parser.add_argument("task", help="Path to a JSON task spec under docs/evals/tasks.")
    parser.add_argument("--baseline-session", help="Reuse an existing baseline session id or directory.")
    parser.add_argument("--candidate-session", help="Grade an existing candidate session id or directory.")
    parser.add_argument("--force-baseline", action="store_true", help="Run baseline even if task has baseline.session.")
    parser.add_argument("--skip-baseline", action="store_true", help="Only run/grade the candidate.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running replay.")
    parser.add_argument("--no-stage-cache", action="store_true", help="Disable stage cache generation.")
    parser.add_argument("--timeout-s", type=float, default=None, help="Optional timeout for each subprocess.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()

    task_path = _project_path(args.task)
    task = _load_json(task_path)
    task_name = _safe_name(str(task.get("name") or task_path.stem))
    output_dir = _project_path(args.output_root) / f"{_stamp()}_{task_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    command_records: list[dict[str, Any]] = []
    baseline_info: dict[str, Any] | None = None
    candidate_info: dict[str, Any] | None = None

    if not args.skip_baseline:
        baseline_cfg = dict(task.get("baseline") or {})
        configured_session = args.baseline_session or baseline_cfg.get("session")
        if configured_session and not args.force_baseline:
            baseline_info = _use_existing_session(str(configured_session), "baseline")
        else:
            baseline_info, command_record = _run_replay(
                task,
                baseline_cfg,
                role="baseline",
                output_dir=output_dir,
                dry_run=args.dry_run,
                timeout_s=args.timeout_s,
            )
            if command_record:
                command_records.append(command_record)

    candidate_cfg = dict(task.get("candidate") or {})
    if args.candidate_session:
        candidate_info = _use_existing_session(args.candidate_session, "candidate")
    else:
        candidate_info, command_record = _run_replay(
            task,
            candidate_cfg,
            role="candidate",
            output_dir=output_dir,
            dry_run=args.dry_run,
            timeout_s=args.timeout_s,
        )
        if command_record:
            command_records.append(command_record)

    if args.dry_run:
        criteria: list[dict[str, Any]] = []
        status = "dry_run"
    else:
        registry_record = _refresh_registry(output_dir, dry_run=False)
        command_records.append(registry_record)
        criteria = _evaluate(task, baseline_session=baseline_info, candidate_session=candidate_info)
        status = "pass" if criteria and all(item["passed"] for item in criteria) else "fail"

        stage_cfg = task.get("stage_cache") or {}
        stage_mode = str(stage_cfg.get("mode", "on_fail")).lower()
        should_build_stage_cache = (
            not args.no_stage_cache
            and candidate_info is not None
            and stage_mode in {"always", "on_fail"}
            and (stage_mode == "always" or status == "fail")
        )
        if should_build_stage_cache:
            command_records.append(
                _build_stage_cache(candidate_info["session_id"], task, output_dir, dry_run=False)
            )

    outcome = {
        "schema_version": 1,
        "generated_at": _now(),
        "task": {
            "name": task.get("name") or task_path.stem,
            "path": str(task_path),
            "description": task.get("description"),
        },
        "status": status,
        "baseline": baseline_info,
        "candidate": candidate_info,
        "criteria": criteria,
        "commands": command_records,
    }
    outcome_path = output_dir / "outcome.json"
    outcome["outcome_path"] = str(outcome_path)
    _write_json(outcome_path, outcome)
    _print_outcome(outcome)

    if status == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
