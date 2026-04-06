from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = [
    ("assessment.overall.score", "Operational score", "higher"),
    ("performance.throughput.render_vs_expected_ratio", "Render FPS vs target", "higher"),
    ("performance.compute.compute_utilization_p95_ratio", "Compute utilization p95", "lower"),
    ("performance.jitter.render_latency_jitter_ms", "Render jitter", "lower"),
    ("performance.continuity.candidate_to_confirmed_ratio", "Candidate/confirmed ratio", "lower"),
    ("performance.continuity.lead_confirmed.switch_count", "Lead confirmed switch", "lower"),
    ("processed.invalid_rate", "Processed invalid rate", "lower"),
    ("render.invalid_rate", "Render invalid rate", "lower"),
    ("processed.capture_to_process_ms.mean", "Processed latency mean", "lower"),
    ("processed.capture_to_process_ms.p95", "Processed latency p95", "lower"),
    ("processed.confirmed_track_count.mean", "Confirmed track mean", "higher"),
    ("processed.multi_confirmed_success_rate", "Processed multi-target success", "higher"),
    ("render.capture_to_render_ms.mean", "Render latency mean", "lower"),
    ("render.capture_to_render_ms.p95", "Render latency p95", "lower"),
    ("render.display_track_count.mean", "Display track mean", "higher"),
    ("render.multi_display_success_rate", "Render multi-target success", "higher"),
    ("render.skipped_render_frames.mean_per_render", "Skipped frames per render", "lower"),
]


def _resolve_summary_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_dir():
        return path / "summary.json"
    return path


def _load_summary(path_value: str):
    summary_path = _resolve_summary_path(path_value)
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8")), summary_path


def _nested_get(data, dotted_key: str):
    current = data
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _round_or_none(value, digits=4):
    if value is None:
        return None
    return round(float(value), digits)


def _percent_change(before_value, after_value):
    if before_value in (None, 0) or after_value is None:
        return None
    return (after_value - before_value) / abs(before_value)


def _judgement(direction: str, delta):
    if delta is None:
        return "n/a"
    if abs(delta) < 1e-12:
        return "same"
    if direction == "lower":
        return "improved" if delta < 0 else "regressed"
    return "improved" if delta > 0 else "regressed"


def build_comparison(before_summary, after_summary, before_path: Path, after_path: Path):
    comparison = {
        "before": {
            "path": str(before_path),
            "session_id": before_summary.get("session_id"),
            "variant": before_summary.get("session_meta", {}).get("variant"),
            "scenario_id": before_summary.get("session_meta", {}).get("scenario_id"),
        },
        "after": {
            "path": str(after_path),
            "session_id": after_summary.get("session_id"),
            "variant": after_summary.get("session_meta", {}).get("variant"),
            "scenario_id": after_summary.get("session_meta", {}).get("scenario_id"),
        },
        "metrics": [],
    }

    for dotted_key, label, direction in METRICS:
        before_value = _nested_get(before_summary, dotted_key)
        after_value = _nested_get(after_summary, dotted_key)
        if isinstance(before_value, (int, float)) and isinstance(after_value, (int, float)):
            delta = after_value - before_value
            pct_change = _percent_change(before_value, after_value)
        else:
            delta = None
            pct_change = None

        comparison["metrics"].append(
            {
                "key": dotted_key,
                "label": label,
                "direction": direction,
                "before": before_value,
                "after": after_value,
                "delta": _round_or_none(delta),
                "percent_change": _round_or_none(pct_change),
                "judgement": _judgement(direction, delta),
            }
        )

    return comparison


def main():
    parser = argparse.ArgumentParser(
        description="Compare two session summaries from live_motion_viewer logs."
    )
    parser.add_argument("before", help="Path to the baseline session directory or summary.json")
    parser.add_argument("after", help="Path to the candidate session directory or summary.json")
    parser.add_argument(
        "--output",
        help="Optional output path for comparison json.",
    )
    args = parser.parse_args()

    before_summary, before_path = _load_summary(args.before)
    after_summary, after_path = _load_summary(args.after)
    comparison = build_comparison(before_summary, after_summary, before_path, after_path)

    output_path = Path(args.output) if args.output else after_path.parent / f"comparison_vs_{before_path.parent.name}.json"
    output_path.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Comparison written to: {output_path}")
    for metric in comparison["metrics"]:
        print(
            f"{metric['label']}: before={metric['before']} "
            f"after={metric['after']} delta={metric['delta']} ({metric['judgement']})"
        )


if __name__ == "__main__":
    main()
