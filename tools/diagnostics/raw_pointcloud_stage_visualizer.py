"""Visualize raw-replay point candidates through detection/tracking stages.

This tool is offline-only. It reads stage-cache data generated from saved raw
captures, then writes diagnostic images under logs/diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tools.lab import stage_cache


STAGE_PANELS = (
    ("Raw-derived angle candidates", "angle_validation"),
    ("After coarse merge", "coarse_merge"),
    ("After body-center", "body_center"),
    ("After final merge", "final_merge"),
    ("After DBSCAN/duplicate", "dbscan"),
    ("Tracker/display output", "tracker"),
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _load_cache(project_root: Path, session_id: str, ablation_mode: str, frame_limit: int | None, force: bool) -> dict:
    manifest = stage_cache.load_stage_cache_manifest(project_root, session_id, ablation_mode)
    if manifest is None or force:
        manifest = stage_cache.build_stage_cache(
            project_root,
            session_id,
            frame_limit=frame_limit,
            force=force,
            ablation_mode=ablation_mode,
        )
    return manifest


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _safe_float(value, default=np.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(number):
        return float(default)
    return number


def _point_from_candidate(candidate: dict, *, stage: str, kind: str = "candidate", label: str = "") -> dict | None:
    x_m = _safe_float(candidate.get("x_m"))
    y_m = _safe_float(candidate.get("y_m"))
    if not np.isfinite(x_m) or not np.isfinite(y_m):
        return None
    return {
        "x_m": x_m,
        "y_m": y_m,
        "range_m": _safe_float(candidate.get("range_m")),
        "angle_deg": _safe_float(candidate.get("angle_deg")),
        "doppler_bin": int(candidate.get("doppler_bin", -1) or -1),
        "score": _safe_float(candidate.get("score"), 0.0),
        "stage": stage,
        "kind": kind,
        "label": label,
        "track_id": candidate.get("track_id"),
    }


def _points_from_candidates(candidates: Iterable[dict], *, stage: str) -> list[dict]:
    points: list[dict] = []
    for index, candidate in enumerate(candidates or [], start=1):
        point = _point_from_candidate(candidate, stage=stage, label=str(index))
        if point is not None:
            points.append(point)
    return points


def _trace_detection(trace: dict) -> dict:
    return trace.get("detection") or {}


def _stage_points(trace: dict, frame: dict, stage_key: str) -> tuple[list[dict], list[tuple[dict, dict]]]:
    detection = _trace_detection(trace)
    arrows: list[tuple[dict, dict]] = []
    if stage_key == "angle_validation":
        return _points_from_candidates(
            (detection.get("angle_validation") or {}).get("top_candidates") or [],
            stage=stage_key,
        ), arrows
    if stage_key == "coarse_merge":
        return _points_from_candidates(
            (detection.get("candidate_merge_coarse") or {}).get("after_top") or [],
            stage=stage_key,
        ), arrows
    if stage_key == "body_center":
        pairs = (detection.get("body_center_refinement") or {}).get("pairs") or []
        after_points: list[dict] = []
        for index, pair in enumerate(pairs, start=1):
            before = _point_from_candidate(pair.get("before") or {}, stage=stage_key, kind="before", label=str(index))
            after = _point_from_candidate(pair.get("after") or {}, stage=stage_key, kind="after", label=str(index))
            if before is not None and after is not None:
                arrows.append((before, after))
                after_points.append(after)
        if after_points:
            return after_points, arrows
        return _points_from_candidates(
            (detection.get("candidate_merge_coarse") or {}).get("after_top") or [],
            stage=stage_key,
        ), arrows
    if stage_key == "final_merge":
        return _points_from_candidates(
            (detection.get("candidate_merge_final") or {}).get("after_top") or [],
            stage=stage_key,
        ), arrows
    if stage_key == "dbscan":
        return _points_from_candidates(
            (detection.get("dbscan") or {}).get("output_top") or (detection.get("final_output") or {}).get("top_detections") or [],
            stage=stage_key,
        ), arrows
    if stage_key == "tracker":
        points: list[dict] = []
        display = trace.get("display_output") or {}
        for item in display.get("confirmed_tracks") or frame.get("confirmed_tracks") or []:
            point = _point_from_candidate(item, stage=stage_key, kind="confirmed", label=f"id {item.get('track_id')}")
            if point is not None:
                point["track_id"] = item.get("track_id")
                points.append(point)
        for item in display.get("tentative_tracks") or frame.get("tentative_tracks") or []:
            point = _point_from_candidate(item, stage=stage_key, kind="tentative", label=f"t {item.get('track_id')}")
            if point is not None:
                point["track_id"] = item.get("track_id")
                points.append(point)
        return points, arrows
    return [], arrows


def _trace_count(trace: dict, key_path: tuple[str, ...], default=0) -> int:
    node = trace
    for key in key_path:
        if not isinstance(node, dict):
            return int(default)
        node = node.get(key)
    try:
        return int(node)
    except (TypeError, ValueError):
        return int(default)


def _frame_score(trace: dict, expected_count: int | None) -> float:
    detection = _trace_detection(trace)
    angle_count = _trace_count(detection, ("angle_validation", "passed_count"))
    final_merge = detection.get("candidate_merge_final") or {}
    final_before = int(final_merge.get("before_count") or 0)
    final_after = int(final_merge.get("after_count") or 0)
    dbscan = detection.get("dbscan") or {}
    dbscan_input = int(dbscan.get("input_count") or 0)
    dbscan_output = int(dbscan.get("output_count") or 0)
    duplicate = detection.get("duplicate_suppression") or {}
    duplicate_before = int(duplicate.get("before_count") or 0)
    duplicate_after = int(duplicate.get("after_count") or 0)
    suspicious = int((trace.get("rai_collapse_diagnostics") or {}).get("suspicious_count") or 0)
    display = trace.get("display_output") or {}
    track_count = len(display.get("confirmed_tracks") or []) + len(display.get("tentative_tracks") or [])
    score = 0.0
    score += max(angle_count - dbscan_output, 0) * 1.2
    score += max(final_before - final_after, 0) * 1.5
    score += max(dbscan_input - dbscan_output, 0) * 1.5
    score += max(duplicate_before - duplicate_after, 0) * 1.8
    score += suspicious * 2.5
    if expected_count is not None and expected_count > 0:
        score += abs(track_count - expected_count) * 2.0
        final_output = _trace_count(detection, ("final_output", "output_count"))
        score += abs(final_output - expected_count) * 1.5
    return score


def _select_ordinals(traces: list[dict], frames: list[dict], frames_arg: str, max_frames: int, expected_count: int | None) -> list[int]:
    if frames_arg.strip().lower() not in {"auto", "suspicious", ""}:
        result = []
        for part in frames_arg.split(","):
            part = part.strip()
            if not part:
                continue
            result.append(int(part))
        return sorted(set(result))
    scored = []
    for index, trace in enumerate(traces):
        ordinal = int((frames[index] if index < len(frames) else {}).get("ordinal", index))
        scored.append((_frame_score(trace, expected_count), ordinal))
    scored.sort(reverse=True)
    selected = [ordinal for score, ordinal in scored if score > 0][: int(max_frames)]
    if selected:
        return sorted(set(selected))
    if not frames:
        return []
    count = min(int(max_frames), len(frames))
    indices = np.unique(np.round(np.linspace(0, len(frames) - 1, count)).astype(int))
    return [int(frames[index].get("ordinal", index)) for index in indices]


def _limits_for_topdown(point_sets: Iterable[list[dict]], manifest: dict) -> tuple[tuple[float, float], tuple[float, float]]:
    xs = [0.0]
    ys = [0.0]
    for points in point_sets:
        for point in points:
            xs.append(float(point["x_m"]))
            ys.append(float(point["y_m"]))
    roi = manifest.get("roi") or {}
    lateral = max(_safe_float(roi.get("lateral_m"), 1.5), max(abs(x) for x in xs) + 0.35, 1.5)
    forward = max(_safe_float(roi.get("forward_m"), 4.0), max(ys) + 0.5, 3.5)
    return (-float(lateral), float(lateral)), (0.0, float(forward))


def _limits_for_radar_pov(point_sets: Iterable[list[dict]], manifest: dict) -> tuple[tuple[float, float], tuple[float, float]]:
    angles = [0.0]
    ranges = [0.0]
    for points in point_sets:
        for point in points:
            angle = _safe_float(point.get("angle_deg"))
            range_m = _safe_float(point.get("range_m"))
            if np.isfinite(angle):
                angles.append(angle)
            if np.isfinite(range_m):
                ranges.append(range_m)
    roi = manifest.get("roi") or {}
    forward = max(_safe_float(roi.get("forward_m"), 4.0), max(ranges) + 0.5, 3.5)
    angle_limit = max(45.0, min(80.0, max(abs(angle) for angle in angles) + 10.0))
    return (-float(angle_limit), float(angle_limit)), (0.0, float(forward))


def _draw_topdown(ax, points: list[dict], arrows: list[tuple[dict, dict]], title: str, xlim, ylim):
    ax.set_title(f"{title}\ncount={len(points)}", fontsize=10)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#dbe3ec", linewidth=0.8)
    ax.scatter([0], [0], marker="s", s=48, color="#111827", label="radar", zorder=5)
    max_radius = max(ylim[1], abs(xlim[0]), abs(xlim[1]))
    theta = np.linspace(np.radians(-70), np.radians(70), 160)
    for radius in np.arange(1.0, max_radius + 0.1, 1.0):
        ax.plot(radius * np.sin(theta), radius * np.cos(theta), color="#eef2f7", linewidth=0.8, zorder=0)
    if arrows:
        for before, after in arrows:
            ax.plot([before["x_m"], after["x_m"]], [before["y_m"], after["y_m"]], color="#94a3b8", linewidth=1.0)
            ax.scatter([before["x_m"]], [before["y_m"]], marker="x", s=36, color="#64748b", zorder=4)
    if points:
        dopplers = np.asarray([point.get("doppler_bin", -1) for point in points], dtype=float)
        sizes = [max(28.0, min(130.0, 35.0 + 25.0 * _safe_float(point.get("score"), 0.0))) for point in points]
        scatter = ax.scatter(
            [point["x_m"] for point in points],
            [point["y_m"] for point in points],
            c=dopplers,
            cmap="coolwarm",
            s=sizes,
            edgecolor="#0f172a",
            linewidth=0.5,
            alpha=0.88,
            zorder=6,
        )
        for point in points:
            label = str(point.get("label") or "")
            if label:
                ax.text(point["x_m"] + 0.03, point["y_m"] + 0.03, label, fontsize=8, color="#111827")
        return scatter
    return None


def _draw_radar_pov(ax, points: list[dict], arrows: list[tuple[dict, dict]], title: str, xlim, ylim):
    ax.set_title(f"{title}\ncount={len(points)}", fontsize=10)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, color="#dbe3ec", linewidth=0.8)
    ax.axvline(0, color="#cbd5e1", linewidth=1.0)
    for angle in np.arange(np.ceil(xlim[0] / 15.0) * 15.0, xlim[1] + 0.1, 15.0):
        ax.axvline(angle, color="#eef2f7", linewidth=0.7, zorder=0)
    for range_m in np.arange(1.0, ylim[1] + 0.1, 1.0):
        ax.axhline(range_m, color="#eef2f7", linewidth=0.7, zorder=0)
    ax.text(
        0.02,
        0.97,
        "radar POV: left/right angle vs depth\ntrue elevation is not retained",
        transform=ax.transAxes,
        va="top",
        fontsize=7,
        color="#64748b",
    )
    if arrows:
        for before, after in arrows:
            before_angle = _safe_float(before.get("angle_deg"))
            after_angle = _safe_float(after.get("angle_deg"))
            before_range = _safe_float(before.get("range_m"))
            after_range = _safe_float(after.get("range_m"))
            if not all(np.isfinite(v) for v in [before_angle, after_angle, before_range, after_range]):
                continue
            ax.plot([before_angle, after_angle], [before_range, after_range], color="#94a3b8", linewidth=1.0)
            ax.scatter([before_angle], [before_range], marker="x", s=36, color="#64748b", zorder=4)
    if points:
        dopplers = np.asarray([point.get("doppler_bin", -1) for point in points], dtype=float)
        sizes = [max(28.0, min(130.0, 35.0 + 25.0 * _safe_float(point.get("score"), 0.0))) for point in points]
        scatter = ax.scatter(
            [point.get("angle_deg") for point in points],
            [point.get("range_m") for point in points],
            c=dopplers,
            cmap="coolwarm",
            s=sizes,
            edgecolor="#0f172a",
            linewidth=0.5,
            alpha=0.88,
            zorder=6,
        )
        for point in points:
            label = str(point.get("label") or "")
            if label:
                ax.text(point.get("angle_deg", 0.0) + 1.0, point.get("range_m", 0.0) + 0.03, label, fontsize=8, color="#111827")
        return scatter
    return None


def _write_frame_image(
    *,
    output_path: Path,
    session_id: str,
    ablation_mode: str,
    frame: dict,
    trace: dict,
    manifest: dict,
    view: str,
) -> dict:
    panel_points: list[list[dict]] = []
    panel_arrows: list[list[tuple[dict, dict]]] = []
    for _title, stage_key in STAGE_PANELS:
        points, arrows = _stage_points(trace, frame, stage_key)
        panel_points.append(points)
        panel_arrows.append(arrows)
    if view == "radar":
        xlim, ylim = _limits_for_radar_pov(panel_points, manifest)
        view_note = "radar POV azimuth-range view"
    else:
        xlim, ylim = _limits_for_topdown(panel_points, manifest)
        view_note = "top-down x-y view"

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    fig.suptitle(
        f"Raw replay point-cloud stages | {view_note} | session={session_id} | frame={frame.get('frame_id')} | ordinal={frame.get('ordinal')} | mode={ablation_mode}",
        fontsize=13,
    )
    last_scatter = None
    for ax, (title, _stage_key), points, arrows in zip(axes.ravel(), STAGE_PANELS, panel_points, panel_arrows):
        if view == "radar":
            scatter = _draw_radar_pov(ax, points, arrows, title, xlim, ylim)
            ax.set_xlabel("azimuth angle (deg)")
            ax.set_ylabel("range / depth (m)")
        else:
            scatter = _draw_topdown(ax, points, arrows, title, xlim, ylim)
            ax.set_xlabel("x lateral (m)")
            ax.set_ylabel("y forward (m)")
        if scatter is not None:
            last_scatter = scatter
    if last_scatter is not None:
        cbar = fig.colorbar(last_scatter, ax=axes.ravel().tolist(), shrink=0.72, pad=0.012)
        cbar.set_label("doppler bin")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    detection = _trace_detection(trace)
    display = trace.get("display_output") or {}
    return {
        "ordinal": int(frame.get("ordinal", -1)),
        "frame_id": int(frame.get("frame_id", -1)),
        "cfar_count": _trace_count(detection, ("cfar", "candidate_count")),
        "angle_validation_count": _trace_count(detection, ("angle_validation", "passed_count")),
        "coarse_merge_count": len(panel_points[1]),
        "body_center_count": len(panel_points[2]),
        "final_merge_count": len(panel_points[3]),
        "dbscan_count": len(panel_points[4]),
        "confirmed_count": len(display.get("confirmed_tracks") or frame.get("confirmed_tracks") or []),
        "tentative_count": len(display.get("tentative_tracks") or frame.get("tentative_tracks") or []),
        "view": view,
        "image_path": str(output_path),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_visuals(
    *,
    project_root: Path,
    session_id: str,
    ablation_mode: str,
    frame_limit: int | None,
    force_cache: bool,
    frames_arg: str,
    max_frames: int,
    expected_count: int | None,
    output_root: Path,
    view: str,
) -> dict:
    manifest = _load_cache(project_root, session_id, ablation_mode, frame_limit, force_cache)
    paths = stage_cache.stage_cache_paths(project_root, session_id, ablation_mode)
    frames = stage_cache.load_stage_cache_frames(project_root, session_id, ablation_mode)
    traces = _read_jsonl(paths["trace_path"])
    selected_ordinals = _select_ordinals(traces, frames, frames_arg, max_frames, expected_count)
    frame_by_ordinal = {int(frame.get("ordinal", index)): frame for index, frame in enumerate(frames)}
    trace_by_ordinal = {int(frames[index].get("ordinal", index)): traces[index] for index in range(min(len(frames), len(traces)))}

    output_dir = output_root / f"raw_pointcloud_stages_{_now_tag()}_{session_id}"
    rows: list[dict] = []
    for ordinal in selected_ordinals:
        frame = frame_by_ordinal.get(int(ordinal))
        trace = trace_by_ordinal.get(int(ordinal))
        if frame is None or trace is None:
            continue
        views = ["radar", "topdown"] if view == "both" else [view]
        for view_name in views:
            suffix = "radar_pov" if view_name == "radar" else "topdown"
            image_path = output_dir / f"frame_{int(ordinal):06d}_{suffix}.png"
            rows.append(
                _write_frame_image(
                    output_path=image_path,
                    session_id=session_id,
                    ablation_mode=ablation_mode,
                    frame=frame,
                    trace=trace,
                    manifest=manifest,
                    view=view_name,
                )
            )

    summary_path = output_dir / "pointcloud_stage_summary.csv"
    _write_csv(summary_path, rows)
    result = {
        "session_id": session_id,
        "ablation_mode": ablation_mode,
        "output_dir": str(output_dir),
        "selected_ordinals": selected_ordinals,
        "view": view,
        "frame_count": len(rows),
        "summary_path": str(summary_path),
        "images": [row["image_path"] for row in rows],
        "notes": [
            "Raw ADC does not contain a point cloud by itself.",
            "The first panel is the raw-replay angle-validated candidate cloud before merge/DBSCAN/tracker.",
            "Radar POV uses azimuth angle vs range/depth because the current Python detection trace does not retain true elevation per point.",
            "All panels are generated from saved raw replay stage-cache data.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    result["manifest_path"] = str(manifest_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create radar-view point-cloud stage images from raw replay stage cache."
    )
    parser.add_argument("--session", required=True, help="live_motion_viewer session id.")
    parser.add_argument("--ablation-mode", default="baseline", help="Stage-cache ablation mode.")
    parser.add_argument("--frame-limit", type=int, default=0, help="Frame limit used when building a missing cache. 0 means all frames.")
    parser.add_argument("--force-cache", action="store_true", help="Rebuild the stage cache before drawing.")
    parser.add_argument("--frames", default="auto", help="Comma-separated ordinals, or auto/suspicious.")
    parser.add_argument("--max-frames", type=int, default=8, help="Maximum auto-selected frames to draw.")
    parser.add_argument("--expected-count", type=int, default=0, help="Expected people count for suspicious-frame scoring.")
    parser.add_argument("--output-root", default="", help="Output root. Default: logs/diagnostics.")
    parser.add_argument("--view", choices=("radar", "topdown", "both"), default="radar", help="Output view. radar means azimuth-vs-range radar POV.")
    args = parser.parse_args()

    project_root = _repo_root()
    output_root = Path(args.output_root) if args.output_root else project_root / "logs" / "diagnostics"
    result = build_visuals(
        project_root=project_root,
        session_id=str(args.session),
        ablation_mode=str(args.ablation_mode),
        frame_limit=None if int(args.frame_limit) <= 0 else int(args.frame_limit),
        force_cache=bool(args.force_cache),
        frames_arg=str(args.frames),
        max_frames=max(1, int(args.max_frames)),
        expected_count=None if int(args.expected_count) <= 0 else int(args.expected_count),
        output_root=output_root,
        view=str(args.view),
    )
    print(f"Output directory: {result['output_dir']}")
    print(f"Summary CSV: {result['summary_path']}")
    for image in result["images"]:
        print(f"Image: {image}")


if __name__ == "__main__":
    main()
