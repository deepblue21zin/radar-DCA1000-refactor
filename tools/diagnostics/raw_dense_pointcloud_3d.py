"""Create dense raw-derived 3D point-cloud images and replays from saved frames.

The live tracker uses a collapsed 2D RAI map. This diagnostic script keeps
azimuth and elevation during offline projection so a radar-frame 3D cloud can
be inspected without changing the runtime pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from tools.lab import registry, stage_cache
from tools.runtime_core import DSP
from tools.runtime_core.radar_runtime import frame_to_radar_cube, remove_static_clutter
from tools.runtime_core.real_time_process import iter_raw_capture_frame_packets, load_raw_capture
from tools.runtime_core.virtual_array import (
    cached_iwr6843isk_virtual_array,
    geometry_range_azimuth_elevation_from_fft,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _resolve_runtime(project_root: Path, session_id: str):
    run_detail = registry.fetch_run_detail(project_root, session_id)
    if run_detail is None:
        registry.refresh_registry(project_root)
        run_detail = registry.fetch_run_detail(project_root, session_id)
    if run_detail is None:
        raise FileNotFoundError(f"Run session not found in registry: {session_id}")
    capture_dir = stage_cache._resolve_capture_dir(project_root, run_detail)
    capture_manifest, _, _ = load_raw_capture(capture_dir)
    runtime_summary = stage_cache._merge_runtime_summary(run_detail, capture_manifest)
    components = stage_cache._build_runtime_components(project_root, runtime_summary, ablation_mode="baseline")
    return run_detail, capture_dir, components


def _load_frame(capture_dir: Path, ordinal: int):
    for index, frame in enumerate(iter_raw_capture_frame_packets(capture_dir)):
        if index == int(ordinal):
            return frame
    raise IndexError(f"Frame ordinal {ordinal} is not present in {capture_dir}.")


def _load_frame_range(capture_dir: Path, start: int, end: int, step: int):
    selected = []
    start = max(0, int(start))
    end = max(start, int(end))
    step = max(1, int(step))
    for index, frame in enumerate(iter_raw_capture_frame_packets(capture_dir)):
        if index < start:
            continue
        if index > end:
            break
        if (index - start) % step == 0:
            selected.append((index, frame))
    if not selected:
        raise IndexError(f"No raw frames selected from {capture_dir} for range {start}..{end} step {step}.")
    return selected


def _dense_response_for_frame(raw_frame, runtime_config):
    radar_cube = frame_to_radar_cube(raw_frame.iq, runtime_config)
    if runtime_config.remove_static:
        radar_cube = remove_static_clutter(radar_cube)
    shared_fft = DSP.shared_range_doppler_fft(
        radar_cube,
        padding_size=[
            runtime_config.doppler_fft_size,
            runtime_config.range_fft_size,
        ],
    )
    model = cached_iwr6843isk_virtual_array(
        str(runtime_config.config_path),
        int(runtime_config.rx_num),
        int(runtime_config.tx_num),
    )
    response = geometry_range_azimuth_elevation_from_fft(
        shared_fft,
        raw_order=model.raw_order,
        x_lambda=model.x_lambda,
        z_lambda=model.z_lambda,
        azimuth_axis_rad=runtime_config.angle_axis_rad,
        elevation_axis_deg=runtime_config.angle_elevation_axis_deg,
        phase_sign=runtime_config.angle_phase_sign,
        channel_coefficients=(
            getattr(runtime_config, "channel_calibration_coefficients", ())
            if getattr(runtime_config, "channel_calibration_enabled", False)
            else None
        ),
    )
    return np.fft.fftshift(response, axes=0)


def _make_points(response, runtime_config, components, *, top_k: int, quantile: float):
    motion = np.asarray(response, dtype=np.float32)
    center = motion.shape[0] // 2
    guard = int(getattr(runtime_config, "doppler_guard_bins", 1))
    lower = max(center - guard, 0)
    upper = min(center + guard + 1, motion.shape[0])
    motion = np.array(motion, copy=True)
    motion[lower:upper, :, :, :] = 0.0

    score = np.max(motion, axis=0)
    doppler = np.argmax(motion, axis=0)

    min_range_bin = int(components.get("min_range_bin", 0))
    max_range_bin = int(components.get("max_range_bin", score.shape[0]))
    score[:min_range_bin, :, :] = 0.0
    score[max_range_bin:, :, :] = 0.0

    positive = score[score > 0]
    if positive.size == 0:
        return []
    threshold = float(np.quantile(positive, min(max(float(quantile), 0.0), 1.0)))
    flat = score.reshape(-1)
    candidate_indices = np.flatnonzero(flat >= threshold)
    if candidate_indices.size == 0:
        candidate_indices = np.flatnonzero(flat > 0)
    if candidate_indices.size > int(top_k):
        strongest = np.argsort(flat[candidate_indices])[-int(top_k) :]
        candidate_indices = candidate_indices[strongest]

    range_count, az_count, el_count = score.shape
    range_bins, az_bins, el_bins = np.unravel_index(candidate_indices, (range_count, az_count, el_count))
    powers = score[range_bins, az_bins, el_bins]
    max_power = float(np.max(powers)) if powers.size else 1.0

    range_axis = np.asarray(runtime_config.range_axis_m, dtype=np.float64)
    az_axis = np.asarray(runtime_config.angle_axis_rad, dtype=np.float64)
    el_axis_deg = np.asarray(runtime_config.angle_elevation_axis_deg, dtype=np.float64)
    points = []
    for rb, ab, eb, power in zip(range_bins, az_bins, el_bins, powers):
        range_m = float(range_axis[int(rb)])
        az_rad = float(az_axis[int(ab)])
        el_deg = float(el_axis_deg[int(eb)])
        el_rad = np.radians(el_deg)
        x_m = range_m * np.cos(el_rad) * np.sin(az_rad)
        y_m = range_m * np.cos(el_rad) * np.cos(az_rad)
        z_m = range_m * np.sin(el_rad)
        points.append(
            {
                "x_m": x_m,
                "y_m": y_m,
                "z_m": z_m,
                "range_m": range_m,
                "azimuth_deg": float(np.degrees(az_rad)),
                "elevation_deg": el_deg,
                "doppler_bin": int(doppler[int(rb), int(ab), int(eb)]),
                "relative_power": float(power) / max(max_power, 1e-6),
                "power": float(power),
            }
        )
    points.sort(key=lambda row: row["power"], reverse=True)
    return points


def _write_points_csv(path: Path, points: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not points:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(points[0].keys()))
        writer.writeheader()
        writer.writerows(points)


def _point_arrays(points: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray([p["x_m"] for p in points], dtype=float)
    y = np.asarray([p["y_m"] for p in points], dtype=float)
    z = np.asarray([p["z_m"] for p in points], dtype=float)
    power = np.asarray([p["relative_power"] for p in points], dtype=float)
    doppler = np.asarray([p["doppler_bin"] for p in points], dtype=float)
    return x, y, z, power, doppler


def _axis_bounds(x, y, z) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    max_radius = max(
        1.0,
        float(np.max(np.abs(x))) if x.size else 1.0,
        float(np.max(y)) if y.size else 1.0,
        float(np.max(np.abs(z))) if z.size else 1.0,
    )
    y_max = max(max_radius, float(np.max(y)) + 0.4 if y.size else 4.0)
    z_radius = max_radius * 0.7
    return max_radius, y_max, z_radius


def _global_axis_bounds(frame_points: list[list[dict]]) -> tuple[float, float, float]:
    xs = []
    ys = []
    zs = []
    for points in frame_points:
        x, y, z, _power, _doppler = _point_arrays(points)
        if x.size:
            xs.append(x)
            ys.append(y)
            zs.append(z)
    if not xs:
        return (1.0, 4.0, 1.0)
    return _axis_bounds(np.concatenate(xs), np.concatenate(ys), np.concatenate(zs))


def _set_equal_3d_axes(ax, x, y, z, axis_bounds: tuple[float, float, float] | None = None) -> None:
    max_radius, y_max, z_radius = axis_bounds if axis_bounds is not None else _axis_bounds(x, y, z)
    ax.set_xlim(-max_radius, max_radius)
    ax.set_ylim(0, y_max)
    ax.set_zlim(-z_radius, z_radius)


def _add_reference_volume(ax, x, y, z, axis_bounds: tuple[float, float, float] | None = None) -> None:
    max_radius, y_max, z_radius = axis_bounds if axis_bounds is not None else _axis_bounds(x, y, z)
    x0, x1 = -max_radius, max_radius
    y0, y1 = 0.0, y_max
    z0, z1 = -z_radius, z_radius
    planes = [
        [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
        [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
    ]
    volume = Poly3DCollection(
        planes,
        facecolors="#93c5fd",
        edgecolors="#60a5fa",
        linewidths=0.7,
        alpha=0.12,
    )
    ax.add_collection3d(volume)


def _plot_cloud(
    path: Path,
    points: list[dict],
    *,
    session_id: str,
    frame_id: int,
    ordinal: int,
    top_k: int,
    quantile: float,
    axis_bounds: tuple[float, float, float] | None = None,
) -> None:
    x, y, z, power, doppler = _point_arrays(points)
    sizes = 12.0 + 70.0 * np.clip(power, 0.0, 1.0)

    fig = plt.figure(figsize=(15, 8.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    ax3d = fig.add_subplot(gs[:, :2], projection="3d")
    ax_xz = fig.add_subplot(gs[0, 2])
    ax_xy = fig.add_subplot(gs[1, 2])

    fig.suptitle(
        f"Dense raw-derived 3D point cloud | session={session_id} | frame={frame_id} | ordinal={ordinal}",
        fontsize=14,
    )
    scatter = ax3d.scatter(x, y, z, c=doppler, cmap="turbo", s=sizes, alpha=0.82, edgecolors="#17202a", linewidths=0.25)
    ax3d.scatter([0], [0], [0], marker="s", s=70, color="#111827", label="radar")
    ax3d.set_xlabel("x lateral (m)")
    ax3d.set_ylabel("y depth (m)")
    ax3d.set_zlabel("z vertical rel. radar (m)")
    ax3d.set_title(f"3D cloud, top_k={top_k}, quantile={quantile}")
    ax3d.view_init(elev=18, azim=-62)
    _add_reference_volume(ax3d, x, y, z, axis_bounds=axis_bounds)
    _set_equal_3d_axes(ax3d, x, y, z, axis_bounds=axis_bounds)

    ax_xz.scatter(x, z, c=doppler, cmap="turbo", s=sizes, alpha=0.82, edgecolors="#17202a", linewidths=0.25)
    ax_xz.axhline(0, color="#cbd5e1", linewidth=1)
    ax_xz.axvline(0, color="#cbd5e1", linewidth=1)
    ax_xz.grid(True, color="#dbe3ec")
    ax_xz.set_title("Radar front projection: x-z")
    ax_xz.set_xlabel("x lateral (m)")
    ax_xz.set_ylabel("z vertical rel. radar (m)")
    if axis_bounds is not None:
        max_radius, _y_max, z_radius = axis_bounds
        ax_xz.set_xlim(-max_radius, max_radius)
        ax_xz.set_ylim(-z_radius, z_radius)

    ax_xy.scatter(x, y, c=doppler, cmap="turbo", s=sizes, alpha=0.82, edgecolors="#17202a", linewidths=0.25)
    ax_xy.scatter([0], [0], marker="s", s=50, color="#111827")
    ax_xy.grid(True, color="#dbe3ec")
    ax_xy.set_title("Top projection: x-y")
    ax_xy.set_xlabel("x lateral (m)")
    ax_xy.set_ylabel("y depth (m)")
    ax_xy.set_aspect("equal", adjustable="box")
    if axis_bounds is not None:
        max_radius, y_max, _z_radius = axis_bounds
        ax_xy.set_xlim(-max_radius, max_radius)
        ax_xy.set_ylim(0, y_max)

    cbar = fig.colorbar(scatter, ax=[ax3d, ax_xz, ax_xy], shrink=0.72, pad=0.015)
    cbar.set_label("doppler bin")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_mp4_from_images(frame_paths: list[Path], video_path: Path, fps: float) -> dict:
    def write_gif_fallback(reason: str) -> dict:
        gif_path = video_path.with_suffix(".gif")
        gif_result = _write_gif_from_images(frame_paths, gif_path, fps)
        if gif_result.get("ok"):
            gif_result["error"] = reason
        return gif_result

    try:
        import cv2
    except ImportError as exc:
        return write_gif_fallback(f"MP4 skipped because OpenCV is not available: {exc}")

    if not frame_paths:
        return {"ok": False, "video_path": "", "error": "No rendered frames to encode."}

    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        return write_gif_fallback(f"OpenCV could not read first rendered frame: {frame_paths[0]}")
    height, width = first.shape[:2]
    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        return write_gif_fallback("OpenCV VideoWriter could not be opened.")

    try:
        writer.write(first)
        for frame_path in frame_paths[1:]:
            image = cv2.imread(str(frame_path))
            if image is None:
                continue
            if image.shape[:2] != (height, width):
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(image)
    finally:
        writer.release()

    return {"ok": True, "video_path": str(video_path), "format": "mp4", "error": ""}


def _write_gif_from_images(frame_paths: list[Path], gif_path: Path, fps: float) -> dict:
    try:
        from PIL import Image
    except ImportError as exc:
        return {
            "ok": False,
            "video_path": "",
            "format": "",
            "error": f"Could not write MP4 and Pillow is not available for GIF fallback: {exc}",
        }

    if not frame_paths:
        return {"ok": False, "video_path": "", "format": "", "error": "No rendered frames to encode."}

    frames = []
    target_size = None
    for frame_path in frame_paths:
        try:
            with Image.open(frame_path) as image:
                frame = image.convert("RGB")
                if target_size is None:
                    target_size = frame.size
                elif frame.size != target_size:
                    frame = frame.resize(target_size, Image.Resampling.LANCZOS)
                frames.append(frame.copy())
        except OSError:
            continue

    if not frames:
        return {"ok": False, "video_path": "", "format": "", "error": "Could not read rendered frames for GIF fallback."}

    gif_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(1, int(round(1000.0 / max(float(fps), 0.1))))
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    return {"ok": True, "video_path": str(gif_path), "format": "gif", "error": ""}


def build_cloud(
    *,
    project_root: Path,
    session_id: str,
    ordinal: int,
    top_k: int,
    quantile: float,
    output_root: Path,
) -> dict:
    _run_detail, capture_dir, components = _resolve_runtime(project_root, session_id)
    runtime_config = components["runtime_config"]
    raw_frame = _load_frame(capture_dir, ordinal)
    response = _dense_response_for_frame(raw_frame, runtime_config)
    points = _make_points(response, runtime_config, components, top_k=top_k, quantile=quantile)

    output_dir = output_root / f"raw_dense_pointcloud_3d_{_now_tag()}_{session_id}"
    image_path = output_dir / f"frame_{ordinal:06d}_dense_3d.png"
    csv_path = output_dir / f"frame_{ordinal:06d}_dense_3d_points.csv"
    _plot_cloud(
        image_path,
        points,
        session_id=session_id,
        frame_id=int(raw_frame.frame_id),
        ordinal=int(ordinal),
        top_k=int(top_k),
        quantile=float(quantile),
    )
    _write_points_csv(csv_path, points)
    result = {
        "session_id": session_id,
        "frame_id": int(raw_frame.frame_id),
        "ordinal": int(ordinal),
        "point_count": int(len(points)),
        "top_k": int(top_k),
        "quantile": float(quantile),
        "output_dir": str(output_dir),
        "image_path": str(image_path),
        "points_csv": str(csv_path),
        "notes": [
            "This is a dense offline diagnostic cloud, not the runtime tracker output.",
            "Coordinates are derived from range-Doppler-azimuth-elevation beamscan energy cells.",
            "z is vertical relative to the radar boresight/array frame, not an absolute floor height.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    result["manifest_path"] = str(manifest_path)
    return result


def build_replay(
    *,
    project_root: Path,
    session_id: str,
    start_frame: int,
    end_frame: int,
    step: int,
    top_k: int,
    quantile: float,
    fps: float,
    output_root: Path,
    keep_frames: bool,
    write_frame_csv: bool,
) -> dict:
    _run_detail, capture_dir, components = _resolve_runtime(project_root, session_id)
    runtime_config = components["runtime_config"]
    raw_frames = _load_frame_range(capture_dir, start_frame, end_frame, step)

    output_dir = output_root / f"raw_dense_pointcloud_3d_replay_{_now_tag()}_{session_id}"
    frames_dir = output_dir / "frames"
    csv_dir = output_dir / "points_csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    per_frame = []
    point_sets = []
    for ordinal, raw_frame in raw_frames:
        response = _dense_response_for_frame(raw_frame, runtime_config)
        points = _make_points(response, runtime_config, components, top_k=top_k, quantile=quantile)
        point_sets.append(points)
        per_frame.append(
            {
                "ordinal": int(ordinal),
                "frame_id": int(raw_frame.frame_id),
                "point_count": int(len(points)),
                "points": points,
            }
        )

    axis_bounds = _global_axis_bounds(point_sets)
    frame_paths = []
    for item in per_frame:
        ordinal = int(item["ordinal"])
        frame_id = int(item["frame_id"])
        points = item["points"]
        frame_path = frames_dir / f"frame_{ordinal:06d}_dense_3d.png"
        _plot_cloud(
            frame_path,
            points,
            session_id=session_id,
            frame_id=frame_id,
            ordinal=ordinal,
            top_k=int(top_k),
            quantile=float(quantile),
            axis_bounds=axis_bounds,
        )
        frame_paths.append(frame_path)
        if write_frame_csv:
            _write_points_csv(csv_dir / f"frame_{ordinal:06d}_dense_3d_points.csv", points)

    video_path = output_dir / "raw_dense_pointcloud_3d_replay.mp4"
    video = _write_mp4_from_images(frame_paths, video_path, fps)

    result = {
        "session_id": session_id,
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "step": int(step),
        "rendered_frame_count": int(len(frame_paths)),
        "top_k": int(top_k),
        "quantile": float(quantile),
        "fps": float(fps),
        "axis_bounds": {
            "x_radius_m": float(axis_bounds[0]),
            "y_max_m": float(axis_bounds[1]),
            "z_radius_m": float(axis_bounds[2]),
        },
        "output_dir": str(output_dir),
        "frames_dir": str(frames_dir),
        "video": video,
        "frame_summary": [
            {
                "ordinal": int(item["ordinal"]),
                "frame_id": int(item["frame_id"]),
                "point_count": int(item["point_count"]),
            }
            for item in per_frame
        ],
        "notes": [
            "This replay shows per-frame raw-derived high-energy 3D cells.",
            "It is not a runtime tracker replay and not a LiDAR-style dense body surface.",
            "Axis limits are fixed across selected frames to reduce visual flicker.",
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    result["manifest_path"] = str(manifest_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dense 3D raw-derived radar point cloud images/replays.")
    parser.add_argument("--session", required=True, help="live_motion_viewer session id.")
    parser.add_argument("--frame", type=int, default=0, help="Frame ordinal in the raw capture, zero-based.")
    parser.add_argument("--start-frame", type=int, default=None, help="First frame ordinal for replay video.")
    parser.add_argument("--end-frame", type=int, default=None, help="Last frame ordinal for replay video.")
    parser.add_argument("--step", type=int, default=1, help="Replay frame step.")
    parser.add_argument("--fps", type=float, default=8.0, help="Replay video frames per second.")
    parser.add_argument("--video", action="store_true", help="Render a frame range to mp4 replay.")
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="Compatibility flag. Rendered PNG frames are always kept with the video output.",
    )
    parser.add_argument("--write-frame-csv", action="store_true", help="Write one point CSV per replay frame.")
    parser.add_argument("--top-k", type=int, default=350, help="Maximum number of energy cells to plot.")
    parser.add_argument("--quantile", type=float, default=0.997, help="Power quantile threshold before top-k selection.")
    parser.add_argument("--output-root", default="", help="Output root. Default: logs/diagnostics.")
    args = parser.parse_args()

    project_root = _repo_root()
    output_root = Path(args.output_root) if args.output_root else project_root / "logs" / "diagnostics"
    if bool(args.video) or args.start_frame is not None or args.end_frame is not None:
        start_frame = int(args.frame if args.start_frame is None else args.start_frame)
        end_frame = int(start_frame if args.end_frame is None else args.end_frame)
        result = build_replay(
            project_root=project_root,
            session_id=str(args.session),
            start_frame=start_frame,
            end_frame=end_frame,
            step=max(1, int(args.step)),
            top_k=max(1, int(args.top_k)),
            quantile=float(args.quantile),
            fps=max(0.1, float(args.fps)),
            output_root=output_root,
            keep_frames=bool(args.keep_frames),
            write_frame_csv=bool(args.write_frame_csv),
        )
        print(f"Output directory: {result['output_dir']}")
        print(f"Rendered frames: {result['rendered_frame_count']}")
        if result["video"].get("ok"):
            print(f"Video: {result['video']['video_path']}")
        else:
            print(f"Video failed: {result['video'].get('error')}")
            print(f"PNG frames: {result['frames_dir']}")
        return

    result = build_cloud(
        project_root=project_root,
        session_id=str(args.session),
        ordinal=int(args.frame),
        top_k=max(1, int(args.top_k)),
        quantile=float(args.quantile),
        output_root=output_root,
    )
    print(f"Output directory: {result['output_dir']}")
    print(f"Image: {result['image_path']}")
    print(f"Points CSV: {result['points_csv']}")
    print(f"Point count: {result['point_count']}")


if __name__ == "__main__":
    main()
