import argparse
import os
from pathlib import Path
import sys
import webbrowser

# Keep pyqtgraph on the same Qt binding as the generated UI module.
os.environ.setdefault('PYQTGRAPH_QT_LIB', 'PyQt5')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.runtime_core.runtime_settings import load_runtime_settings, resolve_project_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay a saved raw radar capture through the live viewer UI.",
    )
    parser.add_argument(
        "--capture",
        help="Path to logs/raw/<session_id> capture directory. If omitted, runtime logging.source_capture is used.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Replay speed multiplier. 1.0 means recorded timing, 2.0 means twice as fast.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop the replay capture continuously.",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Open the replay UI and wait for the Start Replay button instead of auto-starting.",
    )
    parser.add_argument(
        "--tuning",
        help="Optional tuning JSON path, for example config/live_motion_tuning_multi.json.",
    )
    parser.add_argument(
        "--no-open-report",
        action="store_true",
        help="Do not open the generated trajectory replay HTML in a browser.",
    )
    return parser.parse_args()


def resolve_capture_path(project_root: Path, capture_arg: str | None) -> Path:
    settings = load_runtime_settings(project_root)
    configured_capture = str((settings["runtime"].get("logging") or {}).get("source_capture") or "").strip()
    capture_value = str(capture_arg or configured_capture).strip()
    if not capture_value:
        raise SystemExit(
            "Replay capture path is required. Use --capture logs/raw/<session_id> "
            "or set logging.source_capture in config/live_motion_runtime_settings.json."
        )

    capture_path = resolve_project_path(project_root, capture_value)
    if not capture_path.exists():
        shorthand_path = project_root / "logs" / "raw" / capture_value
        if shorthand_path.exists():
            capture_path = shorthand_path
        else:
            raise SystemExit(f"Replay capture directory not found: {capture_path}")
    return capture_path


def main():
    args = parse_args()
    capture_path = resolve_capture_path(PROJECT_ROOT, args.capture)
    if args.tuning:
        tuning_path = resolve_project_path(PROJECT_ROOT, args.tuning)
        if not tuning_path.exists():
            raise SystemExit(f"Tuning file not found: {tuning_path}")
        os.environ["RADAR_TUNING_PATH"] = str(tuning_path)
        print(f"Replay tuning: {tuning_path}")

    from live_motion_viewer import MotionViewer

    print(f"Replay source: {capture_path}")
    viewer = MotionViewer(
        input_mode="replay",
        source_capture=str(capture_path),
        replay_speed=args.speed,
        replay_loop=args.loop,
        auto_start=not args.wait,
        write_raw_capture=False,
    )
    viewer.run()

    trajectory_replay_path = viewer.session_logger.session_dir / "trajectory_replay.html"
    if trajectory_replay_path.exists():
        print(f"Trajectory replay report: {trajectory_replay_path}")
        if not args.no_open_report:
            try:
                webbrowser.open(trajectory_replay_path.resolve().as_uri())
            except Exception as exc:
                print(f"Warning: failed to open trajectory replay report automatically: {exc!r}")
    else:
        print(
            "Replay finished, but trajectory replay report was not found yet: "
            f"{trajectory_replay_path}"
        )


if __name__ == "__main__":
    main()
