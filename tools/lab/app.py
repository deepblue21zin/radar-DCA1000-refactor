from __future__ import annotations

import base64
from datetime import datetime
import html
import io
import json
import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import streamlit as st
except ImportError as error:  # pragma: no cover
    raise SystemExit(
        "Streamlit is not installed. Run `pip install -r requirements-lab.txt` first."
    ) from error

from tools.lab import analytics, registry, stage_cache, wandb_sync
from tools.tuning_loop.run_loop import ISK_SCENARIOS, PARAMETER_SPECS


EVAL_TASKS_DIR = PROJECT_ROOT / "docs" / "evals" / "tasks"
EVAL_RUNS_DIR = PROJECT_ROOT / "docs" / "evals" / "runs"
TUNING_RUNS_DIR = PROJECT_ROOT / "lab_data" / "tuning_runs"

LABEL_OPTIONS = ["", "baseline", "good", "usable", "interesting", "discard"]
BOARD_OPTIONS = ["", "IWR6843ISK", "IWR6843ISK-ODS", "unknown", "mixed"]
MOTION_OPTIONS = [
    "",
    "center",
    "center-round-trip",
    "straight",
    "round-trip",
    "right-diagonal",
    "left-diagonal",
    "circle",
    "square",
    "two-person",
    "custom",
]


def _rerun() -> None:
    rerun = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if rerun is not None:
        rerun()


def _format_float(value, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}{suffix}"


def _format_percent(value) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def _safe_cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def _short_text(value, limit: int = 72) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(limit - 1, 1)] + "..."


def _option_index(options: list[str], value: str) -> int:
    return options.index(value) if value in options else 0


def _normalize_board_label(value) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    compact = text.lower().replace("_", "-").replace(" ", "")
    aliases = {
        "isk": "IWR6843ISK",
        "iwr6843isk": "IWR6843ISK",
        "ods": "IWR6843ISK-ODS",
        "isk-ods": "IWR6843ISK-ODS",
        "iwr6843isk-ods": "IWR6843ISK-ODS",
        "iwr6843iskods": "IWR6843ISK-ODS",
        "unknown": "unknown",
        "mixed": "mixed",
    }
    return aliases.get(compact, text)


def _row_board(row: dict) -> str:
    for value in (
        row.get("annotation_board_type"),
        row.get("board_type"),
        _nested_get(row.get("runtime_config"), "radar_board", default=""),
        _nested_get(row.get("runtime_config"), "runtime_snapshot", "radar_board", default=""),
        _nested_get(row.get("summary"), "runtime_config", "radar_board", default=""),
        _nested_get(row.get("manifest"), "runtime_summary", "radar_board", default=""),
        _nested_get(row.get("manifest"), "raw_capture", "radar_board", default=""),
    ):
        board = _normalize_board_label(value)
        if board:
            return board
    return ""


def _annotation_summary(row: dict, *, include_notes: bool = False) -> str:
    parts = [
        row.get("annotation_label") or "",
        _row_board(row),
        row.get("annotation_motion_pattern") or "",
        row.get("scenario_id") or "",
    ]
    if include_notes:
        parts.append(_short_text(row.get("annotation_notes"), 48))
    text = " / ".join(part for part in parts if part)
    return text or "unlabeled"


def _render_html(markup: str) -> None:
    """Render generated HTML/SVG without Markdown treating indentation as code."""
    cleaned = textwrap.dedent(markup).strip()
    html_renderer = getattr(st, "html", None)
    if html_renderer is not None:
        html_renderer(cleaned)
        return
    st.markdown(cleaned, unsafe_allow_html=True)


def _render_matplotlib_figure(fig, *, caption: str | None = None) -> None:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    buffer.seek(0)
    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:
        pass
    st.image(buffer.getvalue(), width="stretch")
    if caption:
        st.caption(caption)


def _make_figure(*, width: float = 10.0, height: float = 4.2):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(width, height), facecolor="white")
    return fig, plt


def _render_table(rows: list[dict], *, key: str, height: int | None = None) -> None:
    if not rows:
        st.caption("표시할 데이터가 없습니다.")
        return

    columns: list[str] = []
    for row in rows:
        for column in row.keys():
            if column not in columns:
                columns.append(column)
    display_rows = [
        {column: _safe_cell(row.get(column, "")) for column in columns}
        for row in rows
    ]

    try:
        if height is None:
            st.dataframe(display_rows, width="stretch", hide_index=True)
        else:
            st.dataframe(display_rows, width="stretch", hide_index=True, height=int(height))
        return
    except Exception as error:
        st.caption(
            "현재 환경에서 Streamlit dataframe backend가 막혀 plain-text 표로 표시합니다. "
            f"({error.__class__.__name__})"
        )

    widths = {
        column: min(
            max(len(column), *(len(str(row.get(column, ""))) for row in display_rows)),
            36,
        )
        for column in columns
    }

    def fmt_cell(column: str, value: str) -> str:
        text = str(value)
        if len(text) > widths[column]:
            text = text[: widths[column] - 1] + "..."
        return text.ljust(widths[column])

    lines = [
        " | ".join(fmt_cell(column, column) for column in columns),
        "-+-".join("-" * widths[column] for column in columns),
    ]
    for row in display_rows:
        lines.append(" | ".join(fmt_cell(column, row.get(column, "")) for column in columns))
    st.code("\n".join(lines), language="text")


def _registry_export_name() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"radar_lab_registry_{stamp}.db"


def _validate_registry_db(path: Path) -> None:
    expected_tables = {"captures", "runs", "run_parameters", "annotations", "registry_meta"}
    with sqlite3.connect(path) as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if not integrity or str(integrity[0]).lower() != "ok":
            raise ValueError("SQLite integrity check failed.")
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()

    table_names = {str(row[0]) for row in rows}
    missing = sorted(expected_tables - table_names)
    if missing:
        raise ValueError(f"Not a Radar Lab registry DB. Missing tables: {', '.join(missing)}")


def _import_registry_db(uploaded_bytes: bytes) -> Path | None:
    if not uploaded_bytes:
        raise ValueError("Uploaded DB file is empty.")

    db_path = registry.database_path(PROJECT_ROOT)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = db_path.with_name("_incoming_radar_lab_registry.db")
    tmp_path.write_bytes(uploaded_bytes)

    try:
        _validate_registry_db(tmp_path)
        backup_path = None
        if db_path.exists():
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = db_path.with_name(f"radar_lab_registry_backup_{stamp}.db")
            db_path.replace(backup_path)
        tmp_path.replace(db_path)
        return backup_path
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _render_registry_share_tools() -> None:
    db_path = registry.database_path(PROJECT_ROOT)
    with st.expander("DB Import / Export", expanded=False):
        st.caption(
            "라벨, annotation, 세션 인덱스만 공유합니다. raw logs, reports, stage cache 파일은 별도로 공유해야 합니다."
        )
        if db_path.exists():
            st.download_button(
                "Export Registry DB",
                data=db_path.read_bytes(),
                file_name=_registry_export_name(),
                mime="application/vnd.sqlite3",
                width="stretch",
            )
        else:
            st.info("아직 export할 registry DB가 없습니다. Refresh Registry를 먼저 눌러 주세요.")

        uploaded_db = st.file_uploader(
            "Import Registry DB",
            type=["db", "sqlite", "sqlite3"],
            accept_multiple_files=False,
            key="registry-db-import",
        )
        if uploaded_db is not None:
            st.warning(
                "Import하면 현재 로컬 Radar Lab DB가 업로드한 DB로 교체됩니다. "
                "기존 DB는 lab_data에 backup으로 남깁니다."
            )
            if st.button("Import Uploaded DB", width="stretch"):
                try:
                    backup_path = _import_registry_db(uploaded_db.getvalue())
                except Exception as error:
                    st.error(f"DB import failed: {error}")
                else:
                    backup_text = f" Backup: `{backup_path}`" if backup_path else ""
                    st.success(f"DB import complete.{backup_text}")
                    _rerun()


def _file_uri(path_value: Path | str | None) -> str | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        return None
    return path.resolve().as_uri()


def _render_file_links(items: dict[str, Path | str | None]) -> None:
    lines = []
    for label, path in items.items():
        uri = _file_uri(path)
        if uri:
            lines.append(f"- [{label}]({uri})")
    if lines:
        st.markdown("\n".join(lines))
    else:
        st.caption("열 수 있는 로컬 파일이 아직 없습니다.")


def _relative_to_project(path: Path | str | None) -> str:
    if path is None:
        return ""
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(PROJECT_ROOT.resolve()))
    except Exception:
        return str(candidate)


def _eval_task_files(*, include_templates: bool = False) -> list[Path]:
    if not EVAL_TASKS_DIR.exists():
        return []
    tasks = sorted(EVAL_TASKS_DIR.glob("*.json"))
    if not include_templates:
        tasks = [path for path in tasks if ".template." not in path.name]
    return tasks


def _load_json_file(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _eval_outcome_files() -> list[Path]:
    if not EVAL_RUNS_DIR.exists():
        return []
    return sorted(
        EVAL_RUNS_DIR.glob("*/outcome.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _load_eval_outcomes(limit: int = 80) -> list[dict]:
    outcomes = []
    for path in _eval_outcome_files()[:limit]:
        outcome = _load_json_file(path)
        if not outcome:
            continue
        outcome["_outcome_path"] = str(path)
        outcome["_run_dir"] = str(path.parent)
        outcomes.append(outcome)
    return outcomes


def _eval_status(outcome: dict) -> str:
    return str(outcome.get("status") or "unknown").lower()


def _eval_status_mark(status: str) -> str:
    normalized = str(status or "unknown").lower()
    if normalized == "pass":
        return "PASS"
    if normalized == "fail":
        return "FAIL"
    if normalized == "dry_run":
        return "DRY"
    return normalized.upper()


def _eval_status_tone(status: str) -> str:
    normalized = str(status or "").lower()
    if normalized == "pass":
        return "good"
    if normalized == "fail":
        return "bad"
    if normalized == "dry_run":
        return "warn"
    return "neutral"


def _criterion_pass_text(criteria: list[dict]) -> str:
    if not criteria:
        return "n/a"
    passed = sum(1 for item in criteria if item.get("passed"))
    return f"{passed}/{len(criteria)}"


def _eval_outcome_rows(outcomes: list[dict]) -> list[dict]:
    rows = []
    for outcome in outcomes:
        candidate = outcome.get("candidate") or {}
        baseline = outcome.get("baseline") or {}
        task = outcome.get("task") or {}
        criteria = outcome.get("criteria") or []
        rows.append(
            {
                "status": _eval_status_mark(_eval_status(outcome)),
                "task": task.get("name") or "",
                "generated_at": outcome.get("generated_at") or "",
                "criteria": _criterion_pass_text(criteria),
                "candidate": candidate.get("session_id") or "",
                "baseline": baseline.get("session_id") or "",
                "transport": _nested_get(candidate, "summary", "transport_quality", default=""),
                "path_clean": _format_float(
                    _nested_get(candidate, "summary", "path_cleanliness_score_10", default=None),
                    2,
                ),
                "policy": str(_nested_get(candidate, "summary", "path_shape_policy_pass", default="")),
                "outcome": _relative_to_project(outcome.get("_outcome_path")),
            }
        )
    return rows


def _extract_outcome_path(stdout_text: str) -> Path | None:
    for line in reversed((stdout_text or "").splitlines()):
        text = line.strip()
        if text.lower().startswith("outcome:"):
            value = text.split(":", 1)[1].strip()
            path = Path(value)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            if path.exists():
                return path
    return None


def _latest_outcome_for_task(task_name: str, *, after_dirs: set[Path] | None = None) -> Path | None:
    matches = []
    for outcome_path in _eval_outcome_files():
        run_dir = outcome_path.parent.resolve()
        if after_dirs is not None and run_dir not in after_dirs:
            continue
        outcome = _load_json_file(outcome_path)
        name = _nested_get(outcome, "task", "name", default="") or outcome_path.parent.name
        if str(name) == task_name or str(name) in outcome_path.parent.name:
            matches.append(outcome_path)
    return matches[0] if matches else None


def _build_eval_command(
    task_path: Path,
    *,
    mode: str,
    baseline_session: str,
    candidate_session: str,
    no_stage_cache: bool,
    force_baseline: bool,
    skip_baseline: bool,
    dry_run: bool,
    timeout_s: float | None,
) -> list[str]:
    command = [
        sys.executable,
        "-B",
        "-m",
        "tools.eval_harness.run_task",
        str(task_path),
    ]
    if baseline_session.strip():
        command.extend(["--baseline-session", baseline_session.strip()])
    if mode == "기존 candidate session 채점" and candidate_session.strip():
        command.extend(["--candidate-session", candidate_session.strip()])
    if no_stage_cache:
        command.append("--no-stage-cache")
    if force_baseline:
        command.append("--force-baseline")
    if skip_baseline:
        command.append("--skip-baseline")
    if dry_run:
        command.append("--dry-run")
    if timeout_s and timeout_s > 0:
        command.extend(["--timeout-s", str(float(timeout_s))])
    return command


def _run_eval_command(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )


def _heatmap_image(array) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    if values.ndim != 2 or values.size == 0:
        return np.zeros((8, 8, 3), dtype=np.uint8)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    lower = float(np.quantile(values, 0.02))
    upper = float(np.quantile(values, 0.995))
    if upper <= lower:
        lower = float(np.min(values))
        upper = float(np.max(values))
    scale = max(upper - lower, 1e-6)
    normalized = np.clip((values - lower) / scale, 0.0, 1.0)

    palette = np.array(
        [
            [11, 7, 25],
            [58, 31, 112],
            [46, 93, 201],
            [37, 179, 196],
            [253, 231, 76],
        ],
        dtype=np.float32,
    )
    positions = np.linspace(0.0, 1.0, palette.shape[0], dtype=np.float32)
    flat = normalized.ravel()
    rgb = np.empty((flat.size, 3), dtype=np.float32)
    for channel in range(3):
        rgb[:, channel] = np.interp(flat, positions, palette[:, channel])
    return rgb.reshape(values.shape + (3,)).astype(np.uint8)


def _heatmap_data_uri(array) -> str | None:
    try:
        from PIL import Image
    except ImportError:
        return None

    image = Image.fromarray(_heatmap_image(array), mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _render_heatmap(
    title: str,
    array,
    *,
    caption: str | None = None,
    height_px: int = 220,
    fit: str = "contain",
) -> None:
    shape = tuple(np.asarray(array).shape)
    data_uri = _heatmap_data_uri(array)
    if data_uri is None:
        st.markdown(f"#### {title}")
        st.image(_heatmap_image(array), width="stretch")
        if caption:
            st.caption(caption)
        return

    caption_text = caption or f"shape={shape}"
    card_html = f"""
    <div style="
      border: 1px solid #d9e5eb;
      border-radius: 16px;
      background: #ffffff;
      padding: 10px 12px 8px;
      box-shadow: 0 10px 28px rgba(18,40,56,.06);
      margin-bottom: 8px;
    ">
      <div style="font-weight: 800; font-size: 1.02rem; color: #263848; margin-bottom: 8px;">
        {html.escape(title)}
      </div>
      <div style="
        height: {int(height_px)}px;
        width: 100%;
        border-radius: 12px;
        overflow: hidden;
        background: #080717;
        display: flex;
        align-items: center;
        justify-content: center;
      ">
        <img src="{data_uri}" alt="{html.escape(title)}" style="
          width: 100%;
          height: 100%;
          object-fit: {html.escape(fit)};
          image-rendering: pixelated;
          display: block;
        " />
      </div>
      <div style="color:#6d7c88;font-size:.82rem;margin-top:6px;">{html.escape(caption_text)}</div>
    </div>
    """
    _render_html(card_html)


def _metric_value(row: dict, metric: str):
    value = row.get(metric)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def _render_metric_timeline_svg_legacy(title: str, rows: list[dict], metric: str, *, threshold=None, lower_is_better=True) -> None:
    points = [
        (int(row.get("ordinal", index)), _metric_value(row, metric), row)
        for index, row in enumerate(rows)
    ]
    points = [(x, y, row) for x, y, row in points if y is not None]
    if not points:
        st.caption(f"`{metric}` 값을 가진 frame feature가 없습니다.")
        return

    width = 980
    height = 260
    left = 52
    right = 18
    top = 24
    bottom = 42
    plot_w = width - left - right
    plot_h = height - top - bottom
    xs = np.asarray([item[0] for item in points], dtype=float)
    ys = np.asarray([item[1] for item in points], dtype=float)
    x_min, x_max = float(np.min(xs)), float(np.max(xs))
    y_min, y_max = float(np.min(ys)), float(np.max(ys))
    if threshold is not None:
        try:
            y_min = min(y_min, float(threshold))
            y_max = max(y_max, float(threshold))
        except (TypeError, ValueError):
            threshold = None
    if abs(y_max - y_min) < 1e-9:
        y_min -= 1.0
        y_max += 1.0
    if abs(x_max - x_min) < 1e-9:
        x_max += 1.0

    def px(x_value):
        return left + ((float(x_value) - x_min) / (x_max - x_min)) * plot_w

    def py(y_value):
        return top + (1.0 - ((float(y_value) - y_min) / (y_max - y_min))) * plot_h

    polyline = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y, _ in points)
    markers = []
    for x, y, row in points:
        bottleneck = row.get("frame_bottleneck")
        severity = float(row.get("frame_severity_10") or 0.0)
        if bottleneck == "ok" and severity < 5:
            continue
        color = "#d94141" if severity >= 8 else "#d88922"
        markers.append(
            f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="3.8" fill="{color}">'
            f'<title>frame {x}: {html.escape(str(bottleneck))} | {html.escape(str(row.get("frame_evidence") or ""))}</title>'
            "</circle>"
        )

    threshold_line = ""
    if threshold is not None:
        threshold_y = py(float(threshold))
        threshold_color = "#b74040" if lower_is_better else "#237a55"
        threshold_line = (
            f'<line x1="{left}" y1="{threshold_y:.1f}" x2="{width-right}" y2="{threshold_y:.1f}" '
            f'stroke="{threshold_color}" stroke-width="1.4" stroke-dasharray="7 6" />'
            f'<text x="{width-right-4}" y="{threshold_y-6:.1f}" text-anchor="end" fill="{threshold_color}" '
            f'font-size="12">target {float(threshold):.3g}</text>'
        )

    svg = f"""
    <div style="border:1px solid #d9e5eb;border-radius:18px;background:#fff;padding:14px 16px 8px;">
      <div style="font-weight:800;margin-bottom:8px;color:#163044;">{html.escape(title)}</div>
      <svg viewBox="0 0 {width} {height}" style="width:100%;height:auto;display:block;">
        <rect x="0" y="0" width="{width}" height="{height}" rx="14" fill="#fbfdfe" />
        <line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#9eb0be" />
        <line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#9eb0be" />
        <text x="{left}" y="{height-12}" fill="#63788a" font-size="12">frame {int(x_min)} - {int(x_max)}</text>
        <text x="{left-8}" y="{top+6}" fill="#63788a" font-size="12" text-anchor="end">{y_max:.3g}</text>
        <text x="{left-8}" y="{height-bottom}" fill="#63788a" font-size="12" text-anchor="end">{y_min:.3g}</text>
        {threshold_line}
        <polyline points="{polyline}" fill="none" stroke="#0f6fb9" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round" />
        {''.join(markers)}
      </svg>
      <div style="color:#63788a;font-size:12px;margin-top:4px;">orange/red markers = frame-level bottleneck candidates</div>
    </div>
    """
    _render_html(svg)


def _render_path_preview_svg_legacy(rows: list[dict]) -> None:
    points = []
    for row in rows:
        x = _metric_value(row, "lead_x_m")
        y = _metric_value(row, "lead_y_m")
        if x is None or y is None:
            continue
        points.append((x, y, row))
    if not points:
        st.caption("lead track 좌표가 있는 frame feature가 없습니다.")
        return

    width = 560
    height = 460
    left = 48
    right = 24
    top = 24
    bottom = 48
    plot_w = width - left - right
    plot_h = height - top - bottom
    xs = np.asarray([item[0] for item in points], dtype=float)
    ys = np.asarray([item[1] for item in points], dtype=float)
    x_abs = max(float(np.max(np.abs(xs))), 0.6)
    y_max = max(float(np.max(ys)), 3.0)
    y_min = 0.0

    def px(x_value):
        return left + ((float(x_value) + x_abs) / (2.0 * x_abs)) * plot_w

    def py(y_value):
        return top + (1.0 - ((float(y_value) - y_min) / max(y_max - y_min, 1e-6))) * plot_h

    polyline = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y, _ in points)
    markers = []
    for x, y, row in points:
        severity = float(row.get("frame_severity_10") or 0.0)
        if severity < 7:
            continue
        color = "#d94141" if severity >= 8 else "#d88922"
        markers.append(
            f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="4" fill="{color}">'
            f'<title>frame {row.get("ordinal")}: {html.escape(str(row.get("frame_bottleneck") or ""))}</title>'
            "</circle>"
        )
    radar_x = px(0.0)
    radar_y = py(0.0)
    svg = f"""
    <div style="border:1px solid #d9e5eb;border-radius:18px;background:#fff;padding:14px 16px 8px;">
      <div style="font-weight:800;margin-bottom:8px;color:#163044;">Lead Path Preview</div>
      <svg viewBox="0 0 {width} {height}" style="width:100%;height:auto;display:block;">
        <rect x="0" y="0" width="{width}" height="{height}" rx="14" fill="#fbfdfe" />
        <line x1="{left}" y1="{radar_y:.1f}" x2="{width-right}" y2="{radar_y:.1f}" stroke="#9eb0be" />
        <line x1="{radar_x:.1f}" y1="{top}" x2="{radar_x:.1f}" y2="{height-bottom}" stroke="#9eb0be" />
        <polyline points="{polyline}" fill="none" stroke="#0f6fb9" stroke-width="2.6" stroke-linejoin="round" stroke-linecap="round" />
        {''.join(markers)}
        <circle cx="{radar_x:.1f}" cy="{radar_y:.1f}" r="7" fill="#172232" />
        <text x="{radar_x+10:.1f}" y="{radar_y-8:.1f}" fill="#172232" font-size="13">radar</text>
        <text x="{left}" y="{height-12}" fill="#63788a" font-size="12">x: radar-left (-) / radar-right (+) (m)</text>
        <text x="{width-right}" y="{top+14}" fill="#63788a" font-size="12" text-anchor="end">y: forward (m)</text>
      </svg>
    </div>
    """
    _render_html(svg)


def _nested_get(payload: dict | None, *keys: str, default=None):
    current = payload or {}
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def _stage_card(label: str, value: str, hint: str, tone: str = "neutral") -> str:
    colors = {
        "good": ("#eaf8f0", "#237a55"),
        "warn": ("#fff6ea", "#b76b18"),
        "bad": ("#fff0f0", "#b74040"),
        "neutral": ("#f5f8fb", "#24425d"),
    }
    bg, fg = colors.get(tone, colors["neutral"])
    return f"""
    <div style="
      min-width: 132px;
      border: 1px solid #d9e5eb;
      border-radius: 15px;
      background: {bg};
      padding: 10px 11px;
      box-shadow: 0 8px 20px rgba(18,40,56,.06);
    ">
      <div style="font-size:.72rem;letter-spacing:.05em;text-transform:uppercase;color:#667887;font-weight:800;">
        {html.escape(label)}
      </div>
      <div style="font-size:1.12rem;color:{fg};font-weight:850;margin-top:4px;">
        {html.escape(str(value))}
      </div>
      <div style="font-size:.76rem;color:#657989;line-height:1.35;margin-top:4px;">
        {html.escape(str(hint))}
      </div>
    </div>
    """


def _tone_for_count(value, *, zero_bad=False, high_warn=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "neutral"
    if zero_bad and value <= 0:
        return "bad"
    if high_warn is not None and value >= high_warn:
        return "warn"
    return "good"


def _as_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _trace_stage_points(trace: dict, stage: str) -> list[dict]:
    detection = trace.get("detection") or {}
    tracker = trace.get("tracker") or {}
    if stage == "angle":
        return list(_nested_get(detection, "angle_validation", "top_candidates", default=[]) or [])
    if stage == "coarse_merge":
        return list(_nested_get(detection, "candidate_merge_coarse", "after_top", default=[]) or [])
    if stage == "body_center":
        pairs = _nested_get(detection, "body_center_refinement", "pairs", default=[]) or []
        return [pair.get("after") for pair in pairs if isinstance(pair, dict) and isinstance(pair.get("after"), dict)]
    if stage == "final_merge":
        return list(_nested_get(detection, "candidate_merge_final", "after_top", default=[]) or [])
    if stage == "dbscan":
        return list(_nested_get(detection, "dbscan", "output_top", default=[]) or [])
    if stage == "tracker_input":
        return list(tracker.get("measurements") or [])
    if stage == "tracks":
        return list(_nested_get(tracker, "track_lifecycle", "tracks_after_prune", default=[]) or [])
    if stage == "display":
        confirmed = _nested_get(trace, "display_output", "confirmed_tracks", default=[]) or []
        tentative = _nested_get(trace, "display_output", "tentative_tracks", default=[]) or []
        return list(confirmed) + list(tentative)
    return []


def _trace_stage_count(trace: dict, stage: str) -> int:
    detection = trace.get("detection") or {}
    tracker = trace.get("tracker") or {}
    if stage == "cfar":
        return int(_nested_get(detection, "cfar", "candidate_count", default=0) or 0)
    if stage == "angle":
        return int(_nested_get(detection, "angle_validation", "passed_count", default=0) or 0)
    if stage == "coarse_merge":
        return int(_nested_get(detection, "candidate_merge_coarse", "after_count", default=0) or 0)
    if stage == "body_center":
        return int(_nested_get(detection, "body_center_refinement", "refined_count", default=0) or 0)
    if stage == "final_merge":
        return int(_nested_get(detection, "candidate_merge_final", "after_count", default=0) or 0)
    if stage == "dbscan":
        return int(_nested_get(detection, "dbscan", "output_count", default=0) or 0)
    if stage == "tracker_input":
        return int(_nested_get(trace, "tracker_input_filter", "tracker_input_count", default=0) or 0)
    if stage == "prediction":
        return int(_nested_get(tracker, "kalman_prediction", "track_count", default=0) or 0)
    if stage == "association":
        return int(_nested_get(tracker, "association", "matched_count", default=0) or 0)
    if stage == "update":
        return int(_nested_get(tracker, "kalman_update", "updated_count", default=0) or 0)
    if stage == "display":
        confirmed = int(_nested_get(trace, "display_output", "confirmed_count", default=0) or 0)
        tentative = int(_nested_get(trace, "display_output", "tentative_count", default=0) or 0)
        return confirmed + tentative
    return 0


def _representative_stage_point(trace: dict, stage: str) -> dict | None:
    points = [
        point for point in _trace_stage_points(trace, stage)
        if isinstance(point, dict)
        and _as_float(point.get("x_m")) is not None
        and _as_float(point.get("y_m")) is not None
    ]
    if not points:
        return None

    def point_score(point: dict) -> float:
        for key in ("is_primary",):
            if bool(point.get(key)):
                return 1e9
        for key in ("score", "confidence", "rdi_peak", "rai_peak"):
            value = _as_float(point.get(key))
            if value is not None:
                return value
        return 0.0

    return max(points, key=point_score)


def _collect_stage_trajectory(trace_rows: list[dict], stage: str) -> list[dict]:
    trajectory = []
    for index, trace in enumerate(trace_rows):
        point = _representative_stage_point(trace, stage)
        if point is None:
            continue
        trajectory.append(
            {
                "index": index,
                "frame_id": trace.get("frame_id", index),
                "x_m": _as_float(point.get("x_m"), 0.0) or 0.0,
                "y_m": _as_float(point.get("y_m"), 0.0) or 0.0,
                "score": point.get("score", point.get("confidence", "")),
            }
        )
    return trajectory


def _render_sequence_trajectory_overlay_svg_legacy(trace_rows: list[dict], selected_stages: list[tuple[str, str, str]]) -> None:
    trajectories = [
        (stage, label, color, _collect_stage_trajectory(trace_rows, stage))
        for stage, label, color in selected_stages
    ]
    trajectories = [(stage, label, color, rows) for stage, label, color, rows in trajectories if rows]
    if not trajectories:
        st.caption("선택한 stage에서 전체 궤적으로 그릴 x/y 좌표가 없습니다.")
        return

    all_points = [point for _, _, _, rows in trajectories for point in rows]
    xs = np.asarray([point["x_m"] for point in all_points], dtype=float)
    ys = np.asarray([point["y_m"] for point in all_points], dtype=float)
    x_abs = max(float(np.max(np.abs(xs))) + 0.15, 0.6)
    y_max = max(float(np.max(ys)) + 0.25, 3.0)
    y_min = 0.0

    width, height = 900, 520
    left, right, top, bottom = 58, 28, 26, 72
    plot_w = width - left - right
    plot_h = height - top - bottom

    def sx(x_value: float) -> float:
        return left + ((float(x_value) + x_abs) / max(2.0 * x_abs, 1e-6)) * plot_w

    def sy(y_value: float) -> float:
        return top + (1.0 - ((float(y_value) - y_min) / max(y_max - y_min, 1e-6))) * plot_h

    grid = []
    for gx in np.linspace(-x_abs, x_abs, 5):
        x = sx(float(gx))
        grid.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-bottom}" stroke="#e5edf2" />')
        grid.append(f'<text x="{x:.1f}" y="{height-45}" text-anchor="middle" fill="#6a7e8d" font-size="11">{gx:.1f}</text>')
    for gy in np.linspace(y_min, y_max, 5):
        y = sy(float(gy))
        grid.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e5edf2" />')
        grid.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" fill="#6a7e8d" font-size="11">{gy:.1f}</text>')

    paths = []
    legend = []
    for idx, (_, label, color, rows) in enumerate(trajectories):
        polyline = " ".join(f"{sx(row['x_m']):.1f},{sy(row['y_m']):.1f}" for row in rows)
        paths.append(
            f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2.6" stroke-opacity=".82" stroke-linejoin="round" stroke-linecap="round" />'
        )
        first = rows[0]
        last = rows[-1]
        paths.append(f'<circle cx="{sx(first["x_m"]):.1f}" cy="{sy(first["y_m"]):.1f}" r="5.2" fill="{color}" stroke="#fff" stroke-width="1.3"><title>{html.escape(label)} start frame {first["frame_id"]}</title></circle>')
        paths.append(f'<circle cx="{sx(last["x_m"]):.1f}" cy="{sy(last["y_m"]):.1f}" r="7.2" fill="{color}" stroke="#172232" stroke-width="1.3"><title>{html.escape(label)} end frame {last["frame_id"]}</title></circle>')
        lx = left + (idx % 4) * 190
        ly = height - 22 - (idx // 4) * 20
        coverage = len(rows) / max(len(trace_rows), 1)
        legend.append(f'<circle cx="{lx}" cy="{ly-4}" r="5" fill="{color}" />')
        legend.append(f'<text x="{lx+10}" y="{ly}" fill="#475d6e" font-size="12">{html.escape(label)} ({len(rows)}/{len(trace_rows)}, {coverage*100:.0f}%)</text>')

    radar_x = sx(0.0)
    radar_y = sy(0.0)
    svg = f"""
    <div style="border:1px solid #d9e5eb;border-radius:18px;background:#fff;padding:14px 16px 8px;">
      <div style="font-weight:850;margin-bottom:4px;color:#163044;">Whole Sequence Stage Trajectories</div>
      <div style="color:#63788a;font-size:.82rem;margin-bottom:8px;">
        프레임 하나가 아니라 전체 raw replay sequence에서 stage별 대표점이 그린 궤적입니다. 작은 점은 시작, 테두리 있는 큰 점은 마지막 위치입니다.
      </div>
      <svg viewBox="0 0 {width} {height}" style="width:100%;height:auto;display:block;">
        <rect x="0" y="0" width="{width}" height="{height}" rx="16" fill="#fbfdfe" />
        {''.join(grid)}
        <line x1="{left}" y1="{radar_y:.1f}" x2="{width-right}" y2="{radar_y:.1f}" stroke="#9eb0be" stroke-width="1.5" />
        <line x1="{radar_x:.1f}" y1="{top}" x2="{radar_x:.1f}" y2="{height-bottom}" stroke="#9eb0be" stroke-width="1.5" />
        {''.join(paths)}
        <circle cx="{radar_x:.1f}" cy="{radar_y:.1f}" r="7" fill="#172232" />
        <text x="{radar_x+10:.1f}" y="{radar_y-8:.1f}" fill="#172232" font-size="13">radar</text>
        <text x="{left}" y="{height-45}" fill="#63788a" font-size="12">x: radar-left (-) / radar-right (+) (m)</text>
        <text x="{width-right}" y="{top+14}" text-anchor="end" fill="#63788a" font-size="12">y: forward (m)</text>
        {''.join(legend)}
      </svg>
    </div>
    """
    _render_html(svg)


def _render_sequence_count_timeline_svg_legacy(trace_rows: list[dict], selected_stages: list[tuple[str, str, str]]) -> None:
    series = []
    for stage, label, color in selected_stages:
        points = [
            (index, _trace_stage_count(trace, stage))
            for index, trace in enumerate(trace_rows)
        ]
        if any(value for _, value in points):
            series.append((label, color, points))
    if not series:
        st.caption("선택한 stage의 count timeline 데이터가 없습니다.")
        return

    width, height = 900, 260
    left, right, top, bottom = 54, 20, 24, 54
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_min = 0
    x_max = max(len(trace_rows) - 1, 1)
    y_max = max(max(value for _, value in points) for _, _, points in series)
    y_max = max(float(y_max), 1.0)

    def sx(index: float) -> float:
        return left + (float(index) / max(x_max, 1)) * plot_w

    def sy(value: float) -> float:
        return top + (1.0 - (float(value) / y_max)) * plot_h

    lines = []
    legend = []
    for idx, (label, color, points) in enumerate(series):
        polyline = " ".join(f"{sx(index):.1f},{sy(value):.1f}" for index, value in points)
        lines.append(f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2.2" stroke-opacity=".86" />')
        lx = left + (idx % 4) * 190
        ly = height - 16 - (idx // 4) * 18
        legend.append(f'<circle cx="{lx}" cy="{ly-4}" r="5" fill="{color}" />')
        legend.append(f'<text x="{lx+10}" y="{ly}" fill="#475d6e" font-size="12">{html.escape(label)}</text>')

    svg = f"""
    <div style="border:1px solid #d9e5eb;border-radius:18px;background:#fff;padding:14px 16px 8px;">
      <div style="font-weight:850;margin-bottom:4px;color:#163044;">Stage Count Timeline</div>
      <div style="color:#63788a;font-size:.82rem;margin-bottom:8px;">
        전체 시간축에서 각 stage의 후보/track 수가 어떻게 변했는지 봅니다. 급락 지점이 frame drill-down 후보입니다.
      </div>
      <svg viewBox="0 0 {width} {height}" style="width:100%;height:auto;display:block;">
        <rect x="0" y="0" width="{width}" height="{height}" rx="16" fill="#fbfdfe" />
        <line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#9eb0be" />
        <line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#9eb0be" />
        <text x="{left}" y="{height-30}" fill="#63788a" font-size="12">frame 0 - {x_max}</text>
        <text x="{left-8}" y="{top+4}" text-anchor="end" fill="#63788a" font-size="12">{y_max:.0f}</text>
        <text x="{left-8}" y="{height-bottom}" text-anchor="end" fill="#63788a" font-size="12">0</text>
        {''.join(lines)}
        {''.join(legend)}
      </svg>
    </div>
    """
    _render_html(svg)


def _render_stage_sequence_overview(trace_rows: list[dict]) -> None:
    if not trace_rows:
        st.info("전체 sequence 시각화를 만들 `frame_trace.jsonl`이 없습니다. Force Rebuild로 stage cache를 다시 생성해 주세요.")
        return

    stage_options = [
        ("angle", "Angle candidates", "#2d8f7a"),
        ("body_center", "Body-center refined", "#c05d9f"),
        ("final_merge", "Merged candidates", "#5176b8"),
        ("dbscan", "DBSCAN output", "#0e8a7e"),
        ("tracker_input", "Tracker input", "#6155b8"),
        ("tracks", "Tracker state", "#172232"),
        ("display", "Display output", "#1b7a4c"),
    ]
    label_to_stage = {label: (stage, label, color) for stage, label, color in stage_options}
    default_labels = ["Angle candidates", "Body-center refined", "DBSCAN output", "Tracker state", "Display output"]
    selected_labels = st.multiselect(
        "Trajectory Stages",
        list(label_to_stage.keys()),
        default=[label for label in default_labels if label in label_to_stage],
        help="전체 움직임을 처리 단계별 궤적으로 비교합니다. 핵심은 DBSCAN까지 괜찮은지, Tracker/Display에서 깨지는지 보는 것입니다.",
    )
    selected_stages = [label_to_stage[label] for label in selected_labels] or [label_to_stage["DBSCAN output"]]

    _render_sequence_trajectory_overlay(trace_rows, selected_stages)
    _render_sequence_count_timeline(trace_rows, selected_stages)

    coverage_rows = []
    for stage, label, _ in selected_stages:
        trajectory = _collect_stage_trajectory(trace_rows, stage)
        trajectory_summary = _trajectory_stats(trajectory, len(trace_rows))
        counts = [_trace_stage_count(trace, stage) for trace in trace_rows]
        coverage_rows.append(
            {
                "stage": label,
                "trajectory_frames": trajectory_summary["trajectory_frames"],
                "coverage": trajectory_summary["coverage"],
                "path_length_m": trajectory_summary["path_length_m"],
                "max_step_m": trajectory_summary["max_step_m"],
                "p95_step_m": trajectory_summary["p95_step_m"],
                "count_mean": round(float(np.mean(counts)), 3) if counts else 0,
                "count_min": min(counts) if counts else 0,
                "count_max": max(counts) if counts else 0,
            }
        )
    _render_table(coverage_rows, key="stage-sequence-coverage", height=180)


def _render_trace_funnel(trace: dict) -> None:
    detection = trace.get("detection") or {}
    tracker = trace.get("tracker") or {}
    stages = [
        ("CFAR", _nested_get(detection, "cfar", "candidate_count", default=0), "#2869a6"),
        ("Angle", _nested_get(detection, "angle_validation", "passed_count", default=0), "#2d8f7a"),
        ("Coarse Merge", _nested_get(detection, "candidate_merge_coarse", "after_count", default=0), "#d6862c"),
        ("Body Center", _nested_get(detection, "body_center_refinement", "refined_count", default=0), "#c05d9f"),
        ("Final Merge", _nested_get(detection, "candidate_merge_final", "after_count", default=0), "#5176b8"),
        ("DBSCAN", _nested_get(detection, "dbscan", "output_count", default=0), "#0e8a7e"),
        ("Tracker In", _nested_get(trace, "tracker_input_filter", "tracker_input_count", default=0), "#6155b8"),
        ("Matched", _nested_get(tracker, "association", "matched_count", default=0), "#8d63c7"),
        ("Display", _nested_get(trace, "display_output", "confirmed_count", default=0), "#1b7a4c"),
    ]
    numeric_counts = [max(0.0, _as_float(count, 0.0) or 0.0) for _, count, _ in stages]
    max_count = max(numeric_counts) if numeric_counts else 1.0
    max_count = max(max_count, 1.0)

    labels = [label for label, _, _ in stages]
    colors = [color for _, _, color in stages]

    fig, plt = _make_figure(width=5.6, height=3.9)
    ax = fig.add_subplot(111)
    y_positions = np.arange(len(labels))
    ax.barh(y_positions, numeric_counts, color=colors, alpha=0.92)
    ax.set_yticks(y_positions, labels=labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, max_count * 1.18)
    ax.set_title("Stage Count Funnel", loc="left", fontsize=12, fontweight="bold", color="#163044")
    ax.set_xlabel("candidate / track count")
    ax.grid(True, axis="x", color="#e5edf2", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#fbfdfe")
    for y, value in zip(y_positions, numeric_counts):
        ax.text(value + max_count * 0.025, y, f"{value:g}", va="center", fontsize=9, color="#20384d")
    _render_matplotlib_figure(fig, caption="후보 수가 어느 단계에서 급격히 줄어드는지 보는 frame 단위 funnel입니다.")


def _render_trace_spatial_view_svg_legacy(trace: dict) -> None:
    stage_specs = [
        ("angle", "Angle", "#2d8f7a", 4.8, 0.45),
        ("body_center", "Body", "#c05d9f", 5.4, 0.72),
        ("final_merge", "Merge", "#5176b8", 5.8, 0.76),
        ("dbscan", "DBSCAN", "#0e8a7e", 7.2, 0.92),
        ("tracker_input", "Tracker In", "#6155b8", 8.2, 0.95),
        ("tracks", "Track", "#172232", 9.0, 0.95),
    ]
    point_sets = []
    all_points = []
    for key, label, color, radius, opacity in stage_specs:
        points = [
            point for point in _trace_stage_points(trace, key)[:12]
            if _as_float(point.get("x_m")) is not None and _as_float(point.get("y_m")) is not None
        ]
        if points:
            point_sets.append((key, label, color, radius, opacity, points))
            all_points.extend(points)

    if not all_points:
        st.caption("표시할 x/y 후보 좌표가 없습니다.")
        return

    roi = _nested_get(trace, "detection", "roi", default={}) or {}
    lateral = max(0.6, abs(_as_float(roi.get("lateral_limit_m"), 1.5) or 1.5))
    forward = max(3.0, _as_float(roi.get("forward_limit_m"), 4.0) or 4.0)
    xs = [_as_float(point.get("x_m"), 0.0) or 0.0 for point in all_points]
    ys = [_as_float(point.get("y_m"), 0.0) or 0.0 for point in all_points]
    x_min = min(-lateral, min(xs) - 0.15)
    x_max = max(lateral, max(xs) + 0.15)
    y_min = 0.0
    y_max = max(forward, max(ys) + 0.25)

    width, height = 560, 360
    left, right, top, bottom = 48, 22, 26, 42
    plot_w = width - left - right
    plot_h = height - top - bottom

    def sx(x_value: float) -> float:
        return left + ((x_value - x_min) / max(x_max - x_min, 1e-6)) * plot_w

    def sy(y_value: float) -> float:
        return top + (1.0 - ((y_value - y_min) / max(y_max - y_min, 1e-6))) * plot_h

    grid = []
    for gx in np.linspace(x_min, x_max, 5):
        x = sx(float(gx))
        grid.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-bottom}" stroke="#e5edf2" />')
        grid.append(f'<text x="{x:.1f}" y="{height-18}" text-anchor="middle" fill="#6a7e8d" font-size="10">{gx:.1f}</text>')
    for gy in np.linspace(y_min, y_max, 5):
        y = sy(float(gy))
        grid.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e5edf2" />')
        grid.append(f'<text x="{left-8}" y="{y+3:.1f}" text-anchor="end" fill="#6a7e8d" font-size="10">{gy:.1f}</text>')

    body_lines = []
    pairs = _nested_get(trace, "detection", "body_center_refinement", "pairs", default=[]) or []
    for pair in pairs[:10]:
        before = pair.get("before") if isinstance(pair, dict) else None
        after = pair.get("after") if isinstance(pair, dict) else None
        if not isinstance(before, dict) or not isinstance(after, dict):
            continue
        bx = _as_float(before.get("x_m"))
        by = _as_float(before.get("y_m"))
        ax = _as_float(after.get("x_m"))
        ay = _as_float(after.get("y_m"))
        if None in (bx, by, ax, ay):
            continue
        body_lines.append(
            f'<line x1="{sx(bx):.1f}" y1="{sy(by):.1f}" x2="{sx(ax):.1f}" y2="{sy(ay):.1f}" stroke="#c05d9f" stroke-width="1.5" stroke-opacity=".45" stroke-dasharray="4 3" />'
        )

    markers = []
    for _, label, color, radius, opacity, points in point_sets:
        for idx, point in enumerate(points):
            x_value = _as_float(point.get("x_m"), 0.0) or 0.0
            y_value = _as_float(point.get("y_m"), 0.0) or 0.0
            score = point.get("score", point.get("confidence", ""))
            title = f"{label} #{idx + 1}: x={x_value:.3f}, y={y_value:.3f}, score={score}"
            markers.append(
                f"""
                <circle cx="{sx(x_value):.1f}" cy="{sy(y_value):.1f}" r="{radius:.1f}" fill="{color}" fill-opacity="{opacity}" stroke="#ffffff" stroke-width="1.4">
                  <title>{html.escape(title)}</title>
                </circle>
                """
            )

    radar_x = sx(0.0)
    radar_y = sy(0.0)
    legend = []
    lx, ly = left, height - 8
    for index, (_, label, color, _, _, points) in enumerate(point_sets):
        x = lx + (index % 3) * 150
        y = ly - (index // 3) * 18
        legend.append(f'<circle cx="{x}" cy="{y-4}" r="5" fill="{color}" />')
        legend.append(f'<text x="{x+10}" y="{y}" fill="#475d6e" font-size="11">{html.escape(label)} ({len(points)})</text>')

    svg = f"""
    <div style="
      border:1px solid #d9e5eb;
      border-radius:18px;
      padding:14px 14px 8px;
      background:#fbfdfe;
      box-shadow:0 10px 26px rgba(18,40,56,.06);
    ">
      <div style="font-size:.95rem;font-weight:850;color:#20384d;margin-bottom:4px;">Candidate Spatial Evolution</div>
      <div style="font-size:.78rem;color:#617789;margin-bottom:8px;">같은 x/y 좌표계에서 후보가 단계별로 어디에 남는지 봅니다.</div>
      <svg viewBox="0 0 {width} {height}" width="100%" height="360" role="img" aria-label="candidate spatial evolution">
        <rect x="0" y="0" width="{width}" height="{height}" rx="16" fill="#ffffff" />
        {''.join(grid)}
        <line x1="{left}" y1="{radar_y:.1f}" x2="{width-right}" y2="{radar_y:.1f}" stroke="#9eb0be" stroke-width="1.5" />
        <line x1="{radar_x:.1f}" y1="{top}" x2="{radar_x:.1f}" y2="{height-bottom}" stroke="#9eb0be" stroke-width="1.5" />
        {''.join(body_lines)}
        {''.join(markers)}
        <circle cx="{radar_x:.1f}" cy="{radar_y:.1f}" r="7" fill="#172232" />
        <text x="{radar_x + 10:.1f}" y="{radar_y - 8:.1f}" fill="#172232" font-size="12">radar</text>
        <text x="{width-right}" y="{top+14}" text-anchor="end" fill="#63788a" font-size="11">y: forward (m)</text>
        <text x="{left}" y="{height-18}" fill="#63788a" font-size="11">x: radar-left (-) / radar-right (+) (m)</text>
        {''.join(legend)}
      </svg>
    </div>
    """
    _render_html(svg)


def _render_trace_flow(trace: dict) -> None:
    if not trace:
        st.info("이 stage cache에는 아직 상세 trace가 없습니다. Stage Cache를 Force Rebuild로 다시 생성하면 `frame_trace.jsonl`이 만들어집니다.")
        return

    raw_packets = _nested_get(trace, "raw_udp_packets", "packets_in_frame", default="n/a")
    invalid = _nested_get(trace, "frame_parsing", "invalid", default=False)
    cfar_count = _nested_get(trace, "detection", "cfar", "candidate_count", default=0)
    angle_passed = _nested_get(trace, "detection", "angle_validation", "passed_count", default=0)
    body_refined = _nested_get(trace, "detection", "body_center_refinement", "refined_count", default=0)
    merge_after = _nested_get(trace, "detection", "candidate_merge_final", "after_count", default=0)
    dbscan_out = _nested_get(trace, "detection", "dbscan", "output_count", default=0)
    tracker_in = _nested_get(trace, "tracker_input_filter", "tracker_input_count", default=0)
    predicted = _nested_get(trace, "tracker", "kalman_prediction", "track_count", default=0)
    matched = _nested_get(trace, "tracker", "association", "matched_count", default=0)
    updated = _nested_get(trace, "tracker", "kalman_update", "updated_count", default=0)
    births = len(_nested_get(trace, "tracker", "track_lifecycle", "births", default=[]) or [])
    deleted = len(_nested_get(trace, "tracker", "track_lifecycle", "deleted_track_ids", default=[]) or [])
    display_confirmed = _nested_get(trace, "display_output", "confirmed_count", default=0)

    snapshot_rows = [
        {"stage": "raw UDP", "value": raw_packets, "meaning": "packets in frame"},
        {"stage": "parsing", "value": "invalid" if invalid else "ok", "meaning": _nested_get(trace, "frame_parsing", "invalid_reason", default="frame health")},
        {"stage": "radar cube", "value": _nested_get(trace, "radar_cube", "shape", default="ok"), "meaning": "ADC reshape output"},
        {"stage": "static removal", "value": "on" if _nested_get(trace, "static_removal", "enabled", default=False) else "off", "meaning": "clutter removal"},
        {"stage": "shared FFT", "value": _nested_get(trace, "shared_fft", "shape", default="ok"), "meaning": "common FFT cache"},
        {"stage": "RDI", "value": _nested_get(trace, "rdi", "shape", default="n/a"), "meaning": "range-doppler map"},
        {"stage": "RAI", "value": _nested_get(trace, "rai", "shape", default="n/a"), "meaning": "range-angle map"},
        {"stage": "CFAR", "value": cfar_count, "meaning": "raw candidates"},
        {"stage": "angle validation", "value": angle_passed, "meaning": "ROI/angle passed"},
        {"stage": "body center", "value": body_refined, "meaning": "representative points"},
        {"stage": "candidate merge", "value": merge_after, "meaning": "after final merge"},
        {"stage": "DBSCAN", "value": dbscan_out, "meaning": "clusters"},
        {"stage": "tracker input", "value": tracker_in, "meaning": "measurements"},
        {"stage": "prediction", "value": predicted, "meaning": "predicted tracks"},
        {"stage": "association", "value": matched, "meaning": "matched pairs"},
        {"stage": "update", "value": updated, "meaning": "Kalman updates"},
        {"stage": "birth/delete", "value": f"+{births}/-{deleted}", "meaning": "track lifecycle"},
        {"stage": "display", "value": display_confirmed, "meaning": "confirmed tracks"},
    ]
    st.markdown("#### Frame Stage Snapshot")
    _render_table(snapshot_rows, key=f"stage-snapshot-{trace.get('frame_id', 'unknown')}", height=235)

    visual_left, visual_right = st.columns([0.92, 1.08])
    with visual_left:
        _render_trace_funnel(trace)
    with visual_right:
        _render_trace_spatial_view(trace)

    flow_rows = [
        {"stage": "raw UDP packets", "input": "-", "output": raw_packets, "meaning": "DCA1000 packet이 frame 하나로 모이는 전 단계"},
        {"stage": "frame parsing / assembly", "input": raw_packets, "output": "invalid" if invalid else "valid", "meaning": "gap/sequence/byte mismatch로 frame health 판단"},
        {"stage": "radar cube", "input": "IQ samples", "output": _nested_get(trace, "radar_cube", "shape", default="n/a"), "meaning": "ADC raw를 range/chirp/rx 구조로 reshape"},
        {"stage": "static removal", "input": "cube", "output": "enabled" if _nested_get(trace, "static_removal", "enabled", default=False) else "disabled", "meaning": "정적 클러터 제거"},
        {"stage": "shared FFT", "input": "cube", "output": _nested_get(trace, "shared_fft", "shape", default="n/a"), "meaning": "RDI/RAI에 공통으로 쓰는 range-doppler FFT"},
        {"stage": "RDI", "input": "FFT cube", "output": _nested_get(trace, "rdi", "shape", default="n/a"), "meaning": "range-doppler energy map"},
        {"stage": "RAI", "input": "FFT cube", "output": _nested_get(trace, "rai", "shape", default="n/a"), "meaning": "range-angle energy map"},
        {"stage": "CFAR candidates", "input": "RDI", "output": cfar_count, "meaning": "RDI local peak + CFAR threshold 통과 후보"},
        {"stage": "angle validation", "input": cfar_count, "output": angle_passed, "meaning": "ROI, RAI peak, angle contrast, local peak 검증"},
        {"stage": "body-center refinement", "input": _nested_get(trace, "detection", "body_center_refinement", "input_count", default=0), "output": body_refined, "meaning": "강한 반사점 근처 patch에서 대표점을 몸 중심 쪽으로 보정"},
        {"stage": "candidate merge", "input": _nested_get(trace, "detection", "candidate_merge_final", "before_count", default=0), "output": merge_after, "meaning": "거리/도플러 기준으로 가까운 후보 결합"},
        {"stage": "DBSCAN clustering", "input": _nested_get(trace, "detection", "dbscan", "input_count", default=0), "output": dbscan_out, "meaning": "최종 detection cluster 구성"},
        {"stage": "tracker input filter", "input": dbscan_out, "output": tracker_in, "meaning": "invalid frame 정책과 birth block 적용"},
        {"stage": "Kalman prediction", "input": "existing tracks", "output": predicted, "meaning": "기존 track의 현재 frame 예측"},
        {"stage": "association", "input": f"tracks={predicted}, meas={tracker_in}", "output": matched, "meaning": "예측 track과 measurement 매칭"},
        {"stage": "Kalman update", "input": matched, "output": updated, "meaning": "매칭된 track 상태 업데이트"},
        {"stage": "birth / miss / delete", "input": "unmatched", "output": f"births={births}, deleted={deleted}", "meaning": "새 track 생성, miss 누적, 삭제"},
        {"stage": "display output", "input": "tracks", "output": display_confirmed, "meaning": "화면/리포트에 남는 confirmed track"},
    ]
    _render_table(flow_rows, key=f"trace-flow-{trace.get('frame_id', 'unknown')}", height=280)


def _render_metric_timeline(title: str, rows: list[dict], metric: str, *, threshold=None, lower_is_better=True) -> None:
    points = [
        (int(row.get("ordinal", index)), _metric_value(row, metric), row)
        for index, row in enumerate(rows)
    ]
    points = [(x, y, row) for x, y, row in points if y is not None]
    if not points:
        st.caption(f"`{metric}` 값을 가진 frame feature가 없습니다.")
        return

    fig, plt = _make_figure(width=10.5, height=3.3)
    ax = fig.add_subplot(111)
    xs = [x for x, _, _ in points]
    ys = [y for _, y, _ in points]
    ax.plot(xs, ys, color="#0f6fb9", linewidth=2.1, marker="o", markersize=3.2)

    for x, y, row in points:
        bottleneck = row.get("frame_bottleneck")
        severity = float(row.get("frame_severity_10") or 0.0)
        if bottleneck == "ok" and severity < 5:
            continue
        color = "#d94141" if severity >= 8 else "#d88922"
        ax.scatter([x], [y], s=34, color=color, zorder=5)

    if threshold is not None:
        try:
            threshold_value = float(threshold)
        except (TypeError, ValueError):
            threshold_value = None
        if threshold_value is not None:
            threshold_color = "#b74040" if lower_is_better else "#237a55"
            ax.axhline(threshold_value, color=threshold_color, linestyle="--", linewidth=1.2, label=f"target {threshold_value:g}")
            ax.legend(loc="best", frameon=False)

    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color="#163044")
    ax.set_xlabel("frame index")
    ax.set_ylabel(metric)
    ax.grid(True, color="#e5edf2", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#fbfdfe")
    _render_matplotlib_figure(fig, caption="orange/red marker = frame-level bottleneck candidate")


def _render_path_preview(rows: list[dict]) -> None:
    points = []
    for row in rows:
        x = _metric_value(row, "lead_x_m")
        y = _metric_value(row, "lead_y_m")
        if x is None or y is None:
            continue
        points.append((x, y, row))
    if not points:
        st.caption("lead track 좌표가 있는 frame feature가 없습니다.")
        return

    fig, plt = _make_figure(width=6.2, height=5.0)
    ax = fig.add_subplot(111)
    xs = [x for x, _, _ in points]
    ys = [y for _, y, _ in points]
    ax.plot(xs, ys, color="#0f6fb9", linewidth=2.0, marker="o", markersize=3)
    ax.scatter([xs[0]], [ys[0]], s=55, color="#0f6fb9", label="start", zorder=5)
    ax.scatter([xs[-1]], [ys[-1]], s=70, color="#172232", label="end", zorder=5)
    ax.scatter([0.0], [0.0], s=90, color="#172232", marker="s", label="radar", zorder=6)
    for x, y, row in points:
        severity = float(row.get("frame_severity_10") or 0.0)
        if severity < 7:
            continue
        color = "#d94141" if severity >= 8 else "#d88922"
        ax.scatter([x], [y], s=42, color=color, zorder=7)
    x_abs = max(max(abs(value) for value in xs), 0.6)
    y_max = max(max(ys), 3.0)
    ax.set_xlim(-x_abs * 1.12, x_abs * 1.12)
    ax.set_ylim(0.0, y_max * 1.08)
    ax.set_title("Lead Path Preview", loc="left", fontsize=12, fontweight="bold", color="#163044")
    ax.set_xlabel("x: radar-left (-) / radar-right (+) (m)")
    ax.set_ylabel("y: forward (m)")
    ax.grid(True, color="#e5edf2", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#fbfdfe")
    ax.legend(loc="best", frameon=False)
    _render_matplotlib_figure(fig)


def _trajectory_stats(trajectory: list[dict], frame_count: int) -> dict:
    if not trajectory:
        return {
            "trajectory_frames": 0,
            "coverage": "0.0%",
            "path_length_m": "n/a",
            "max_step_m": "n/a",
            "p95_step_m": "n/a",
        }
    steps = []
    for before, after in zip(trajectory, trajectory[1:]):
        steps.append(float(np.hypot(after["x_m"] - before["x_m"], after["y_m"] - before["y_m"])))
    return {
        "trajectory_frames": len(trajectory),
        "coverage": _format_percent(len(trajectory) / max(frame_count, 1)),
        "path_length_m": round(float(np.sum(steps)), 3) if steps else 0.0,
        "max_step_m": round(float(np.max(steps)), 3) if steps else 0.0,
        "p95_step_m": round(float(np.quantile(steps, 0.95)), 3) if steps else 0.0,
    }


def _load_compare_final_trajectory(session_id: str) -> dict:
    trace_rows = stage_cache.load_stage_traces(PROJECT_ROOT, session_id)
    manifest = stage_cache.load_stage_cache_manifest(PROJECT_ROOT, session_id) or {}
    if not trace_rows:
        return {
            "session_id": session_id,
            "trace_rows": [],
            "stage": "display",
            "stage_label": "Display output",
            "trajectory": [],
            "stats": _trajectory_stats([], 0),
            "manifest": manifest,
        }

    for stage, label in [("display", "Display output"), ("tracks", "Tracker state")]:
        trajectory = _collect_stage_trajectory(trace_rows, stage)
        if trajectory:
            return {
                "session_id": session_id,
                "trace_rows": trace_rows,
                "stage": stage,
                "stage_label": label,
                "trajectory": trajectory,
                "stats": _trajectory_stats(trajectory, len(trace_rows)),
                "manifest": manifest,
            }
    return {
        "session_id": session_id,
        "trace_rows": trace_rows,
        "stage": "display",
        "stage_label": "Display output",
        "trajectory": [],
        "stats": _trajectory_stats([], len(trace_rows)),
        "manifest": manifest,
    }


def _render_compare_final_trajectory_grid(run_roles: list[tuple[str, dict]]) -> None:
    if not run_roles:
        return

    loaded = [(role, detail, _load_compare_final_trajectory(detail["session_id"])) for role, detail in run_roles]
    trajectories = [item[2]["trajectory"] for item in loaded if item[2]["trajectory"]]
    if not trajectories:
        st.info(
            "비교할 final tracking trajectory가 없습니다. Stage Debug에서 각 replay run의 stage cache를 먼저 생성해 주세요."
        )
        return

    all_points = [point for trajectory in trajectories for point in trajectory]
    xs = np.asarray([point["x_m"] for point in all_points], dtype=float)
    ys = np.asarray([point["y_m"] for point in all_points], dtype=float)
    x_abs = max(float(np.max(np.abs(xs))) + 0.15, 0.6)
    y_max = max(float(np.max(ys)) + 0.25, 3.0)

    panels = len(loaded)
    colors = ["#172232", "#1b7a4c", "#d88922", "#5176b8"]
    fig, plt = _make_figure(width=max(4.0 * panels, 8.0), height=4.3)
    axes = fig.subplots(1, panels, squeeze=False)[0]
    fig.suptitle(
        "Final Tracking Trajectory Comparison",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="bold",
        color="#163044",
    )

    stat_rows = []
    for index, (role, detail, payload) in enumerate(loaded):
        ax = axes[index]
        trajectory = payload["trajectory"]
        color = colors[index % len(colors)]
        title_id = detail["session_id"]
        annotation = _annotation_summary(detail, include_notes=True)
        if trajectory:
            tx = [point["x_m"] for point in trajectory]
            ty = [point["y_m"] for point in trajectory]
            ax.plot(tx, ty, color=color, linewidth=1.9, marker="o", markersize=2.6)
            ax.scatter([tx[0]], [ty[0]], s=48, color=color, edgecolors="white", linewidth=0.8, zorder=5)
            ax.scatter([tx[-1]], [ty[-1]], s=64, color=color, edgecolors="#172232", linewidth=1.0, zorder=5)
        else:
            ax.text(
                0.5,
                0.52,
                "No stage cache\nor no final track",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="#63788a",
                fontsize=10,
            )
        ax.scatter([0.0], [0.0], s=56, color="#172232", marker="s", zorder=6)
        stats = payload["stats"]
        manifest = payload.get("manifest") or {}
        limit_requested = manifest.get("frame_limit_requested")
        cache_note = "partial cache" if limit_requested else "full cache"
        ax.set_title(
            f"{role}: {title_id}\n{annotation}\n{cache_note} | coverage {stats['coverage']} | max step {stats['max_step_m']}m",
            fontsize=9,
            color="#20384d",
        )
        ax.set_xlim(-x_abs * 1.08, x_abs * 1.08)
        ax.set_ylim(0.0, y_max * 1.06)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.grid(True, color="#e5edf2", linewidth=0.8)
        ax.set_facecolor("#fbfdfe")
        ax.spines[["top", "right"]].set_visible(False)

        stat_rows.append(
            {
                "role": role,
                "session_id": detail["session_id"],
                "stage": payload["stage_label"],
                "cached_frames": len(payload.get("trace_rows") or []),
                "cache_scope": f"limit {limit_requested}" if limit_requested else "all",
                "trajectory_frames": stats["trajectory_frames"],
                "coverage": stats["coverage"],
                "path_length_m": stats["path_length_m"],
                "max_step_m": stats["max_step_m"],
                "p95_step_m": stats["p95_step_m"],
            }
        )

    fig.tight_layout(rect=[0, 0, 1, 0.88])
    _render_matplotlib_figure(
        fig,
        caption=(
            "Baseline과 replay 후보를 같은 x/y 축으로 비교합니다. "
            "Display output이 있으면 그것을 우선 사용하고, 없으면 Tracker state로 fallback합니다."
        ),
    )
    _render_table(stat_rows, key="compare-final-trajectory-stats", height=180)


def _render_sequence_trajectory_overlay(trace_rows: list[dict], selected_stages: list[tuple[str, str, str]]) -> None:
    trajectories = [
        (stage, label, color, _collect_stage_trajectory(trace_rows, stage))
        for stage, label, color in selected_stages
    ]
    trajectories = [(stage, label, color, rows) for stage, label, color, rows in trajectories if rows]
    if not trajectories:
        st.caption("선택한 stage에서 전체 궤적으로 그릴 x/y 좌표가 없습니다.")
        return

    all_points = [point for _, _, _, rows in trajectories for point in rows]
    xs = np.asarray([point["x_m"] for point in all_points], dtype=float)
    ys = np.asarray([point["y_m"] for point in all_points], dtype=float)
    x_abs = max(float(np.max(np.abs(xs))) + 0.15, 0.6)
    y_max = max(float(np.max(ys)) + 0.25, 3.0)
    columns = min(3, len(trajectories))
    rows_count = int(np.ceil(len(trajectories) / columns))
    fig, plt = _make_figure(width=4.2 * columns, height=3.7 * rows_count)
    axes = fig.subplots(rows_count, columns, squeeze=False)
    fig.suptitle("Raw Replay Stage Output Trajectory Comparison", x=0.01, ha="left", fontsize=13, fontweight="bold", color="#163044")

    for index, (_, label, color, trajectory) in enumerate(trajectories):
        ax = axes[index // columns][index % columns]
        tx = [point["x_m"] for point in trajectory]
        ty = [point["y_m"] for point in trajectory]
        ax.plot(tx, ty, color=color, linewidth=1.8, marker="o", markersize=2.5)
        ax.scatter([tx[0]], [ty[0]], s=46, color=color, edgecolors="white", linewidth=0.8, zorder=5)
        ax.scatter([tx[-1]], [ty[-1]], s=58, color=color, edgecolors="#172232", linewidth=1.0, zorder=5)
        ax.scatter([0.0], [0.0], s=52, color="#172232", marker="s", zorder=6)
        stats = _trajectory_stats(trajectory, len(trace_rows))
        ax.set_title(f"{label}\ncoverage {stats['coverage']} | max step {stats['max_step_m']}m", fontsize=10, color="#20384d")
        ax.set_xlim(-x_abs * 1.08, x_abs * 1.08)
        ax.set_ylim(0.0, y_max * 1.06)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.grid(True, color="#e5edf2", linewidth=0.8)
        ax.set_facecolor("#fbfdfe")
        ax.spines[["top", "right"]].set_visible(False)

    for index in range(len(trajectories), rows_count * columns):
        axes[index // columns][index % columns].axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _render_matplotlib_figure(
        fig,
        caption="각 패널은 같은 x/y 축으로 전체 raw replay를 처리한 stage별 대표 궤적입니다. 시작점은 작은 원, 마지막점은 테두리 있는 원입니다.",
    )


def _render_sequence_count_timeline(trace_rows: list[dict], selected_stages: list[tuple[str, str, str]]) -> None:
    series = []
    for stage, label, color in selected_stages:
        points = [(index, _trace_stage_count(trace, stage)) for index, trace in enumerate(trace_rows)]
        if any(value for _, value in points):
            series.append((label, color, points))
    if not series:
        st.caption("선택한 stage의 count timeline 데이터가 없습니다.")
        return

    fig, plt = _make_figure(width=10.5, height=3.4)
    ax = fig.add_subplot(111)
    for label, color, points in series:
        ax.plot([x for x, _ in points], [y for _, y in points], label=label, color=color, linewidth=1.9)
    ax.set_title("Stage Count Timeline", loc="left", fontsize=12, fontweight="bold", color="#163044")
    ax.set_xlabel("frame index")
    ax.set_ylabel("candidate / track count")
    ax.grid(True, color="#e5edf2", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#fbfdfe")
    ax.legend(loc="best", frameon=False, ncols=min(3, len(series)))
    _render_matplotlib_figure(fig, caption="후보 수가 급락하는 구간을 찾은 뒤 아래 frame slider로 상세 원인을 봅니다.")


def _render_trace_spatial_view(trace: dict) -> None:
    stage_specs = [
        ("angle", "Angle", "#2d8f7a"),
        ("body_center", "Body", "#c05d9f"),
        ("final_merge", "Merge", "#5176b8"),
        ("dbscan", "DBSCAN", "#0e8a7e"),
        ("tracker_input", "Tracker In", "#6155b8"),
        ("display", "Display", "#1b7a4c"),
    ]
    point_sets = []
    all_points = []
    for key, label, color in stage_specs:
        points = [
            point for point in _trace_stage_points(trace, key)[:12]
            if _as_float(point.get("x_m")) is not None and _as_float(point.get("y_m")) is not None
        ]
        if points:
            point_sets.append((label, color, points))
            all_points.extend(points)
    if not all_points:
        st.info("이 frame에는 x/y로 표시할 후보가 없습니다.")
        return

    fig, plt = _make_figure(width=6.4, height=4.6)
    ax = fig.add_subplot(111)
    xs = [_as_float(point.get("x_m"), 0.0) or 0.0 for point in all_points]
    ys = [_as_float(point.get("y_m"), 0.0) or 0.0 for point in all_points]
    x_abs = max(max(abs(value) for value in xs), 0.6)
    y_max = max(max(ys), 3.0)
    for label, color, points in point_sets:
        px = [_as_float(point.get("x_m"), 0.0) or 0.0 for point in points]
        py = [_as_float(point.get("y_m"), 0.0) or 0.0 for point in points]
        ax.scatter(px, py, s=38, color=color, alpha=0.78, label=f"{label} ({len(points)})", edgecolors="white", linewidth=0.5)
    pairs = _nested_get(trace, "detection", "body_center_refinement", "pairs", default=[]) or []
    for pair in pairs[:10]:
        before = pair.get("before") if isinstance(pair, dict) else None
        after = pair.get("after") if isinstance(pair, dict) else None
        if not isinstance(before, dict) or not isinstance(after, dict):
            continue
        bx = _as_float(before.get("x_m"))
        by = _as_float(before.get("y_m"))
        axv = _as_float(after.get("x_m"))
        ay = _as_float(after.get("y_m"))
        if None in (bx, by, axv, ay):
            continue
        ax.plot([bx, axv], [by, ay], color="#c05d9f", alpha=0.35, linestyle="--", linewidth=1.0)
    ax.scatter([0.0], [0.0], s=70, color="#172232", marker="s", label="radar", zorder=6)
    ax.set_xlim(-x_abs * 1.12, x_abs * 1.12)
    ax.set_ylim(0.0, y_max * 1.08)
    ax.set_title("Candidate Spatial Evolution", loc="left", fontsize=12, fontweight="bold", color="#163044")
    ax.set_xlabel("x: radar-left (-) / radar-right (+) (m)")
    ax.set_ylabel("y: forward (m)")
    ax.grid(True, color="#e5edf2", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#fbfdfe")
    ax.legend(loc="best", frameon=False, fontsize=8)
    _render_matplotlib_figure(fig, caption="선택 frame에서 stage별 후보가 x/y 평면 어디에 남았는지 보여 줍니다.")


def _run_filters(
    rows: list[dict],
    *,
    transport: str,
    input_mode: str,
    label: str,
    board: str,
    motion: str,
    benchmark_only: bool,
) -> list[dict]:
    filtered = rows
    if transport != "all":
        filtered = [row for row in filtered if (row.get("transport_category") or "unknown") == transport]
    if input_mode != "all":
        filtered = [row for row in filtered if (row.get("input_mode") or "unknown") == input_mode]
    if label != "all":
        filtered = [row for row in filtered if (row.get("annotation_label") or "") == label]
    if board != "all":
        filtered = [row for row in filtered if _row_board(row) == board]
    if motion != "all":
        filtered = [row for row in filtered if (row.get("annotation_motion_pattern") or "") == motion]
    if benchmark_only:
        filtered = [row for row in filtered if bool(row.get("annotation_keep_flag"))]
    return filtered


def _short_git(value) -> str:
    text = str(value or "")
    return text[:10] if len(text) > 10 else text


def _run_context_row(role: str, row: dict | None) -> dict:
    row = row or {}
    return {
        "role": role,
        "session_id": row.get("session_id") or "",
        "variant": row.get("variant") or "",
        "scenario_id": row.get("scenario_id") or "",
        "board": _row_board(row),
        "label": row.get("annotation_label") or "",
        "motion": row.get("annotation_motion_pattern") or "",
        "description": _short_text(row.get("annotation_notes"), 56),
        "git_commit": _short_git(row.get("git_commit")),
        "git_dirty": "yes" if bool(row.get("git_dirty")) else "no",
        "capture_id": row.get("capture_id") or "",
    }


def _parameter_diff_rows(role_parameters: dict[str, list[dict]], *, show_unchanged: bool = False) -> list[dict]:
    by_role = {
        role: {
            item["param_key"]: item
            for item in parameters
            if item.get("param_key")
        }
        for role, parameters in role_parameters.items()
    }
    keys = sorted({key for parameters in by_role.values() for key in parameters.keys()})
    rows = []
    for param_key in keys:
        role_values = {
            role: by_role.get(role, {}).get(param_key, {}).get("param_value")
            for role in role_parameters.keys()
        }
        visible_values = [value for value in role_values.values() if value is not None]
        missing_in_some_roles = len(visible_values) != len(role_values)
        changed = len(set(visible_values)) > 1 or (bool(visible_values) and missing_in_some_roles)
        if not changed and not show_unchanged:
            continue
        first_item = next(
            (
                by_role.get(role, {}).get(param_key)
                for role in role_parameters.keys()
                if by_role.get(role, {}).get(param_key)
            ),
            {},
        )
        row = {
            "param_group": first_item.get("param_group") or "",
            "param_key": param_key,
        }
        row.update({role: role_values[role] if role_values[role] is not None else "" for role in role_parameters.keys()})
        row["changed"] = "yes" if changed else "no"
        rows.append(row)
    return rows


def _run_select_label(session_id: str, runs_by_id: dict[str, dict]) -> str:
    if not session_id:
        return "None"
    row = runs_by_id.get(session_id, {})
    return f"{session_id} | {_annotation_summary(row, include_notes=True)}"


def _compare_metric_display(value) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(numeric) >= 100:
        return f"{numeric:.1f}"
    if abs(numeric) >= 10:
        return f"{numeric:.2f}"
    return f"{numeric:.4f}".rstrip("0").rstrip(".")


def _compare_metric_delta(value, baseline_value, direction: str) -> str:
    if value is None or baseline_value is None:
        return "n/a"
    try:
        delta = float(value) - float(baseline_value)
    except (TypeError, ValueError):
        return "n/a"
    if abs(delta) < 1e-9:
        verdict = "same"
    elif direction == "lower":
        verdict = "improved" if delta < 0 else "regressed"
    else:
        verdict = "improved" if delta > 0 else "regressed"
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:.4f}".rstrip("0").rstrip(".") + f" ({verdict})"


def _compare_metric_rows(run_roles: list[tuple[str, dict]], metrics: list[tuple[str, str, str]]) -> list[dict]:
    if not run_roles:
        return []
    baseline = run_roles[0][1]
    rows = []
    for label, key, direction in metrics:
        row = {"metric": label}
        baseline_value = baseline.get(key)
        for role, detail in run_roles:
            value = detail.get(key)
            row[role] = _compare_metric_display(value)
            if role != run_roles[0][0]:
                row[f"{role}_delta"] = _compare_metric_delta(value, baseline_value, direction)
        rows.append(row)
    return rows


def _capture_filters(rows: list[dict], *, transport: str, label: str, board: str, motion: str, benchmark_only: bool) -> list[dict]:
    filtered = rows
    if transport != "all":
        filtered = [row for row in filtered if (row.get("transport_category") or "unknown") == transport]
    if label != "all":
        filtered = [row for row in filtered if (row.get("annotation_label") or "") == label]
    if board != "all":
        filtered = [row for row in filtered if _row_board(row) == board]
    if motion != "all":
        filtered = [row for row in filtered if (row.get("annotation_motion_pattern") or "") == motion]
    if benchmark_only:
        filtered = [row for row in filtered if bool(row.get("annotation_keep_flag"))]
    return filtered


def _annotation_form(target_type: str, target_id: str, row: dict) -> None:
    label_default = row.get("annotation_label") or ""
    board_default = _row_board(row)
    people_default = int(row.get("annotation_people_count") or 1)
    motion_default = row.get("annotation_motion_pattern") or ""
    notes_default = row.get("annotation_notes") or ""
    keep_default = bool(row.get("annotation_keep_flag"))

    with st.form(f"annotation-{target_type}-{target_id}"):
        col1, col2, col3 = st.columns(3)
        with col1:
            label = st.selectbox(
                "Label",
                LABEL_OPTIONS,
                index=_option_index(LABEL_OPTIONS, label_default),
            )
            people_count = st.number_input("People Count", min_value=0, max_value=20, value=people_default, step=1)
        with col2:
            board_type = st.selectbox(
                "Board",
                BOARD_OPTIONS,
                index=_option_index(BOARD_OPTIONS, board_default),
                help="이 측정이 어떤 안테나 보드 기준인지 기록합니다.",
            )
            keep_flag = st.checkbox("Benchmark Set에 포함", value=keep_default)
        with col3:
            motion_pattern = st.selectbox(
                "Motion / Scenario",
                MOTION_OPTIONS,
                index=_option_index(MOTION_OPTIONS, motion_default),
            )
        notes = st.text_area(
            "Description / Notes",
            value=notes_default,
            height=120,
            placeholder="예: 레이더 기준 오른쪽 대각선 왕복, 중앙 왕복, 축 반전 확인용, Wi-Fi off",
        )
        if st.form_submit_button("Save Annotation"):
            registry.save_annotation(
                PROJECT_ROOT,
                target_type=target_type,
                target_id=target_id,
                label=label,
                keep_flag=keep_flag,
                people_count=int(people_count) if people_count > 0 else None,
                motion_pattern=motion_pattern,
                board_type=board_type,
                notes=notes,
            )
            st.success("annotation을 저장했습니다.")
            _rerun()


def _wandb_export_section(detail: dict) -> None:
    session_id = str(detail["session_id"])
    session_dir = Path(detail["session_dir"])
    readiness = wandb_sync.sync_readiness(detail)
    last_sync = wandb_sync.read_sync_result(PROJECT_ROOT, session_id)
    wandb_installed = wandb_sync.wandb_available()

    st.markdown("#### W&B Export")
    st.caption(
        "검토가 끝난 run을 W&B에 누적 기록하는 구간입니다. 기본은 offline 모드로 저장하고, "
        "원하면 나중에 온라인 sync로 밀어 올릴 수 있게 설계했습니다."
    )

    if readiness["hard_blockers"]:
        for message in readiness["hard_blockers"]:
            st.error(message)
    if readiness["soft_blockers"]:
        for message in readiness["soft_blockers"]:
            st.warning(message)
    for message in readiness["warnings"]:
        st.info(message)

    if not wandb_installed:
        st.warning("현재 Python 환경에는 `wandb`가 설치되어 있지 않습니다. contract 저장은 가능하지만 실제 sync는 아직 막혀 있습니다.")
        st.code("pip install wandb", language="bash")

    if last_sync:
        a, b, c = st.columns(3)
        a.metric("Last Sync Mode", last_sync.get("mode") or "n/a")
        b.metric("Artifacts", last_sync.get("artifact_count") or 0)
        c.metric("Frame Metric Steps", last_sync.get("frame_metric_steps_logged") or 0)
        if last_sync.get("url"):
            st.markdown(f"[Open W&B Run]({last_sync['url']})")
        elif (last_sync.get("mode") or "") == "offline":
            st.caption("offline run으로 저장되어 아직 웹 URL은 없습니다. 나중에 `python -m wandb sync <local_run_dir>`로 밀어 올리면 됩니다.")
        _render_file_links(
            {
                "wandb contract": session_dir / "wandb_run_contract.json",
                "wandb sync result": session_dir / "wandb_sync_result.json",
                "local wandb run dir": last_sync.get("local_run_dir"),
            }
        )

    mode = st.selectbox(
        "W&B Mode",
        ["offline", "online"],
        index=0,
        key=f"wandb-mode-{session_id}",
        help="offline은 login 없이 로컬에 기록만 남깁니다. online은 W&B 계정으로 바로 업로드합니다.",
    )
    project_name = st.text_input(
        "W&B Project",
        value=wandb_sync.DEFAULT_PROJECT,
        key=f"wandb-project-{session_id}",
    )
    phase_options = ["benchmark", "debug", "paper"]
    recommended_phase = readiness["recommended_phase"]
    phase = st.selectbox(
        "Phase Tag",
        phase_options,
        index=phase_options.index(recommended_phase) if recommended_phase in phase_options else 0,
        key=f"wandb-phase-{session_id}",
    )
    col1, col2 = st.columns(2)
    with col1:
        include_frame_features = st.checkbox(
            "Attach frame_features.jsonl if present",
            value=False,
            key=f"wandb-frame-features-{session_id}",
        )
    with col2:
        log_frame_metrics = st.checkbox(
            "Log frame timeline metrics if present",
            value=False,
            key=f"wandb-frame-metrics-{session_id}",
            help="compute_total_ms, detection_count, confirmed_track_count 같은 대표 frame metric만 step chart로 보냅니다.",
        )
    force_export = False
    if readiness["soft_blockers"]:
        force_export = st.checkbox(
            "review 경고를 무시하고 이 run도 내보내기",
            value=False,
            key=f"wandb-force-{session_id}",
        )

    contract = wandb_sync.build_run_contract(
        PROJECT_ROOT,
        session_id,
        project=project_name,
        mode=mode,
        phase=phase,
        include_frame_features=include_frame_features or log_frame_metrics,
    )
    with st.expander("Preview W&B Payload"):
        st.code(json.dumps(contract, ensure_ascii=False, indent=2), language="json")

    can_sync = (
        wandb_installed
        and not readiness["hard_blockers"]
        and (not readiness["soft_blockers"] or force_export)
    )
    button_a, button_b = st.columns(2)
    if button_a.button("Save W&B Contract", key=f"wandb-contract-{session_id}", width="stretch"):
        path = wandb_sync.write_run_contract(
            PROJECT_ROOT,
            session_id,
            project=project_name,
            mode=mode,
            phase=phase,
            include_frame_features=include_frame_features or log_frame_metrics,
        )
        st.success(f"contract를 저장했습니다: {path.name}")
        _rerun()
    if button_b.button(
        "Sync Run to W&B",
        key=f"wandb-sync-{session_id}",
        width="stretch",
        disabled=not can_sync,
    ):
        try:
            result = wandb_sync.sync_run(
                PROJECT_ROOT,
                session_id,
                project=project_name,
                mode=mode,
                phase=phase,
                include_frame_features=include_frame_features,
                log_frame_metrics=log_frame_metrics,
            )
        except Exception as error:
            st.error(f"W&B sync 실패: {error}")
        else:
            st.success(
                "W&B sync를 마쳤습니다. "
                f"mode={result.get('mode')}, artifacts={result.get('artifact_count')}, "
                f"frame_steps={result.get('frame_metric_steps_logged')}"
            )
            _rerun()


def _overview_page() -> None:
    overview = registry.get_registry_overview(PROJECT_ROOT)
    runs = registry.fetch_runs(PROJECT_ROOT)
    captures = registry.fetch_captures(PROJECT_ROOT)

    st.subheader("Overview")
    a, b, c, d = st.columns(4)
    a.metric("Runs", overview["run_total"])
    b.metric("Raw Captures", overview["capture_total"])
    c.metric("Clean Runs", overview["run_transport_counts"].get("clean", 0))
    d.metric(
        "Benchmark-tagged Runs",
        overview["run_annotation_counts"].get("baseline", 0)
        + overview["run_annotation_counts"].get("good", 0),
    )
    st.caption(f"DB: `{overview['db_path']}` | last refresh: `{overview['last_refresh_at'] or 'n/a'}`")

    st.markdown("### Run Transport Quality")
    _render_table(
        [{"category": key, "count": value} for key, value in sorted(overview["run_transport_counts"].items())],
        key="overview-run-transport",
    )

    st.markdown("### Recent Runs")
    _render_table(
        [
            {
                "session_id": row["session_id"],
                "input_mode": row.get("input_mode"),
                "transport": row.get("transport_category"),
                "op_score": row.get("operational_score"),
                "perf_score": _format_float(row.get("performance_score"), 1),
                "path_clean": _format_float(row.get("path_cleanliness_score_10"), 2),
                "board": _row_board(row),
                "annotation": _annotation_summary(row),
                "motion": row.get("annotation_motion_pattern") or "",
                "notes": _short_text(row.get("annotation_notes"), 56),
                "capture_id": row.get("capture_id") or "",
            }
            for row in runs[:12]
        ],
        key="overview-recent-runs",
    )

    st.markdown("### Recent Raw Captures")
    _render_table(
        [
            {
                "capture_id": row["capture_id"],
                "transport": row.get("transport_category"),
                "frames": row.get("frame_count"),
                "invalid_rate": _format_percent(row.get("invalid_rate")),
                "linked_runs": row.get("linked_run_count"),
                "board": _row_board(row),
                "annotation": _annotation_summary(row),
                "motion": row.get("annotation_motion_pattern") or "",
                "notes": _short_text(row.get("annotation_notes"), 56),
            }
            for row in captures[:12]
        ],
        key="overview-recent-captures",
    )


def _runs_page() -> None:
    all_runs = registry.fetch_runs(PROJECT_ROOT)
    transport = st.sidebar.selectbox("Run Transport Filter", ["all", "clean", "noisy", "unusable", "insufficient"])
    input_mode = st.sidebar.selectbox("Run Input Mode", ["all", "live", "replay"])
    label = st.sidebar.selectbox("Run Label Filter", ["all", *LABEL_OPTIONS[1:]])
    board = st.sidebar.selectbox("Run Board Filter", ["all", *BOARD_OPTIONS[1:]])
    motion = st.sidebar.selectbox("Run Motion Filter", ["all", *MOTION_OPTIONS[1:]])
    benchmark_only = st.sidebar.checkbox("Benchmark-tagged only", value=False)
    runs = _run_filters(
        all_runs,
        transport=transport,
        input_mode=input_mode,
        label=label,
        board=board,
        motion=motion,
        benchmark_only=benchmark_only,
    )

    st.subheader("Run Library")
    st.caption(
        f"{len(runs)} runs matched the current filter. "
        "Detail Session을 고른 뒤 Annotation에서 motion/description을 저장하면 표에 바로 보입니다."
    )
    _render_table(
        [
            {
                "session_id": row["session_id"],
                "input_mode": row.get("input_mode"),
                "transport": row.get("transport_category"),
                "op_score": row.get("operational_score"),
                "perf_score": _format_float(row.get("performance_score"), 1),
                "path_clean": _format_float(row.get("path_cleanliness_score_10"), 2),
                "board": _row_board(row),
                "label": row.get("annotation_label") or "",
                "motion": row.get("annotation_motion_pattern") or "",
                "description": _short_text(row.get("annotation_notes"), 72),
                "benchmark": "yes" if bool(row.get("annotation_keep_flag")) else "",
                "scenario_id": row.get("scenario_id") or "",
                "capture_id": row.get("capture_id") or "",
            }
            for row in runs
        ],
        key="runs-library",
    )
    if not runs:
        return

    runs_by_session = {row["session_id"]: row for row in runs}
    selected_session = st.selectbox(
        "Detail Session",
        [row["session_id"] for row in runs],
        index=0,
        format_func=lambda session_id: f"{session_id} | {_annotation_summary(runs_by_session.get(session_id, {}), include_notes=True)}",
    )
    detail = registry.fetch_run_detail(PROJECT_ROOT, selected_session)
    if detail is None:
        return

    st.markdown("### Run Detail")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transport", detail.get("transport_category") or "n/a")
    c2.metric("Operational", _format_float(detail.get("operational_score"), 0))
    c3.metric("Performance", _format_float(detail.get("performance_score"), 1))
    c4.metric("Path Cleanliness", _format_float(detail.get("path_cleanliness_score_10"), 2))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Render P95", _format_float(detail.get("render_latency_p95_ms"), 1, " ms"))
    c6.metric("Compute Util P95", _format_percent(detail.get("compute_utilization_p95")))
    c7.metric("Lead Switch Rate", _format_percent(detail.get("lead_confirmed_switch_rate")))
    c8.metric("Capture Link", detail.get("capture_id") or "none")

    stage_data = (detail.get("summary") or {}).get("diagnostics", {}).get("preferred_stage_timings_ms", {})
    timings = stage_data.get("timings", {})
    if timings:
        st.markdown("#### Preferred Stage Timings")
        _render_table(
            [
                {
                    "stage": name,
                    "mean_ms": values.get("mean"),
                    "p95_ms": values.get("p95"),
                    "max_ms": values.get("max"),
                }
                for name, values in sorted(timings.items())
            ],
            key="run-detail-stage-timings",
        )

    session_dir = Path(detail["session_dir"])
    st.markdown("#### Local Reports / Files")
    _render_file_links(
        {
            "session index": session_dir / "index.html",
            "performance report": session_dir / "performance_report.html",
            "trajectory replay": session_dir / "trajectory_replay.html",
            "ops report": session_dir / "ops_report.html",
            "summary.json": session_dir / "summary.json",
        }
    )
    if detail.get("capture_id"):
        st.caption(f"linked raw capture: `{detail['capture_id']}`")

    st.markdown("#### Annotation")
    _annotation_form("run", detail["session_id"], detail)
    _wandb_export_section(detail)


def _captures_page() -> None:
    all_captures = registry.fetch_captures(PROJECT_ROOT)
    transport = st.sidebar.selectbox("Capture Transport Filter", ["all", "clean", "noisy", "unusable", "insufficient"], key="capture-transport")
    label = st.sidebar.selectbox("Capture Label Filter", ["all", *LABEL_OPTIONS[1:]], key="capture-label")
    board = st.sidebar.selectbox("Capture Board Filter", ["all", *BOARD_OPTIONS[1:]], key="capture-board")
    motion = st.sidebar.selectbox("Capture Motion Filter", ["all", *MOTION_OPTIONS[1:]], key="capture-motion")
    benchmark_only = st.sidebar.checkbox("Benchmark-tagged captures only", value=False, key="capture-benchmark")
    captures = _capture_filters(
        all_captures,
        transport=transport,
        label=label,
        board=board,
        motion=motion,
        benchmark_only=benchmark_only,
    )

    st.subheader("Raw Capture Library")
    st.caption(f"{len(captures)} raw captures matched the current filter.")
    _render_table(
        [
            {
                "capture_id": row["capture_id"],
                "transport": row.get("transport_category"),
                "frames": row.get("frame_count"),
                "invalid_rate": _format_percent(row.get("invalid_rate")),
                "linked_runs": row.get("linked_run_count"),
                "board": _row_board(row),
                "label": row.get("annotation_label") or "",
                "motion": row.get("annotation_motion_pattern") or "",
                "description": _short_text(row.get("annotation_notes"), 72),
                "benchmark": "yes" if bool(row.get("annotation_keep_flag")) else "",
                "scenario_id": row.get("scenario_id") or "",
            }
            for row in captures
        ],
        key="captures-library",
    )
    if not captures:
        return

    selected_capture = st.selectbox("Detail Capture", [row["capture_id"] for row in captures], index=0)
    detail = registry.fetch_capture_detail(PROJECT_ROOT, selected_capture)
    if detail is None:
        return

    st.markdown("### Capture Detail")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transport", detail.get("transport_category") or "n/a")
    c2.metric("Frames", detail.get("frame_count") or 0)
    c3.metric("Invalid Rate", _format_percent(detail.get("invalid_rate")))
    c4.metric("Linked Runs", detail.get("linked_run_count") or 0)

    capture_dir = Path(detail["capture_dir"])
    st.markdown("#### Local Files")
    _render_file_links(
        {
            "capture manifest": capture_dir / "capture_manifest.json",
            "raw frame index": capture_dir / "raw_frames_index.jsonl",
            "source session index": Path(detail["source_session_dir"]) / "index.html" if detail.get("source_session_dir") else None,
        }
    )

    linked_runs = detail.get("linked_runs") or []
    if linked_runs:
        st.markdown("#### Linked Runs")
        _render_table(
            [
                {
                    "session_id": row["session_id"],
                    "input_mode": row.get("input_mode"),
                    "transport": row.get("transport_category"),
                    "perf_score": _format_float(row.get("performance_score"), 1),
                    "board": _row_board(row),
                    "label": row.get("annotation_label") or "",
                    "motion": row.get("annotation_motion_pattern") or "",
                    "description": _short_text(row.get("annotation_notes"), 56),
                }
                for row in linked_runs
            ],
            key="capture-linked-runs",
        )
    else:
        st.caption("이 raw capture에 연결된 run이 아직 없습니다.")

    st.markdown("#### Annotation")
    _annotation_form("capture", detail["capture_id"], detail)


def _compare_page() -> None:
    runs = registry.fetch_runs(PROJECT_ROOT)
    if len(runs) < 2:
        st.info("비교할 run이 2개 이상 필요합니다.")
        return
    st.subheader("Compare Runs")
    st.caption(
        "한 raw capture를 baseline으로 고정하고 replay 결과를 최대 3개까지 나란히 올려서 "
        "알고리즘 수정 전후의 KPI와 최종 tracking 궤적 변화를 봅니다."
    )
    runs_by_id = {row["session_id"]: row for row in runs}
    capture_counts: dict[str, int] = {}
    for row in runs:
        capture_id = row.get("capture_id")
        if capture_id:
            capture_counts[capture_id] = capture_counts.get(capture_id, 0) + 1

    baseline_options = [row["session_id"] for row in runs]
    baseline_default = next(
        (
            index
            for index, row in enumerate(runs)
            if (row.get("annotation_label") or "").lower() == "baseline"
            or (row.get("variant") or "").lower() == "baseline"
        ),
        0,
    )
    baseline_id = st.selectbox(
        "Baseline",
        baseline_options,
        index=baseline_default,
        format_func=lambda session_id: _run_select_label(session_id, runs_by_id),
        help="기준으로 둘 baseline run입니다. 보통 같은 raw capture를 replay한 튜닝 결과들과 비교합니다.",
    )
    baseline_seed = runs_by_id.get(baseline_id, {})
    baseline_capture = baseline_seed.get("capture_id")
    same_capture_default = bool(baseline_capture and capture_counts.get(baseline_capture, 0) >= 2)
    same_capture_only = st.checkbox(
        "Compare only runs from the baseline raw capture",
        value=same_capture_default,
        help="같은 raw capture replay끼리 비교하면 튜닝 변경 효과를 더 공정하게 볼 수 있습니다.",
    )
    replay_candidates = (
        [
            row
            for row in runs
            if row["session_id"] != baseline_id
            and baseline_capture
            and row.get("capture_id") == baseline_capture
        ]
        if same_capture_only
        else [row for row in runs if row["session_id"] != baseline_id]
    )
    replay_options = [""] + [row["session_id"] for row in replay_candidates]
    if len(replay_options) == 1:
        st.warning("선택한 baseline과 비교할 replay run이 없습니다. 필터를 끄거나 다른 baseline을 골라 주세요.")
        return

    slot_columns = st.columns(3)
    replay_ids: list[str] = []
    for slot_index, column in enumerate(slot_columns, start=1):
        default_index = slot_index if slot_index < len(replay_options) else 0
        selected = column.selectbox(
            f"Replay {slot_index}",
            replay_options,
            index=default_index,
            key=f"compare-replay-slot-{slot_index}",
            format_func=lambda session_id: _run_select_label(session_id, runs_by_id),
        )
        if selected and selected not in replay_ids:
            replay_ids.append(selected)

    if not replay_ids:
        st.info("비교할 replay run을 하나 이상 선택해 주세요.")
        return

    baseline = registry.fetch_run_detail(PROJECT_ROOT, baseline_id)
    replay_details = [
        detail
        for detail in (registry.fetch_run_detail(PROJECT_ROOT, session_id) for session_id in replay_ids)
        if detail is not None
    ]
    if baseline is None or not replay_details:
        return
    run_roles = [("baseline", baseline)] + [
        (f"replay{index}", detail)
        for index, detail in enumerate(replay_details, start=1)
    ]

    st.markdown("### Run Context")
    _render_table(
        [_run_context_row(role, detail) for role, detail in run_roles],
        key="compare-run-context",
    )

    metrics = [
        ("Operational Score", "operational_score", "higher"),
        ("Performance Score", "performance_score", "higher"),
        ("Render P95 ms", "render_latency_p95_ms", "lower"),
        ("Compute Utilization P95", "compute_utilization_p95", "lower"),
        ("Candidate/Confirmed", "candidate_to_confirmed_ratio", "lower"),
        ("Lead Switch Rate", "lead_confirmed_switch_rate", "lower"),
        ("Path Cleanliness", "path_cleanliness_score_10", "higher"),
        ("Path Max Gap", "path_max_gap_frames", "lower"),
        ("Path Residual RMS", "path_local_residual_rms_m", "lower"),
        ("Path Jump Ratio", "path_jump_ratio", "lower"),
    ]

    st.markdown("### KPI vs Baseline")
    _render_table(_compare_metric_rows(run_roles, metrics), key="compare-metrics")

    st.markdown("### Final Tracking Trajectory")
    _render_compare_final_trajectory_grid(run_roles)

    st.markdown("### Parameter Diff")
    show_unchanged_params = st.checkbox("Show unchanged parameters", value=False, key="compare-show-unchanged-params")
    role_parameters = {
        role: registry.fetch_run_parameters(PROJECT_ROOT, detail["session_id"])
        for role, detail in run_roles
    }
    parameter_diff = _parameter_diff_rows(role_parameters, show_unchanged=show_unchanged_params)
    if not any(role_parameters.values()):
        st.info("아직 run_parameters 인덱스가 비어 있습니다. 왼쪽 Refresh Registry를 눌러 다시 인덱싱해 주세요.")
    else:
        _render_table(parameter_diff, key="compare-parameter-diff", height=360)

    st.markdown("### Local Reports")
    report_columns = st.columns(min(len(run_roles), 4))
    for index, (role, detail) in enumerate(run_roles):
        with report_columns[index % len(report_columns)]:
            st.markdown(f"#### {role} `{detail['session_id']}`")
            _render_file_links(
                {
                    "performance report": Path(detail["session_dir"]) / "performance_report.html",
                    "trajectory replay": Path(detail["session_dir"]) / "trajectory_replay.html",
                }
            )

    selected_capture_ids = {detail.get("capture_id") for _, detail in run_roles if detail.get("capture_id")}
    if len(selected_capture_ids) == 1:
        st.success(f"모든 선택 run이 같은 raw capture `{next(iter(selected_capture_ids))}` 기준입니다.")
    else:
        st.warning("선택한 run들의 raw capture가 다를 수 있습니다. 공정 비교인지 다시 확인하는 편이 좋습니다.")


def _analytics_page() -> None:
    all_runs = registry.fetch_runs(PROJECT_ROOT)
    transport = st.sidebar.selectbox(
        "Analytics Transport Filter",
        ["all", "clean", "noisy", "unusable", "insufficient"],
        key="analytics-transport",
    )
    input_mode = st.sidebar.selectbox(
        "Analytics Input Mode",
        ["all", "live", "replay"],
        key="analytics-input-mode",
    )
    label = st.sidebar.selectbox(
        "Analytics Label Filter",
        ["all", *LABEL_OPTIONS[1:]],
        key="analytics-label",
    )
    board = st.sidebar.selectbox(
        "Analytics Board Filter",
        ["all", *BOARD_OPTIONS[1:]],
        key="analytics-board",
    )
    motion = st.sidebar.selectbox(
        "Analytics Motion Filter",
        ["all", *MOTION_OPTIONS[1:]],
        key="analytics-motion",
    )
    benchmark_only = st.sidebar.checkbox(
        "Benchmark-tagged analytics only",
        value=False,
        key="analytics-benchmark",
    )
    runs = _run_filters(
        all_runs,
        transport=transport,
        input_mode=input_mode,
        label=label,
        board=board,
        motion=motion,
        benchmark_only=benchmark_only,
    )
    diagnosed = analytics.build_diagnosed_run_rows(runs)

    st.subheader("Analytics / Bottleneck Triage")
    st.write(
        "여러 세션을 한 번에 모아서 PMF/ECDF 스타일의 분포와 rule-based 병목 진단을 보는 화면입니다. "
        "지금 단계의 목표는 AI 모델을 바로 학습시키는 것이 아니라, AI가 먹을 수 있는 feature와 label 후보를 "
        "먼저 안정적으로 만드는 것입니다."
    )
    if not diagnosed:
        st.info("현재 필터에 맞는 run이 없습니다.")
        return

    clean_count = sum(1 for row in diagnosed if row.get("transport_category") == "clean")
    severe_count = sum(1 for row in diagnosed if float(row.get("severity_score_10") or 0.0) >= 7.0)
    baseline_count = sum(1 for row in diagnosed if row.get("primary_bottleneck") == "baseline_candidate")
    path_values = [
        float(row["path_cleanliness_score_10"])
        for row in diagnosed
        if row.get("path_cleanliness_score_10") is not None
    ]
    median_path = float(np.percentile(np.asarray(path_values, dtype=float), 50)) if path_values else None
    top_counts = analytics.bottleneck_counts(diagnosed)
    top_label = top_counts[0]["primary_bottleneck"] if top_counts else "n/a"

    a, b, c, d, e = st.columns(5)
    a.metric("Runs", len(diagnosed))
    b.metric("Clean Transport", clean_count)
    c.metric("Severe Bottleneck", severe_count)
    d.metric("Baseline Candidates", baseline_count)
    e.metric("Median Path Clean", _format_float(median_path, 2))
    st.caption(f"most common bottleneck: `{top_label}`")

    export_col, explain_col = st.columns([1, 3])
    with export_col:
        if st.button("Export Analytics Snapshot", width="stretch"):
            output_path = analytics.write_snapshot(PROJECT_ROOT)
            st.success(f"written: `{output_path}`")
    with explain_col:
        st.caption(
            "export 결과는 `lab_data/analytics/run_triage_snapshot.json`에 저장됩니다. "
            "이 파일은 나중에 AI 분류 모델이나 논문용 통계 테이블의 입력으로 사용할 수 있습니다."
        )

    st.markdown("### 1. Bottleneck PMF")
    st.caption("PMF처럼 각 병목 label이 전체 세션 중 몇 %를 차지하는지 봅니다.")
    _render_table(top_counts, key="analytics-bottleneck-counts")

    st.markdown("### 2. KPI Target ECDF")
    st.caption(
        "ECDF 관점으로 `target 이하/이상인 세션 비율`을 봅니다. "
        "예를 들어 pass rate 80%는 전체 세션 중 80%가 그 기준을 만족한다는 뜻입니다."
    )
    _render_table(analytics.ecdf_target_rows(diagnosed), key="analytics-ecdf-targets")

    st.markdown("### 3. Metric Summary")
    st.caption("평균보다 p90/p95를 더 중요하게 봅니다. p95가 나쁘면 간헐적 실패가 숨어 있을 가능성이 큽니다.")
    _render_table(analytics.metric_summary(diagnosed), key="analytics-metric-summary")

    st.markdown("### 4. Discrete / Binned PMF")
    st.caption("candidate 비율, path cleanliness, lead switch rate를 구간화해서 분포를 봅니다.")
    _render_table(analytics.pmf_rows(diagnosed), key="analytics-pmf")

    st.markdown("### 5. Parameter Impact")
    st.caption("현재 필터에 남은 run들을 `run_parameters` 값별로 묶어 KPI 평균과 주요 병목 label을 같이 봅니다.")
    impact_metric = st.selectbox(
        "Impact Metric",
        list(analytics.METRIC_DEFINITIONS.keys()),
        format_func=lambda key: analytics.METRIC_DEFINITIONS.get(key, {}).get("label", key),
        index=0,
        key="analytics-impact-metric",
    )
    show_constant_params = st.checkbox(
        "Include parameters with a single value",
        value=False,
        key="analytics-impact-constant",
    )
    impact_rows = analytics.parameter_impact_rows(
        PROJECT_ROOT,
        diagnosed,
        metric=impact_metric,
        varying_only=not show_constant_params,
    )
    if not impact_rows:
        st.info("표시할 파라미터 impact가 없습니다. registry refresh 후 다시 보거나, single-value 파라미터 포함 옵션을 켜 보세요.")
    else:
        _render_table(impact_rows[:120], key="analytics-parameter-impact", height=360)

    st.markdown("### 6. Triage Board")
    st.caption("severity가 높은 세션부터 보면서 `raw 문제인지, detection 문제인지, tracking 문제인지`를 빠르게 분리합니다.")
    _render_table(
        [
            {
                "session_id": row.get("session_id"),
                "input": row.get("input_mode"),
                "transport": row.get("transport_category"),
                "bottleneck": row.get("primary_bottleneck"),
                "severity_10": row.get("severity_score_10"),
                "evidence": row.get("primary_evidence"),
                "next_action": row.get("recommended_action"),
                "perf": _format_float(row.get("performance_score"), 1),
                "path": _format_float(row.get("path_cleanliness_score_10"), 2),
                "residual_m": _format_float(row.get("path_local_residual_rms_m"), 3),
                "jump": _format_percent(row.get("path_jump_ratio")),
                "lead_switch": _format_percent(row.get("lead_confirmed_switch_rate")),
                "cand/conf": _format_float(row.get("candidate_to_confirmed_ratio"), 2),
                "capture": row.get("capture_id") or "",
            }
            for row in diagnosed
        ],
        key="analytics-triage-board",
    )

    selected_session = st.selectbox(
        "Diagnosis Detail",
        [row["session_id"] for row in diagnosed if row.get("session_id")],
        index=0,
    )
    selected = next((row for row in diagnosed if row.get("session_id") == selected_session), None)
    if not selected:
        return

    st.markdown(f"### Detail `{selected_session}`")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Bottleneck", selected.get("primary_bottleneck") or "n/a")
    d2.metric("Severity", _format_float(selected.get("severity_score_10"), 2))
    d3.metric("Transport", selected.get("transport_category") or "n/a")
    d4.metric("Capture", selected.get("capture_id") or "none")

    st.markdown("#### Recommended Tuning Parameters")
    _render_table(
        analytics.recommended_parameters_for_bottleneck(selected.get("primary_bottleneck")),
        key=f"analytics-recommended-params-{selected_session}",
    )

    st.markdown("#### Issues")
    _render_table(
        selected.get("issues") or [],
        key=f"analytics-issues-{selected_session}",
    )

    session_dir = Path(selected["session_dir"])
    _render_file_links(
        {
            "performance report": session_dir / "performance_report.html",
            "trajectory replay": session_dir / "trajectory_replay.html",
            "summary.json": session_dir / "summary.json",
        }
    )


def _tuning_config_options() -> list[str]:
    config_dir = PROJECT_ROOT / "config"
    paths = sorted(config_dir.glob("live_motion_tuning*.json"))
    baseline_dir = config_dir / "baselines"
    if baseline_dir.exists():
        paths.extend(sorted(baseline_dir.glob("*.json")))
    options = []
    for path in paths:
        try:
            options.append(path.relative_to(PROJECT_ROOT).as_posix())
        except ValueError:
            options.append(str(path))
    return options


def _runtime_config_options() -> list[str]:
    config_dir = PROJECT_ROOT / "config"
    paths = sorted(config_dir.glob("live_motion_runtime*.json"))
    options = []
    for path in paths:
        try:
            options.append(path.relative_to(PROJECT_ROOT).as_posix())
        except ValueError:
            options.append(str(path))
    return options


def _tuning_result_files() -> list[Path]:
    if not TUNING_RUNS_DIR.exists():
        return []
    return sorted(
        TUNING_RUNS_DIR.glob("*/result.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _extract_tuning_result_path(stdout_text: str) -> Path | None:
    for line in reversed((stdout_text or "").splitlines()):
        text = line.strip()
        if text.lower().startswith("tuning loop result:"):
            value = text.split(":", 1)[1].strip()
            path = Path(value)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            if path.exists():
                return path
    return None


def _run_tuning_loop_command(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )


def _format_tuning_change(changes: list[dict]) -> str:
    if not changes:
        return "seed"
    parts = []
    for change in changes[:3]:
        parts.append(f"{change.get('path')}: {change.get('before')} -> {change.get('after')}")
    if len(changes) > 3:
        parts.append(f"+{len(changes) - 3} more")
    return "; ".join(parts)


def _format_tuning_reasons(reasons: list[str] | None) -> str:
    if not reasons:
        return ""
    return ", ".join(str(reason) for reason in reasons)


def _tuning_kpi_rows(result: dict) -> list[dict]:
    baseline = ((result.get("baseline") or {}).get("kpis") or {})
    best = ((result.get("best") or {}).get("kpis") or {})
    metrics = [
        ("score", "Loop Score", "", "higher"),
        ("policy_overall_pass", "Policy Overall Pass", "", "must pass"),
        ("policy_preserves_tracking_shape", "Preserves Tracking Shape", "", "must pass"),
        ("policy_smooths_jumpy_raw", "Smooths Jumpy Raw", "", "must pass"),
        ("path_cleanliness_score_10", "Path Cleanliness", "/10", "higher"),
        ("output_x_span_m", "Output X Span", "m", "scenario"),
        ("output_y_span_m", "Output Y Span", "m", "scenario"),
        ("output_vs_tracking_x_span_ratio", "Output/Tracking X Ratio", "", "higher"),
        ("output_width_ratio", "Output Width Ratio", "", "scenario"),
        ("output_step_p95_m", "Output Step P95", "m", "lower"),
        ("output_max_step_m", "Output Max Step", "m", "lower"),
        ("trajectory_distance_p95_m", "Trajectory Distance P95", "m", "lower"),
        ("candidate_to_confirmed_ratio", "Candidate/Confirmed", "", "lower"),
        ("lead_switch_count", "Lead Switch Count", "", "lower"),
    ]
    rows = []
    for key, label, unit, direction in metrics:
        before = (result.get("baseline") or {}).get("score") if key == "score" else baseline.get(key)
        after = (result.get("best") or {}).get("score") if key == "score" else best.get(key)
        delta = None
        try:
            if before is not None and after is not None:
                delta = float(after) - float(before)
        except (TypeError, ValueError):
            delta = None
        rows.append(
            {
                "metric": label,
                "baseline": _safe_cell(before),
                "best": _safe_cell(after),
                "delta": _safe_cell(delta),
                "unit": unit,
                "direction": direction,
            }
        )
    return rows


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _lead_track_point(record: dict) -> dict | None:
    for key in ("display_tracks", "tentative_display_tracks", "confirmed_tracks", "tentative_tracks"):
        items = record.get(key) or []
        if not isinstance(items, list) or not items:
            continue
        candidates = []
        for item in items:
            if not isinstance(item, dict):
                continue
            x_m = _as_float(item.get("x_m"))
            y_m = _as_float(item.get("y_m"))
            if x_m is None or y_m is None:
                continue
            candidates.append(
                {
                    "x_m": x_m,
                    "y_m": y_m,
                    "score": _as_float(item.get("score"), 0.0) or 0.0,
                    "confidence": _as_float(item.get("confidence"), 0.0) or 0.0,
                    "is_primary": bool(item.get("is_primary")),
                }
            )
        if candidates:
            return max(
                candidates,
                key=lambda item: (
                    1 if item.get("is_primary") else 0,
                    item.get("confidence") or 0.0,
                    item.get("score") or 0.0,
                ),
            )
    return None


def _session_output_trajectory(session_dir: Path) -> list[dict]:
    rows = _load_jsonl(session_dir / "render_frames.jsonl")
    if not rows:
        rows = _load_jsonl(session_dir / "processed_frames.jsonl")
    trajectory = []
    for index, row in enumerate(rows):
        point = _lead_track_point(row)
        if point is None:
            continue
        trajectory.append(
            {
                "index": int(row.get("frame_index", row.get("frame_id", index)) or index),
                "x_m": float(point["x_m"]),
                "y_m": float(point["y_m"]),
            }
        )
    return trajectory


def _render_tuning_trajectory_compare(result: dict) -> None:
    baseline = result.get("baseline") or {}
    best = result.get("best") or {}
    items = [
        ("baseline", baseline.get("session_id"), baseline.get("session_dir"), "#172232"),
        ("best", best.get("session_id"), best.get("session_dir"), "#1b7a4c"),
    ]
    trajectories = []
    for role, session_id, session_dir, color in items:
        if not session_dir:
            continue
        trajectory = _session_output_trajectory(Path(session_dir))
        if trajectory:
            trajectories.append((role, session_id, color, trajectory))
    if not trajectories:
        st.info("비교할 output trajectory가 없습니다. replay 결과의 render_frames/processed_frames를 확인해 주세요.")
        return

    all_points = [point for _, _, _, trajectory in trajectories for point in trajectory]
    xs = np.asarray([point["x_m"] for point in all_points], dtype=float)
    ys = np.asarray([point["y_m"] for point in all_points], dtype=float)
    x_abs = max(float(np.max(np.abs(xs))) + 0.2, 0.8)
    y_max = max(float(np.max(ys)) + 0.25, 3.0)
    y_min = min(float(np.min(ys)) - 0.2, 0.0)

    fig, plt = _make_figure(width=9.5, height=5.0)
    ax = fig.add_subplot(111)
    for role, session_id, color, trajectory in trajectories:
        tx = [point["x_m"] for point in trajectory]
        ty = [point["y_m"] for point in trajectory]
        ax.plot(tx, ty, color=color, linewidth=1.9, alpha=0.9, label=f"{role} {session_id}")
        ax.scatter([tx[0]], [ty[0]], s=46, color=color, marker="o", zorder=5)
        ax.scatter([tx[-1]], [ty[-1]], s=64, facecolors="white", edgecolors=color, linewidths=2, zorder=6)
    ax.scatter([0], [0], s=54, marker="s", color="#263848", label="radar")
    ax.set_xlim(-x_abs, x_abs)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Before/After Output Trajectory", loc="left", fontsize=12, fontweight="bold", color="#163044")
    ax.grid(True, color="#e5edf2", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="best", frameon=False)
    ax.set_facecolor("#fbfdfe")
    _render_matplotlib_figure(fig, caption="작은 원은 시작점, 테두리 원은 마지막점입니다.")


def _tuning_page() -> None:
    st.subheader("Tuning Loop")
    st.caption(
        "ISK 4개 raw 중 하나를 고정 입력으로 선택하고, baseline tuning과 candidate tuning을 같은 raw로 replay해 "
        "config 파라미터 후보를 자동 비교합니다. 선택한 candidate tuning 원본은 덮어쓰지 않고 run별 tuning 복사본을 만듭니다."
    )

    scenario_keys = list(ISK_SCENARIOS.keys())
    scenario = st.selectbox(
        "Baseline Scenario / Raw Capture",
        scenario_keys,
        index=scenario_keys.index("right-diagonal") if "right-diagonal" in scenario_keys else 0,
        format_func=lambda key: f"{key} - {ISK_SCENARIOS[key]['capture']}",
    )
    capture = st.text_input("Raw Capture", value=ISK_SCENARIOS[scenario]["capture"])

    config_options = _tuning_config_options()
    if not config_options:
        st.error("config/live_motion_tuning*.json 파일을 찾지 못했습니다.")
        return
    default_isk = "config/live_motion_tuning_isk.json"
    baseline_tuning = st.selectbox(
        "Baseline Tuning",
        config_options,
        index=_option_index(config_options, default_isk),
        help="비교 기준으로만 사용하는 tuning입니다.",
    )
    candidate_tuning = st.selectbox(
        "Candidate Tuning Seed",
        config_options,
        index=_option_index(config_options, default_isk),
        help="후보 tuning의 시작점입니다. 원본은 덮어쓰지 않고 복사본으로 trial을 만듭니다.",
    )
    runtime_options = _runtime_config_options()
    default_runtime = "config/live_motion_runtime_isk.json"
    runtime_settings = st.selectbox(
        "Runtime Settings",
        runtime_options or [default_runtime],
        index=_option_index(runtime_options or [default_runtime], default_runtime),
        help="replay 실행 시 RADAR_RUNTIME_SETTINGS_PATH로 고정됩니다. ISK 측정 비교는 ISK runtime을 선택하세요.",
    )
    if baseline_tuning == candidate_tuning:
        st.warning(
            "baseline과 candidate seed가 같은 파일입니다. 첫 seed 결과는 baseline과 비슷할 수 있지만, "
            "trial tuning 복사본은 선택 파라미터를 바꿔 비교합니다."
        )

    param_labels = {spec["key"]: f"{spec['label']} ({spec['key']})" for spec in PARAMETER_SPECS}
    selected_params = st.multiselect(
        "Tunable Parameters",
        [spec["key"] for spec in PARAMETER_SPECS],
        default=[spec["key"] for spec in PARAMETER_SPECS],
        format_func=lambda key: param_labels.get(key, key),
        help="이번 1차 버전은 config-only 파라미터 탐색입니다. 코드 파일은 자동 수정하지 않습니다.",
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        max_trials = st.number_input("Max Candidate Trials", min_value=1, max_value=40, value=6, step=1)
    with col2:
        speed = st.number_input("Replay Speed", min_value=0.5, max_value=20.0, value=5.0, step=0.5)
    with col3:
        target_score = st.number_input("Target Score", min_value=1.0, max_value=100.0, value=80.0, step=1.0)
    timeout_s = st.number_input(
        "Per-command timeout seconds",
        min_value=0,
        max_value=3600,
        value=0,
        step=30,
        help="0이면 timeout을 걸지 않습니다.",
    )

    command = [
        sys.executable,
        "-B",
        "-m",
        "tools.tuning_loop.run_loop",
        "--scenario",
        scenario,
        "--capture",
        capture,
        "--baseline-tuning",
        baseline_tuning,
        "--candidate-tuning",
        candidate_tuning,
        "--runtime-settings",
        runtime_settings,
        "--max-trials",
        str(int(max_trials)),
        "--speed",
        str(float(speed)),
        "--target-score",
        str(float(target_score)),
    ]
    if int(timeout_s) > 0:
        command.extend(["--timeout-s", str(int(timeout_s))])
    if selected_params:
        command.append("--params")
        command.extend(selected_params)

    with st.expander("Command Preview"):
        st.code(" ".join(command), language="powershell")

    if st.button("Run Tuning Loop", type="primary", width="stretch", disabled=not selected_params):
        with st.spinner("Replay tuning loop 실행 중입니다. 선택한 trial 수만큼 시간이 걸릴 수 있습니다."):
            completed = _run_tuning_loop_command(command)
        st.code(completed.stdout or "", language="text")
        if completed.stderr:
            st.code(completed.stderr, language="text")
        if completed.returncode != 0:
            st.error(f"Tuning loop 실패: exit={completed.returncode}")
        else:
            result_path = _extract_tuning_result_path(completed.stdout or "")
            if result_path:
                st.success(f"결과 저장: {result_path}")
            else:
                st.success("Tuning loop가 끝났습니다. 아래 Recent Results에서 최신 결과를 확인하세요.")
            registry.refresh_registry(PROJECT_ROOT)

    result_files = _tuning_result_files()
    st.markdown("### Recent Results")
    if not result_files:
        st.info("아직 tuning loop 결과가 없습니다.")
        return

    selected_result_path = st.selectbox(
        "Result",
        result_files,
        index=0,
        format_func=lambda path: str(path.parent.name),
    )
    result = _load_json_file(Path(selected_result_path))
    status = str(result.get("status") or "unknown").upper()
    best = result.get("best") or {}
    baseline = result.get("baseline") or {}

    a, b, c, d = st.columns(4)
    a.metric("Status", status)
    b.metric("Best Score", _format_float(best.get("score"), 2))
    c.metric("Baseline Session", baseline.get("session_id") or "n/a")
    d.metric("Best Session", best.get("session_id") or "n/a")
    if best.get("reject_reasons"):
        st.warning("Best 후보 제외 사유: " + _format_tuning_reasons(best.get("reject_reasons") or []))
    elif best.get("accepted"):
        st.success("Best 후보가 baseline safety 정책을 통과했습니다.")

    _render_file_links(
        {
            "result.json": selected_result_path,
            "best tuning": best.get("best_tuning_path"),
            "runtime settings": result.get("runtime_settings"),
            "baseline summary": baseline.get("summary_path"),
            "best summary": best.get("summary_path"),
        }
    )

    st.markdown("#### KPI Before / After")
    _render_table(_tuning_kpi_rows(result), key=f"tuning-kpis-{Path(selected_result_path).parent.name}", height=330)

    st.markdown("#### Trajectory Before / After")
    _render_tuning_trajectory_compare(result)

    st.markdown("#### Trial History")
    trial_rows = [
        {
            "trial": trial.get("label"),
            "session": trial.get("session_id"),
            "score": trial.get("score"),
            "accepted": trial.get("accepted"),
            "target_pass": trial.get("target_pass"),
            "reject_reasons": _format_tuning_reasons(trial.get("reject_reasons") or []),
            "changes": _format_tuning_change(trial.get("changes") or []),
            "tuning": trial.get("tuning"),
        }
        for trial in (result.get("trials") or [])
    ]
    _render_table(trial_rows, key=f"tuning-trials-{Path(selected_result_path).parent.name}", height=360)


def _eval_page() -> None:
    include_templates = st.sidebar.checkbox("Show template tasks", value=False, key="eval-show-templates")
    status_filter = st.sidebar.selectbox(
        "Eval Status Filter",
        ["all", "pass", "fail", "dry_run"],
        key="eval-status-filter",
    )

    tasks = _eval_task_files(include_templates=include_templates)
    all_outcomes = _load_eval_outcomes()
    outcomes = all_outcomes
    if status_filter != "all":
        outcomes = [outcome for outcome in outcomes if _eval_status(outcome) == status_filter]

    st.subheader("Eval Harness")
    st.caption(
        "같은 raw capture를 replay하고, task JSON의 acceptance criteria로 candidate run을 PASS/FAIL 판정합니다. "
        "새 replay를 실행하면 일반 run처럼 `logs/live_motion_viewer/<session_id>` 아래에 세션이 생깁니다."
    )

    status_counts = {"pass": 0, "fail": 0, "dry_run": 0, "unknown": 0}
    for outcome in all_outcomes:
        status = _eval_status(outcome)
        status_counts[status] = status_counts.get(status, 0) + 1
    a, b, c, d = st.columns(4)
    a.metric("Eval Runs", len(all_outcomes))
    b.metric("PASS", status_counts.get("pass", 0))
    c.metric("FAIL", status_counts.get("fail", 0))
    d.metric("DRY", status_counts.get("dry_run", 0))

    st.markdown("### Run Eval")
    if not tasks:
        st.info("`docs/evals/tasks` 아래에 실행할 task JSON이 없습니다.")
    else:
        task_by_name = {path.name: path for path in tasks}
        task_names = list(task_by_name.keys())
        default_name = (
            "diagonal_positive_x_trajectory_fidelity.json"
            if "diagonal_positive_x_trajectory_fidelity.json" in task_by_name
            else task_names[0]
        )
        selected_task_name = st.selectbox(
            "Task",
            task_names,
            index=task_names.index(default_name),
            key="eval-task",
        )
        task_path = task_by_name[selected_task_name]
        task = _load_json_file(task_path)
        task_name = str(task.get("name") or task_path.stem)

        latest_for_task = next(
            (
                outcome
                for outcome in all_outcomes
                if _nested_get(outcome, "task", "name", default="") == task_name
            ),
            None,
        )
        latest_baseline = _nested_get(latest_for_task, "baseline", "session_id", default="") if latest_for_task else ""

        runs = registry.fetch_runs(PROJECT_ROOT)
        runs_by_id = {row["session_id"]: row for row in runs}
        run_options = [""] + [row["session_id"] for row in runs]
        baseline_default = str(_nested_get(task, "baseline", "session", default="") or latest_baseline or "")
        if not baseline_default:
            baseline_default = next(
                (
                    row["session_id"]
                    for row in runs
                    if (row.get("annotation_label") or "").lower() == "baseline"
                ),
                "",
            )
        baseline_index = run_options.index(baseline_default) if baseline_default in run_options else 0

        top_left, top_right = st.columns([1.2, 1])
        with top_left:
            st.markdown("#### Task Summary")
            _render_table(
                [
                    {
                        "name": task_name,
                        "capture": task.get("capture") or "",
                        "speed": task.get("speed") or "",
                        "baseline_tuning": _nested_get(task, "baseline", "tuning", default=""),
                        "candidate_tuning": _nested_get(task, "candidate", "tuning", default=""),
                        "criteria": len(task.get("acceptance") or []),
                    }
                ],
                key=f"eval-task-summary-{task_name}",
            )
            if task.get("description"):
                st.caption(str(task.get("description")))
        with top_right:
            st.markdown("#### Local Files")
            _render_file_links(
                {
                    "task json": task_path,
                    "eval runs": EVAL_RUNS_DIR,
                    "eval README": PROJECT_ROOT / "docs" / "evals" / "README.md",
                }
            )

        mode = st.radio(
            "Execution Mode",
            ["새 candidate replay 실행", "기존 candidate session 채점"],
            horizontal=True,
            key="eval-execution-mode",
        )
        c1, c2 = st.columns(2)
        with c1:
            baseline_session = st.selectbox(
                "Baseline Session",
                run_options,
                index=baseline_index,
                format_func=lambda session_id: _run_select_label(session_id, runs_by_id),
                help="delta 기준이 되는 baseline입니다. 비워두면 task 설정대로 baseline replay를 실행할 수 있습니다.",
                key="eval-baseline-session",
            )
        with c2:
            candidate_session = ""
            if mode == "기존 candidate session 채점":
                candidate_session = st.selectbox(
                    "Candidate Session",
                    run_options,
                    index=0,
                    format_func=lambda session_id: _run_select_label(session_id, runs_by_id),
                    help="이미 생성된 replay/live run을 task criteria로 채점합니다.",
                    key="eval-candidate-session",
                )
            else:
                st.info("실행하면 새 replay session이 생성되고, 그 session이 candidate가 됩니다.")

        option_cols = st.columns(4)
        with option_cols[0]:
            build_stage_cache = st.checkbox("Fail 시 Stage Cache", value=True, key="eval-stage-cache")
        with option_cols[1]:
            force_baseline = st.checkbox("Force Baseline Replay", value=False, key="eval-force-baseline")
        with option_cols[2]:
            skip_baseline = st.checkbox("Skip Baseline", value=False, key="eval-skip-baseline")
        with option_cols[3]:
            dry_run = st.checkbox("Dry Run", value=False, key="eval-dry-run")
        timeout_s = st.number_input(
            "Subprocess Timeout Seconds (0 = no timeout)",
            min_value=0,
            max_value=3600,
            value=0,
            step=30,
            key="eval-timeout",
        )

        command = _build_eval_command(
            task_path,
            mode=mode,
            baseline_session=baseline_session,
            candidate_session=candidate_session,
            no_stage_cache=not build_stage_cache,
            force_baseline=force_baseline,
            skip_baseline=skip_baseline,
            dry_run=dry_run,
            timeout_s=float(timeout_s) if timeout_s else None,
        )
        with st.expander("Command Preview", expanded=False):
            st.code(subprocess.list2cmdline(command), language="powershell")

        disabled = mode == "기존 candidate session 채점" and not candidate_session
        if st.button("Run Eval Task", width="stretch", disabled=disabled):
            before_dirs = {path.parent.resolve() for path in _eval_outcome_files()}
            with st.spinner("eval task를 실행하는 중입니다. replay 모드면 새 candidate session이 생성됩니다..."):
                completed = _run_eval_command(command)
            st.code(completed.stdout[-6000:] if completed.stdout else "(no stdout)", language="text")
            if completed.stderr:
                with st.expander("stderr", expanded=False):
                    st.code(completed.stderr[-6000:], language="text")

            outcome_path = _extract_outcome_path(completed.stdout or "")
            after_dirs = {path.parent.resolve() for path in _eval_outcome_files()} - before_dirs
            if outcome_path is None and after_dirs:
                outcome_path = _latest_outcome_for_task(task_name, after_dirs=after_dirs)
            outcome = _load_json_file(outcome_path) if outcome_path else {}
            if outcome:
                status = _eval_status(outcome)
                if status == "pass":
                    st.success(f"PASS: `{task_name}`")
                elif status == "fail":
                    st.error(f"FAIL: `{task_name}`")
                else:
                    st.warning(f"{_eval_status_mark(status)}: `{task_name}`")
                candidate = outcome.get("candidate") or {}
                st.caption(
                    f"candidate=`{candidate.get('session_id') or 'n/a'}` | "
                    f"outcome=`{_relative_to_project(outcome_path)}`"
                )
                if completed.returncode == 0 or status in {"pass", "fail", "dry_run"}:
                    registry.refresh_registry(PROJECT_ROOT)
                    all_outcomes = _load_eval_outcomes()
                    outcomes = all_outcomes if status_filter == "all" else [
                        item for item in all_outcomes if _eval_status(item) == status_filter
                    ]
            elif completed.returncode != 0:
                st.error(f"eval command failed before writing outcome.json (exit {completed.returncode}).")
            else:
                st.warning("eval command는 끝났지만 outcome.json을 찾지 못했습니다.")

    st.markdown("### Recent Eval Outcomes")
    if outcomes:
        _render_table(_eval_outcome_rows(outcomes[:40]), key="eval-outcomes", height=360)

        selected_outcome_path = st.selectbox(
            "Outcome Detail",
            [outcome["_outcome_path"] for outcome in outcomes],
            format_func=lambda value: _relative_to_project(value),
            key="eval-outcome-detail",
        )
        selected = next((outcome for outcome in outcomes if outcome["_outcome_path"] == selected_outcome_path), None)
        if selected:
            status = _eval_status(selected)
            cards = st.columns(4)
            candidate = selected.get("candidate") or {}
            baseline = selected.get("baseline") or {}
            cards[0].markdown(
                _stage_card("status", _eval_status_mark(status), "task verdict", _eval_status_tone(status)),
                unsafe_allow_html=True,
            )
            cards[1].markdown(
                _stage_card("criteria", _criterion_pass_text(selected.get("criteria") or []), "passed / total"),
                unsafe_allow_html=True,
            )
            cards[2].markdown(
                _stage_card("candidate", candidate.get("session_id") or "n/a", "graded run"),
                unsafe_allow_html=True,
            )
            cards[3].markdown(
                _stage_card("baseline", baseline.get("session_id") or "n/a", "comparison run"),
                unsafe_allow_html=True,
            )

            st.markdown("#### Criteria")
            _render_table(
                [
                    {
                        "result": "PASS" if item.get("passed") else "FAIL",
                        "name": item.get("name") or "",
                        "metric": item.get("metric") or "",
                        "mode": item.get("mode") or "",
                        "actual": item.get("actual"),
                        "op": item.get("op") or "",
                        "expected": item.get("expected"),
                    }
                    for item in selected.get("criteria") or []
                ],
                key="eval-outcome-criteria",
            )

            candidate_dir = candidate.get("session_dir")
            run_dir = selected.get("_run_dir")
            st.markdown("#### Local Reports / Files")
            _render_file_links(
                {
                    "outcome.json": selected_outcome_path,
                    "eval run folder": run_dir,
                    "candidate trajectory": Path(candidate_dir) / "trajectory_replay.html" if candidate_dir else None,
                    "candidate summary": Path(candidate_dir) / "summary.json" if candidate_dir else None,
                }
            )
    else:
        st.info("아직 저장된 eval outcome이 없습니다. 위에서 task를 실행하면 PASS/FAIL 기록이 여기에 쌓입니다.")


def _stage_timeline_page() -> None:
    runs = registry.fetch_runs(PROJECT_ROOT)
    raw_linked_runs = [row for row in runs if row.get("capture_id")]

    st.subheader("Stage Timeline")
    st.write(
        "raw capture를 같은 처리 루프로 다시 태운 뒤, frame별 feature와 bottleneck label을 시간축으로 보는 화면입니다. "
        "이 화면의 목적은 `결과가 별로다`에서 멈추지 않고 `몇 프레임부터 어느 단계가 깨졌는지`를 찾는 것입니다."
    )
    if not raw_linked_runs:
        st.warning("raw capture가 연결된 run이 아직 없습니다.")
        return

    selected_session = st.selectbox(
        "Run for Stage Timeline",
        [row["session_id"] for row in raw_linked_runs],
        index=0,
    )
    detail = registry.fetch_run_detail(PROJECT_ROOT, selected_session)
    if detail is None:
        return

    manifest = stage_cache.load_stage_cache_manifest(PROJECT_ROOT, selected_session)
    feature_summary = stage_cache.load_stage_feature_summary(PROJECT_ROOT, selected_session)
    features = stage_cache.load_stage_features(PROJECT_ROOT, selected_session)
    cache_paths = stage_cache.stage_cache_paths(PROJECT_ROOT, selected_session)

    a, b, c, d = st.columns(4)
    a.metric("Session", selected_session)
    b.metric("Capture", detail.get("capture_id") or "n/a")
    c.metric("Transport", detail.get("transport_category") or "n/a")
    d.metric("Cached Frames", (manifest or {}).get("frame_count", 0))

    build_col, limit_col, force_col = st.columns([2, 1, 1])
    with limit_col:
        frame_limit = st.number_input(
            "Frame Limit",
            min_value=0,
            max_value=5000,
            value=int((manifest or {}).get("frame_count") or 120),
            step=20,
            help="0이면 전체 frame을 처리합니다. 처음에는 120 정도로 빠르게 확인하는 것을 권장합니다.",
        )
    with force_col:
        force_rebuild = st.checkbox("Force Rebuild", value=False, key="timeline-force")
    with build_col:
        if st.button("Build / Refresh Feature Timeline", width="stretch"):
            try:
                with st.spinner("raw capture를 다시 처리해 frame feature timeline을 생성하는 중입니다..."):
                    manifest = stage_cache.build_stage_cache(
                        PROJECT_ROOT,
                        selected_session,
                        frame_limit=(int(frame_limit) or None),
                        force=bool(force_rebuild),
                    )
                st.success(f"feature timeline ready: frames={manifest.get('frame_count', 0)}")
                _rerun()
            except Exception as error:
                st.error(f"feature timeline 생성 중 오류가 발생했습니다: {error}")
                return

    if not features:
        st.info("아직 frame feature cache가 없습니다. 위 버튼으로 timeline을 먼저 생성해 주세요.")
        return

    if not feature_summary:
        feature_summary = {"frame_count": len(features)}

    st.markdown("### Summary")
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Feature Frames", feature_summary.get("frame_count", len(features)))
    s2.metric("OK Frame Rate", _format_percent(feature_summary.get("ok_frame_rate")))
    s3.metric("Invalid Frames", feature_summary.get("invalid_frame_count", 0))
    s4.metric("Lead Switches", feature_summary.get("lead_switch_count", 0))
    s5.metric("Top Bottleneck", feature_summary.get("top_frame_bottleneck") or "n/a")
    st.caption(f"feature file: `{cache_paths['features_path']}`")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("### Frame Bottleneck PMF")
        _render_table(
            feature_summary.get("frame_bottleneck_counts") or [],
            key=f"timeline-bottleneck-counts-{selected_session}",
        )
    with right:
        st.markdown("### Slowest Stage PMF")
        _render_table(
            feature_summary.get("slowest_stage_counts") or [],
            key=f"timeline-slowest-stage-counts-{selected_session}",
        )

    st.markdown("### Timeline View")
    metric_options = {
        "Compute Total ms": ("compute_total_ms", 100.0, True),
        "Detection Count": ("detection_count", None, False),
        "Confirmed Track Count": ("confirmed_track_count", None, False),
        "Lead Step m": ("lead_step_m", 0.45, True),
        "Lead Residual m": ("lead_measurement_residual_m", 0.18, True),
        "RAI Peak / Median": ("rai_peak_to_median", 2.0, False),
        "Detect Stage ms": ("detect_ms", None, True),
        "Track Stage ms": ("track_ms", None, True),
    }
    selected_metric_label = st.selectbox("Timeline Metric", list(metric_options.keys()), index=0)
    metric, threshold, lower_is_better = metric_options[selected_metric_label]
    _render_metric_timeline(
        selected_metric_label,
        features,
        metric,
        threshold=threshold,
        lower_is_better=lower_is_better,
    )

    st.markdown("### Path Preview")
    _render_path_preview(features)

    st.markdown("### High Severity Frames")
    high_severity = sorted(
        [
            row
            for row in features
            if float(row.get("frame_severity_10") or 0.0) >= 6.0
        ],
        key=lambda row: float(row.get("frame_severity_10") or 0.0),
        reverse=True,
    )
    _render_table(
        [
            {
                "ordinal": row.get("ordinal"),
                "frame_id": row.get("frame_id"),
                "bottleneck": row.get("frame_bottleneck"),
                "severity": row.get("frame_severity_10"),
                "evidence": row.get("frame_evidence"),
                "invalid": row.get("invalid"),
                "compute_ms": row.get("compute_total_ms"),
                "detections": row.get("detection_count"),
                "confirmed": row.get("confirmed_track_count"),
                "lead_id": row.get("lead_track_id"),
                "lead_step_m": row.get("lead_step_m"),
                "lead_residual_m": row.get("lead_measurement_residual_m"),
                "slowest_stage": row.get("slowest_stage_name"),
            }
            for row in high_severity[:80]
        ],
        key=f"timeline-high-severity-{selected_session}",
    )

    st.markdown("### Metric Summary")
    metric_summary = feature_summary.get("metrics") or {}
    _render_table(
        [
            {"metric": name, **values}
            for name, values in metric_summary.items()
        ],
        key=f"timeline-metric-summary-{selected_session}",
    )

    _render_file_links(
        {
            "frame features": cache_paths["features_path"],
            "feature summary": cache_paths["feature_summary_path"],
            "stage cache frames": cache_paths["frames_path"],
            "stage cache manifest": cache_paths["manifest_path"],
        }
    )


def _stage_page() -> None:
    runs = registry.fetch_runs(PROJECT_ROOT)
    st.subheader("Stage Debug")
    st.write(
        "같은 raw를 기준으로 `cube / RDI / RAI / detections / tracker state`를 프레임 단위로 다시 확인하는 "
        "디버그 화면입니다. 1차 버전은 stage cache를 생성한 뒤 heatmap과 serialized 후보/track을 열어보는 방식입니다."
    )
    if not runs:
        return
    raw_linked_runs = [row for row in runs if row.get("capture_id")]
    show_runs_without_raw = st.checkbox("Show runs without raw capture", value=False)
    selectable_runs = runs if show_runs_without_raw else raw_linked_runs
    if not selectable_runs:
        st.warning("raw capture가 연결된 run이 아직 없습니다. live 측정 시 raw capture를 먼저 남겨 주세요.")
        return

    selected_session = st.selectbox(
        "Run for Stage Debug",
        [row["session_id"] for row in selectable_runs],
        index=0,
    )
    detail = registry.fetch_run_detail(PROJECT_ROOT, selected_session)
    if detail is None:
        return

    session_dir = Path(detail["session_dir"])
    capture_id = detail.get("capture_id") or "none"
    st.markdown("### Session Context")
    a, b, c, d = st.columns(4)
    a.metric("Session", selected_session)
    b.metric("Capture Link", capture_id)
    c.metric("Transport", detail.get("transport_category") or "n/a")
    d.metric("Input Mode", detail.get("input_mode") or "n/a")

    with st.expander("Available Artifact Links", expanded=False):
        _render_file_links(
            {
                "processed report": session_dir / "processed_report.html",
                "render report": session_dir / "render_report.html",
                "trajectory replay": session_dir / "trajectory_replay.html",
                "summary.json": session_dir / "summary.json",
            }
        )

    manifest = stage_cache.load_stage_cache_manifest(PROJECT_ROOT, selected_session)
    cache_paths = stage_cache.stage_cache_paths(PROJECT_ROOT, selected_session)
    st.markdown("### Stage Cache")
    stage_cache_enabled = bool(detail.get("capture_id"))
    if not stage_cache_enabled:
        st.info(
            "이 세션은 raw capture가 연결되어 있지 않아 stage cache를 만들 수 없습니다. "
            "Stage Cache는 `Capture Link`가 있는 replay/live 세션에서만 생성할 수 있습니다."
        )
        if raw_linked_runs:
            recommended = ", ".join(row["session_id"] for row in raw_linked_runs[:5])
            st.caption(f"예시 raw-linked run: {recommended}")

    build_col, force_col = st.columns([3, 1])
    with build_col:
        frame_limit = st.number_input(
            "Frame Limit (0 = all frames)",
            min_value=0,
            max_value=5000,
            value=0 if not manifest else int((manifest or {}).get("frame_limit_requested") or 0),
            step=10,
            help="Compare/논문용 분석은 0으로 전체 프레임을 생성하는 편이 안전합니다. 작은 값은 빠른 샘플 디버깅용입니다.",
        )
    with force_col:
        force_rebuild = st.checkbox("Force Rebuild", value=False)

    if st.button(
        "Generate / Refresh Stage Cache",
        width="stretch",
        disabled=not stage_cache_enabled,
    ):
        try:
            with st.spinner("raw capture를 다시 처리해서 stage cache를 생성하는 중입니다..."):
                manifest = stage_cache.build_stage_cache(
                    PROJECT_ROOT,
                    selected_session,
                    frame_limit=(int(frame_limit) or None),
                    force=bool(force_rebuild),
                )
            st.success(
                f"stage cache ready: frames={manifest.get('frame_count', 0)} | "
                f"capture={manifest.get('capture_id') or detail.get('capture_id') or 'n/a'}"
            )
            _rerun()
        except Exception as error:
            st.error(f"stage cache 생성 중 오류가 발생했습니다: {error}")

    manifest = stage_cache.load_stage_cache_manifest(PROJECT_ROOT, selected_session)
    if manifest:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Cached Frames", manifest.get("frame_count", 0))
        m2.metric("Capture", manifest.get("capture_id") or detail.get("capture_id") or "n/a")
        m3.metric("Range FFT", (manifest.get("runtime") or {}).get("range_fft_size") or "n/a")
        m4.metric("Angle FFT", (manifest.get("runtime") or {}).get("angle_fft_size") or "n/a")
        st.caption(
            f"generated_at={manifest.get('generated_at') or 'n/a'} | "
            f"cache_dir=`{cache_paths['cache_dir']}`"
        )
        with st.expander("Stage Cache Files", expanded=False):
            _render_file_links(
                {
                    "stage cache manifest": cache_paths["manifest_path"],
                "stage cache frames": cache_paths["frames_path"],
                "frame features": cache_paths["features_path"],
                "feature summary": cache_paths["feature_summary_path"],
                "frame trace": cache_paths["trace_path"],
                "trace summary": cache_paths["trace_summary_path"],
            }
        )
    else:
        st.info("아직 stage cache가 없습니다. 위 버튼으로 raw replay cache를 생성해 주세요.")

    stage_data = (detail.get("summary") or {}).get("diagnostics", {}).get("preferred_stage_timings_ms", {})
    timings = stage_data.get("timings", {})
    if timings:
        with st.expander("Session Stage Timing Summary", expanded=False):
            _render_table(
                [
                    {
                        "stage": stage_name,
                        "count": values.get("count"),
                        "mean_ms": values.get("mean"),
                        "p50_ms": values.get("p50"),
                        "p95_ms": values.get("p95"),
                        "max_ms": values.get("max"),
                    }
                    for stage_name, values in sorted(timings.items())
                ],
                key="stage-debug-timings",
            )
            slowest = stage_data.get("slowest_stage", {})
            if slowest:
                st.info(
                    f"slowest stage: `{slowest.get('name')}` | "
                    f"p95={_format_float(slowest.get('p95_ms'), 3, ' ms')} | "
                    f"mean={_format_float(slowest.get('mean_ms'), 3, ' ms')}"
                )
    else:
        st.caption("이 세션에는 stage timing summary가 없습니다.")

    if not manifest:
        return

    frames = stage_cache.load_stage_cache_frames(PROJECT_ROOT, selected_session)
    if not frames:
        st.warning("stage cache manifest는 있지만 frame record가 비어 있습니다. 다시 생성해 보는 편이 좋습니다.")
        return

    trace_rows = stage_cache.load_stage_traces(PROJECT_ROOT, selected_session)
    st.markdown("### Whole Sequence Stage View")
    st.caption(
        "raw를 다시 처리한 뒤 stage별 출력 궤적을 전체 이동 기준으로 비교합니다. "
        "먼저 전체 궤적 정확도와 끊김을 보고, 급락한 구간만 아래 slider로 frame drill-down합니다."
    )
    _render_stage_sequence_overview(trace_rows)

    selected_ordinal = st.slider("Cached Frame Index", 0, len(frames) - 1, 0)
    frame_record, arrays = stage_cache.load_stage_cache_frame(PROJECT_ROOT, selected_session, selected_ordinal)
    feature_rows = stage_cache.load_stage_features(PROJECT_ROOT, selected_session)
    feature_record = next(
        (row for row in feature_rows if int(row.get("ordinal", -1)) == int(selected_ordinal)),
        {},
    )
    trace_record = next(
        (row for row in trace_rows if int(row.get("frame_id", -1)) == int(frame_record.get("frame_id", -2))),
        {},
    )

    st.markdown("### Selected Frame Summary")
    f1, f2, f3, f4, f5, f6, f7 = st.columns(7)
    f1.metric("Frame ID", frame_record.get("frame_id"))
    f2.metric("Invalid", "yes" if frame_record.get("invalid") else "no")
    f3.metric("Tracker Policy", frame_record.get("tracker_policy") or "n/a")
    f4.metric("Detections", len(frame_record.get("detections") or []))
    f5.metric("Confirmed", len(frame_record.get("confirmed_tracks") or []))
    f6.metric("Tentative", len(frame_record.get("tentative_tracks") or []))
    f7.metric("Bottleneck", feature_record.get("frame_bottleneck") or "n/a")
    st.caption(
        f"udp_gap={frame_record.get('udp_gap_count', 0)} | "
        f"out_of_sequence={frame_record.get('out_of_sequence_count', 0)} | "
        f"byte_mismatch={frame_record.get('byte_mismatch_count', 0)} | "
        f"evidence={feature_record.get('frame_evidence') or 'n/a'}"
    )

    st.markdown("### Compact Stage Images")
    heatmap_cols = st.columns(3)
    with heatmap_cols[0]:
        _render_heatmap(
            "Cube Preview",
            arrays.get("cube_preview"),
            caption=f"shape={tuple(np.asarray(arrays.get('cube_preview')).shape)}",
            height_px=190,
        )
    with heatmap_cols[1]:
        _render_heatmap(
            "RDI",
            arrays.get("rdi"),
            caption=f"shape={tuple(np.asarray(arrays.get('rdi')).shape)}",
            height_px=190,
        )
    with heatmap_cols[2]:
        _render_heatmap(
            "RAI",
            arrays.get("rai"),
            caption=f"shape={tuple(np.asarray(arrays.get('rai')).shape)}",
            height_px=190,
        )

    st.markdown("### Detailed Stage Trace")
    st.caption(
        "이 영역은 실시간 로그가 아니라 저장된 raw capture를 replay하면서 만든 `frame_trace.jsonl`입니다. "
        "따라서 실시간 측정 성능에는 영향을 주지 않습니다."
    )
    _render_trace_flow(trace_record)

    st.markdown("### Processing Loop Outputs")
    frame_timings = frame_record.get("stage_timings_ms") or {}
    output_cols = st.columns(4)
    if frame_timings:
        with output_cols[0]:
            st.markdown("#### 1. Stage Timings")
            _render_table(
                [{"stage": key, "ms": value} for key, value in frame_timings.items()],
                key=f"stage-cache-timings-{selected_session}-{selected_ordinal}",
                height=210,
            )
    else:
        with output_cols[0]:
            st.markdown("#### 1. Stage Timings")
            st.caption("stage timing payload가 없습니다.")

    with output_cols[1]:
        st.markdown("#### 2. Detection Output")
        _render_table(
            frame_record.get("detections") or [],
            key=f"stage-cache-detections-{selected_session}-{selected_ordinal}",
            height=210,
        )
    with output_cols[2]:
        st.markdown("#### 3. Tracker Input")
        _render_table(
            frame_record.get("tracker_input_detections") or [],
            key=f"stage-cache-tracker-input-{selected_session}-{selected_ordinal}",
            height=210,
        )
    with output_cols[3]:
        st.markdown("#### 4. Tracker Output")
        track_rows = [
            {"state": "confirmed", **track}
            for track in (frame_record.get("confirmed_tracks") or [])
        ] + [
            {"state": "tentative", **track}
            for track in (frame_record.get("tentative_tracks") or [])
        ]
        _render_table(
            track_rows,
            key=f"stage-cache-tracks-{selected_session}-{selected_ordinal}",
            height=210,
        )


def main() -> None:
    st.set_page_config(page_title="Radar Lab", layout="wide")
    st.title("Radar Lab")
    st.caption(
        "로컬 전용 세션 관리 앱입니다. raw capture 원본은 그대로 두고, SQLite 인덱스로 "
        "세션 분류, 태깅, 비교, stage debug를 돕습니다."
    )

    overview = registry.get_registry_overview(PROJECT_ROOT)
    if overview["run_total"] == 0 and overview["capture_total"] == 0:
        registry.refresh_registry(PROJECT_ROOT)
        overview = registry.get_registry_overview(PROJECT_ROOT)

    with st.sidebar:
        st.markdown("### Registry")
        if st.button("Refresh Registry", width="stretch"):
            stats = registry.refresh_registry(PROJECT_ROOT)
            st.success(f"refreshed: captures={stats['captures_indexed']}, runs={stats['runs_indexed']}")
            _rerun()
        st.caption(f"DB: `{overview['db_path']}`")
        st.caption(f"Last refresh: `{overview['last_refresh_at'] or 'n/a'}`")
        _render_registry_share_tools()
        page = st.radio(
            "Page",
            [
                "Dashboard",
                "Runs",
                "Captures",
                "Compare",
                "Tuning Loop",
                "Eval Harness",
                "Analytics/Triage",
                "Stage Timeline",
                "Stage Debug",
            ],
        )

    if page == "Dashboard":
        _overview_page()
    elif page == "Runs":
        _runs_page()
    elif page == "Captures":
        _captures_page()
    elif page == "Compare":
        _compare_page()
    elif page == "Tuning Loop":
        _tuning_page()
    elif page == "Eval Harness":
        _eval_page()
    elif page == "Analytics/Triage":
        _analytics_page()
    elif page == "Stage Timeline":
        _stage_timeline_page()
    else:
        _stage_page()


if __name__ == "__main__":
    main()
