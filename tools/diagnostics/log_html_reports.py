from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from .session_compare import METRICS, build_comparison
from .session_report import build_summary


COMMON_STYLE = """
  :root {
    --bg: #f4f7fb;
    --panel: #ffffff;
    --panel-soft: #fbfdff;
    --text: #142033;
    --muted: #5f6f86;
    --line: #d7deea;
    --brand: #0f6cbd;
    --brand-soft: #eaf3ff;
    --good: #027a48;
    --good-soft: #ecfdf3;
    --warn: #b54708;
    --warn-soft: #fff4e5;
    --danger: #b42318;
    --danger-soft: #fff1f1;
    --shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    color: var(--text);
    background: linear-gradient(180deg, #eef4ff 0%, var(--bg) 220px);
    font-family: "Segoe UI", "Noto Sans KR", sans-serif;
    line-height: 1.65;
  }
  a { color: var(--brand); text-decoration: none; }
  .wrap { max-width: 1180px; margin: 0 auto; padding: 36px 24px 80px; }
  .hero {
    background: linear-gradient(135deg, #0f6cbd 0%, #0c4f8a 100%);
    color: #fff;
    border-radius: 24px;
    padding: 30px 34px;
    box-shadow: var(--shadow);
  }
  .hero h1 { margin: 0 0 8px; font-size: 2rem; line-height: 1.2; }
  .hero p { margin: 0; color: rgba(255,255,255,0.9); max-width: 880px; }
  .nav { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 18px; }
  .nav a {
    color: #fff;
    background: rgba(255,255,255,0.14);
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 999px;
    padding: 10px 14px;
  }
  section { margin-top: 24px; }
  .card {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 22px;
    padding: 24px;
    box-shadow: var(--shadow);
  }
  .card h2 { margin: 0 0 10px; font-size: 1.4rem; }
  .card h3 { margin: 0 0 8px; font-size: 1.05rem; }
  .subtle { color: var(--muted); }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; }
  .metric {
    background: var(--panel-soft);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 16px;
  }
  .metric .label { color: var(--muted); font-size: 0.92rem; margin-bottom: 6px; }
  .metric .value { font-size: 1.7rem; font-weight: 700; line-height: 1.1; }
  .metric .hint { color: var(--muted); font-size: 0.9rem; margin-top: 6px; }
  .pill {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 700;
  }
  .pill.brand { background: var(--brand-soft); color: var(--brand); }
  .pill.good { background: var(--good-soft); color: var(--good); }
  .pill.warn { background: var(--warn-soft); color: var(--warn); }
  .pill.danger { background: var(--danger-soft); color: var(--danger); }
  .chart-card {
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 18px;
    background: var(--panel-soft);
  }
  .chart { min-height: 280px; }
  .chart svg { width: 100%; height: auto; display: block; }
  .empty {
    display: grid;
    place-items: center;
    min-height: 180px;
    color: var(--muted);
    border: 1px dashed var(--line);
    border-radius: 16px;
    background: #fff;
  }
  table { width: 100%; border-collapse: collapse; }
  th, td { text-align: left; vertical-align: top; padding: 12px 10px; border-bottom: 1px solid var(--line); }
  th { color: var(--muted); font-size: 0.92rem; font-weight: 700; }
  code, pre { font-family: "Consolas", "Courier New", monospace; }
  .note {
    border-left: 4px solid var(--brand);
    background: #f8fbff;
    padding: 14px 16px;
    border-radius: 12px;
    color: var(--muted);
  }
  .timeline { position: relative; padding-left: 22px; }
  .timeline::before {
    content: "";
    position: absolute;
    left: 6px;
    top: 0;
    bottom: 0;
    width: 2px;
    background: var(--line);
  }
  .timeline-item {
    position: relative;
    margin-bottom: 16px;
    padding: 14px 16px;
    border-radius: 16px;
    border: 1px solid var(--line);
    background: var(--panel-soft);
  }
  .timeline-item::before {
    content: "";
    position: absolute;
    left: -21px;
    top: 18px;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: var(--brand);
    border: 2px solid #fff;
    box-shadow: 0 0 0 2px var(--brand-soft);
  }
  .legend { display: flex; flex-wrap: wrap; gap: 14px; margin-top: 12px; }
  .legend span { display: inline-flex; align-items: center; gap: 8px; color: var(--muted); font-size: 0.9rem; }
  .legend i { width: 12px; height: 12px; border-radius: 999px; display: inline-block; }
  .compare-list { display: grid; gap: 12px; }
  .compare-row {
    display: grid;
    grid-template-columns: minmax(200px, 260px) minmax(0, 1fr);
    gap: 14px;
    align-items: center;
    padding: 12px 0;
    border-bottom: 1px solid var(--line);
  }
  .compare-bars { display: grid; gap: 10px; }
  .compare-track {
    background: #edf2f9;
    border-radius: 999px;
    height: 12px;
    position: relative;
    overflow: hidden;
  }
  .compare-fill { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 999px; }
  .compare-fill.before { background: #5b8def; }
  .compare-fill.after { background: #f97316; }
  .compare-meta { display: flex; gap: 10px; flex-wrap: wrap; color: var(--muted); font-size: 0.9rem; }
  .controls { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; }
  .control { display: grid; gap: 8px; }
  .control label { font-weight: 700; }
  .control select {
    width: 100%;
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 12px;
    font: inherit;
    background: #fff;
  }
"""


COMMON_SCRIPT = """
function fmt(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
  return Number(value).toFixed(digits).replace(/\\.0+$/, "").replace(/(\\.\\d*[1-9])0+$/, "$1");
}

function esc(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderLineChart(targetId, seriesList, options = {}) {
  const target = document.getElementById(targetId);
  if (!target) return;
  const width = options.width || 960;
  const height = options.height || 280;
  const padding = { top: 18, right: 18, bottom: 28, left: 46 };
  const values = [];
  for (const series of seriesList) {
    for (const value of series.values) {
      if (Number.isFinite(value)) values.push(Number(value));
    }
  }
  if (!values.length) {
    target.innerHTML = '<div class="empty">표시할 데이터가 없습니다.</div>';
    return;
  }
  let minValue = options.min !== undefined ? options.min : (options.zeroFloor ? 0 : Math.min(...values));
  let maxValue = options.max !== undefined ? options.max : Math.max(...values);
  if (minValue === maxValue) {
    if (minValue === 0) maxValue = 1;
    else {
      minValue = Math.min(0, minValue);
      maxValue = maxValue * 1.1;
    }
  }
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const xFor = (index, length) => length <= 1
    ? padding.left + plotWidth / 2
    : padding.left + (plotWidth * index) / (length - 1);
  const yFor = (value) => {
    const ratio = (value - minValue) / (maxValue - minValue);
    return padding.top + (1 - ratio) * plotHeight;
  };
  let svg = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(options.title || "chart")}">`;
  svg += `<rect x="0" y="0" width="${width}" height="${height}" rx="18" fill="#ffffff"></rect>`;
  for (let step = 0; step <= 4; step += 1) {
    const value = minValue + ((maxValue - minValue) * step) / 4;
    const y = padding.top + plotHeight - (plotHeight * step) / 4;
    svg += `<line x1="${padding.left}" y1="${y}" x2="${width - padding.right}" y2="${y}" stroke="#d7deea" stroke-width="1"></line>`;
    svg += `<text x="${padding.left - 8}" y="${y + 4}" text-anchor="end" fill="#5f6f86" font-size="12">${esc(fmt(value, options.digits || 1))}</text>`;
  }
  svg += `<line x1="${padding.left}" y1="${height - padding.bottom}" x2="${width - padding.right}" y2="${height - padding.bottom}" stroke="#8ea0bb" stroke-width="1.3"></line>`;
  for (const series of seriesList) {
    const segments = [];
    let active = [];
    series.values.forEach((rawValue, index) => {
      if (!Number.isFinite(rawValue)) {
        if (active.length) { segments.push(active); active = []; }
        return;
      }
      active.push(`${xFor(index, series.values.length)},${yFor(Number(rawValue))}`);
    });
    if (active.length) segments.push(active);
    for (const points of segments) {
      svg += `<polyline fill="none" stroke="${series.color}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round" points="${points.join(" ")}"></polyline>`;
    }
  }
  svg += `<text x="${padding.left}" y="${height - 6}" fill="#5f6f86" font-size="12">${esc(options.startLabel || "start")}</text>`;
  svg += `<text x="${width - padding.right}" y="${height - 6}" text-anchor="end" fill="#5f6f86" font-size="12">${esc(options.endLabel || "end")}</text>`;
  svg += `</svg>`;
  const legend = seriesList.map((series) => `<span><i style="background:${series.color}"></i>${esc(series.label)}</span>`).join("");
  target.innerHTML = svg + `<div class="legend">${legend}</div>`;
}

function nestedGet(data, dottedKey) {
  let current = data;
  for (const part of dottedKey.split(".")) {
    if (!current || typeof current !== "object" || !(part in current)) return null;
    current = current[part];
  }
  return current;
}

function judgementPillClass(value) {
  if (value === "improved") return "good";
  if (value === "regressed") return "danger";
  if (value === "same") return "brand";
  return "warn";
}
"""


def _load_json(path: Path, default: Any):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _write_json(path: Path, data: Any):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_text(path: Path, text: str):
    path.write_text(text, encoding="utf-8")


def _round_or_none(value, digits=3):
    if value is None:
        return None
    return round(float(value), digits)


def _fmt(value, digits=3, suffix=""):
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}".rstrip("0").rstrip(".") + suffix


def _fmt_pct(value):
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def _pill(label: str, tone: str):
    return f'<span class="pill {tone}">{label}</span>'


def _judgement_tone(label: str):
    if label == "improved":
        return "good"
    if label == "regressed":
        return "danger"
    if label == "same":
        return "brand"
    return "warn"


def _resolve_summary_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_dir():
        return path / "summary.json"
    return path


def _session_health(summary: dict):
    render_invalid = summary.get("render", {}).get("invalid_rate")
    display_mean = summary.get("render", {}).get("display_track_count", {}).get("mean")
    render_latency = summary.get("render", {}).get("capture_to_render_ms", {}).get("p95")
    if render_invalid is None:
        return {"label": "데이터 부족", "tone": "warn", "detail": "summary는 있지만 render 로그가 충분하지 않습니다."}
    if render_invalid >= 0.8 or (display_mean is not None and display_mean < 0.2):
        return {"label": "위험", "tone": "danger", "detail": "통신 무결성 또는 최종 표시 품질이 매우 나쁩니다."}
    if render_invalid >= 0.4 or (render_latency is not None and render_latency > 220):
        return {"label": "주의", "tone": "warn", "detail": "동작은 하지만 현업 기준으로 안정성이 부족합니다."}
    return {"label": "양호", "tone": "good", "detail": "현 시점 로그 기준으로는 비교적 안정적인 편입니다."}


def _event_summary(events: list[dict]):
    if not events:
        return {
            "event_count": 0,
            "dca_config_complete": False,
            "radar_open_complete": False,
            "first_rendered_frame": False,
            "first_render_elapsed_s": None,
            "session_duration_s": None,
            "opengl_unavailable": False,
            "session_error": False,
            "session_error_repr": None,
        }

    def first(event_type: str):
        for event in events:
            if event.get("event_type") == event_type:
                return event
        return None

    def parse_time(event: dict | None):
        if not event:
            return None
        try:
            return datetime.fromisoformat(event["wall_time"])
        except (KeyError, ValueError):
            return None

    radar_open_start = first("radar_open_start")
    shutdown_start = first("shutdown_start")
    first_render = first("first_rendered_frame")
    session_error = first("session_error")

    duration = None
    open_time = parse_time(radar_open_start)
    shutdown_time = parse_time(shutdown_start)
    if open_time and shutdown_time:
        duration = max((shutdown_time - open_time).total_seconds(), 0.0)

    return {
        "event_count": len(events),
        "dca_config_complete": first("dca_config_complete") is not None,
        "radar_open_complete": first("radar_open_complete") is not None,
        "first_rendered_frame": first_render is not None,
        "first_render_elapsed_s": _round_or_none(first_render.get("elapsed_since_stream_start_s") if first_render else None, digits=4),
        "session_duration_s": _round_or_none(duration, digits=3),
        "opengl_unavailable": first("opengl_unavailable") is not None,
        "session_error": session_error is not None,
        "session_error_repr": None if session_error is None else session_error.get("error"),
    }


def _simplify_processed_records(records: list[dict]):
    rows = []
    for record in records:
        detections = record.get("detections") or []
        lead = max(detections, key=lambda item: item.get("score", 0.0)) if detections else None
        rows.append(
            {
                "frame_id": int(record.get("frame_id", 0)),
                "capture_to_process_ms": record.get("capture_to_process_ms"),
                "udp_gap_count": int(record.get("udp_gap_count", 0)),
                "byte_mismatch_count": int(record.get("byte_mismatch_count", 0)),
                "out_of_sequence_count": int(record.get("out_of_sequence_count", 0)),
                "invalid": bool(record.get("invalid")),
                "candidate_count": int(record.get("candidate_count", 0)),
                "tracker_input_count": int(record.get("tracker_input_count", 0)),
                "confirmed_track_count": int(record.get("confirmed_track_count", 0)),
                "tentative_track_count": int(record.get("tentative_track_count", 0)),
                "lead_angle_deg": None if lead is None else lead.get("angle_deg"),
            }
        )
    return rows


def _simplify_render_records(records: list[dict]):
    rows = []
    for record in records:
        detections = record.get("detections") or []
        lead = max(detections, key=lambda item: item.get("score", 0.0)) if detections else None
        rows.append(
            {
                "frame_id": int(record.get("frame_id", 0)),
                "capture_to_render_ms": record.get("capture_to_render_ms"),
                "process_to_render_ms": record.get("process_to_render_ms"),
                "invalid": bool(record.get("invalid")),
                "candidate_count": int(record.get("candidate_count", 0)),
                "display_track_count": int(record.get("display_track_count", 0)),
                "tentative_display_track_count": int(record.get("tentative_display_track_count", 0)),
                "skipped_render_frames": int(record.get("skipped_render_frames", 0)),
                "status_text": record.get("status_text", ""),
                "lead_angle_deg": None if lead is None else lead.get("angle_deg"),
            }
        )
    return rows


def _overview_cards(summary: dict):
    health = _session_health(summary)
    render = summary.get("render", {})
    processed = summary.get("processed", {})
    return [
        ("세션 상태", health["label"], health["detail"], health["tone"]),
        ("Processed Invalid", _fmt_pct(processed.get("invalid_rate")), "처리 입력 무결성 관점", "danger" if (processed.get("invalid_rate") or 0) >= 0.5 else "brand"),
        ("Render Invalid", _fmt_pct(render.get("invalid_rate")), "사용자 체감 품질 관점", "danger" if (render.get("invalid_rate") or 0) >= 0.5 else "brand"),
        ("Display Track Mean", _fmt(render.get("display_track_count", {}).get("mean")), "화면에 실제로 보인 트랙 수", "warn"),
        ("Render P95", _fmt(render.get("capture_to_render_ms", {}).get("p95"), suffix=" ms"), "수집부터 표시까지 상위 95%", "brand"),
    ]


def _build_session_index_html(session_dir: Path, summary: dict, event_summary: dict):
    cards_html = "".join(
        f"""
        <div class="metric">
          <div class="label">{label}</div>
          <div class="value">{value}</div>
          <div class="hint">{hint}</div>
        </div>
        """
        for label, value, hint, _tone in _overview_cards(summary)
    )
    health = _session_health(summary)
    session_id = summary.get("session_id", session_dir.name)
    event_note = "정상" if event_summary["first_rendered_frame"] else "first_rendered_frame 미확인"
    if event_summary["session_error"]:
        event_note = f"세션 오류: {event_summary['session_error_repr']}"

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{session_id} 로그 리포트</title>
  <style>{COMMON_STYLE}</style>
</head>
<body>
  <div class="wrap">
    <header class="hero">
      <h1>{session_id} 로그 리포트</h1>
      <p>세션 전체 개요입니다. 아래 링크로 processed, render, event 전용 리포트로 이동할 수 있습니다.</p>
      <nav class="nav">
        <a href="./processed_report.html">processed 리포트</a>
        <a href="./render_report.html">render 리포트</a>
        <a href="./event_report.html">event 리포트</a>
        <a href="../index.html">전체 비교 대시보드</a>
      </nav>
    </header>

    <section class="card">
      <h2>세션 개요</h2>
      <p class="subtle">
        Variant: <code>{summary.get("session_meta", {}).get("variant") or "n/a"}</code> |
        Input: <code>{summary.get("session_meta", {}).get("input_mode") or "n/a"}</code> |
        상태: {_pill(health["label"], health["tone"])}
      </p>
      <div class="grid">{cards_html}</div>
    </section>

    <section class="card">
      <h2>빠른 해석</h2>
      <div class="note">
        <strong>이 세션의 핵심 상태:</strong> {health["detail"]}<br />
        <strong>Event 로그 관점:</strong> {event_note}<br />
        <strong>First render 지연:</strong> {_fmt(event_summary.get("first_render_elapsed_s"), digits=3, suffix=" s")} |
        <strong>세션 길이:</strong> {_fmt(event_summary.get("session_duration_s"), digits=3, suffix=" s")}
      </div>
    </section>

    <section class="card">
      <h2>파일 바로가기</h2>
      <table>
        <thead>
          <tr>
            <th>리포트</th>
            <th>설명</th>
            <th>링크</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><code>processed_report.html</code></td>
            <td>처리 파이프라인 자체의 품질, invalid, tracker 입력/출력</td>
            <td><a href="./processed_report.html">열기</a></td>
          </tr>
          <tr>
            <td><code>render_report.html</code></td>
            <td>화면에 실제로 그려진 결과와 지연</td>
            <td><a href="./render_report.html">열기</a></td>
          </tr>
          <tr>
            <td><code>event_report.html</code></td>
            <td>세션 시작, DCA1000 설정, 첫 렌더, 종료 타임라인</td>
            <td><a href="./event_report.html">열기</a></td>
          </tr>
          <tr>
            <td><code>summary.json</code></td>
            <td>자동 요약 통계 원본</td>
            <td><a href="./summary.json">열기</a></td>
          </tr>
        </tbody>
      </table>
    </section>
  </div>
</body>
</html>"""


def _build_processed_html(session_dir: Path, summary: dict, processed_records: list[dict]):
    simplified = _simplify_processed_records(processed_records)
    sample_rows = "".join(
        f"""
        <tr>
          <td>{row['frame_id']}</td>
          <td>{'true' if row['invalid'] else 'false'}</td>
          <td>{row['udp_gap_count']}</td>
          <td>{row['byte_mismatch_count']}</td>
          <td>{row['candidate_count']}</td>
          <td>{row['tracker_input_count']}</td>
          <td>{row['confirmed_track_count']}</td>
          <td>{_fmt(row['lead_angle_deg'])}</td>
        </tr>
        """
        for row in simplified[-10:]
    ) or '<tr><td colspan="8">표시할 processed frame이 없습니다.</td></tr>'

    payload = {
        "series": {
            "frameIds": [row["frame_id"] for row in simplified],
            "captureToProcessMs": [row["capture_to_process_ms"] for row in simplified],
            "udpGapCount": [row["udp_gap_count"] for row in simplified],
            "byteMismatchCount": [row["byte_mismatch_count"] for row in simplified],
            "outOfSequenceCount": [row["out_of_sequence_count"] for row in simplified],
            "candidateCount": [row["candidate_count"] for row in simplified],
            "trackerInputCount": [row["tracker_input_count"] for row in simplified],
            "confirmedTrackCount": [row["confirmed_track_count"] for row in simplified],
            "tentativeTrackCount": [row["tentative_track_count"] for row in simplified],
            "leadAngleDeg": [row["lead_angle_deg"] for row in simplified],
        }
    }

    processed = summary.get("processed", {})
    health = _session_health(summary)
    script = f"""
    <script>
      {COMMON_SCRIPT}
      const REPORT_DATA = {json.dumps(payload, ensure_ascii=False)};
      renderLineChart('processed-latency-chart', [
        {{ label: 'capture_to_process_ms', color: '#0f6cbd', values: REPORT_DATA.series.captureToProcessMs }}
      ], {{ title: 'processed latency', digits: 1, zeroFloor: true, startLabel: 'frame 1', endLabel: 'last frame' }});

      renderLineChart('processed-integrity-chart', [
        {{ label: 'udp_gap_count', color: '#b42318', values: REPORT_DATA.series.udpGapCount }},
        {{ label: 'byte_mismatch_count', color: '#f97316', values: REPORT_DATA.series.byteMismatchCount }},
        {{ label: 'out_of_sequence_count', color: '#8b5cf6', values: REPORT_DATA.series.outOfSequenceCount }}
      ], {{ title: 'integrity', digits: 0, zeroFloor: true, startLabel: 'frame 1', endLabel: 'last frame' }});

      renderLineChart('processed-track-chart', [
        {{ label: 'candidate_count', color: '#0f6cbd', values: REPORT_DATA.series.candidateCount }},
        {{ label: 'tracker_input_count', color: '#10b981', values: REPORT_DATA.series.trackerInputCount }},
        {{ label: 'confirmed_track_count', color: '#f97316', values: REPORT_DATA.series.confirmedTrackCount }},
        {{ label: 'tentative_track_count', color: '#8b5cf6', values: REPORT_DATA.series.tentativeTrackCount }}
      ], {{ title: 'track flow', digits: 0, zeroFloor: true, startLabel: 'frame 1', endLabel: 'last frame' }});

      renderLineChart('processed-angle-chart', [
        {{ label: 'lead_angle_deg', color: '#ef4444', values: REPORT_DATA.series.leadAngleDeg }}
      ], {{ title: 'lead angle', digits: 1, startLabel: 'frame 1', endLabel: 'last frame' }});
    </script>
    """

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{session_dir.name} processed report</title>
  <style>{COMMON_STYLE}</style>
</head>
<body>
  <div class="wrap">
    <header class="hero">
      <h1>processed_frames.jsonl 리포트</h1>
      <p>처리 파이프라인 내부 상태를 보는 페이지입니다. detection 후보, tracker 입력, invalid 신호를 먼저 읽습니다.</p>
      <nav class="nav">
        <a href="./index.html">세션 개요</a>
        <a href="./render_report.html">render 리포트</a>
        <a href="./event_report.html">event 리포트</a>
        <a href="../index.html">전체 비교 대시보드</a>
      </nav>
    </header>

    <section class="card">
      <h2>핵심 지표</h2>
      <p class="subtle">세션 상태: {_pill(health["label"], health["tone"])}</p>
      <div class="grid">
        <div class="metric"><div class="label">Processed Frame Count</div><div class="value">{processed.get("frame_count", 0)}</div></div>
        <div class="metric"><div class="label">Invalid Rate</div><div class="value">{_fmt_pct(processed.get("invalid_rate"))}</div></div>
        <div class="metric"><div class="label">Birth Block Rate</div><div class="value">{_fmt_pct(processed.get("birth_block_rate"))}</div></div>
        <div class="metric"><div class="label">Latency P95</div><div class="value">{_fmt(processed.get("capture_to_process_ms", {}).get("p95"), suffix=" ms")}</div></div>
        <div class="metric"><div class="label">Confirmed Track Mean</div><div class="value">{_fmt(processed.get("confirmed_track_count", {}).get("mean"))}</div></div>
      </div>
      <p class="note" style="margin-top:16px;"><strong>읽는 법:</strong> candidate가 높아도 confirmed가 낮고 invalid가 높으면 detection보다 입력 무결성이나 tracker 정책이 더 큰 병목일 수 있습니다.</p>
    </section>

    <section class="grid">
      <div class="chart-card"><h3>처리 지연</h3><div id="processed-latency-chart" class="chart"></div></div>
      <div class="chart-card"><h3>통신 무결성 신호</h3><div id="processed-integrity-chart" class="chart"></div></div>
      <div class="chart-card"><h3>검출에서 추적으로 가는 흐름</h3><div id="processed-track-chart" class="chart"></div></div>
      <div class="chart-card"><h3>대표 detection 각도 변화</h3><div id="processed-angle-chart" class="chart"></div></div>
    </section>

    <section class="card">
      <h2>최근 프레임 샘플</h2>
      <table>
        <thead>
          <tr>
            <th>frame</th><th>invalid</th><th>gap</th><th>byte</th><th>candidate</th><th>tracker_input</th><th>confirmed</th><th>lead_angle</th>
          </tr>
        </thead>
        <tbody>{sample_rows}</tbody>
      </table>
    </section>
    {script}
  </div>
</body>
</html>"""


def _build_render_html(session_dir: Path, summary: dict, render_records: list[dict]):
    simplified = _simplify_render_records(render_records)
    sample_rows = "".join(
        f"""
        <tr>
          <td>{row['frame_id']}</td>
          <td>{'true' if row['invalid'] else 'false'}</td>
          <td>{_fmt(row['capture_to_render_ms'], suffix=' ms')}</td>
          <td>{row['candidate_count']}</td>
          <td>{row['display_track_count']}</td>
          <td>{row['tentative_display_track_count']}</td>
          <td>{row['skipped_render_frames']}</td>
          <td>{row['status_text']}</td>
        </tr>
        """
        for row in simplified[-10:]
    ) or '<tr><td colspan="8">표시할 render frame이 없습니다.</td></tr>'

    payload = {
        "series": {
            "frameIds": [row["frame_id"] for row in simplified],
            "captureToRenderMs": [row["capture_to_render_ms"] for row in simplified],
            "processToRenderMs": [row["process_to_render_ms"] for row in simplified],
            "candidateCount": [row["candidate_count"] for row in simplified],
            "displayTrackCount": [row["display_track_count"] for row in simplified],
            "tentativeDisplayTrackCount": [row["tentative_display_track_count"] for row in simplified],
            "leadAngleDeg": [row["lead_angle_deg"] for row in simplified],
        }
    }

    render = summary.get("render", {})
    health = _session_health(summary)
    script = f"""
    <script>
      {COMMON_SCRIPT}
      const REPORT_DATA = {json.dumps(payload, ensure_ascii=False)};
      renderLineChart('render-latency-chart', [
        {{ label: 'capture_to_render_ms', color: '#0f6cbd', values: REPORT_DATA.series.captureToRenderMs }},
        {{ label: 'process_to_render_ms', color: '#10b981', values: REPORT_DATA.series.processToRenderMs }}
      ], {{ title: 'render latency', digits: 1, zeroFloor: true, startLabel: 'frame 1', endLabel: 'last frame' }});

      renderLineChart('render-track-chart', [
        {{ label: 'candidate_count', color: '#0f6cbd', values: REPORT_DATA.series.candidateCount }},
        {{ label: 'display_track_count', color: '#f97316', values: REPORT_DATA.series.displayTrackCount }},
        {{ label: 'tentative_display_track_count', color: '#8b5cf6', values: REPORT_DATA.series.tentativeDisplayTrackCount }}
      ], {{ title: 'visible tracks', digits: 0, zeroFloor: true, startLabel: 'frame 1', endLabel: 'last frame' }});

      renderLineChart('render-angle-chart', [
        {{ label: 'lead_angle_deg', color: '#ef4444', values: REPORT_DATA.series.leadAngleDeg }}
      ], {{ title: 'lead angle', digits: 1, startLabel: 'frame 1', endLabel: 'last frame' }});
    </script>
    """

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{session_dir.name} render report</title>
  <style>{COMMON_STYLE}</style>
</head>
<body>
  <div class="wrap">
    <header class="hero">
      <h1>render_frames.jsonl 리포트</h1>
      <p>사용자가 실제로 본 결과 기준 리포트입니다. 화면 지연과 최종 display track을 중심으로 읽습니다.</p>
      <nav class="nav">
        <a href="./index.html">세션 개요</a>
        <a href="./processed_report.html">processed 리포트</a>
        <a href="./event_report.html">event 리포트</a>
        <a href="../index.html">전체 비교 대시보드</a>
      </nav>
    </header>

    <section class="card">
      <h2>핵심 지표</h2>
      <p class="subtle">세션 상태: {_pill(health["label"], health["tone"])}</p>
      <div class="grid">
        <div class="metric"><div class="label">Render Frame Count</div><div class="value">{render.get("frame_count", 0)}</div></div>
        <div class="metric"><div class="label">Invalid Rate</div><div class="value">{_fmt_pct(render.get("invalid_rate"))}</div></div>
        <div class="metric"><div class="label">Display Track Mean</div><div class="value">{_fmt(render.get("display_track_count", {}).get("mean"))}</div></div>
        <div class="metric"><div class="label">Render P95</div><div class="value">{_fmt(render.get("capture_to_render_ms", {}).get("p95"), suffix=" ms")}</div></div>
        <div class="metric"><div class="label">Multi Display Success</div><div class="value">{_fmt_pct(render.get("multi_display_success_rate"))}</div></div>
      </div>
      <p class="note" style="margin-top:16px;"><strong>읽는 법:</strong> candidate는 많은데 display track이 거의 0이면 내부 후보는 있지만 화면에 남는 결과는 거의 없다는 뜻입니다.</p>
    </section>

    <section class="grid">
      <div class="chart-card"><h3>화면 지연</h3><div id="render-latency-chart" class="chart"></div></div>
      <div class="chart-card"><h3>후보와 실제 표시 트랙</h3><div id="render-track-chart" class="chart"></div></div>
      <div class="chart-card"><h3>대표 detection 각도 변화</h3><div id="render-angle-chart" class="chart"></div></div>
    </section>

    <section class="card">
      <h2>최근 프레임 샘플</h2>
      <table>
        <thead>
          <tr>
            <th>frame</th><th>invalid</th><th>capture_to_render</th><th>candidate</th><th>display</th><th>tentative_display</th><th>skipped</th><th>status_text</th>
          </tr>
        </thead>
        <tbody>{sample_rows}</tbody>
      </table>
    </section>
    {script}
  </div>
</body>
</html>"""


def _build_event_html(session_dir: Path, events: list[dict], event_summary: dict):
    items = []
    for event in events:
        payload = []
        for key, value in event.items():
            if key in {"event_type", "wall_time", "session_id", "frame_index"}:
                continue
            payload.append(f"{key}={value}")
        payload_text = " | ".join(payload) if payload else "추가 payload 없음"
        items.append(
            f"""
            <div class="timeline-item">
              <strong>{event.get('wall_time')} — <code>{event.get('event_type')}</code></strong>
              <div class="subtle">frame_index={event.get('frame_index', 0)}</div>
              <div class="subtle">{payload_text}</div>
            </div>
            """
        )
    timeline_html = "".join(items) or '<div class="timeline-item"><strong>event 로그가 없습니다.</strong></div>'

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{session_dir.name} event report</title>
  <style>{COMMON_STYLE}</style>
</head>
<body>
  <div class="wrap">
    <header class="hero">
      <h1>event_log.jsonl 리포트</h1>
      <p>세션 시작, DCA1000 설정, radar open, first render, 종료 흐름을 시간순으로 보는 리포트입니다.</p>
      <nav class="nav">
        <a href="./index.html">세션 개요</a>
        <a href="./processed_report.html">processed 리포트</a>
        <a href="./render_report.html">render 리포트</a>
        <a href="../index.html">전체 비교 대시보드</a>
      </nav>
    </header>

    <section class="card">
      <h2>핵심 상태</h2>
      <div class="grid">
        <div class="metric"><div class="label">Event Count</div><div class="value">{event_summary.get('event_count', 0)}</div></div>
        <div class="metric"><div class="label">DCA Config</div><div class="value">{'OK' if event_summary.get('dca_config_complete') else 'Missing'}</div></div>
        <div class="metric"><div class="label">Radar Open</div><div class="value">{'OK' if event_summary.get('radar_open_complete') else 'Missing'}</div></div>
        <div class="metric"><div class="label">First Render</div><div class="value">{_fmt(event_summary.get('first_render_elapsed_s'), digits=3, suffix=' s')}</div></div>
        <div class="metric"><div class="label">Session Duration</div><div class="value">{_fmt(event_summary.get('session_duration_s'), digits=3, suffix=' s')}</div></div>
      </div>
      <p class="note" style="margin-top:16px;">
        <strong>OpenGL 상태:</strong> {'3D view unavailable' if event_summary.get('opengl_unavailable') else '정상'} |
        <strong>Session Error:</strong> {event_summary.get('session_error_repr') or '없음'}
      </p>
    </section>

    <section class="card">
      <h2>타임라인</h2>
      <div class="timeline">{timeline_html}</div>
    </section>
  </div>
</body>
</html>"""


def _collect_session_rows(log_root: Path):
    rows = []
    for session_dir in sorted(
        [path for path in log_root.iterdir() if path.is_dir()],
        key=lambda path: path.name,
        reverse=True,
    ):
        try:
            generate_session_artifacts(session_dir)
            summary = _load_json(session_dir / "summary.json", {})
        except Exception:
            continue

        events = _load_jsonl(session_dir / "event_log.jsonl")
        event_summary = _event_summary(events)
        health = _session_health(summary)
        rows.append(
            {
                "session_id": summary.get("session_id", session_dir.name),
                "session_dir": session_dir.name,
                "created_at": summary.get("session_meta", {}).get("created_at"),
                "variant": summary.get("session_meta", {}).get("variant"),
                "scenario_id": summary.get("session_meta", {}).get("scenario_id"),
                "input_mode": summary.get("session_meta", {}).get("input_mode"),
                "health": health,
                "summary": summary,
                "event_summary": event_summary,
                "links": {
                    "index": f"./{session_dir.name}/index.html",
                    "processed": f"./{session_dir.name}/processed_report.html",
                    "render": f"./{session_dir.name}/render_report.html",
                    "event": f"./{session_dir.name}/event_report.html",
                },
            }
        )
    return rows


def _build_root_dashboard_html(session_rows: list[dict]):
    table_rows = "".join(
        f"""
        <tr>
          <td><strong>{row['session_id']}</strong></td>
          <td>{row['created_at'] or 'n/a'}</td>
          <td>{row['variant'] or 'n/a'}</td>
          <td>{row['scenario_id'] or 'n/a'}</td>
          <td>{_pill(row['health']['label'], row['health']['tone'])}</td>
          <td>{_fmt_pct(row['summary']['render']['invalid_rate'])}</td>
          <td>{_fmt(row['summary']['render']['display_track_count']['mean'])}</td>
          <td>{_fmt(row['summary']['render']['capture_to_render_ms']['p95'], suffix=' ms')}</td>
          <td>
            <a href="{row['links']['index']}">개요</a> |
            <a href="{row['links']['processed']}">processed</a> |
            <a href="{row['links']['render']}">render</a> |
            <a href="{row['links']['event']}">event</a>
          </td>
        </tr>
        """
        for row in session_rows
    ) or '<tr><td colspan="9">표시할 세션이 없습니다.</td></tr>'

    payload = {
        "sessions": session_rows,
        "metrics": [
            {"key": key, "label": label, "direction": direction}
            for key, label, direction in METRICS
        ],
    }

    script = f"""
    <script>
      {COMMON_SCRIPT}
      const DASHBOARD = {json.dumps(payload, ensure_ascii=False)};

      function comparisonRows(beforeSession, afterSession) {{
        return DASHBOARD.metrics.map((metric) => {{
          const beforeValue = nestedGet(beforeSession.summary, metric.key);
          const afterValue = nestedGet(afterSession.summary, metric.key);
          let delta = null;
          let judgement = 'n/a';
          if (Number.isFinite(beforeValue) && Number.isFinite(afterValue)) {{
            delta = Number(afterValue) - Number(beforeValue);
            if (Math.abs(delta) < 1e-12) {{
              judgement = 'same';
            }} else if (metric.direction === 'lower') {{
              judgement = delta < 0 ? 'improved' : 'regressed';
            }} else {{
              judgement = delta > 0 ? 'improved' : 'regressed';
            }}
          }}
          return {{ ...metric, before: beforeValue, after: afterValue, delta, judgement }};
        }});
      }}

      function renderComparison() {{
        const beforeId = document.getElementById('before-session').value;
        const afterId = document.getElementById('after-session').value;
        const beforeSession = DASHBOARD.sessions.find((item) => item.session_id === beforeId);
        const afterSession = DASHBOARD.sessions.find((item) => item.session_id === afterId);
        if (!beforeSession || !afterSession) return;

        const rows = comparisonRows(beforeSession, afterSession);
        const metricsTarget = document.getElementById('comparison-metrics');
        const chartTarget = document.getElementById('comparison-chart');
        const eventTarget = document.getElementById('comparison-events');

        metricsTarget.innerHTML = rows.map((row) => `
          <div class="metric">
            <div class="label">${{esc(row.label)}}</div>
            <div class="value">${{fmt(row.after, 3)}}</div>
            <div class="hint">
              before=${{fmt(row.before, 3)}} |
              delta=${{row.delta === null ? 'n/a' : fmt(row.delta, 3)}} |
              <span class="pill ${{judgementPillClass(row.judgement)}}">${{esc(row.judgement)}}</span>
            </div>
          </div>
        `).join("");

        chartTarget.innerHTML = '<div class="compare-list">' + rows.map((row) => {{
          if (!Number.isFinite(row.before) || !Number.isFinite(row.after)) {{
            return `
              <div class="compare-row">
                <div><strong>${{esc(row.label)}}</strong><div class="compare-meta">비교 불가</div></div>
                <div class="compare-bars"><div class="subtle">숫자 데이터가 부족합니다.</div></div>
              </div>`;
          }}
          const maxValue = Math.max(Math.abs(row.before), Math.abs(row.after), 1e-9);
          const beforeWidth = Math.max(6, (Math.abs(row.before) / maxValue) * 100);
          const afterWidth = Math.max(6, (Math.abs(row.after) / maxValue) * 100);
          return `
            <div class="compare-row">
              <div>
                <strong>${{esc(row.label)}}</strong>
                <div class="compare-meta">
                  <span>before=${{fmt(row.before, 3)}}</span>
                  <span>after=${{fmt(row.after, 3)}}</span>
                  <span class="pill ${{judgementPillClass(row.judgement)}}">${{esc(row.judgement)}}</span>
                </div>
              </div>
              <div class="compare-bars">
                <div>
                  <div class="subtle">before</div>
                  <div class="compare-track"><div class="compare-fill before" style="width:${{beforeWidth}}%"></div></div>
                </div>
                <div>
                  <div class="subtle">after</div>
                  <div class="compare-track"><div class="compare-fill after" style="width:${{afterWidth}}%"></div></div>
                </div>
              </div>
            </div>`;
        }}).join('') + '</div>';

        const beforeEvent = beforeSession.event_summary;
        const afterEvent = afterSession.event_summary;
        eventTarget.innerHTML = `
          <div class="grid">
            <div class="metric"><div class="label">Before First Render</div><div class="value">${{fmt(beforeEvent.first_render_elapsed_s, 3)}} s</div></div>
            <div class="metric"><div class="label">After First Render</div><div class="value">${{fmt(afterEvent.first_render_elapsed_s, 3)}} s</div></div>
            <div class="metric"><div class="label">Before Session Duration</div><div class="value">${{fmt(beforeEvent.session_duration_s, 3)}} s</div></div>
            <div class="metric"><div class="label">After Session Duration</div><div class="value">${{fmt(afterEvent.session_duration_s, 3)}} s</div></div>
          </div>
          <p class="note" style="margin-top:16px;">
            before OpenGL unavailable: ${{beforeEvent.opengl_unavailable ? 'yes' : 'no'}} |
            after OpenGL unavailable: ${{afterEvent.opengl_unavailable ? 'yes' : 'no'}} |
            before session error: ${{beforeEvent.session_error ? beforeEvent.session_error_repr : 'none'}} |
            after session error: ${{afterEvent.session_error ? afterEvent.session_error_repr : 'none'}}
          </p>
          <p class="subtle">
            before: <a href="${{beforeSession.links.index}}">세션 개요 열기</a> |
            after: <a href="${{afterSession.links.index}}">세션 개요 열기</a>
          </p>
        `;
      }}

      function initDashboard() {{
        const beforeSelect = document.getElementById('before-session');
        const afterSelect = document.getElementById('after-session');
        beforeSelect.innerHTML = '';
        afterSelect.innerHTML = '';
        DASHBOARD.sessions.forEach((session) => {{
          const beforeOption = document.createElement('option');
          beforeOption.value = session.session_id;
          beforeOption.textContent = `${{session.session_id}} | ${{session.variant || 'n/a'}}`;
          beforeSelect.appendChild(beforeOption);

          const afterOption = document.createElement('option');
          afterOption.value = session.session_id;
          afterOption.textContent = `${{session.session_id}} | ${{session.variant || 'n/a'}}`;
          afterSelect.appendChild(afterOption);
        }});
        if (DASHBOARD.sessions.length >= 2) {{
          afterSelect.value = DASHBOARD.sessions[0].session_id;
          beforeSelect.value = DASHBOARD.sessions[1].session_id;
        }} else if (DASHBOARD.sessions.length === 1) {{
          beforeSelect.value = DASHBOARD.sessions[0].session_id;
          afterSelect.value = DASHBOARD.sessions[0].session_id;
        }}
        beforeSelect.addEventListener('change', renderComparison);
        afterSelect.addEventListener('change', renderComparison);
        renderComparison();
      }}

      initDashboard();
    </script>
    """

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>live_motion_viewer 로그 대시보드</title>
  <style>{COMMON_STYLE}</style>
</head>
<body>
  <div class="wrap">
    <header class="hero">
      <h1>live_motion_viewer 로그 대시보드</h1>
      <p>세션이 저장될 때마다 자동으로 생성되는 HTML 리포트입니다. 아래에서 두 세션을 선택해 비교할 수 있습니다.</p>
    </header>

    <section class="card">
      <h2>세션 비교</h2>
      <div class="controls">
        <div class="control"><label for="before-session">비교 기준 세션</label><select id="before-session"></select></div>
        <div class="control"><label for="after-session">비교 대상 세션</label><select id="after-session"></select></div>
      </div>
      <div id="comparison-metrics" class="grid" style="margin-top:18px;"></div>
    </section>

    <section class="card">
      <h2>비교 그래프</h2>
      <div id="comparison-chart"></div>
    </section>

    <section class="card">
      <h2>이벤트 비교</h2>
      <div id="comparison-events"></div>
    </section>

    <section class="card">
      <h2>세션 목록</h2>
      <table>
        <thead>
          <tr>
            <th>세션</th><th>생성 시각</th><th>variant</th><th>scenario</th><th>상태</th><th>render invalid</th><th>display mean</th><th>render p95</th><th>리포트</th>
          </tr>
        </thead>
        <tbody>{table_rows}</tbody>
      </table>
    </section>
    {script}
  </div>
</body>
</html>"""


def _day_metric_context(log_root: Path, day_prefix: str):
    metric_specs = [
        ("render.invalid_rate", "Render Invalid", "lower"),
        ("render.capture_to_render_ms.p95", "Render P95", "lower"),
        ("render.display_track_count.mean", "Display Track Mean", "higher"),
        ("render.multi_display_success_rate", "Multi Display Success", "higher"),
    ]
    rows = []
    for session_dir in sorted(path for path in log_root.iterdir() if path.is_dir() and path.name.startswith(day_prefix)):
        summary_path = session_dir / "summary.json"
        if not summary_path.exists():
            continue
        summary = _load_json(summary_path, {})
        metrics = {}
        for dotted_key, _label, _direction in metric_specs:
            current = summary
            for part in dotted_key.split("."):
                if not isinstance(current, dict) or part not in current:
                    current = None
                    break
                current = current[part]
            metrics[dotted_key] = current
        rows.append({"session_id": session_dir.name, "metrics": metrics})
    return {"metrics": metric_specs, "rows": rows}


def _comparison_summary_note(comparison: dict):
    metric_map = {item["key"]: item for item in comparison["metrics"]}
    render_invalid = metric_map.get("render.invalid_rate", {})
    render_p95 = metric_map.get("render.capture_to_render_ms.p95", {})
    display_mean = metric_map.get("render.display_track_count.mean", {})
    multi_success = metric_map.get("render.multi_display_success_rate", {})

    notes = []
    if render_invalid.get("judgement") == "improved":
        notes.append("입력 무결성과 최종 렌더 안정성은 비교 기준보다 좋아졌습니다.")
    elif render_invalid.get("judgement") == "regressed":
        notes.append("입력 무결성이 비교 기준보다 나빠졌습니다.")

    if render_p95.get("judgement") == "regressed":
        notes.append("다만 상위 95% 지연은 아직 더 높아서 체감 응답성은 열세입니다.")
    elif render_p95.get("judgement") == "improved":
        notes.append("상위 95% 지연도 함께 줄어들어 체감 응답성이 좋아졌습니다.")

    if display_mean.get("judgement") == "improved":
        notes.append("화면에 실제로 남는 트랙 수는 늘었습니다.")
    elif display_mean.get("judgement") == "regressed":
        notes.append("화면에 실제로 남는 트랙 수는 아직 소폭 부족합니다.")

    if multi_success.get("judgement") == "improved":
        notes.append("다중 타깃 유지력도 개선 방향입니다.")
    elif multi_success.get("judgement") == "regressed":
        notes.append("다중 타깃 유지력은 아직 더 끌어올릴 여지가 있습니다.")

    if not notes:
        notes.append("일부 지표는 비슷한 수준이라, 현 시점 평가는 환경 재현 여부까지 함께 봐야 합니다.")
    return notes


def _comparison_bar_rows(comparison: dict):
    rows = []
    numeric_metrics = [
        item for item in comparison["metrics"]
        if isinstance(item.get("before"), (int, float)) and isinstance(item.get("after"), (int, float))
    ]
    if not numeric_metrics:
        return '<div class="subtle">비교 가능한 숫자 지표가 없습니다.</div>'

    for item in numeric_metrics:
        before_value = float(item["before"])
        after_value = float(item["after"])
        max_value = max(abs(before_value), abs(after_value), 1e-9)
        before_width = max(6.0, abs(before_value) / max_value * 100.0)
        after_width = max(6.0, abs(after_value) / max_value * 100.0)
        if item["key"].endswith("invalid_rate") or item["key"].endswith("success_rate"):
            before_text = _fmt_pct(before_value)
            after_text = _fmt_pct(after_value)
            delta_text = _fmt_pct(item.get("delta"))
        elif "ms" in item["key"]:
            before_text = _fmt(before_value, suffix=" ms")
            after_text = _fmt(after_value, suffix=" ms")
            delta_text = _fmt(item.get("delta"), suffix=" ms")
        else:
            before_text = _fmt(before_value)
            after_text = _fmt(after_value)
            delta_text = _fmt(item.get("delta"))

        rows.append(
            f"""
            <div class="compare-row">
              <div>
                <strong>{item['label']}</strong>
                <div class="compare-meta">
                  <span>before={before_text}</span>
                  <span>after={after_text}</span>
                  <span>delta={delta_text}</span>
                  {_pill(item['judgement'], _judgement_tone(item['judgement']))}
                </div>
              </div>
              <div class="compare-bars">
                <div>
                  <div class="subtle">before</div>
                  <div class="compare-track"><div class="compare-fill before" style="width:{before_width:.2f}%"></div></div>
                </div>
                <div>
                  <div class="subtle">after</div>
                  <div class="compare-track"><div class="compare-fill after" style="width:{after_width:.2f}%"></div></div>
                </div>
              </div>
            </div>
            """
        )
    return '<div class="compare-list">' + "".join(rows) + "</div>"


def _build_static_comparison_html(
    before_summary: dict,
    after_summary: dict,
    comparison: dict,
    before_event_summary: dict,
    after_event_summary: dict,
    before_day_context: dict,
):
    before_session = comparison["before"]["session_id"] or "before"
    after_session = comparison["after"]["session_id"] or "after"
    notes = "".join(f"<li>{note}</li>" for note in _comparison_summary_note(comparison))
    metric_map = {item["key"]: item for item in comparison["metrics"]}

    overview_specs = [
        ("render.invalid_rate", "Render Invalid", "percent"),
        ("render.capture_to_render_ms.p95", "Render P95", "ms"),
        ("render.display_track_count.mean", "Display Track Mean", "number"),
        ("render.multi_display_success_rate", "Multi Display Success", "percent"),
    ]
    overview_cards = []
    for key, label, fmt_type in overview_specs:
        item = metric_map.get(key, {})
        before_value = item.get("before")
        after_value = item.get("after")
        delta_value = item.get("delta")
        if fmt_type == "percent":
            before_text = _fmt_pct(before_value)
            after_text = _fmt_pct(after_value)
            delta_text = _fmt_pct(delta_value)
        elif fmt_type == "ms":
            before_text = _fmt(before_value, suffix=" ms")
            after_text = _fmt(after_value, suffix=" ms")
            delta_text = _fmt(delta_value, suffix=" ms")
        else:
            before_text = _fmt(before_value)
            after_text = _fmt(after_value)
            delta_text = _fmt(delta_value)
        overview_cards.append(
            f"""
            <div class="metric">
              <div class="label">{label}</div>
              <div class="value">{after_text}</div>
              <div class="hint">
                before={before_text} | delta={delta_text}<br />
                {_pill(item.get('judgement', 'n/a'), _judgement_tone(item.get('judgement', 'n/a')))}
              </div>
            </div>
            """
        )

    context_rows = []
    for dotted_key, label, direction in before_day_context["metrics"]:
        values = [
            row["metrics"].get(dotted_key)
            for row in before_day_context["rows"]
            if isinstance(row["metrics"].get(dotted_key), (int, float))
        ]
        if not values:
            continue
        values.sort()
        middle_index = len(values) // 2
        median_value = values[middle_index] if len(values) % 2 == 1 else (values[middle_index - 1] + values[middle_index]) / 2
        current_value = metric_map.get(dotted_key, {}).get("after")
        if dotted_key.endswith("invalid_rate") or dotted_key.endswith("success_rate"):
            min_text = _fmt_pct(values[0])
            median_text = _fmt_pct(median_value)
            max_text = _fmt_pct(values[-1])
            current_text = _fmt_pct(current_value)
        elif "ms" in dotted_key:
            min_text = _fmt(values[0], suffix=" ms")
            median_text = _fmt(median_value, suffix=" ms")
            max_text = _fmt(values[-1], suffix=" ms")
            current_text = _fmt(current_value, suffix=" ms")
        else:
            min_text = _fmt(values[0])
            median_text = _fmt(median_value)
            max_text = _fmt(values[-1])
            current_text = _fmt(current_value)

        verdict = "중간권"
        if isinstance(current_value, (int, float)):
            if direction == "lower":
                if current_value <= values[0]:
                    verdict = "상위권"
                elif current_value >= values[-1]:
                    verdict = "하위권"
            else:
                if current_value >= values[-1]:
                    verdict = "상위권"
                elif current_value <= values[0]:
                    verdict = "하위권"
        context_rows.append(
            f"""
            <tr>
              <td><strong>{label}</strong></td>
              <td>{current_text}</td>
              <td>{min_text}</td>
              <td>{median_text}</td>
              <td>{max_text}</td>
              <td>{verdict}</td>
            </tr>
            """
        )
    context_table = "".join(context_rows) or '<tr><td colspan="6">비교 가능한 3월 25일 분포가 없습니다.</td></tr>'

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{after_session} vs {before_session} 비교 리포트</title>
  <style>{COMMON_STYLE}</style>
</head>
<body>
  <div class="wrap">
    <header class="hero">
      <h1>{after_session} vs {before_session}</h1>
      <p>최신 안정 세션과 2026년 3월 25일 대표 세션을 비교한 정적 보고서입니다. 입력 무결성, 지연, 표시 트랙, 다중 타깃 유지력을 함께 봅니다.</p>
      <nav class="nav">
        <a href="../index.html">전체 비교 대시보드</a>
        <a href="../{after_session}/index.html">최신 세션 개요</a>
        <a href="../{before_session}/index.html">3월 25일 기준 세션 개요</a>
      </nav>
    </header>

    <section class="card">
      <h2>요약 평가</h2>
      <p class="subtle">
        before: <code>{before_session}</code> |
        after: <code>{after_session}</code> |
        before variant: <code>{before_summary.get("session_meta", {}).get("variant") or "n/a"}</code> |
        after variant: <code>{after_summary.get("session_meta", {}).get("variant") or "n/a"}</code>
      </p>
      <div class="grid">
        {''.join(overview_cards)}
      </div>
      <div class="note" style="margin-top:16px;">
        <strong>현업 해석:</strong>
        <ul style="margin:10px 0 0 18px; padding:0;">{notes}</ul>
      </div>
    </section>

    <section class="card">
      <h2>핵심 비교 시각화</h2>
      {_comparison_bar_rows(comparison)}
    </section>

    <section class="card">
      <h2>이벤트 / 시스템 상태 비교</h2>
      <div class="grid">
        <div class="metric">
          <div class="label">Before First Render</div>
          <div class="value">{_fmt(before_event_summary.get('first_render_elapsed_s'), suffix=' s')}</div>
          <div class="hint">3월 25일 기준 세션</div>
        </div>
        <div class="metric">
          <div class="label">After First Render</div>
          <div class="value">{_fmt(after_event_summary.get('first_render_elapsed_s'), suffix=' s')}</div>
          <div class="hint">최신 세션</div>
        </div>
        <div class="metric">
          <div class="label">Before Session Duration</div>
          <div class="value">{_fmt(before_event_summary.get('session_duration_s'), suffix=' s')}</div>
          <div class="hint">OpenGL unavailable: {'yes' if before_event_summary.get('opengl_unavailable') else 'no'}</div>
        </div>
        <div class="metric">
          <div class="label">After Session Duration</div>
          <div class="value">{_fmt(after_event_summary.get('session_duration_s'), suffix=' s')}</div>
          <div class="hint">OpenGL unavailable: {'yes' if after_event_summary.get('opengl_unavailable') else 'no'}</div>
        </div>
      </div>
    </section>

    <section class="card">
      <h2>3월 25일 로그 범위 안에서 현재 위치</h2>
      <p class="subtle">대표 세션 1개만 보는 대신, 같은 날짜의 저장된 세션 범위와 함께 비교한 표입니다.</p>
      <table>
        <thead>
          <tr>
            <th>지표</th>
            <th>현재 세션</th>
            <th>3월 25일 최저</th>
            <th>3월 25일 중앙값</th>
            <th>3월 25일 최고</th>
            <th>판정</th>
          </tr>
        </thead>
        <tbody>{context_table}</tbody>
      </table>
    </section>
  </div>
</body>
</html>"""


def generate_static_comparison_report(
    before_path: str | Path,
    after_path: str | Path,
    output_path: str | Path | None = None,
):
    before_summary_path = _resolve_summary_path(before_path)
    after_summary_path = _resolve_summary_path(after_path)
    before_summary = _load_json(before_summary_path, {})
    after_summary = _load_json(after_summary_path, {})
    comparison = build_comparison(before_summary, after_summary, before_summary_path, after_summary_path)

    before_event_summary = _event_summary(_load_jsonl(before_summary_path.parent / "event_log.jsonl"))
    after_event_summary = _event_summary(_load_jsonl(after_summary_path.parent / "event_log.jsonl"))
    before_day_context = _day_metric_context(after_summary_path.parent.parent, before_summary_path.parent.name[:8])

    if output_path is None:
        output_path = after_summary_path.parent / f"comparison_vs_{before_summary_path.parent.name}.html"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_text(
        output_path,
        _build_static_comparison_html(
            before_summary,
            after_summary,
            comparison,
            before_event_summary,
            after_event_summary,
            before_day_context,
        ),
    )
    return output_path


def generate_session_artifacts(session_dir: str | Path):
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(session_dir)
    _write_json(session_dir / "summary.json", summary)

    processed_records = _load_jsonl(session_dir / "processed_frames.jsonl")
    render_records = _load_jsonl(session_dir / "render_frames.jsonl")
    if not render_records:
        render_records = _load_jsonl(session_dir / "status_log.jsonl")
    events = _load_jsonl(session_dir / "event_log.jsonl")
    event_summary = _event_summary(events)

    _write_text(session_dir / "index.html", _build_session_index_html(session_dir, summary, event_summary))
    _write_text(session_dir / "processed_report.html", _build_processed_html(session_dir, summary, processed_records))
    _write_text(session_dir / "render_report.html", _build_render_html(session_dir, summary, render_records))
    _write_text(session_dir / "event_report.html", _build_event_html(session_dir, events, event_summary))

    return {
        "summary_path": session_dir / "summary.json",
        "index_path": session_dir / "index.html",
        "processed_report_path": session_dir / "processed_report.html",
        "render_report_path": session_dir / "render_report.html",
        "event_report_path": session_dir / "event_report.html",
    }


def generate_root_dashboard(log_root: str | Path):
    log_root = Path(log_root)
    log_root.mkdir(parents=True, exist_ok=True)
    session_rows = _collect_session_rows(log_root)
    dashboard_path = log_root / "index.html"
    _write_text(dashboard_path, _build_root_dashboard_html(session_rows))
    return dashboard_path


def generate_reports(target_path: str | Path):
    target_path = Path(target_path)
    is_session_dir = target_path.is_dir() and any(
        (target_path / name).exists()
        for name in (
            "session_meta.json",
            "runtime_config.json",
            "processed_frames.jsonl",
            "render_frames.jsonl",
            "status_log.jsonl",
            "summary.json",
        )
    )
    if is_session_dir:
        generate_session_artifacts(target_path)
        dashboard_path = generate_root_dashboard(target_path.parent)
        return {"mode": "session", "session_dir": target_path, "dashboard_path": dashboard_path}

    log_root = target_path
    if log_root.name != "live_motion_viewer":
        log_root = log_root / "logs" / "live_motion_viewer"
    dashboard_path = generate_root_dashboard(log_root)
    return {"mode": "root", "log_root": log_root, "dashboard_path": dashboard_path}


def main():
    parser = argparse.ArgumentParser(description="Generate HTML reports for live_motion_viewer session logs.")
    parser.add_argument("target", nargs="?", help="Session directory or live_motion_viewer log root.")
    parser.add_argument("--compare-before", help="Baseline session directory or summary.json for static comparison HTML.")
    parser.add_argument("--compare-after", help="Candidate session directory or summary.json for static comparison HTML.")
    parser.add_argument("--output", help="Optional output path for static comparison HTML.")
    args = parser.parse_args()
    if args.compare_before and args.compare_after:
        output_path = generate_static_comparison_report(
            args.compare_before,
            args.compare_after,
            args.output,
        )
        print(json.dumps({"mode": "compare", "output_path": str(output_path)}, ensure_ascii=False, indent=2))
        return
    if not args.target:
        parser.error("target or --compare-before/--compare-after must be provided")
    result = generate_reports(args.target)
    print(json.dumps({key: str(value) for key, value in result.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
