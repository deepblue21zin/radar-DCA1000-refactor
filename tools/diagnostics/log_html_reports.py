from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from .operational_assessment import build_event_summary
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
  .steps { margin: 0; padding-left: 18px; }
  .steps li + li { margin-top: 8px; }
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

function renderTrajectoryChart(targetId, seriesList, options = {}) {
  const target = document.getElementById(targetId);
  if (!target) return;
  const width = options.width || 520;
  const height = options.height || 420;
  const padding = { top: 18, right: 24, bottom: 40, left: 44 };
  const palette = ['#0f6cbd', '#ef4444', '#10b981', '#f97316', '#8b5cf6', '#14b8a6'];
  const points = [];
  (seriesList || []).forEach((series) => {
    (series.points || []).forEach((point) => {
      const x = Number(point.x_m);
      const y = Number(point.y_m);
      if (Number.isFinite(x) && Number.isFinite(y)) points.push({ x, y });
    });
  });
  if (!points.length) {
    target.innerHTML = `<div class="empty">${esc(options.emptyMessage || '표시할 궤적 데이터가 없습니다.')}</div>`;
    return;
  }

  let maxAbsX = Math.max(...points.map((point) => Math.abs(point.x)), 0.5);
  let minY = Math.min(...points.map((point) => point.y), 0);
  let maxY = Math.max(...points.map((point) => point.y), 0.5);
  maxAbsX = Math.max(0.5, maxAbsX * 1.15);
  minY = Math.min(0, minY);
  maxY = Math.max(0.5, maxY * 1.12);
  if (Math.abs(maxY - minY) < 0.5) maxY = minY + 0.5;

  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const xFor = (x) => padding.left + ((Number(x) + maxAbsX) / (maxAbsX * 2)) * plotWidth;
  const yFor = (y) => padding.top + (1 - ((Number(y) - minY) / (maxY - minY))) * plotHeight;

  let svg = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(options.title || 'trajectory')}">`;
  svg += `<rect x="0" y="0" width="${width}" height="${height}" rx="18" fill="#ffffff"></rect>`;

  for (let step = 0; step <= 4; step += 1) {
    const yValue = minY + ((maxY - minY) * step) / 4;
    const y = yFor(yValue);
    svg += `<line x1="${padding.left}" y1="${y}" x2="${width - padding.right}" y2="${y}" stroke="#d7deea" stroke-width="1"></line>`;
    svg += `<text x="${padding.left - 8}" y="${y + 4}" text-anchor="end" fill="#5f6f86" font-size="12">${esc(fmt(yValue, 1))}</text>`;
  }
  for (let step = 0; step <= 4; step += 1) {
    const xValue = -maxAbsX + ((maxAbsX * 2) * step) / 4;
    const x = xFor(xValue);
    svg += `<line x1="${x}" y1="${padding.top}" x2="${x}" y2="${height - padding.bottom}" stroke="#eef2f7" stroke-width="1"></line>`;
    svg += `<text x="${x}" y="${height - 14}" text-anchor="middle" fill="#5f6f86" font-size="12">${esc(fmt(xValue, 1))}</text>`;
  }

  svg += `<line x1="${xFor(0)}" y1="${padding.top}" x2="${xFor(0)}" y2="${height - padding.bottom}" stroke="#8ea0bb" stroke-width="1.4"></line>`;
  svg += `<line x1="${padding.left}" y1="${yFor(0)}" x2="${width - padding.right}" y2="${yFor(0)}" stroke="#8ea0bb" stroke-width="1.4"></line>`;

  (seriesList || []).forEach((series, index) => {
    const color = series.color || palette[index % palette.length];
    const validPoints = (series.points || []).filter((point) => Number.isFinite(Number(point.x_m)) && Number.isFinite(Number(point.y_m)));
    if (!validPoints.length) return;
    const polyline = validPoints.map((point) => `${xFor(point.x_m)},${yFor(point.y_m)}`).join(" ");
    const start = validPoints[0];
    const end = validPoints[validPoints.length - 1];
    svg += `<polyline fill="none" stroke="${color}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round" points="${polyline}"></polyline>`;
    svg += `<circle cx="${xFor(start.x_m)}" cy="${yFor(start.y_m)}" r="4.5" fill="#ffffff" stroke="${color}" stroke-width="2"></circle>`;
    svg += `<circle cx="${xFor(end.x_m)}" cy="${yFor(end.y_m)}" r="5.2" fill="${color}" stroke="#ffffff" stroke-width="2"></circle>`;
  });

  svg += `<circle cx="${xFor(0)}" cy="${yFor(0)}" r="6" fill="#0f172a"></circle>`;
  svg += `<text x="${xFor(0) + 10}" y="${yFor(0) - 8}" fill="#142033" font-size="12">radar</text>`;
  svg += `<text x="${padding.left}" y="${height - 4}" fill="#5f6f86" font-size="12">x: left / right (m)</text>`;
  svg += `<text x="${width - padding.right}" y="${padding.top + 12}" text-anchor="end" fill="#5f6f86" font-size="12">y: forward (m)</text>`;
  svg += `</svg>`;

  const legend = (seriesList || []).map((series, index) => {
    const color = series.color || palette[index % palette.length];
    return `<span><i style="background:${color}"></i>${esc(series.label)}</span>`;
  }).join("");
  target.innerHTML = svg
    + `<div class="legend">${legend}</div>`
    + `<p class="subtle" style="margin:10px 0 0;">range/angle 값을 레이더 기준 x/y 좌표로 바꾼 경로입니다. 빈 원은 시작점, 채워진 원은 마지막 위치입니다.</p>`;
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


TRAJECTORY_SOURCE_LABELS = {
    "display_tracks": "display track",
    "tentative_display_tracks": "tentative display track",
    "tentative_tracks": "tentative track",
    "confirmed_tracks": "confirmed track",
    "detections": "lead detection fallback",
}

TRAJECTORY_COLORS = ["#0f6cbd", "#ef4444", "#10b981", "#f97316", "#8b5cf6", "#14b8a6"]

PROCESSED_FIELD_GUIDE = [
    ("capture_to_process_ms", "수집부터 처리 완료까지 걸린 시간", "알고리즘 지연을 봅니다. 프레임 주기보다 계속 크면 backlog 위험이 있습니다."),
    ("udp_gap_count", "UDP 패킷이 중간에 빠진 개수", "0에 가까워야 정상입니다. 두 자릿수 이상이 계속 나오면 통신 문제를 먼저 의심합니다."),
    ("byte_mismatch_count", "예상 바이트 카운트와 실제 값 불일치 횟수", "DCA1000 스트림 무결성 경고입니다."),
    ("invalid, invalid_reason", "이 프레임을 신뢰할 수 있는지 여부", "<code>sequence</code>, <code>byte_count</code>가 반복되면 알고리즘보다 먼저 네트워크 무결성을 의심합니다."),
    ("candidate_count", "검출 후보 수", "높다고 무조건 좋은 게 아닙니다. 노이즈가 많아도 증가합니다."),
    ("tracker_input_count", "실제로 tracker에 넘긴 후보 수", "<code>candidate_count</code>보다 작으면 정책상 버린 것입니다."),
    ("tracker_policy", "<code>full</code>, <code>no_birth</code>, <code>drop</code>", "<code>drop</code>이면 detection이 있어도 tracker가 아예 받지 않습니다."),
    ("confirmed_track_count", "확정 트랙 개수", "현업에서 실제 사용 가능한 결과는 보통 이쪽을 더 중요하게 봅니다."),
    ("tentative_track_count", "가설 단계 트랙 개수", "후보는 보이지만 아직 안정적이지 않다는 뜻입니다."),
    ("stage_timings_ms.shared_fft2_ms", "공통 range/doppler FFT 시간", "DSP 병목 1순위 후보입니다. shared FFT를 넣은 뒤에는 이 값을 우선 봅니다."),
    ("stage_timings_ms.range_angle_project_ms", "공통 FFT를 RAI로 투영하는 시간", "angle 쪽 비용입니다. 이 값이 크면 angle update 정책을 검토합니다."),
    ("stage_timings_ms.detect_ms", "detection 전체 시간", "CFAR와 후보 정제가 느린지 볼 때 씁니다."),
    ("stage_timings_ms.track_ms", "tracker update 시간", "타깃 수가 많을 때 association 비용이 늘어나는지 확인합니다."),
    ("stage_timings_ms.log_write_ms", "processed 로그 쓰기 시간", "평균보다 p95가 튀면 logging이 지터를 만드는지 확인합니다."),
]

RENDER_FIELD_GUIDE = [
    ("capture_to_render_ms", "수집부터 실제 표시까지 걸린 총 시간", "화면 반응성이 느린지 판단합니다. 이 값이 크고 <code>process_to_render_ms</code>는 작으면 원인은 앞단 processing일 가능성이 큽니다."),
    ("process_to_render_ms", "처리 완료 후 UI에 그리기까지 걸린 시간", "렌더링 병목을 볼 때 씁니다."),
    ("display_track_count", "화면에 실제로 표시된 확정 트랙 수", "사용자 입장에서 가장 중요한 숫자입니다."),
    ("tentative_display_track_count", "디버그용 가설 트랙 표시 수", "confirmed가 없어도 tentative가 있으면 '후보는 보인다'로 해석할 수 있습니다."),
    ("status_text", "상태바에 표시된 요약 문장", "현장에서 가장 빨리 읽을 수 있는 1줄 요약입니다."),
    ("skipped_render_frames", "렌더링 중 건너뛴 프레임 수", "0이 아니면 UI가 처리 속도를 못 따라가는 것입니다."),
    ("invalid, invalid_reason", "렌더된 프레임이 신뢰 가능한 입력에서 왔는지 여부", "render 문제처럼 보여도 실제 원인이 UDP 무결성일 수 있다는 뜻입니다."),
    ("stage_timings_ms", "처리단 stage timing 사본", "render 문서만 열어도 slow stage를 같이 확인할 수 있게 해 줍니다."),
]

EVENT_FIELD_GUIDE = [
    ("system_snapshot_captured", "전원 계획, host IP 일치 여부, process priority 기록", "성능 회귀 원인이 코드가 아니라 환경일 가능성"),
    ("dca_config_complete", "DCA1000 설정 성공", "없으면 host IP, fpga IP, UDP config port, 방화벽을 먼저 봅니다."),
    ("workers_started", "수집/처리 스레드 시작", "없으면 앱 시작 로직, 큐 초기화, 예외 발생을 의심합니다."),
    ("radar_open_complete", "레이더 CLI cfg 적용 완료", "없으면 CLI COM 포트, baudrate, cfg 파일 내용을 봅니다."),
    ("first_rendered_frame", "첫 프레임이 UI에 도달", "없으면 LVDS 스트림, UDP 데이터 수신, frame assembly를 의심합니다."),
    ("opengl_unavailable", "3D 뷰 비활성화", "PyOpenGL 미설치 가능성이 큽니다. 다만 2D 실패 원인과는 별개일 수 있습니다."),
]

PROCESSED_READING_STEPS = [
    "<strong>1단계: invalid rate부터 본다</strong><br />invalid가 높으면 알고리즘 튜닝 전에 NIC, IP, 방화벽, 케이블부터 확인합니다.",
    "<strong>2단계: stage timing을 본다</strong><br /><code>shared_fft2_ms</code>, <code>detect_ms</code>, <code>track_ms</code> 중 어디가 가장 큰지 먼저 구분합니다.",
    "<strong>3단계: 후보 손실 위치를 본다</strong><br /><code>candidate_count → tracker_input_count → confirmed_track_count</code> 순서로 어디서 줄었는지 봅니다.",
    "<strong>4단계: ops report와 system snapshot을 같이 본다</strong><br />같은 코드인데 느려졌다면 <code>system_snapshot.json</code>과 <code>ops_report.html</code>를 같이 봐야 합니다.",
]

RENDER_READING_STEPS = [
    "<strong>1순위: total latency를 processing과 render로 분리</strong><br /><code>process_to_render_ms</code>와 <code>stage_timings_ms</code>를 같이 봅니다.",
    "<strong>2순위: invalid와 skipped_render_frames 확인</strong><br />입력 무결성 문제인지, UI가 밀리는 문제인지 먼저 구분합니다.",
    "<strong>3순위: display 정책 확인</strong><br /><code>display_min_confidence</code>, <code>report_miss_tolerance</code>, <code>confirm_hits</code>가 너무 보수적인지 확인합니다.",
    "<strong>4순위: ops report와 비교</strong><br /><code>ops_report.html</code>의 운영 점수와 slowest stage를 같이 보면 설명이 훨씬 쉬워집니다.",
]

EVENT_READING_STEPS = [
    "<strong>1단계: event 로그로 시작 단계가 정상인지 먼저 확인</strong><br />이 단계가 비정상이면 processing 튜닝보다 먼저 장비/포트/IP를 봐야 합니다.",
    "<strong>2단계: system snapshot payload를 확인</strong><br />전원 모드, host IP mismatch, priority class를 같이 봅니다.",
    "<strong>3단계: processed 로그로 invalid와 stage timing을 확인</strong><br />세션이 시작된 뒤 병목이 어디인지 찾습니다.",
    "<strong>4단계: render 로그로 사용자 체감 결과를 확인</strong><br />display track이 왜 적은지 최종적으로 설명합니다.",
]


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


def _load_render_records_with_fallback(session_dir: Path):
    records = _load_jsonl(session_dir / "render_frames.jsonl")
    if not records:
        records = _load_jsonl(session_dir / "status_log.jsonl")
    return records


def _trajectory_priority_label(priority_keys: list[str]):
    labels = [TRAJECTORY_SOURCE_LABELS.get(key, key) for key in priority_keys]
    return " -> ".join(labels)


def _downsample_points(points: list[dict], max_points: int = 120):
    if len(points) <= max_points:
        return points
    index_map = sorted({round(index * (len(points) - 1) / (max_points - 1)) for index in range(max_points)})
    return [points[index] for index in index_map]


def _trajectory_point(item: dict, frame_id: int):
    x_m = item.get("x_m")
    y_m = item.get("y_m")
    if not isinstance(x_m, (int, float)) or not isinstance(y_m, (int, float)):
        return None
    return {
        "frame_id": int(frame_id),
        "x_m": round(float(x_m), 4),
        "y_m": round(float(y_m), 4),
        "angle_deg": _round_or_none(item.get("angle_deg")),
        "range_m": _round_or_none(item.get("range_m"), digits=4),
    }


def _build_track_trajectory_bundle(
    records: list[dict],
    priority_keys: list[str],
    *,
    max_tracks: int = 4,
    min_points: int = 2,
):
    key_hits = {key: 0 for key in priority_keys}
    grouped: dict[str, list[dict]] = {}

    for record in records:
        frame_id = int(record.get("frame_id", record.get("frame_index", 0)) or 0)
        selected_key = None
        selected_items = []
        for key in priority_keys:
            items = record.get(key) or []
            if items:
                selected_key = key
                selected_items = items
                break
        if not selected_key:
            continue
        key_hits[selected_key] += 1
        for item in selected_items:
            track_id = item.get("track_id")
            if track_id is None:
                continue
            point = _trajectory_point(item, frame_id)
            if point is None:
                continue
            grouped.setdefault(str(track_id), []).append(point)

    series = []
    for index, (track_id, points) in enumerate(
        sorted(grouped.items(), key=lambda entry: (-len(entry[1]), entry[0]))
    ):
        points.sort(key=lambda point: point["frame_id"])
        if len(points) < min_points:
            continue
        sampled_points = _downsample_points(points)
        series.append(
            {
                "track_id": track_id,
                "label": f"track {track_id}",
                "color": TRAJECTORY_COLORS[index % len(TRAJECTORY_COLORS)],
                "point_count": len(sampled_points),
                "points": sampled_points,
            }
        )
        if len(series) >= max_tracks:
            break

    source_key = None
    if key_hits and max(key_hits.values()) > 0:
        source_key = max(key_hits.items(), key=lambda item: item[1])[0]

    fallback_used = False
    if not series:
        detection_points = []
        for record in records:
            detections = record.get("detections") or []
            if not detections:
                continue
            lead = max(detections, key=lambda item: float(item.get("score", 0.0) or 0.0))
            point = _trajectory_point(lead, int(record.get("frame_id", record.get("frame_index", 0)) or 0))
            if point is not None:
                detection_points.append(point)
        detection_points.sort(key=lambda point: point["frame_id"])
        if len(detection_points) >= min_points:
            sampled_points = _downsample_points(detection_points)
            series.append(
                {
                    "track_id": "lead",
                    "label": "lead detection",
                    "color": TRAJECTORY_COLORS[0],
                    "point_count": len(sampled_points),
                    "points": sampled_points,
                }
            )
            source_key = "detections"
            fallback_used = True

    source_label = TRAJECTORY_SOURCE_LABELS.get(source_key, "n/a") if source_key else "n/a"
    empty_message = (
        "궤적 데이터가 없습니다. 이 세션은 payload가 비활성화됐거나 track 좌표가 저장되지 않았습니다."
    )
    if fallback_used:
        empty_message = "track 궤적이 없어 lead detection 기준 경로만 표시합니다."

    return {
        "series": series,
        "track_count": len(series),
        "source_key": source_key,
        "source_label": source_label,
        "priority_label": _trajectory_priority_label(priority_keys),
        "fallback_used": fallback_used,
        "empty_message": empty_message,
    }


def _load_session_trajectory_bundle(session_dir: Path):
    render_bundle = _build_track_trajectory_bundle(
        _load_render_records_with_fallback(session_dir),
        ["display_tracks", "tentative_display_tracks", "tentative_tracks"],
    )
    if render_bundle["series"]:
        return render_bundle
    return _build_track_trajectory_bundle(
        _load_jsonl(session_dir / "processed_frames.jsonl"),
        ["confirmed_tracks", "tentative_tracks"],
    )


def _trajectory_summary_text(bundle: dict):
    if bundle.get("series"):
        fallback_text = " | fallback=detection" if bundle.get("fallback_used") else ""
        return (
            f"priority={bundle.get('priority_label', 'n/a')} | "
            f"source={bundle.get('source_label', 'n/a')} | "
            f"tracks={bundle.get('track_count', 0)}{fallback_text}"
        )
    return bundle.get("empty_message") or "궤적 데이터가 없습니다."


def _event_summary(events: list[dict]):
    return build_event_summary(events)


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
    assessment = summary.get("assessment", {})
    overall = assessment.get("overall", {})
    system = summary.get("system", {})
    preferred_stage = (summary.get("diagnostics", {}).get("preferred_stage_timings_ms") or {}).get("slowest_stage") or {}
    power_tone = "brand"
    if system.get("power_plan_recommended") is True:
        power_tone = "good"
    elif system.get("power_plan_recommended") is False:
        power_tone = "warn"
    host_ip_tone = "brand"
    if system.get("host_ip_present") is True:
        host_ip_tone = "good"
    elif system.get("host_ip_present") is False:
        host_ip_tone = "danger"
    return [
        (
            "현업 점수",
            f"{overall.get('score', 'n/a')}/100",
            overall.get("label", "평가 없음"),
            overall.get("tone", "brand"),
        ),
        ("세션 상태", health["label"], health["detail"], health["tone"]),
        ("Processed Invalid", _fmt_pct(processed.get("invalid_rate")), "처리 입력 무결성 관점", "danger" if (processed.get("invalid_rate") or 0) >= 0.5 else "brand"),
        ("Render Invalid", _fmt_pct(render.get("invalid_rate")), "사용자 체감 품질 관점", "danger" if (render.get("invalid_rate") or 0) >= 0.5 else "brand"),
        ("Display Track Mean", _fmt(render.get("display_track_count", {}).get("mean")), "화면에 실제로 보인 트랙 수", "warn"),
        ("Render P95", _fmt(render.get("capture_to_render_ms", {}).get("p95"), suffix=" ms"), "수집부터 표시까지 상위 95%", "brand"),
        ("Power Plan", system.get("power_plan_name") or "n/a", "실험/측정용으로는 High performance 권장", power_tone),
        ("Host IP Match", _yes_no_unknown(system.get("host_ip_present"), yes_text="match", no_text="mismatch"), f"expected={system.get('expected_host_ip') or 'n/a'}", host_ip_tone),
        ("Slowest Stage", preferred_stage.get("name") or "n/a", f"p95={_fmt(preferred_stage.get('p95_ms'), suffix=' ms')}", "warn"),
    ]


def _build_text_list_html(items: list[str], empty_message: str):
    if not items:
        return f'<p class="subtle">{empty_message}</p>'
    rows = "".join(f"<li>{item}</li>" for item in items)
    return f"<ul>{rows}</ul>"


def _build_steps_html(items: list[str], empty_message: str = "표시할 단계가 없습니다."):
    if not items:
        return f'<p class="subtle">{empty_message}</p>'
    rows = "".join(f"<li>{item}</li>" for item in items)
    return f'<ol class="steps">{rows}</ol>'


def _build_field_guide_rows(items: list[tuple[str, str, str]]):
    if not items:
        return '<tr><td colspan="3">표시할 필드 설명이 없습니다.</td></tr>'
    rows = []
    for field_name, meaning, interpretation in items:
        rows.append(
            f"""
            <tr>
              <td><code>{field_name}</code></td>
              <td>{meaning}</td>
              <td>{interpretation}</td>
            </tr>
            """
        )
    return "".join(rows)


def _mean_or_none(values):
    finite_values = [float(value) for value in values if isinstance(value, (int, float))]
    if not finite_values:
        return None
    return sum(finite_values) / len(finite_values)


def _format_counter(counter: Counter, limit: int = 2):
    items = [f"{key} x{count}" for key, count in counter.most_common(limit) if count > 0]
    return ", ".join(items) if items else "n/a"


def _frame_examples(frame_ids: list[int], limit: int = 6):
    unique_ids = sorted({int(frame_id) for frame_id in frame_ids if frame_id is not None})
    if not unique_ids:
        return "n/a"
    preview = ", ".join(str(frame_id) for frame_id in unique_ids[:limit])
    if len(unique_ids) > limit:
        preview += ", ..."
    return preview


def _longest_streak(frame_ids: list[int]):
    unique_ids = sorted({int(frame_id) for frame_id in frame_ids if frame_id is not None})
    if not unique_ids:
        return None
    best_start = best_end = current_start = unique_ids[0]
    best_length = current_length = 1
    previous = unique_ids[0]
    for frame_id in unique_ids[1:]:
        if frame_id == previous + 1:
            current_length += 1
        else:
            if current_length > best_length:
                best_start, best_end, best_length = current_start, previous, current_length
            current_start = frame_id
            current_length = 1
        previous = frame_id
    if current_length > best_length:
        best_start, best_end, best_length = current_start, previous, current_length
    return {"start": best_start, "end": best_end, "length": best_length}


def _format_streak(streak: dict | None):
    if not streak:
        return "n/a"
    if streak["length"] <= 1:
        return f"1 frame ({streak['start']})"
    return f"{streak['length']} frames ({streak['start']}-{streak['end']})"


def _classify_processed_row(row: dict):
    if row.get("invalid"):
        return "invalid_input"
    if int(row.get("confirmed_track_count", 0)) > 0:
        return "confirmed_tracking"
    if int(row.get("tentative_track_count", 0)) > 0:
        return "tentative_only"
    if int(row.get("candidate_count", 0)) > 0:
        return "candidate_without_track"
    return "empty_scene"


def _build_processed_pattern_rows(rows: list[dict]):
    if not rows:
        return '<tr><td colspan="7">표시할 processed frame이 없습니다.</td></tr>'

    labels = {
        "invalid_input": ("입력 이상 프레임", "invalid=true 또는 tracker policy degraded"),
        "confirmed_tracking": ("confirmed 추적 활성", "확정 track이 1개 이상 나온 정상 처리 구간"),
        "tentative_only": ("tentative만 존재", "후보는 있지만 아직 확정 track이 안 붙은 구간"),
        "candidate_without_track": ("후보만 있고 track 없음", "candidate는 있으나 tentative/confirmed로 이어지지 않은 구간"),
        "empty_scene": ("후보 없음", "candidate와 track이 모두 거의 없는 구간"),
    }
    grouped: dict[str, list[dict]] = {key: [] for key in labels}
    for row in rows:
        grouped[_classify_processed_row(row)].append(row)

    html_rows = []
    total_count = len(rows)
    for key in ("invalid_input", "confirmed_tracking", "tentative_only", "candidate_without_track", "empty_scene"):
        matched = grouped[key]
        if not matched:
            continue
        frame_ids = [row["frame_id"] for row in matched]
        latency_mean = _mean_or_none(row.get("capture_to_process_ms") for row in matched)
        candidate_mean = _mean_or_none(row.get("candidate_count") for row in matched)
        confirmed_mean = _mean_or_none(row.get("confirmed_track_count") for row in matched)
        tentative_mean = _mean_or_none(row.get("tentative_track_count") for row in matched)
        reason_counter = Counter((row.get("invalid_reason") or "none") for row in matched)
        policy_counter = Counter((row.get("tracker_policy") or "n/a") for row in matched)

        if key == "invalid_input":
            signal_text = f"reason={_format_counter(reason_counter)} | policy={_format_counter(policy_counter)}"
        elif key == "confirmed_tracking":
            signal_text = f"candidate≈{_fmt(candidate_mean)} | confirmed≈{_fmt(confirmed_mean)} | tentative≈{_fmt(tentative_mean)}"
        elif key == "tentative_only":
            signal_text = f"candidate≈{_fmt(candidate_mean)} | tentative≈{_fmt(tentative_mean)}"
        elif key == "candidate_without_track":
            signal_text = f"candidate≈{_fmt(candidate_mean)} | tracker_input≈{_fmt(_mean_or_none(row.get('tracker_input_count') for row in matched))}"
        else:
            signal_text = "scene가 비거나 threshold에 거의 걸리지 않은 구간"

        title, description = labels[key]
        html_rows.append(
            f"""
            <tr>
              <td><strong>{title}</strong><br /><span class="subtle">{description}</span></td>
              <td>{len(matched)}</td>
              <td>{_fmt(len(matched) / max(total_count, 1) * 100.0, suffix=' %')}</td>
              <td>{_format_streak(_longest_streak(frame_ids))}</td>
              <td>{_fmt(latency_mean, suffix=' ms')}</td>
              <td>{signal_text}</td>
              <td>{_frame_examples(frame_ids)}</td>
            </tr>
            """
        )
    return "".join(html_rows)


def _classify_render_row(row: dict):
    if int(row.get("skipped_render_frames", 0)) > 0:
        return "render_backlog"
    if row.get("invalid"):
        return "invalid_input"
    if int(row.get("display_track_count", 0)) >= 2:
        return "multi_display"
    if int(row.get("display_track_count", 0)) == 1:
        return "single_display"
    if int(row.get("tentative_display_track_count", 0)) > 0:
        return "tentative_only"
    if int(row.get("candidate_count", 0)) > 0:
        return "candidate_but_blank"
    return "empty_scene"


def _build_render_pattern_rows(rows: list[dict]):
    if not rows:
        return '<tr><td colspan="7">표시할 render frame이 없습니다.</td></tr>'

    labels = {
        "render_backlog": ("render backlog", "UI가 최신 processed frame을 따라가지 못한 구간"),
        "invalid_input": ("invalid 입력 표시", "화면에 보였지만 입력 무결성이 깨진 구간"),
        "multi_display": ("여러 confirmed track 표시", "동시에 2개 이상 display된 구간"),
        "single_display": ("1개 confirmed track 표시", "단일 타깃이 화면에 유지된 구간"),
        "tentative_only": ("tentative만 표시", "confirmed는 없지만 가설 track은 보인 구간"),
        "candidate_but_blank": ("후보는 있으나 화면 비어 있음", "candidate는 있었지만 표시 정책 때문에 화면에 안 남은 구간"),
        "empty_scene": ("후보 없음", "candidate와 표시가 모두 거의 없는 구간"),
    }
    grouped: dict[str, list[dict]] = {key: [] for key in labels}
    for row in rows:
        grouped[_classify_render_row(row)].append(row)

    html_rows = []
    total_count = len(rows)
    for key in ("render_backlog", "invalid_input", "multi_display", "single_display", "tentative_only", "candidate_but_blank", "empty_scene"):
        matched = grouped[key]
        if not matched:
            continue
        frame_ids = [row["frame_id"] for row in matched]
        latency_mean = _mean_or_none(row.get("capture_to_render_ms") for row in matched)
        display_mean = _mean_or_none(row.get("display_track_count") for row in matched)
        tentative_mean = _mean_or_none(row.get("tentative_display_track_count") for row in matched)
        status_counter = Counter((row.get("status_text") or "status unavailable") for row in matched)
        invalid_counter = Counter((row.get("invalid_reason") or "none") for row in matched)

        if key == "invalid_input":
            signal_text = f"reason={_format_counter(invalid_counter)} | status={_format_counter(status_counter, limit=1)}"
        elif key == "render_backlog":
            signal_text = f"skipped mean={_fmt(_mean_or_none(row.get('skipped_render_frames') for row in matched))} | status={_format_counter(status_counter, limit=1)}"
        else:
            signal_text = f"display≈{_fmt(display_mean)} | tentative≈{_fmt(tentative_mean)} | status={_format_counter(status_counter, limit=1)}"

        title, description = labels[key]
        html_rows.append(
            f"""
            <tr>
              <td><strong>{title}</strong><br /><span class="subtle">{description}</span></td>
              <td>{len(matched)}</td>
              <td>{_fmt(len(matched) / max(total_count, 1) * 100.0, suffix=' %')}</td>
              <td>{_format_streak(_longest_streak(frame_ids))}</td>
              <td>{_fmt(latency_mean, suffix=' ms')}</td>
              <td>{signal_text}</td>
              <td>{_frame_examples(frame_ids)}</td>
            </tr>
            """
        )
    return "".join(html_rows)


def _build_issue_table_rows(issues: list[dict]):
    if not issues:
        return '<tr><td colspan="3">표시할 이슈가 없습니다.</td></tr>'
    return "".join(
        f"""
        <tr>
          <td>{_pill(issue.get('severity', 'info').upper(), 'danger' if issue.get('severity') == 'high' else ('warn' if issue.get('severity') == 'medium' else 'brand'))}</td>
          <td><strong>{issue.get('title', 'n/a')}</strong></td>
          <td>{issue.get('detail', '')}</td>
        </tr>
        """
        for issue in issues
    )


def _ops_metric_fmt(value):
    if isinstance(value, bool):
        return "yes" if value else "no"
    return _fmt(value)


def _tone_label(tone: str):
    return {
        "good": "양호",
        "brand": "보통",
        "warn": "주의",
        "danger": "위험",
    }.get(tone, tone)


def _yes_no_unknown(value, *, yes_text="yes", no_text="no", unknown_text="n/a"):
    if value is True:
        return yes_text
    if value is False:
        return no_text
    return unknown_text


def _stage_timing_rows(stage_summary: dict):
    timings = (stage_summary or {}).get("timings") or {}
    if not timings:
        return '<tr><td colspan="5">stage timing 데이터가 없습니다.</td></tr>'
    rows = []
    for stage_name, stats in sorted(
        timings.items(),
        key=lambda item: (item[1].get("p95") is None, -(item[1].get("p95") or item[1].get("mean") or 0.0)),
    ):
        rows.append(
            f"""
            <tr>
              <td><code>{stage_name}</code></td>
              <td>{stats.get('count', 0)}</td>
              <td>{_fmt(stats.get('mean'), suffix=' ms')}</td>
              <td>{_fmt(stats.get('p50'), suffix=' ms')}</td>
              <td>{_fmt(stats.get('p95'), suffix=' ms')}</td>
            </tr>
            """
        )
    return "".join(rows)


def _system_snapshot_rows(system_summary: dict):
    if not system_summary.get("snapshot_present"):
        return '<tr><td colspan="2">system_snapshot.json이 없습니다.</td></tr>'

    thread_env = system_summary.get("thread_env") or {}
    thread_env_text = ", ".join(
        f"{key}={value}"
        for key, value in thread_env.items()
        if value not in (None, "")
    ) or "설정 없음"

    rows = [
        ("Power Plan", system_summary.get("power_plan_name") or "n/a"),
        (
            "Recommended For Benchmarking",
            _yes_no_unknown(
                system_summary.get("power_plan_recommended"),
                yes_text="yes",
                no_text="no",
            ),
        ),
        ("Process Priority", system_summary.get("process_priority_class") or "n/a"),
        ("Expected Host IP", system_summary.get("expected_host_ip") or "n/a"),
        (
            "Host IP Present",
            _yes_no_unknown(
                system_summary.get("host_ip_present"),
                yes_text="match",
                no_text="mismatch",
            ),
        ),
        ("IPv4 Addresses", ", ".join(system_summary.get("ipv4_addresses") or []) or "n/a"),
        ("Firewall Profiles", ", ".join(system_summary.get("enabled_firewall_profiles") or []) or "n/a"),
        ("Thread Env", thread_env_text),
    ]
    return "".join(
        f"""
        <tr>
          <td><strong>{label}</strong></td>
          <td>{value}</td>
        </tr>
        """
        for label, value in rows
    )


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
    assessment = summary.get("assessment", {})
    overall = assessment.get("overall", {})
    system = summary.get("system", {})
    session_id = summary.get("session_id", session_dir.name)
    event_note = "정상" if event_summary["first_rendered_frame"] else "first_rendered_frame 미확인"
    if event_summary["session_error"]:
        event_note = f"세션 오류: {event_summary['session_error_repr']}"
    system_file_row = ""
    if summary.get("log_files_present", {}).get("system_snapshot"):
        system_file_row = """
          <tr>
            <td><code>system_snapshot.json</code></td>
            <td>전원 모드, NIC/IP, 방화벽, 프로세스 priority 스냅샷</td>
            <td><a href="./system_snapshot.json">열기</a></td>
          </tr>
        """

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
        <a href="./ops_report.html">현업 평가 리포트</a>
        <a href="./processed_report.html">processed 리포트</a>
        <a href="./render_report.html">render 리포트</a>
        <a href="./event_report.html">event 리포트</a>
        <a href="../index.html">전체 비교 대시보드</a>
      </nav>
    </header>

    <section class="card">
      <h2>로그 해석 가이드</h2>
      <p class="subtle">필드 이름과 뜻을 바로 확인할 수 있도록 문서 링크를 같이 둡니다.</p>
      <div class="grid">
        <div class="metric"><div class="label">processed guide</div><div class="value"><a href="../../../docs/log_guides/processed_frames_guide.html">열기</a></div><div class="hint">처리단 필드와 stage timing 설명</div></div>
        <div class="metric"><div class="label">render guide</div><div class="value"><a href="../../../docs/log_guides/render_frames_guide.html">열기</a></div><div class="hint">화면 지연과 표시 정책 설명</div></div>
        <div class="metric"><div class="label">event guide</div><div class="value"><a href="../../../docs/log_guides/event_log_guide.html">열기</a></div><div class="hint">세션 시작/종료 이벤트 설명</div></div>
      </div>
    </section>

    <section class="card">
      <h2>세션 개요</h2>
      <p class="subtle">
        Variant: <code>{summary.get("session_meta", {}).get("variant") or "n/a"}</code> |
        Input: <code>{summary.get("session_meta", {}).get("input_mode") or "n/a"}</code> |
        상태: {_pill(health["label"], health["tone"])} |
        현업 점수: <strong>{overall.get("score", "n/a")}/100</strong> {_pill(overall.get("grade", "n/a"), overall.get("tone", "brand"))}
      </p>
      <div class="grid">{cards_html}</div>
    </section>

    <section class="card">
      <h2>빠른 해석</h2>
      <div class="note">
        <strong>이 세션의 핵심 상태:</strong> {health["detail"]}<br />
        <strong>현업 판정:</strong> {overall.get("summary", "평가 데이터가 없습니다.")}<br />
        <strong>Event 로그 관점:</strong> {event_note}<br />
        <strong>First render 지연:</strong> {_fmt(event_summary.get("first_render_elapsed_s"), digits=3, suffix=" s")} |
        <strong>세션 길이:</strong> {_fmt(event_summary.get("session_duration_s"), digits=3, suffix=" s")}<br />
        <strong>Power plan:</strong> {system.get("power_plan_name") or "n/a"} |
        <strong>Host IP match:</strong> {_yes_no_unknown(system.get("host_ip_present"), yes_text="match", no_text="mismatch")}
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
            <td><code>ops_report.html</code></td>
            <td>현업 기준 점수, 주요 문제, 권장 조치</td>
            <td><a href="./ops_report.html">열기</a></td>
          </tr>
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
          {system_file_row}
        </tbody>
      </table>
    </section>
  </div>
</body>
</html>"""


def _build_ops_html(session_dir: Path, summary: dict, event_summary: dict):
    assessment = summary.get("assessment", {})
    overall = assessment.get("overall", {})
    category_scores = assessment.get("category_scores", {})
    system = summary.get("system", {})
    preferred_stage_timings = summary.get("diagnostics", {}).get("preferred_stage_timings_ms") or {}
    preferred_slowest_stage = preferred_stage_timings.get("slowest_stage") or {}
    category_cards = "".join(
        f"""
        <div class="metric">
          <div class="label">{category.get('label', name)}</div>
          <div class="value">{category.get('score', 0)}/{category.get('max_score', 0)}</div>
          <div class="hint">
            {_pill(_tone_label(category.get('tone', 'brand')), category.get('tone', 'brand'))}
            {' | '.join(f"{metric_key}={_ops_metric_fmt(metric_value)}" for metric_key, metric_value in category.get('metrics', {}).items())}
          </div>
        </div>
        """
        for name, category in category_scores.items()
    )
    issue_rows = _build_issue_table_rows(assessment.get("issues", []))
    strengths_html = _build_text_list_html(
        assessment.get("strengths", []),
        "강점으로 자동 추출된 항목이 없습니다.",
    )
    recommendations_html = _build_text_list_html(
        assessment.get("recommendations", []),
        "추가 권장 조치가 없습니다.",
    )
    derived = assessment.get("derived_metrics", {})
    stage_rows = _stage_timing_rows(preferred_stage_timings)
    system_rows = _system_snapshot_rows(system)

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{session_dir.name} operational assessment</title>
  <style>{COMMON_STYLE}</style>
</head>
<body>
  <div class="wrap">
    <header class="hero">
      <h1>{session_dir.name} 현업 평가 리포트</h1>
      <p>최근 세션 로그를 기준으로 운영 적합도를 100점 만점으로 환산한 요약입니다. 절대적인 인증 점수라기보다, 현장 투입 전 품질 판단과 세션 간 비교를 돕기 위한 내부 루브릭입니다.</p>
      <nav class="nav">
        <a href="./index.html">세션 개요</a>
        <a href="./processed_report.html">processed 리포트</a>
        <a href="./render_report.html">render 리포트</a>
        <a href="./event_report.html">event 리포트</a>
        <a href="../../../docs/problem/environment_checklist.html">환경 체크리스트</a>
        <a href="../../../docs/problem/retest_plan.html">재실험 계획서</a>
        <a href="../index.html">전체 비교 대시보드</a>
      </nav>
    </header>

    <section class="card">
      <h2>종합 판정</h2>
      <div class="grid">
        <div class="metric">
          <div class="label">Operational Score</div>
          <div class="value">{overall.get('score', 'n/a')}/100</div>
          <div class="hint">{overall.get('summary', '평가 데이터가 없습니다.')}</div>
        </div>
        <div class="metric">
          <div class="label">Grade</div>
          <div class="value">{overall.get('grade', 'n/a')}</div>
          <div class="hint">{overall.get('label', 'n/a')}</div>
        </div>
        <div class="metric">
          <div class="label">First Render</div>
          <div class="value">{_fmt(event_summary.get('first_render_elapsed_s'), digits=3, suffix=' s')}</div>
          <div class="hint">세션 기동 응답성</div>
        </div>
        <div class="metric">
          <div class="label">Display/Confirmed Ratio</div>
          <div class="value">{_fmt(derived.get('display_to_confirmed_ratio'))}</div>
          <div class="hint">화면 표시 트랙 대비 내부 confirmed track 비율</div>
        </div>
        <div class="metric">
          <div class="label">Slowest Stage</div>
          <div class="value">{preferred_slowest_stage.get('name', 'n/a')}</div>
          <div class="hint">source={preferred_stage_timings.get('source', 'n/a')} | p95={_fmt(preferred_slowest_stage.get('p95_ms'), suffix=' ms')}</div>
        </div>
        <div class="metric">
          <div class="label">Power Plan</div>
          <div class="value">{system.get('power_plan_name') or 'n/a'}</div>
          <div class="hint">recommended={_yes_no_unknown(system.get('power_plan_recommended'))}</div>
        </div>
      </div>
    </section>

    <section class="card">
      <h2>카테고리 점수</h2>
      <div class="grid">{category_cards}</div>
    </section>

    <section class="card">
      <h2>처리 Stage Timing</h2>
      <p class="subtle">
        source=<code>{preferred_stage_timings.get('source', 'n/a')}</code> |
        frames with timings=<code>{preferred_stage_timings.get('frame_count_with_timings', 0)}</code>
      </p>
      <table>
        <thead>
          <tr>
            <th>Stage</th>
            <th>Frames</th>
            <th>Mean</th>
            <th>P50</th>
            <th>P95</th>
          </tr>
        </thead>
        <tbody>{stage_rows}</tbody>
      </table>
    </section>

    <section class="card">
      <h2>실행 환경 Snapshot</h2>
      <table>
        <thead>
          <tr>
            <th>항목</th>
            <th>값</th>
          </tr>
        </thead>
        <tbody>{system_rows}</tbody>
      </table>
    </section>

    <section class="card">
      <h2>주요 문제</h2>
      <table>
        <thead>
          <tr>
            <th>심각도</th>
            <th>문제</th>
            <th>근거</th>
          </tr>
        </thead>
        <tbody>{issue_rows}</tbody>
      </table>
    </section>

    <section class="card">
      <h2>강점</h2>
      {strengths_html}
    </section>

    <section class="card">
      <h2>권장 조치</h2>
      {recommendations_html}
    </section>
  </div>
</body>
</html>"""


def _build_processed_html(session_dir: Path, summary: dict, processed_records: list[dict]):
    simplified = _simplify_processed_records(processed_records)
    trajectory = _build_track_trajectory_bundle(processed_records, ["confirmed_tracks", "tentative_tracks"])
    pattern_rows = _build_processed_pattern_rows(processed_records)

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
        },
        "trajectory": trajectory,
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

      renderTrajectoryChart(
        'processed-trajectory-chart',
        REPORT_DATA.trajectory.series,
        {{ title: 'processed trajectory', emptyMessage: REPORT_DATA.trajectory.empty_message }}
      );
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
        <a href="../../../docs/log_guides/processed_frames_guide.html">processed 가이드</a>
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
      <h2>레이더 기준 궤적</h2>
      <p class="subtle">{_trajectory_summary_text(trajectory)}</p>
      <div id="processed-trajectory-chart" class="chart"></div>
    </section>

    <section class="card">
      <h2>필드별 해석법</h2>
      <p class="subtle">docs/log_guides의 processed 가이드를 리포트 안에서 바로 볼 수 있게 다시 넣은 표입니다.</p>
      <table>
        <thead>
          <tr>
            <th>필드</th>
            <th>의미</th>
            <th>실무 해석</th>
          </tr>
        </thead>
        <tbody>{_build_field_guide_rows(PROCESSED_FIELD_GUIDE)}</tbody>
      </table>
    </section>

    <section class="card">
      <h2>이 리포트 읽는 순서</h2>
      {_build_steps_html(PROCESSED_READING_STEPS)}
    </section>

    <section class="card">
      <h2>유사 프레임 패턴 묶음</h2>
      <p class="subtle">최근 10프레임 대신, 비슷한 상태의 프레임을 세션 전체에서 묶어 경향을 보여줍니다.</p>
      <table>
        <thead>
          <tr>
            <th>패턴</th><th>프레임 수</th><th>비율</th><th>최장 연속 구간</th><th>평균 지연</th><th>대표 신호</th><th>예시 frame</th>
          </tr>
        </thead>
        <tbody>{pattern_rows}</tbody>
      </table>
    </section>
    {script}
  </div>
</body>
</html>"""


def _build_render_html(session_dir: Path, summary: dict, render_records: list[dict]):
    simplified = _simplify_render_records(render_records)
    trajectory = _build_track_trajectory_bundle(
        render_records,
        ["display_tracks", "tentative_display_tracks", "tentative_tracks"],
    )
    pattern_rows = _build_render_pattern_rows(render_records)

    payload = {
        "series": {
            "frameIds": [row["frame_id"] for row in simplified],
            "captureToRenderMs": [row["capture_to_render_ms"] for row in simplified],
            "processToRenderMs": [row["process_to_render_ms"] for row in simplified],
            "candidateCount": [row["candidate_count"] for row in simplified],
            "displayTrackCount": [row["display_track_count"] for row in simplified],
            "tentativeDisplayTrackCount": [row["tentative_display_track_count"] for row in simplified],
            "leadAngleDeg": [row["lead_angle_deg"] for row in simplified],
        },
        "trajectory": trajectory,
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

      renderTrajectoryChart(
        'render-trajectory-chart',
        REPORT_DATA.trajectory.series,
        {{ title: 'render trajectory', emptyMessage: REPORT_DATA.trajectory.empty_message }}
      );
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
        <a href="../../../docs/log_guides/render_frames_guide.html">render 가이드</a>
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
      <h2>레이더 기준 궤적</h2>
      <p class="subtle">{_trajectory_summary_text(trajectory)}</p>
      <div id="render-trajectory-chart" class="chart"></div>
    </section>

    <section class="card">
      <h2>필드별 해석법</h2>
      <p class="subtle">render 로그에서 자주 보는 값들을 이름 그대로 정리한 표입니다.</p>
      <table>
        <thead>
          <tr>
            <th>필드</th>
            <th>의미</th>
            <th>실무 해석</th>
          </tr>
        </thead>
        <tbody>{_build_field_guide_rows(RENDER_FIELD_GUIDE)}</tbody>
      </table>
    </section>

    <section class="card">
      <h2>이 리포트 읽는 순서</h2>
      {_build_steps_html(RENDER_READING_STEPS)}
    </section>

    <section class="card">
      <h2>유사 프레임 패턴 묶음</h2>
      <p class="subtle">세션 전체를 대상으로 비슷한 화면 상태를 묶어서, 무엇이 자주 반복됐는지 보여줍니다.</p>
      <table>
        <thead>
          <tr>
            <th>패턴</th><th>프레임 수</th><th>비율</th><th>최장 연속 구간</th><th>평균 지연</th><th>대표 신호</th><th>예시 frame</th>
          </tr>
        </thead>
        <tbody>{pattern_rows}</tbody>
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
        <a href="../../../docs/log_guides/event_log_guide.html">event 가이드</a>
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

    <section class="card">
      <h2>필드와 이벤트 해석</h2>
      <p class="subtle">세션 수명주기에서 자주 보는 이벤트 이름을 그대로 남긴 표입니다.</p>
      <table>
        <thead>
          <tr>
            <th>이벤트</th>
            <th>뜻</th>
            <th>이벤트가 없거나 실패하면 의심할 것</th>
          </tr>
        </thead>
        <tbody>{_build_field_guide_rows(EVENT_FIELD_GUIDE)}</tbody>
      </table>
    </section>

    <section class="card">
      <h2>이 리포트 읽는 순서</h2>
      {_build_steps_html(EVENT_READING_STEPS)}
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
        event_summary = summary.get("event") or _event_summary(events)
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
                "trajectory": _load_session_trajectory_bundle(session_dir),
                "links": {
                    "index": f"./{session_dir.name}/index.html",
                    "ops": f"./{session_dir.name}/ops_report.html",
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
          <td><strong>{row['summary'].get('assessment', {}).get('overall', {}).get('score', 'n/a')}</strong></td>
          <td>{_pill(row['summary'].get('assessment', {}).get('overall', {}).get('grade', 'n/a'), row['summary'].get('assessment', {}).get('overall', {}).get('tone', 'brand'))}</td>
          <td>{_pill(row['health']['label'], row['health']['tone'])}</td>
          <td>{_fmt_pct(row['summary']['render']['invalid_rate'])}</td>
          <td>{_fmt(row['summary']['render']['display_track_count']['mean'])}</td>
          <td>{_fmt(row['summary']['render']['capture_to_render_ms']['p95'], suffix=' ms')}</td>
          <td>
            <a href="{row['links']['index']}">개요</a> |
            <a href="{row['links']['ops']}">ops</a> |
            <a href="{row['links']['processed']}">processed</a> |
            <a href="{row['links']['render']}">render</a> |
            <a href="{row['links']['event']}">event</a>
          </td>
        </tr>
        """
        for row in session_rows
    ) or '<tr><td colspan="11">표시할 세션이 없습니다.</td></tr>'

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
        const beforeTrajectory = beforeSession.trajectory || {{ series: [], empty_message: 'before trajectory unavailable' }};
        const afterTrajectory = afterSession.trajectory || {{ series: [], empty_message: 'after trajectory unavailable' }};

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

        document.getElementById('before-trajectory-title').textContent = `before | ${{beforeSession.session_id}}`;
        document.getElementById('after-trajectory-title').textContent = `after | ${{afterSession.session_id}}`;
        document.getElementById('before-trajectory-note').textContent =
          `priority=${{beforeTrajectory.priority_label || 'n/a'}} | source=${{beforeTrajectory.source_label || 'n/a'}} | tracks=${{beforeTrajectory.track_count ?? 0}}`;
        document.getElementById('after-trajectory-note').textContent =
          `priority=${{afterTrajectory.priority_label || 'n/a'}} | source=${{afterTrajectory.source_label || 'n/a'}} | tracks=${{afterTrajectory.track_count ?? 0}}`;
        renderTrajectoryChart('before-trajectory-chart', beforeTrajectory.series, {{
          title: 'before trajectory',
          emptyMessage: beforeTrajectory.empty_message
        }});
        renderTrajectoryChart('after-trajectory-chart', afterTrajectory.series, {{
          title: 'after trajectory',
          emptyMessage: afterTrajectory.empty_message
        }});
      }}

      function initDashboard() {{
        const beforeSelect = document.getElementById('before-session');
        const afterSelect = document.getElementById('after-session');
        beforeSelect.innerHTML = '';
        afterSelect.innerHTML = '';
        DASHBOARD.sessions.forEach((session) => {{
          const score = nestedGet(session.summary, 'assessment.overall.score');
          const beforeOption = document.createElement('option');
          beforeOption.value = session.session_id;
          beforeOption.textContent = `${{session.session_id}} | score=${{score ?? 'n/a'}} | ${{session.variant || 'n/a'}}`;
          beforeSelect.appendChild(beforeOption);

          const afterOption = document.createElement('option');
          afterOption.value = session.session_id;
          afterOption.textContent = `${{session.session_id}} | score=${{score ?? 'n/a'}} | ${{session.variant || 'n/a'}}`;
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
      <h2>로그 가이드 바로가기</h2>
      <div class="grid">
        <div class="metric"><div class="label">processed</div><div class="value"><a href="../../docs/log_guides/processed_frames_guide.html">가이드</a></div><div class="hint">candidate, invalid, stage timing 해석</div></div>
        <div class="metric"><div class="label">render</div><div class="value"><a href="../../docs/log_guides/render_frames_guide.html">가이드</a></div><div class="hint">capture_to_render, display 정책 해석</div></div>
        <div class="metric"><div class="label">event</div><div class="value"><a href="../../docs/log_guides/event_log_guide.html">가이드</a></div><div class="hint">세션 시작/종료, system snapshot 해석</div></div>
      </div>
    </section>

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
      <h2>레이더 기준 궤적 비교</h2>
      <div class="grid">
        <div class="chart-card">
          <h3 id="before-trajectory-title">before</h3>
          <div id="before-trajectory-chart" class="chart"></div>
          <p id="before-trajectory-note" class="subtle"></p>
        </div>
        <div class="chart-card">
          <h3 id="after-trajectory-title">after</h3>
          <div id="after-trajectory-chart" class="chart"></div>
          <p id="after-trajectory-note" class="subtle"></p>
        </div>
      </div>
    </section>

    <section class="card">
      <h2>세션 목록</h2>
      <table>
        <thead>
          <tr>
            <th>세션</th><th>생성 시각</th><th>variant</th><th>scenario</th><th>점수</th><th>등급</th><th>상태</th><th>render invalid</th><th>display mean</th><th>render p95</th><th>리포트</th>
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
    before_trajectory: dict,
    after_trajectory: dict,
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

    payload = {
        "beforeTrajectory": before_trajectory,
        "afterTrajectory": after_trajectory,
    }
    script = f"""
    <script>
      {COMMON_SCRIPT}
      const COMPARISON_DATA = {json.dumps(payload, ensure_ascii=False)};
      renderTrajectoryChart('static-before-trajectory-chart', COMPARISON_DATA.beforeTrajectory.series, {{
        title: 'before trajectory',
        emptyMessage: COMPARISON_DATA.beforeTrajectory.empty_message
      }});
      renderTrajectoryChart('static-after-trajectory-chart', COMPARISON_DATA.afterTrajectory.series, {{
        title: 'after trajectory',
        emptyMessage: COMPARISON_DATA.afterTrajectory.empty_message
      }});
    </script>
    """

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
      <h2>레이더 기준 궤적 비교</h2>
      <div class="grid">
        <div class="chart-card">
          <h3>before | {before_session}</h3>
          <p class="subtle">{_trajectory_summary_text(before_trajectory)}</p>
          <div id="static-before-trajectory-chart" class="chart"></div>
        </div>
        <div class="chart-card">
          <h3>after | {after_session}</h3>
          <p class="subtle">{_trajectory_summary_text(after_trajectory)}</p>
          <div id="static-after-trajectory-chart" class="chart"></div>
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
    {script}
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
    before_trajectory = _load_session_trajectory_bundle(before_summary_path.parent)
    after_trajectory = _load_session_trajectory_bundle(after_summary_path.parent)

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
            before_trajectory,
            after_trajectory,
        ),
    )
    return output_path


def generate_session_artifacts(session_dir: str | Path):
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(session_dir)
    _write_json(session_dir / "summary.json", summary)

    processed_records = _load_jsonl(session_dir / "processed_frames.jsonl")
    render_records = _load_render_records_with_fallback(session_dir)
    events = _load_jsonl(session_dir / "event_log.jsonl")
    event_summary = summary.get("event") or _event_summary(events)

    _write_text(session_dir / "index.html", _build_session_index_html(session_dir, summary, event_summary))
    _write_text(session_dir / "ops_report.html", _build_ops_html(session_dir, summary, event_summary))
    _write_text(session_dir / "processed_report.html", _build_processed_html(session_dir, summary, processed_records))
    _write_text(session_dir / "render_report.html", _build_render_html(session_dir, summary, render_records))
    _write_text(session_dir / "event_report.html", _build_event_html(session_dir, events, event_summary))

    return {
        "summary_path": session_dir / "summary.json",
        "index_path": session_dir / "index.html",
        "ops_report_path": session_dir / "ops_report.html",
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
