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
  .replay-shell { display: grid; gap: 16px; }
  .replay-stage {
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 16px;
    background: var(--panel-soft);
  }
  .replay-controls {
    display: grid;
    grid-template-columns: auto minmax(220px, 1fr) auto auto auto;
    gap: 12px;
    align-items: center;
  }
  .replay-controls button,
  .replay-controls select {
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 10px 12px;
    font: inherit;
    background: #fff;
    color: var(--text);
  }
  .replay-controls input[type="range"] { width: 100%; }
  .replay-frame-label {
    color: var(--muted);
    font-size: 0.92rem;
    font-weight: 700;
    white-space: nowrap;
  }
  .replay-meta-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
  }
  .replay-meta-card {
    border: 1px solid var(--line);
    border-radius: 14px;
    background: #fff;
    padding: 12px 14px;
  }
  .replay-meta-card .label {
    color: var(--muted);
    font-size: 0.86rem;
    margin-bottom: 6px;
  }
  .replay-meta-card .value {
    font-size: 1.12rem;
    font-weight: 700;
    line-height: 1.2;
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
  const breakFrameGap = Number.isFinite(Number(options.breakFrameGap)) ? Number(options.breakFrameGap) : 2;
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
    const segments = [];
    let currentSegment = [];
    validPoints.forEach((point, pointIndex) => {
      if (pointIndex > 0) {
        const previousPoint = validPoints[pointIndex - 1];
        const frameGap = Number(point.frame_id) - Number(previousPoint.frame_id);
        if (Number.isFinite(frameGap) && frameGap > breakFrameGap) {
          if (currentSegment.length) segments.push(currentSegment);
          currentSegment = [];
        }
      }
      currentSegment.push(point);
    });
    if (currentSegment.length) segments.push(currentSegment);
    const start = validPoints[0];
    const end = validPoints[validPoints.length - 1];
    segments.forEach((segment) => {
      if (segment.length < 2) return;
      const polyline = segment.map((point) => `${xFor(point.x_m)},${yFor(point.y_m)}`).join(" ");
      svg += `<polyline fill="none" stroke="${color}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round" points="${polyline}"></polyline>`;
    });
    svg += `<circle cx="${xFor(start.x_m)}" cy="${yFor(start.y_m)}" r="4.5" fill="#ffffff" stroke="${color}" stroke-width="2"></circle>`;
    svg += `<circle cx="${xFor(end.x_m)}" cy="${yFor(end.y_m)}" r="5.2" fill="${color}" stroke="#ffffff" stroke-width="2"></circle>`;
  });

  svg += `<circle cx="${xFor(0)}" cy="${yFor(0)}" r="6" fill="#0f172a"></circle>`;
  svg += `<text x="${xFor(0) + 10}" y="${yFor(0) - 8}" fill="#142033" font-size="12">radar</text>`;
  svg += `<text x="${padding.left}" y="${height - 4}" fill="#5f6f86" font-size="12">x: radar-left (-) / radar-right (+) (m)</text>`;
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

function renderTrajectoryReplay(targetId, playback, options = {}) {
  const target = document.getElementById(targetId);
  if (!target) return;
  const frames = Array.isArray(playback?.frames) ? playback.frames : [];
  const seriesList = Array.isArray(playback?.series) ? playback.series : [];
  const breakFrameGap = Number.isFinite(Number(playback?.gap_break_frames))
    ? Number(playback.gap_break_frames)
    : 2;
  if (!frames.length) {
    target.innerHTML = `<div class="empty">${esc(playback?.empty_message || options.emptyMessage || '재생할 프레임 데이터가 없습니다.')}</div>`;
    return;
  }

  const width = options.width || 760;
  const height = options.height || 520;
  const padding = { top: 18, right: 24, bottom: 40, left: 44 };
  const palette = ['#0f6cbd', '#ef4444', '#10b981', '#f97316', '#8b5cf6', '#14b8a6'];
  const frameDelayMs = { '0.5': 700, '1': 380, '2': 190, '4': 95 };

  const allPoints = [];
  frames.forEach((frame) => {
    (frame.tracks || []).forEach((track) => {
      const x = Number(track.x_m);
      const y = Number(track.y_m);
      if (Number.isFinite(x) && Number.isFinite(y)) allPoints.push({ x, y });
    });
  });
  if (!allPoints.length) {
    target.innerHTML = `<div class="empty">${esc(playback?.empty_message || options.emptyMessage || '재생할 좌표 데이터가 없습니다.')}</div>`;
    return;
  }

  let maxAbsX = Math.max(...allPoints.map((point) => Math.abs(point.x)), 0.5);
  let minY = Math.min(...allPoints.map((point) => point.y), 0);
  let maxY = Math.max(...allPoints.map((point) => point.y), 0.5);
  maxAbsX = Math.max(0.5, maxAbsX * 1.15);
  minY = Math.min(0, minY);
  maxY = Math.max(0.5, maxY * 1.12);
  if (Math.abs(maxY - minY) < 0.5) maxY = minY + 0.5;

  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const xFor = (x) => padding.left + ((Number(x) + maxAbsX) / (maxAbsX * 2)) * plotWidth;
  const yFor = (y) => padding.top + (1 - ((Number(y) - minY) / (maxY - minY))) * plotHeight;

  const seriesMeta = new Map();
  (seriesList || []).forEach((series, index) => {
    const trackId = String(series.track_id);
    seriesMeta.set(trackId, {
      color: series.color || palette[index % palette.length],
      label: series.label || `track ${trackId}`,
    });
  });
  frames.forEach((frame) => {
    (frame.tracks || []).forEach((track) => {
      const trackId = String(track.track_id);
      if (!seriesMeta.has(trackId)) {
        const color = palette[seriesMeta.size % palette.length];
        seriesMeta.set(trackId, { color, label: track.label || `track ${trackId}` });
      }
    });
  });

  const preparedFrames = frames.map((frame) => {
    const trackMap = new Map();
    (frame.tracks || []).forEach((track) => {
      trackMap.set(String(track.track_id), track);
    });
    return { ...frame, trackMap };
  });

  target.innerHTML = `
    <div class="replay-shell">
      <div class="replay-stage">
        <div id="${targetId}-chart" class="chart"></div>
      </div>
      <div class="replay-controls">
        <button type="button" id="${targetId}-play">재생</button>
        <input type="range" id="${targetId}-slider" min="0" max="${Math.max(preparedFrames.length - 1, 0)}" step="1" value="0" />
        <span class="replay-frame-label" id="${targetId}-frame-label"></span>
        <select id="${targetId}-trail">
          <option value="12">최근 12프레임</option>
          <option value="24" selected>최근 24프레임</option>
          <option value="48">최근 48프레임</option>
          <option value="9999">전체</option>
        </select>
        <select id="${targetId}-speed">
          <option value="0.5">0.5x</option>
          <option value="1" selected>1x</option>
          <option value="2">2x</option>
          <option value="4">4x</option>
        </select>
      </div>
      <div id="${targetId}-meta" class="replay-meta-grid"></div>
      <p class="subtle" style="margin:0;">최근 N프레임 궤적과 현재 위치를 함께 보여 줍니다. render 기준은 실제 화면에 보인 결과, processed 기준은 내부 detection/tracker 상태를 의미합니다.</p>
    </div>
  `;

  const chartEl = document.getElementById(`${targetId}-chart`);
  const sliderEl = document.getElementById(`${targetId}-slider`);
  const playButtonEl = document.getElementById(`${targetId}-play`);
  const frameLabelEl = document.getElementById(`${targetId}-frame-label`);
  const trailEl = document.getElementById(`${targetId}-trail`);
  const speedEl = document.getElementById(`${targetId}-speed`);
  const metaEl = document.getElementById(`${targetId}-meta`);

  let activeIndex = 0;
  let timer = null;

  function stopPlayback() {
    if (timer !== null) {
      window.clearTimeout(timer);
      timer = null;
    }
    playButtonEl.textContent = '재생';
  }

  function trackSegmentsForFrame(trackId, frameIndex, trailWindow) {
    const segments = [];
    let current = [];
    const startIndex = Math.max(0, frameIndex - trailWindow + 1);
    for (let index = startIndex; index <= frameIndex; index += 1) {
      const point = preparedFrames[index].trackMap.get(trackId);
      if (!point) {
        if (current.length) {
          segments.push(current);
          current = [];
        }
        continue;
      }
      if (current.length) {
        const previousPoint = current[current.length - 1];
        const frameGap = Number(point.frame_id) - Number(previousPoint.frame_id);
        if (Number.isFinite(frameGap) && frameGap > breakFrameGap) {
          segments.push(current);
          current = [];
        }
      }
      current.push(point);
    }
    if (current.length) segments.push(current);
    return segments;
  }

  function renderActiveFrame() {
    const frame = preparedFrames[activeIndex];
    const trailWindow = Number(trailEl.value) || 24;
    let svg = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(options.title || 'trajectory replay')}">`;
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

    Array.from(seriesMeta.entries()).forEach(([trackId, meta]) => {
      const segments = trackSegmentsForFrame(trackId, activeIndex, trailWindow);
      segments.forEach((segment) => {
        if (segment.length < 2) return;
        const polyline = segment.map((point) => `${xFor(point.x_m)},${yFor(point.y_m)}`).join(' ');
        svg += `<polyline fill="none" stroke="${meta.color}" stroke-width="2.7" stroke-linejoin="round" stroke-linecap="round" opacity="0.92" points="${polyline}"></polyline>`;
      });
      const currentPoint = frame.trackMap.get(trackId);
      if (currentPoint) {
        svg += `<circle cx="${xFor(currentPoint.x_m)}" cy="${yFor(currentPoint.y_m)}" r="6.4" fill="${meta.color}" stroke="#ffffff" stroke-width="2.2"></circle>`;
      }
    });

    svg += `<circle cx="${xFor(0)}" cy="${yFor(0)}" r="6" fill="#0f172a"></circle>`;
    svg += `<text x="${xFor(0) + 10}" y="${yFor(0) - 8}" fill="#142033" font-size="12">radar</text>`;
    svg += `<text x="${padding.left}" y="${height - 4}" fill="#5f6f86" font-size="12">x: radar-left (-) / radar-right (+) (m)</text>`;
    svg += `<text x="${width - padding.right}" y="${padding.top + 12}" text-anchor="end" fill="#5f6f86" font-size="12">y: forward (m)</text>`;
    svg += `</svg>`;

    const legend = Array.from(seriesMeta.entries()).map(([trackId, meta]) => (
      `<span><i style="background:${meta.color}"></i>${esc(meta.label)}</span>`
    )).join('');
    chartEl.innerHTML = svg + `<div class="legend">${legend}</div>`;

    const metrics = [
      ['frame', `#${frame.frame_id}`],
      ['visible track', String((frame.tracks || []).length)],
      ['invalid', frame.invalid ? 'yes' : 'no'],
      ['latency', frame.capture_latency_ms === null || frame.capture_latency_ms === undefined ? 'n/a' : `${fmt(frame.capture_latency_ms, 1)} ms`],
      [playback.count_label || 'track', frame.candidate_count === null || frame.candidate_count === undefined ? 'n/a' : String(frame.candidate_count)],
      ['status', frame.status_text || frame.tracker_policy || 'n/a'],
    ];
    metaEl.innerHTML = metrics.map(([label, value]) => (
      `<div class="replay-meta-card"><div class="label">${esc(label)}</div><div class="value">${esc(String(value))}</div></div>`
    )).join('');
    frameLabelEl.textContent = `${activeIndex + 1} / ${preparedFrames.length}`;
    sliderEl.value = String(activeIndex);
  }

  function stepPlayback() {
    if (activeIndex >= preparedFrames.length - 1) {
      stopPlayback();
      return;
    }
    activeIndex += 1;
    renderActiveFrame();
    timer = window.setTimeout(stepPlayback, frameDelayMs[speedEl.value] || 380);
  }

  playButtonEl.addEventListener('click', () => {
    if (timer !== null) {
      stopPlayback();
      return;
    }
    playButtonEl.textContent = '일시정지';
    timer = window.setTimeout(stepPlayback, frameDelayMs[speedEl.value] || 380);
  });
  sliderEl.addEventListener('input', () => {
    activeIndex = Number(sliderEl.value) || 0;
    stopPlayback();
    renderActiveFrame();
  });
  trailEl.addEventListener('change', () => {
    renderActiveFrame();
  });
  speedEl.addEventListener('change', () => {
    if (timer !== null) {
      stopPlayback();
      playButtonEl.textContent = '재생';
    }
  });

  renderActiveFrame();
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
    "postprocessed": "log postprocess",
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


def _extract_record_count(record: dict, count_key: str | None = None, fallback_key: str | None = None):
    if count_key:
        value = record.get(count_key)
        if isinstance(value, (int, float)):
            return int(value)
    if fallback_key:
        fallback = record.get(fallback_key)
        if isinstance(fallback, list):
            return len(fallback)
        if isinstance(fallback, (int, float)):
            return int(fallback)
    return None


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


def _transport_quality(summary: dict):
    quality = summary.get("transport_quality")
    if isinstance(quality, dict) and quality.get("label"):
        return quality
    return {
        "category": "insufficient",
        "label": "data 부족",
        "tone": "warn",
        "suitability": "판정 불가",
        "detail": "transport quality 정보가 아직 생성되지 않았습니다.",
    }


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
        "is_primary": bool(item.get("is_primary")),
    }


def _longest_contiguous_run(points: list[dict], max_gap_frames: int = 2) -> int:
    if not points:
        return 0
    longest = 1
    current = 1
    for previous, current_point in zip(points, points[1:]):
        frame_gap = int(current_point["frame_id"]) - int(previous["frame_id"])
        if frame_gap <= max_gap_frames:
            current += 1
        else:
            longest = max(longest, current)
            current = 1
    return max(longest, current)


def _build_track_trajectory_bundle(
    records: list[dict],
    priority_keys: list[str],
    *,
    max_tracks: int = 4,
    min_points: int = 2,
    lead_only: bool = False,
    gap_break_frames: int = 2,
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
        sorted(grouped.items(), key=lambda entry: entry[0])
    ):
        points.sort(key=lambda point: point["frame_id"])
        if len(points) < min_points:
            continue
        sampled_points = _downsample_points(points)
        primary_hits = sum(1 for point in points if point.get("is_primary"))
        longest_run = _longest_contiguous_run(points, max_gap_frames=gap_break_frames)
        series.append(
            {
                "track_id": track_id,
                "label": f"track {track_id}",
                "color": TRAJECTORY_COLORS[index % len(TRAJECTORY_COLORS)],
                "point_count": len(sampled_points),
                "full_point_count": len(points),
                "primary_hits": primary_hits,
                "longest_run": longest_run,
                "points": sampled_points,
            }
        )

    series.sort(
        key=lambda item: (
            -int(item.get("primary_hits", 0)),
            -int(item.get("longest_run", 0)),
            -int(item.get("full_point_count", 0)),
            item.get("track_id", ""),
        )
    )

    if lead_only and series:
        lead_series = dict(series[0])
        lead_series["label"] = f"lead track {lead_series['track_id']}"
        series = [lead_series]
    else:
        for index, item in enumerate(series):
            item["color"] = TRAJECTORY_COLORS[index % len(TRAJECTORY_COLORS)]
        series = series[:max_tracks]

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
                    "full_point_count": len(detection_points),
                    "primary_hits": 0,
                    "longest_run": _longest_contiguous_run(detection_points, max_gap_frames=gap_break_frames),
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
        "lead_only": bool(lead_only),
        "gap_break_frames": int(gap_break_frames),
        "empty_message": empty_message,
    }


def _load_session_trajectory_bundle(session_dir: Path):
    render_bundle = _build_track_trajectory_bundle(
        _load_render_records_with_fallback(session_dir),
        ["display_tracks", "tentative_display_tracks", "tentative_tracks"],
        lead_only=True,
    )
    if render_bundle["series"]:
        return render_bundle
    return _build_track_trajectory_bundle(
        _load_jsonl(session_dir / "processed_frames.jsonl"),
        ["confirmed_tracks", "tentative_tracks"],
        lead_only=True,
    )


def _playback_point(item: dict, frame_id: int):
    point = _trajectory_point(item, frame_id)
    if point is None:
        return None
    point["track_id"] = str(item.get("track_id"))
    point["label"] = f"track {point['track_id']}"
    point["confidence"] = _round_or_none(item.get("confidence"), digits=4)
    point["score"] = _round_or_none(item.get("score"), digits=4)
    return point


def _track_item_rank(item: dict):
    return (
        1 if item.get("is_primary") else 0,
        float(item.get("confidence", 0.0) or 0.0),
        float(item.get("score", 0.0) or 0.0),
        -abs(float(item.get("x_m", 0.0) or 0.0)),
    )


def _select_lead_point_from_record(record: dict, priority_keys: list[str]):
    frame_id = int(record.get("frame_id", record.get("frame_index", 0)) or 0)
    for key in priority_keys:
        items = record.get(key) or []
        if not items:
            continue
        if key == "detections":
            lead = max(items, key=lambda item: float(item.get("score", 0.0) or 0.0))
            point = _playback_point({**lead, "track_id": "lead"}, frame_id)
            if point is not None:
                point["source_key"] = key
                return point
            continue
        lead = max(items, key=_track_item_rank)
        point = _playback_point(lead, frame_id)
        if point is not None:
            point["source_key"] = key
            return point
    return None


def _interpolate_postprocessed_points(points: list[dict], max_gap_frames: int = 6):
    if not points:
        return []
    merged = [dict(points[0], postprocess_state="measured")]
    for previous, current in zip(points, points[1:]):
        gap = int(current["frame_id"]) - int(previous["frame_id"])
        if 1 < gap <= max_gap_frames:
            for step in range(1, gap):
                ratio = step / gap
                merged.append(
                    {
                        "frame_id": int(previous["frame_id"]) + step,
                        "x_m": round(float(previous["x_m"]) + (float(current["x_m"]) - float(previous["x_m"])) * ratio, 4),
                        "y_m": round(float(previous["y_m"]) + (float(current["y_m"]) - float(previous["y_m"])) * ratio, 4),
                        "angle_deg": None,
                        "range_m": None,
                        "is_primary": bool(previous.get("is_primary") or current.get("is_primary")),
                        "track_id": "post",
                        "label": "postprocessed lead",
                        "confidence": None,
                        "score": None,
                        "source_key": "postprocessed",
                        "postprocess_state": "interpolated",
                    }
                )
        merged.append(dict(current, postprocess_state="measured"))
    return merged


def _smooth_postprocessed_points(points: list[dict], alpha: float = 0.35):
    if not points:
        return []
    smoothed = []
    prev_x = None
    prev_y = None
    previous_frame_id = None
    for point in points:
        current_x = float(point["x_m"])
        current_y = float(point["y_m"])
        frame_id = int(point["frame_id"])
        if (
            prev_x is None
            or prev_y is None
            or previous_frame_id is None
            or frame_id - previous_frame_id > 1
        ):
            smoothed_x = current_x
            smoothed_y = current_y
        else:
            smoothed_x = alpha * current_x + (1.0 - alpha) * prev_x
            smoothed_y = alpha * current_y + (1.0 - alpha) * prev_y
        prev_x = smoothed_x
        prev_y = smoothed_y
        previous_frame_id = frame_id
        smoothed.append(
            {
                **point,
                "x_m": round(smoothed_x, 4),
                "y_m": round(smoothed_y, 4),
            }
        )
    return smoothed


def _build_postprocessed_trajectory_bundle(
    render_records: list[dict],
    processed_records: list[dict],
    *,
    interpolation_gap_frames: int = 6,
    smoothing_alpha: float = 0.35,
):
    render_priority = ["display_tracks", "tentative_display_tracks", "tentative_tracks", "detections"]
    processed_priority = ["confirmed_tracks", "tentative_tracks", "detections"]
    measured_points = []
    source_label = "render lead"
    priority_label = _trajectory_priority_label(render_priority)

    for record in render_records:
        point = _select_lead_point_from_record(record, render_priority)
        if point is not None:
            measured_points.append(point)

    if len(measured_points) < 2:
        measured_points = []
        source_label = "processed lead"
        priority_label = _trajectory_priority_label(processed_priority)
        for record in processed_records:
            point = _select_lead_point_from_record(record, processed_priority)
            if point is not None:
                measured_points.append(point)

    measured_points.sort(key=lambda item: item["frame_id"])
    interpolated_points = _interpolate_postprocessed_points(
        measured_points,
        max_gap_frames=interpolation_gap_frames,
    )
    smoothed_points = _smooth_postprocessed_points(interpolated_points, alpha=smoothing_alpha)

    if not smoothed_points:
        return {
            "series": [],
            "frames": [],
            "track_count": 0,
            "source_key": None,
            "source_label": source_label,
            "priority_label": priority_label,
            "fallback_used": False,
            "lead_only": True,
            "gap_break_frames": int(interpolation_gap_frames),
            "empty_message": "후처리할 lead trajectory가 충분하지 않습니다.",
            "count_label": "postprocessed lead",
            "postprocess": {
                "input_points": 0,
                "output_points": 0,
                "interpolated_points": 0,
                "smoothing_alpha": smoothing_alpha,
                "interpolation_gap_frames": interpolation_gap_frames,
            },
        }

    series_points = _downsample_points(smoothed_points)
    frames = [
        {
            "frame_id": int(point["frame_id"]),
            "tracks": [point],
            "source_key": "postprocessed",
            "invalid": False,
            "status_text": point.get("postprocess_state", "smoothed"),
            "tracker_policy": None,
            "capture_latency_ms": None,
            "candidate_count": 1,
        }
        for point in smoothed_points
    ]
    interpolated_count = sum(1 for point in smoothed_points if point.get("postprocess_state") == "interpolated")
    return {
        "series": [
            {
                "track_id": "post",
                "label": "postprocessed lead",
                "color": TRAJECTORY_COLORS[0],
                "point_count": len(series_points),
                "full_point_count": len(smoothed_points),
                "primary_hits": sum(1 for point in smoothed_points if point.get("is_primary")),
                "longest_run": _longest_contiguous_run(smoothed_points, max_gap_frames=interpolation_gap_frames),
                "points": series_points,
            }
        ],
        "frames": frames,
        "track_count": 1,
        "source_key": "postprocessed",
        "source_label": source_label,
        "priority_label": priority_label,
        "fallback_used": False,
        "lead_only": True,
        "gap_break_frames": int(interpolation_gap_frames),
        "empty_message": "후처리 trajectory 데이터가 없습니다.",
        "count_label": "postprocessed lead",
        "postprocess": {
            "input_points": len(measured_points),
            "output_points": len(smoothed_points),
            "interpolated_points": interpolated_count,
            "smoothing_alpha": smoothing_alpha,
            "interpolation_gap_frames": interpolation_gap_frames,
        },
    }


def _build_track_playback_bundle(
    records: list[dict],
    priority_keys: list[str],
    *,
    max_tracks: int = 4,
    lead_only: bool = False,
    gap_break_frames: int = 2,
    latency_key: str | None = None,
    count_key: str | None = None,
    fallback_count_key: str | None = None,
    count_label: str = "track",
):
    trajectory = _build_track_trajectory_bundle(
        records,
        priority_keys,
        max_tracks=max_tracks,
        lead_only=lead_only,
        gap_break_frames=gap_break_frames,
    )
    selected_track_ids = {str(item.get("track_id")) for item in trajectory.get("series", [])}
    frames = []
    used_detection_fallback = bool(trajectory.get("fallback_used"))

    for record in records:
        frame_id = int(record.get("frame_id", record.get("frame_index", 0)) or 0)
        frame_tracks = []
        selected_items = []
        selected_key = None

        if used_detection_fallback:
            detections = record.get("detections") or []
            if detections:
                lead = max(detections, key=lambda item: float(item.get("score", 0.0) or 0.0))
                selected_key = "detections"
                point = _playback_point({**lead, "track_id": "lead"}, frame_id)
                if point is not None:
                    frame_tracks.append(point)
        else:
            for key in priority_keys:
                items = record.get(key) or []
                if items:
                    selected_key = key
                    selected_items = items
                    break
            for item in selected_items:
                track_id = item.get("track_id")
                if track_id is None:
                    continue
                if selected_track_ids and str(track_id) not in selected_track_ids:
                    continue
                point = _playback_point(item, frame_id)
                if point is not None:
                    frame_tracks.append(point)

        frames.append(
            {
                "frame_id": frame_id,
                "tracks": frame_tracks,
                "source_key": selected_key,
                "invalid": bool(record.get("invalid")),
                "status_text": record.get("status_text", ""),
                "tracker_policy": record.get("tracker_policy"),
                "capture_latency_ms": record.get(latency_key) if latency_key else None,
                "candidate_count": _extract_record_count(record, count_key, fallback_count_key)
                if count_key or fallback_count_key
                else None,
            }
        )

    return {
        **trajectory,
        "frames": frames,
        "count_label": count_label,
    }


def _trajectory_summary_text(bundle: dict):
    if bundle.get("series"):
        fallback_text = " | fallback=detection" if bundle.get("fallback_used") else ""
        mode_text = " | mode=lead-only" if bundle.get("lead_only") else ""
        gap_text = f" | gap_break>{int(bundle.get('gap_break_frames', 2))}" if bundle.get("gap_break_frames") else ""
        return (
            f"priority={bundle.get('priority_label', 'n/a')} | "
            f"source={bundle.get('source_label', 'n/a')} | "
            f"tracks={bundle.get('track_count', 0)}{mode_text}{gap_text}{fallback_text}"
        )
    return bundle.get("empty_message") or "궤적 데이터가 없습니다."


def _postprocess_summary_text(bundle: dict):
    meta = bundle.get("postprocess") or {}
    if not bundle.get("series"):
        return bundle.get("empty_message") or "후처리 trajectory 데이터가 없습니다."
    return (
        f"source={bundle.get('source_label', 'n/a')} | "
        f"input={meta.get('input_points', 0)} | "
        f"output={meta.get('output_points', 0)} | "
        f"interpolated={meta.get('interpolated_points', 0)} | "
        f"gap_fill<={meta.get('interpolation_gap_frames', 'n/a')} | "
        f"ema_alpha={meta.get('smoothing_alpha', 'n/a')}"
    )


def _build_trajectory_replay_html(
    session_dir: Path,
    render_records: list[dict],
    processed_records: list[dict],
):
    postprocessed_playback = _build_postprocessed_trajectory_bundle(render_records, processed_records)
    render_playback = _build_track_playback_bundle(
        render_records,
        ["display_tracks", "tentative_display_tracks", "tentative_tracks"],
        max_tracks=4,
        lead_only=False,
        latency_key="capture_to_render_ms",
        count_key="display_track_count",
        fallback_count_key="display_tracks",
        count_label="display track",
    )
    processed_playback = _build_track_playback_bundle(
        processed_records,
        ["confirmed_tracks", "tentative_tracks"],
        max_tracks=4,
        lead_only=False,
        latency_key="capture_to_process_ms",
        count_key="confirmed_track_count",
        fallback_count_key="confirmed_tracks",
        count_label="confirmed track",
    )
    payload = {
        "render": render_playback,
        "processed": processed_playback,
        "postprocessed": postprocessed_playback,
    }
    script = f"""
    <script>
      {COMMON_SCRIPT}
      const REPLAY_DATA = {json.dumps(payload, ensure_ascii=False)};
      function replaySummaryText(bundle, modeLabel) {{
        if (!bundle || !bundle.series || !bundle.series.length) {{
          return `${{modeLabel}} | ${{
            (bundle && (bundle.empty_message || bundle.emptyMessage)) || '재생할 데이터가 없습니다.'
          }}`;
        }}
        const fallbackText = bundle.fallback_used ? ' | fallback=detection' : '';
        const gapText = bundle.gap_break_frames ? ` | gap_break>${{bundle.gap_break_frames}}` : '';
        return `${{modeLabel}} | priority=${{bundle.priority_label || 'n/a'}} | source=${{bundle.source_label || 'n/a'}} | tracks=${{bundle.track_count || 0}}${{gapText}}${{fallbackText}}`;
      }}
      function renderSelectedReplay() {{
        const selected = document.getElementById('replay-source-select').value;
        const bundle = REPLAY_DATA[selected] || {{ frames: [], series: [], empty_message: '선택한 재생 데이터가 없습니다.' }};
        const modeLabel = selected === 'render'
          ? 'render 기준: 실제 화면에 표시된 track'
          : selected === 'processed'
            ? 'processed 기준: 내부 confirmed/tentative track'
            : '후처리 기준: lead trajectory 보간 + smoothing 결과';
        document.getElementById('replay-summary').textContent = replaySummaryText(bundle, modeLabel);
        renderTrajectoryReplay('trajectory-replay-root', bundle, {{
          title: selected === 'render'
            ? 'render trajectory replay'
            : selected === 'processed'
              ? 'processed trajectory replay'
              : 'postprocessed trajectory replay'
        }});
      }}
      document.getElementById('replay-source-select').addEventListener('change', renderSelectedReplay);
      renderSelectedReplay();
    </script>
    """

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{session_dir.name} trajectory replay</title>
  <style>{COMMON_STYLE}</style>
</head>
<body>
  <div class="wrap">
    <header class="hero">
      <h1>trajectory_replay.html</h1>
      <p>정적 발자취 대신, 프레임 순서대로 이동을 재생해서 직선/왕복/원형/네모 경로가 실제로 어떻게 무너지는지 디버깅하는 페이지입니다.</p>
      <nav class="nav">
        <a href="./index.html">세션 개요</a>
        <a href="./performance_report.html">성능 KPI 리포트</a>
        <a href="./processed_report.html">processed 리포트</a>
        <a href="./render_report.html">render 리포트</a>
        <a href="./event_report.html">event 리포트</a>
        <a href="../index.html">전체 비교 대시보드</a>
      </nav>
    </header>

    <section class="card">
      <h2>재생 소스 선택</h2>
      <div class="controls">
        <div class="control">
          <label for="replay-source-select">데이터 기준</label>
          <select id="replay-source-select">
            <option value="render">render 기준</option>
            <option value="processed">processed 기준</option>
            <option value="postprocessed">로그 후처리 기준</option>
          </select>
        </div>
      </div>
      <p id="replay-summary" class="subtle" style="margin-top:14px;"></p>
    </section>

    <section class="card">
      <h2>시간축 재생</h2>
      <p class="subtle">최근 N프레임 trail과 현재 위치를 함께 표시합니다. render 기준은 화면에서 실제 보인 결과를, processed 기준은 내부 tracker 상태를, 후처리 기준은 로그에서 lead trajectory를 보간·smoothing한 결과를 보여 줍니다.</p>
      <div id="trajectory-replay-root"></div>
    </section>

    <section class="card">
      <h2>읽는 법</h2>
      <ol class="steps">
        <li>우선 <code>render</code> 기준으로 재생해, 사용자가 실제로 본 끊김과 튐이 어디서 생기는지 확인합니다.</li>
        <li>같은 구간을 <code>processed</code> 기준으로 바꿔서 보면, 내부 track은 유지되는데 display에서만 숨겨졌는지 구분할 수 있습니다.</li>
        <li><code>로그 후처리 기준</code>은 짧은 gap 보간과 EMA smoothing이 들어간 offline 디버그 경로라, 원본과 얼마나 다른지 비교해 representative point drift를 해석할 때 씁니다.</li>
        <li><code>최근 12/24/48프레임</code> trail을 바꿔 보면, 직선운동이 왜 원처럼 말려 보였는지와 코너에서 왜 끊겼는지 해석하기 쉽습니다.</li>
      </ol>
    </section>
    {script}
  </div>
</body>
</html>"""


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
    transport = _transport_quality(summary)
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
        (
            "Transport Quality",
            transport.get("label", "n/a"),
            transport.get("suitability", "n/a"),
            transport.get("tone", "brand"),
        ),
        ("Processed Invalid", _fmt_pct(processed.get("invalid_rate")), "처리 입력 무결성 관점", "danger" if (processed.get("invalid_rate") or 0) >= 0.5 else "brand"),
        ("Render Invalid", _fmt_pct(render.get("invalid_rate")), "사용자 체감 품질 관점", "danger" if (render.get("invalid_rate") or 0) >= 0.5 else "brand"),
        ("Display Track Mean", _fmt(render.get("display_track_count", {}).get("mean")), "화면에 실제로 보인 트랙 수", "warn"),
        ("Held Display Mean", _fmt(render.get("display_held_track_count", {}).get("mean")), "display hysteresis 개입량", "warn"),
        ("Render P95", _fmt(render.get("capture_to_render_ms", {}).get("p95"), suffix=" ms"), "수집부터 표시까지 상위 95%", "brand"),
        ("Power Plan", system.get("power_plan_name") or "n/a", "실험/측정용으로는 High performance 권장", power_tone),
        ("Host IP Match", _yes_no_unknown(system.get("host_ip_present"), yes_text="match", no_text="mismatch"), f"expected={system.get('expected_host_ip') or 'n/a'}", host_ip_tone),
        ("Slowest Stage", preferred_stage.get("name") or "n/a", f"p95={_fmt(preferred_stage.get('p95_ms'), suffix=' ms')}", "warn"),
    ]


def _performance_overview_cards(summary: dict):
    performance = summary.get("performance", {})
    scoring = performance.get("scoring", {})
    budget = performance.get("frame_budget", {})
    throughput = performance.get("throughput", {})
    compute = performance.get("compute", {})
    jitter = performance.get("jitter", {})
    continuity = performance.get("continuity", {})
    geometry = performance.get("geometry", {})
    geometry_reference = geometry.get("reference", {})

    processed_ratio = throughput.get("processed_vs_expected_ratio")
    render_ratio = throughput.get("render_vs_expected_ratio")
    compute_ratio = compute.get("compute_utilization_p95_ratio")
    render_jitter = jitter.get("render_latency_jitter_ms")
    lead_switches = (continuity.get("lead_confirmed") or {}).get("switch_count")
    candidate_ratio = continuity.get("candidate_to_confirmed_ratio")
    path_cleanliness = geometry_reference.get("path_cleanliness_score_10")
    path_max_gap = geometry_reference.get("max_gap_frames")
    path_residual = geometry_reference.get("local_residual_rms_m")
    overall_score_10 = scoring.get("overall_score_10")
    overall_score_100 = scoring.get("overall_score_100")

    def ratio_tone(value):
        if value is None:
            return "brand"
        if value >= 0.9:
            return "good"
        if value >= 0.75:
            return "brand"
        if value >= 0.5:
            return "warn"
        return "danger"

    def lower_tone(value, good_limit, brand_limit, warn_limit):
        if value is None:
            return "brand"
        if value <= good_limit:
            return "good"
        if value <= brand_limit:
            return "brand"
        if value <= warn_limit:
            return "warn"
        return "danger"

    return [
        (
            "Performance Score",
            f"{_fmt(overall_score_10)}/10",
            f"{_fmt(overall_score_100)}/100 | {scoring.get('label') or '평가 없음'}",
            scoring.get("tone", "brand"),
        ),
        (
            "Frame Budget",
            _fmt(budget.get("configured_frame_period_ms"), suffix=" ms"),
            f"target={_fmt(budget.get('expected_fps'))} fps",
            "brand",
        ),
        (
            "Processed FPS",
            _fmt(throughput.get("processed_fps")),
            _fps_target_hint(throughput.get("processed_fps"), budget.get("expected_fps"), processed_ratio),
            ratio_tone(processed_ratio),
        ),
        (
            "Render FPS",
            _fmt(throughput.get("render_fps")),
            _fps_target_hint(throughput.get("render_fps"), budget.get("expected_fps"), render_ratio),
            ratio_tone(render_ratio),
        ),
        (
            "Compute Util P95",
            _fmt_pct(compute_ratio),
            _budget_hint((compute.get("compute_total_ms") or {}).get("p95"), budget.get("configured_frame_period_ms"), compute_ratio),
            lower_tone(compute_ratio, 0.6, 0.85, 1.0),
        ),
        (
            "Render Jitter",
            _fmt(render_jitter, suffix=" ms"),
            "render latency p95 - p50",
            lower_tone(render_jitter, 20, 40, 70),
        ),
        (
            "Lead Confirmed Switch",
            _fmt(lead_switches, digits=0),
            f"coverage={_fmt_pct((continuity.get('lead_confirmed') or {}).get('coverage_rate'))}",
            lower_tone(lead_switches, 5, 12, 20),
        ),
        (
            "Candidate/Confirmed",
            _fmt(candidate_ratio),
            "높을수록 한 사람 후보가 여러 개로 나뉘기 쉬움",
            lower_tone(candidate_ratio, 1.3, 1.8, 2.5),
        ),
        (
            "Display/Confirmed",
            _fmt(continuity.get("display_to_confirmed_ratio")),
            "화면에 실제로 남는 track 비율",
            ratio_tone(continuity.get("display_to_confirmed_ratio")),
        ),
        (
            "Path Cleanliness",
            f"{_fmt(path_cleanliness)}/10",
            f"gap={_fmt(path_max_gap, digits=0)} frames | residual={_fmt(path_residual, suffix=' m')}",
            lower_tone(10.0 - float(path_cleanliness) if path_cleanliness is not None else None, 1, 3, 5),
        ),
    ]


def _fps_target_hint(actual_fps, expected_fps, ratio):
    if actual_fps is None or expected_fps is None or ratio is None:
        return "목표 FPS 대비 비율을 계산할 데이터가 부족합니다."
    return f"실제 {_fmt(actual_fps)} fps / 목표 {_fmt(expected_fps)} fps = {_fmt_pct(ratio)}"


def _budget_hint(used_ms, budget_ms, ratio):
    if used_ms is None or budget_ms is None or ratio is None:
        return "프레임 예산 대비 사용량을 계산할 데이터가 부족합니다."
    return f"{_fmt(used_ms, suffix=' ms')} / 예산 {_fmt(budget_ms, suffix=' ms')} = {_fmt_pct(ratio)}"


def _performance_category_cards(summary: dict):
    categories = ((summary.get("performance") or {}).get("scoring") or {}).get("categories") or {}
    rows = []
    for key in ("throughput", "efficiency", "stability", "continuity", "geometry"):
        category = categories.get(key) or {}
        rows.append(
            f"""
            <div class="metric">
              <div class="label">{category.get('label', key)}</div>
              <div class="value">{_fmt(category.get('score_10'))}/10</div>
              <div class="hint">{_fmt(category.get('score_100'))}/100 | {_pill(category.get('grade', 'n/a'), category.get('tone', 'brand'))}</div>
            </div>
            """
        )
    return "".join(rows)


def _performance_kpi_rows(summary: dict):
    kpis = (((summary.get("performance") or {}).get("scoring") or {}).get("kpis") or {})
    if not kpis:
        return '<tr><td colspan="7">성능 KPI 점수를 계산할 데이터가 없습니다.</td></tr>'

    rows = []
    order = (
        "processed_vs_target",
        "render_vs_target",
        "compute_utilization_p95",
        "render_latency_p95",
        "render_jitter",
        "candidate_to_confirmed",
        "display_to_confirmed",
        "lead_confirmed_switch",
        "path_cleanliness",
        "path_max_gap_frames",
        "path_local_residual_rms",
        "path_jump_ratio",
    )
    for key in order:
        item = kpis.get(key)
        if not item:
            continue
        rows.append(
            f"""
            <tr>
              <td><strong>{item.get('label', key)}</strong><br /><span class="subtle"><code>{key}</code></span></td>
              <td>{item.get('value_display', 'n/a')}</td>
              <td>{_fmt(item.get('score_10'))}/10<br />{_pill(_tone_label(item.get('tone', 'brand')), item.get('tone', 'brand'))}</td>
              <td>{item.get('target', 'n/a')}<br /><span class="subtle">식: <code>{item.get('calculation', 'n/a')}</code></span></td>
              <td>{item.get('meaning', '')}</td>
              <td>{item.get('industry_standard', '')}</td>
              <td>{item.get('interpretation', '')}</td>
            </tr>
            """
        )
    return "".join(rows)


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
        frame_ids = [int(row.get("frame_id", row.get("frame_index", 0)) or 0) for row in matched]
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
        frame_ids = [int(row.get("frame_id", row.get("frame_index", 0)) or 0) for row in matched]
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


def _build_performance_html(session_dir: Path, summary: dict):
    performance = summary.get("performance", {})
    scoring = performance.get("scoring", {})
    budget = performance.get("frame_budget", {})
    throughput = performance.get("throughput", {})
    compute = performance.get("compute", {})
    jitter = performance.get("jitter", {})
    continuity = performance.get("continuity", {})
    geometry = performance.get("geometry", {})
    geometry_reference = geometry.get("reference", {})
    transport = _transport_quality(summary)
    render_geometry = geometry.get("render_lead", {})
    processed_geometry = geometry.get("processed_lead", {})
    overview_cards = "".join(
        f"""
        <div class="metric">
          <div class="label">{label}</div>
          <div class="value">{value}</div>
          <div class="hint">{hint}</div>
        </div>
        """
        for label, value, hint, _tone in _performance_overview_cards(summary)
    )
    category_cards = _performance_category_cards(summary)
    kpi_rows = _performance_kpi_rows(summary)
    highlights_html = _build_text_list_html(
        performance.get("highlights", []),
        "성능 해석용 하이라이트가 없습니다.",
    )
    read_steps = _build_steps_html(
        [
            "1순위: frame budget과 processed/render FPS를 같이 봅니다. target 대비 비율이 낮으면 실시간성부터 흔들립니다.",
            "2순위: compute utilization을 봅니다. compute_total p95가 frame budget에 가까우면 계산 최적화가 우선입니다.",
            "3순위: render jitter와 non-compute latency를 같이 봅니다. compute는 여유인데 jitter가 크면 큐, 환경, 표시 경로를 의심합니다.",
            "4순위: continuity 지표를 봅니다. candidate/confirmed 비율이 높고 lead switch가 많으면 detection/tracking 설계 병목일 가능성이 큽니다.",
            "5순위: geometry/path quality를 봅니다. gap, local residual, jump ratio가 크면 같은 ID여도 눈으로 보는 경로는 지저분할 수 있습니다.",
        ],
        empty_message="표시할 해석 순서가 없습니다.",
    )

    throughput_rows = "".join(
        f"""
        <tr><td><strong>{label}</strong></td><td>{value}</td><td>{hint}</td></tr>
        """
        for label, value, hint in [
            (
                "Configured Frame Period",
                _fmt(budget.get("configured_frame_period_ms"), suffix=" ms"),
                f"설정 파일 기준 1프레임 예산입니다. cfg={budget.get('cfg_path') or 'unavailable'}",
            ),
            (
                "Expected FPS",
                _fmt(budget.get("expected_fps")),
                "frameCfg가 기대하는 목표 처리량입니다. 이후 target 대비 xx%는 모두 이 값을 100%로 두고 계산합니다.",
            ),
            (
                "Processed FPS",
                _fmt(throughput.get("processed_fps")),
                _fps_target_hint(
                    throughput.get("processed_fps"),
                    budget.get("expected_fps"),
                    throughput.get("processed_vs_expected_ratio"),
                ) + ". 즉 알고리즘 처리단이 계획한 처리량을 얼마나 따라갔는지 뜻합니다.",
            ),
            (
                "Render FPS",
                _fmt(throughput.get("render_fps")),
                _fps_target_hint(
                    throughput.get("render_fps"),
                    budget.get("expected_fps"),
                    throughput.get("render_vs_expected_ratio"),
                ) + ". 즉 사용자가 실제로 본 화면 갱신이 목표 대비 어느 정도인지 뜻합니다.",
            ),
            (
                "Render / Processed",
                _fmt_pct(throughput.get("render_to_processed_ratio")),
                "처리 완료된 프레임 중 실제 화면까지 간 비율입니다. 100%에 가까울수록 UI가 처리 출력을 잘 따라갑니다.",
            ),
            (
                "Session Duration",
                _fmt(throughput.get("session_duration_s"), suffix=" s"),
                "FPS와 switch rate 계산에 사용된 세션 길이입니다.",
            ),
        ]
    )

    compute_rows = "".join(
        f"""
        <tr><td><strong>{label}</strong></td><td>{value}</td><td>{hint}</td></tr>
        """
        for label, value, hint in [
            (
                "Compute Total Mean",
                _fmt((compute.get("compute_total_ms") or {}).get("mean"), suffix=" ms"),
                _budget_hint(
                    (compute.get("compute_total_ms") or {}).get("mean"),
                    budget.get("configured_frame_period_ms"),
                    compute.get("compute_utilization_mean_ratio"),
                ) + ". 평균적으로 계산이 프레임 예산을 얼마나 차지하는지 봅니다.",
            ),
            (
                "Compute Total P95",
                _fmt((compute.get("compute_total_ms") or {}).get("p95"), suffix=" ms"),
                _budget_hint(
                    (compute.get("compute_total_ms") or {}).get("p95"),
                    budget.get("configured_frame_period_ms"),
                    compute.get("compute_utilization_p95_ratio"),
                ) + ". 느린 프레임 상위 5%에서의 계산 여유를 보는 핵심 지표입니다.",
            ),
            (
                "Pipeline Total P95",
                _fmt((compute.get("pipeline_total_ms") or {}).get("p95"), suffix=" ms"),
                _budget_hint(
                    (compute.get("pipeline_total_ms") or {}).get("p95"),
                    budget.get("configured_frame_period_ms"),
                    compute.get("pipeline_utilization_p95_ratio"),
                ) + ". compute뿐 아니라 큐/후처리까지 포함한 파이프라인 시간입니다.",
            ),
            (
                "Render Overhead Mean",
                _fmt(compute.get("render_overhead_mean_ms"), suffix=" ms"),
                "processed frame이 준비된 뒤 실제 render 제출까지 걸린 평균 시간입니다.",
            ),
            (
                "Non-compute Capture->Process Mean",
                _fmt(compute.get("non_compute_capture_to_process_mean_ms"), suffix=" ms"),
                "frame 수집, 조립, 큐 대기 등 compute 바깥 구간의 평균 지연입니다. compute가 낮은데 전체 지연이 크면 이 값을 먼저 봅니다.",
            ),
            (
                "Slowest Substage",
                compute.get("slowest_stage_name") or "n/a",
                f"p95={_fmt(compute.get('slowest_stage_p95_ms'), suffix=' ms')} | compute p95 중 {_fmt_pct(compute.get('slowest_stage_share_of_compute_p95_ratio'))} 차지",
            ),
            (
                "Log Write Mean",
                _fmt((compute.get("log_write_ms") or {}).get("mean"), suffix=" ms"),
                f"source={compute.get('source') or 'n/a'} | 핫패스 로그 쓰기 비용입니다.",
            ),
        ]
    )

    jitter_rows = "".join(
        f"""
        <tr><td><strong>{label}</strong></td><td>{value}</td><td>{hint}</td></tr>
        """
        for label, value, hint in [
            (
                "Processed Latency P50",
                _fmt(jitter.get("processed_latency_p50_ms"), suffix=" ms"),
                "capture_to_process 중앙값입니다. 평소에 가장 자주 겪는 처리 지연에 가깝습니다.",
            ),
            (
                "Processed Latency P95",
                _fmt(jitter.get("processed_latency_p95_ms"), suffix=" ms"),
                "느린 프레임 상위 5%의 capture_to_process 지연입니다. 평균보다 운영 위험을 더 잘 드러냅니다.",
            ),
            (
                "Processed Jitter",
                _fmt(jitter.get("processed_latency_jitter_ms"), suffix=" ms"),
                "processed latency의 p95 - p50입니다. 값이 클수록 세션 중 흔들림이 큽니다.",
            ),
            (
                "Render Latency P50",
                _fmt(jitter.get("render_latency_p50_ms"), suffix=" ms"),
                "capture_to_render 중앙값입니다. 평소 화면 체감 지연에 가깝습니다.",
            ),
            (
                "Render Latency P95",
                _fmt(jitter.get("render_latency_p95_ms"), suffix=" ms"),
                "느린 화면 프레임 상위 5%의 지연입니다. 데모에서 '갑자기 느린 순간'을 잡는 값입니다.",
            ),
            (
                "Render Jitter",
                _fmt(jitter.get("render_latency_jitter_ms"), suffix=" ms"),
                "render latency의 p95 - p50입니다. 20ms 이하가 안정적이고, 40ms를 넘기면 버벅임 체감이 커질 수 있습니다.",
            ),
            (
                "Compute Jitter",
                _fmt(jitter.get("compute_total_jitter_ms"), suffix=" ms"),
                "compute_total의 p95 - p50입니다. 계산 경로 자체의 흔들림만 따로 본 값입니다.",
            ),
        ]
    )

    continuity_rows = "".join(
        f"""
        <tr><td><strong>{label}</strong></td><td>{value}</td><td>{hint}</td></tr>
        """
        for label, value, hint in [
            (
                "Candidate / Confirmed",
                _fmt(continuity.get("candidate_to_confirmed_ratio")),
                "confirmed track 1개를 만들기 위해 candidate가 몇 개 필요한지 보는 값입니다. 단일 인원은 1.0~1.3 수준이 이상적입니다.",
            ),
            (
                "Display / Confirmed",
                _fmt(continuity.get("display_to_confirmed_ratio")),
                "내부 confirmed track이 화면 표시까지 얼마나 유지되는지 보는 값입니다. 높을수록 내부 추적이 사용자 화면에 잘 전달됩니다.",
            ),
            (
                "Lead Confirmed Switch",
                _fmt((continuity.get("lead_confirmed") or {}).get("switch_count"), digits=0),
                f"coverage={_fmt_pct((continuity.get('lead_confirmed') or {}).get('coverage_rate'))} | switch rate={_fmt_pct((continuity.get('lead_confirmed') or {}).get('switch_rate'))}",
            ),
            (
                "Lead Display Switch",
                _fmt((continuity.get("lead_display") or {}).get("switch_count"), digits=0),
                f"coverage={_fmt_pct((continuity.get('lead_display') or {}).get('coverage_rate'))} | switch rate={_fmt_pct((continuity.get('lead_display') or {}).get('switch_rate'))}",
            ),
            (
                "Unique Confirmed Track IDs",
                _fmt(continuity.get("unique_confirmed_track_ids"), digits=0),
                "세션 전체에서 내부 confirmed로 관측된 ID 수입니다. 단일 인원인데 값이 크면 ID fragmentation을 의심합니다.",
            ),
            (
                "Unique Display Track IDs",
                _fmt(continuity.get("unique_display_track_ids"), digits=0),
                "세션 전체에서 화면에 실제로 보인 ID 수입니다. 내부 continuity와 사용자 체감 continuity를 함께 봅니다.",
            ),
        ]
    )

    geometry_rows = "".join(
        f"""
        <tr><td><strong>{label}</strong></td><td>{render_value}</td><td>{processed_value}</td><td>{hint}</td></tr>
        """
        for label, render_value, processed_value, hint in [
            (
                "Reference Source",
                geometry.get("reference_source") or "n/a",
                f"{_fmt(geometry_reference.get('path_cleanliness_score_10'))}/10",
                "점수 계산에 실제로 사용한 경로 기준입니다. render lead가 충분하면 사용자 화면 기준을 우선하고, 아니면 processed lead를 참조합니다. 오른쪽 값은 그 참조 경로의 cleanliness 점수입니다.",
            ),
            (
                "Path Cleanliness",
                _fmt((render_geometry or {}).get("path_cleanliness_score_10")),
                _fmt((processed_geometry or {}).get("path_cleanliness_score_10")),
                f"이번 점수 계산 기준 값은 {_fmt(geometry_reference.get('path_cleanliness_score_10'))}/10 입니다. gap, local residual, jump ratio를 합친 종합 경로 품질 점수입니다.",
            ),
            (
                "Coverage Ratio",
                _fmt_pct((render_geometry or {}).get("coverage_ratio")),
                _fmt_pct((processed_geometry or {}).get("coverage_ratio")),
                "전체 프레임 중 lead path가 실제 좌표로 남은 비율입니다. 낮을수록 경로가 듬성듬성 비어 보입니다.",
            ),
            (
                "Max Gap Frames",
                _fmt((render_geometry or {}).get("max_gap_frames"), digits=0),
                _fmt((processed_geometry or {}).get("max_gap_frames"), digits=0),
                "lead path가 가장 길게 비는 구간 길이입니다. 0~1이면 매우 좋고, 5를 넘기면 눈으로도 뚜렷한 끊김이 느껴질 수 있습니다.",
            ),
            (
                "Local Residual RMS",
                _fmt((render_geometry or {}).get("local_residual_rms_m"), suffix=" m"),
                _fmt((processed_geometry or {}).get("local_residual_rms_m"), suffix=" m"),
                "인접한 세 점을 이은 local 직선에서 가운데 점이 얼마나 벗어나는지의 RMS입니다. 값이 클수록 지그재그/말림이 큽니다.",
            ),
            (
                "Jump Ratio",
                _fmt_pct((render_geometry or {}).get("jump_ratio")),
                _fmt_pct((processed_geometry or {}).get("jump_ratio")),
                "정상적인 연속 이동보다 갑자기 멀리 튄 스텝 비율입니다. 10%를 넘기면 대표점이 프레임 사이에서 자주 뛰는 것으로 볼 수 있습니다.",
            ),
            (
                "Path Efficiency",
                _fmt((render_geometry or {}).get("path_efficiency_ratio")),
                _fmt((processed_geometry or {}).get("path_efficiency_ratio")),
                "시작-끝 직선 거리 / 실제 누적 경로 길이입니다. 직선/대각선 테스트에서는 1에 가까울수록 자연스럽고, 값이 낮으면 우회/말림이 큽니다.",
            ),
        ]
    )

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{session_dir.name} performance report</title>
  <style>{COMMON_STYLE}</style>
</head>
<body>
  <div class="wrap">
    <header class="hero">
      <h1>{session_dir.name} 성능 KPI 리포트</h1>
      <p>운영 안정성(fail-safe)과 별도로, 실시간 처리 성능을 엔지니어링 관점에서 정리한 페이지입니다. 이제 각 KPI마다 10점 만점 점수, target의 정확한 의미, 현업 기준을 함께 보여줍니다.</p>
      <nav class="nav">
        <a href="./index.html">세션 개요</a>
        <a href="./ops_report.html">현업 평가 리포트</a>
        <a href="./processed_report.html">processed 리포트</a>
        <a href="./render_report.html">render 리포트</a>
        <a href="./event_report.html">event 리포트</a>
        <a href="../index.html">전체 비교 대시보드</a>
      </nav>
    </header>

    <section class="card">
      <h2>종합 성능 점수</h2>
      <div class="grid">
        <div class="metric">
          <div class="label">Performance Score</div>
          <div class="value">{_fmt(scoring.get('overall_score_10'))}/10</div>
          <div class="hint">{_fmt(scoring.get('overall_score_100'))}/100 | {_pill(scoring.get('grade', 'n/a'), scoring.get('tone', 'brand'))} | {scoring.get('summary', 'n/a')}</div>
        </div>
        <div class="metric">
          <div class="label">처리량/목표 달성</div>
          <div class="value">{_fmt(((scoring.get('categories') or {}).get('throughput') or {}).get('score_10'))}/10</div>
          <div class="hint">FPS가 목표치에 얼마나 가까운지</div>
        </div>
        <div class="metric">
          <div class="label">계산 여유</div>
          <div class="value">{_fmt(((scoring.get('categories') or {}).get('efficiency') or {}).get('score_10'))}/10</div>
          <div class="hint">프레임 예산 안에서 compute가 얼마나 여유로운지</div>
        </div>
        <div class="metric">
          <div class="label">지연 안정성</div>
          <div class="value">{_fmt(((scoring.get('categories') or {}).get('stability') or {}).get('score_10'))}/10</div>
          <div class="hint">세션 중 latency/jitter가 얼마나 흔들리는지</div>
        </div>
        <div class="metric">
          <div class="label">추적 연속성</div>
          <div class="value">{_fmt(((scoring.get('categories') or {}).get('continuity') or {}).get('score_10'))}/10</div>
          <div class="hint">한 사람을 한 ID로 유지하고 화면까지 전달하는 능력</div>
        </div>
        <div class="metric">
          <div class="label">경로 기하 품질</div>
          <div class="value">{_fmt(((scoring.get('categories') or {}).get('geometry') or {}).get('score_10'))}/10</div>
          <div class="hint">끊김, 지그재그, 좌표 점프를 얼마나 줄였는지</div>
        </div>
        <div class="metric">
          <div class="label">Transport Quality</div>
          <div class="value">{transport.get('label', 'n/a')}</div>
          <div class="hint">{transport.get('suitability', 'n/a')}</div>
        </div>
      </div>
      <p class="subtle" style="margin-top:16px;">
        이 점수는 내부 실험용 루브릭입니다. 절대적인 인증 점수는 아니지만, 세션 간 추세 비교와 병목 우선순위 판단에 쓰기 좋도록 설계했습니다.
      </p>
      <p class="note" style="margin-top:16px;"><strong>해석 주의:</strong> transport quality가 <code>noisy</code> 또는 <code>unusable</code>면 performance score가 높아도 raw 입력 자체의 불연속이 섞였을 수 있으므로, 알고리즘 before/after baseline 판정은 clean capture로 다시 확인하는 편이 안전합니다.</p>
    </section>

    <section class="card">
      <h2>핵심 KPI</h2>
      <div class="grid">{overview_cards}</div>
    </section>

    <section class="card">
      <h2>성능 카테고리 점수</h2>
      <div class="grid">{category_cards}</div>
    </section>

    <section class="card">
      <h2>지표별 10점 만점 평가</h2>
      <p class="subtle">
        여기서 말하는 <strong>target 대비 92.5%</strong> 같은 값은, 예를 들어 목표가 10fps일 때 실제가 9.25fps였다는 뜻입니다.
        즉 <strong>실제 값 / 목표 값</strong>으로 계산한 비율이며, 100%면 목표와 동일합니다.
      </p>
      <table>
        <thead>
          <tr>
            <th>KPI</th>
            <th>현재 값</th>
            <th>점수</th>
            <th>목표 / 기준</th>
            <th>무슨 뜻인가</th>
            <th>현업 기준</th>
            <th>이번 세션 해석</th>
          </tr>
        </thead>
        <tbody>{kpi_rows}</tbody>
      </table>
    </section>

    <section class="card">
      <h2>성능 해석 하이라이트</h2>
      {highlights_html}
    </section>

    <section class="card">
      <h2>Throughput &amp; Budget</h2>
      <table>
        <thead><tr><th>지표</th><th>값</th><th>해석</th></tr></thead>
        <tbody>{throughput_rows}</tbody>
      </table>
    </section>

    <section class="card">
      <h2>Compute &amp; Overhead</h2>
      <table>
        <thead><tr><th>지표</th><th>값</th><th>해석</th></tr></thead>
        <tbody>{compute_rows}</tbody>
      </table>
    </section>

    <section class="card">
      <h2>Jitter</h2>
      <table>
        <thead><tr><th>지표</th><th>값</th><th>해석</th></tr></thead>
        <tbody>{jitter_rows}</tbody>
      </table>
    </section>

    <section class="card">
      <h2>Continuity &amp; Fragmentation</h2>
      <table>
        <thead><tr><th>지표</th><th>값</th><th>해석</th></tr></thead>
        <tbody>{continuity_rows}</tbody>
      </table>
    </section>

    <section class="card">
      <h2>Geometry / Path Quality</h2>
      <p class="subtle">
        render lead는 사용자가 실제로 본 경로 기준, processed lead는 내부 tracker가 유지한 경로 기준입니다.
        같은 ID를 유지해도 이 표의 값이 나쁘면 눈으로 보는 궤적은 여전히 지저분할 수 있습니다.
      </p>
      <table>
        <thead><tr><th>지표</th><th>Render Lead</th><th>Processed Lead</th><th>해석</th></tr></thead>
        <tbody>{geometry_rows}</tbody>
      </table>
    </section>

    <section class="card">
      <h2>이 페이지 읽는 순서</h2>
      {read_steps}
    </section>
  </div>
</body>
</html>"""


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
    performance_cards_html = "".join(
        f"""
        <div class="metric">
          <div class="label">{label}</div>
          <div class="value">{value}</div>
          <div class="hint">{hint}</div>
        </div>
        """
        for label, value, hint, _tone in _performance_overview_cards(summary)
    )
    health = _session_health(summary)
    transport = _transport_quality(summary)
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
    replay_nav = ""
    replay_file_row = ""
    if str(summary.get("session_meta", {}).get("input_mode") or "").strip().lower() == "replay":
        replay_nav = '<a href="./replay_report.html">replay 요약</a>'
        replay_file_row = """
          <tr>
            <td><code>replay_report.html</code></td>
            <td>source capture, replay 속도, 주요 프레임/지연, trajectory replay 바로가기</td>
            <td><a href="./replay_report.html">열기</a></td>
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
        <a href="./performance_report.html">성능 KPI 리포트</a>
        <a href="./trajectory_replay.html">움직임 재생</a>
        {replay_nav}
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
        Transport: {_pill(transport.get("label", "n/a"), transport.get("tone", "brand"))} |
        현업 점수: <strong>{overall.get("score", "n/a")}/100</strong> {_pill(overall.get("grade", "n/a"), overall.get("tone", "brand"))}
      </p>
      <div class="grid">{cards_html}</div>
    </section>

    <section class="card">
      <h2>성능 KPI 요약</h2>
      <p class="subtle">운영 안정성 점수와 별도로, 실제 실시간 처리 품질을 보는 엔지니어링 지표입니다.</p>
      <div class="grid">{performance_cards_html}</div>
    </section>

    <section class="card">
      <h2>빠른 해석</h2>
      <div class="note">
        <strong>이 세션의 핵심 상태:</strong> {health["detail"]}<br />
        <strong>Transport 등급:</strong> {_pill(transport.get("label", "n/a"), transport.get("tone", "brand"))}
        | {transport.get("suitability", "n/a")}<br />
        <strong>Transport 해석:</strong> {transport.get("detail", "n/a")}<br />
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
            <td><code>performance_report.html</code></td>
            <td>frame budget, FPS, compute utilization, jitter, continuity KPI</td>
            <td><a href="./performance_report.html">열기</a></td>
          </tr>
          <tr>
            <td><code>processed_report.html</code></td>
            <td>처리 파이프라인 자체의 품질, invalid, tracker 입력/출력</td>
            <td><a href="./processed_report.html">열기</a></td>
          </tr>
          <tr>
            <td><code>trajectory_replay.html</code></td>
            <td>시간축 기준 움직임 재생, render/processed 전환, 최근 N프레임 trail 디버깅</td>
            <td><a href="./trajectory_replay.html">열기</a></td>
          </tr>
          {replay_file_row}
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


def _build_replay_report_html(session_dir: Path, summary: dict, event_summary: dict):
    session_meta = summary.get("session_meta") or {}
    runtime_config = summary.get("runtime_config") or {}
    source_capture = session_meta.get("source_capture") or runtime_config.get("log_source_capture") or "n/a"
    replay_speed = runtime_config.get("log_replay_speed")
    replay_loop = runtime_config.get("log_replay_loop")
    processed = summary.get("processed") or {}
    render = summary.get("render") or {}
    event = summary.get("event") or event_summary or {}
    transport = _transport_quality(summary)
    source_capture_path = str(source_capture)

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{session_dir.name} replay report</title>
  <style>{COMMON_STYLE}</style>
</head>
<body>
  <div class="wrap">
    <header class="hero">
      <h1>{session_dir.name} replay_report.html</h1>
      <p>raw capture를 다시 태운 replay 세션의 간단 요약입니다. source capture와 replay 조건, 결과 프레임 수, 즉시 확인할 링크를 한 곳에 모았습니다.</p>
      <nav class="nav">
        <a href="./index.html">세션 개요</a>
        <a href="./trajectory_replay.html">움직임 재생</a>
        <a href="./performance_report.html">성능 KPI 리포트</a>
        <a href="./processed_report.html">processed 리포트</a>
        <a href="./render_report.html">render 리포트</a>
        <a href="../index.html">전체 비교 대시보드</a>
      </nav>
    </header>

    <section class="card">
      <h2>Replay 입력</h2>
      <div class="grid">
        <div class="metric"><div class="label">input mode</div><div class="value">{session_meta.get("input_mode") or "n/a"}</div><div class="hint">live가 아니라 저장된 raw capture를 다시 태운 세션입니다.</div></div>
        <div class="metric"><div class="label">source capture</div><div class="value"><code>{source_capture_path}</code></div><div class="hint">이번 replay가 읽은 raw capture 경로입니다.</div></div>
        <div class="metric"><div class="label">transport quality</div><div class="value">{transport.get("label", "n/a")}</div><div class="hint">{transport.get("suitability", "n/a")}</div></div>
        <div class="metric"><div class="label">replay speed</div><div class="value">{_fmt(replay_speed, digits=2, suffix='x') if replay_speed is not None else "n/a"}</div><div class="hint">1.0x는 녹화 타이밍 그대로, 2.0x는 두 배 빠른 재생입니다.</div></div>
        <div class="metric"><div class="label">loop</div><div class="value">{_yes_no_unknown(replay_loop, yes_text="on", no_text="off")}</div><div class="hint">loop가 on이면 raw capture를 반복 재생합니다.</div></div>
      </div>
    </section>

    <section class="card">
      <h2>Replay 결과 요약</h2>
      <div class="grid">
        <div class="metric"><div class="label">processed frames</div><div class="value">{processed.get("frame_count", 0)}</div><div class="hint">raw replay에서 실제 처리된 프레임 수입니다.</div></div>
        <div class="metric"><div class="label">rendered frames</div><div class="value">{render.get("frame_count", 0)}</div><div class="hint">화면까지 실제 반영된 프레임 수입니다.</div></div>
        <div class="metric"><div class="label">render p95</div><div class="value">{_fmt((render.get("capture_to_render_ms") or {}).get("p95"), digits=1, suffix=' ms')}</div><div class="hint">replay에서도 처리와 렌더 지연은 다시 측정됩니다.</div></div>
        <div class="metric"><div class="label">first render</div><div class="value">{_fmt(event.get("first_render_elapsed_s"), digits=3, suffix=' s')}</div><div class="hint">replay 시작 후 첫 결과가 보이기까지 걸린 시간입니다.</div></div>
      </div>
    </section>

    <section class="card">
      <h2>어떻게 읽으면 되나</h2>
      <div class="note">
        <strong>1.</strong> <a href="./trajectory_replay.html">trajectory_replay.html</a>에서 시간축으로 실제 경로를 먼저 봅니다.<br />
        <strong>2.</strong> 경로가 끊기면 <a href="./render_report.html">render_report.html</a>에서 display와 invalid를 확인합니다.<br />
        <strong>3.</strong> 계산 회귀가 의심되면 <a href="./performance_report.html">performance_report.html</a>에서 detect_ms, compute_total_ms, jitter를 같이 봅니다.<br />
        <strong>4.</strong> 같은 source capture로 코드 전후를 비교해야 replay의 의미가 있습니다.
      </div>
    </section>
  </div>
</body>
</html>"""


def _build_ops_html(session_dir: Path, summary: dict, event_summary: dict):
    assessment = summary.get("assessment", {})
    overall = assessment.get("overall", {})
    category_scores = assessment.get("category_scores", {})
    evaluation_mode = assessment.get("evaluation_mode", {})
    transport = _transport_quality(summary)
    system = summary.get("system", {})
    preferred_stage_timings = summary.get("diagnostics", {}).get("preferred_stage_timings_ms") or {}
    preferred_slowest_stage = preferred_stage_timings.get("slowest_stage") or {}
    performance_cards = "".join(
        f"""
        <div class="metric">
          <div class="label">{label}</div>
          <div class="value">{value}</div>
          <div class="hint">{hint}</div>
        </div>
        """
        for label, value, hint, _tone in _performance_overview_cards(summary)
    )
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
      <p class="subtle">평가 모드: <strong>{evaluation_mode.get('label', 'n/a')}</strong> | {evaluation_mode.get('description', '평가 모드 설명이 없습니다.')}</p>
      <nav class="nav">
        <a href="./index.html">세션 개요</a>
        <a href="./performance_report.html">성능 KPI 리포트</a>
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
          <div class="label">평가 모드</div>
          <div class="value">{evaluation_mode.get('label', 'n/a')}</div>
          <div class="hint">{evaluation_mode.get('description', 'n/a')}</div>
        </div>
        <div class="metric">
          <div class="label">Transport Quality</div>
          <div class="value">{transport.get('label', 'n/a')}</div>
          <div class="hint">{transport.get('suitability', 'n/a')}</div>
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
      <h2>엔지니어링 성능 KPI</h2>
      <p class="subtle">운영 점수와 별도로, 실시간 처리 관점에서 throughput, compute budget, jitter, continuity를 바로 읽을 수 있는 요약입니다.</p>
      <div class="grid">{performance_cards}</div>
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
    trajectory = _build_track_trajectory_bundle(
        processed_records,
        ["confirmed_tracks", "tentative_tracks"],
        lead_only=True,
    )
    postprocessed = _build_postprocessed_trajectory_bundle(
        _load_render_records_with_fallback(session_dir),
        processed_records,
    )
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
        "postprocessed": postprocessed,
    }

    processed = summary.get("processed", {})
    health = _session_health(summary)
    transport = _transport_quality(summary)
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
        {{
          title: 'processed trajectory',
          emptyMessage: REPORT_DATA.trajectory.empty_message,
          breakFrameGap: REPORT_DATA.trajectory.gap_break_frames || 2
        }}
      );

      renderTrajectoryChart(
        'processed-postprocessed-chart',
        REPORT_DATA.postprocessed.series,
        {{
          title: 'postprocessed trajectory',
          emptyMessage: REPORT_DATA.postprocessed.empty_message,
          breakFrameGap: REPORT_DATA.postprocessed.gap_break_frames || 2
        }}
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
        <a href="./performance_report.html">성능 KPI 리포트</a>
        <a href="./trajectory_replay.html">움직임 재생</a>
        <a href="./render_report.html">render 리포트</a>
        <a href="./event_report.html">event 리포트</a>
        <a href="../../../docs/log_guides/processed_frames_guide.html">processed 가이드</a>
        <a href="../index.html">전체 비교 대시보드</a>
      </nav>
    </header>

    <section class="card">
      <h2>핵심 지표</h2>
      <p class="subtle">세션 상태: {_pill(health["label"], health["tone"])} | transport: {_pill(transport.get("label", "n/a"), transport.get("tone", "brand"))} | {transport.get("suitability", "n/a")}</p>
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
      <p class="subtle" style="margin-top:12px;">발자취만으로 부족하면 <a href="./trajectory_replay.html">trajectory replay</a>에서 시간축 기준으로 frame-by-frame 재생을 확인합니다.</p>
    </section>

    <section class="card">
      <h2>로그 기반 후처리 궤적</h2>
      <p class="subtle">{_postprocess_summary_text(postprocessed)}</p>
      <div id="processed-postprocessed-chart" class="chart"></div>
      <p class="subtle" style="margin-top:12px;">이 경로는 실시간 출력이 아니라, 저장된 로그에서 lead trajectory를 다시 골라 짧은 gap을 보간하고 EMA smoothing을 적용한 offline 디버그용 결과입니다.</p>
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
        lead_only=True,
    )
    postprocessed = _build_postprocessed_trajectory_bundle(
        render_records,
        _load_jsonl(session_dir / "processed_frames.jsonl"),
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
        "postprocessed": postprocessed,
    }

    render = summary.get("render", {})
    health = _session_health(summary)
    transport = _transport_quality(summary)
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
        {{
          title: 'render trajectory',
          emptyMessage: REPORT_DATA.trajectory.empty_message,
          breakFrameGap: REPORT_DATA.trajectory.gap_break_frames || 2
        }}
      );

      renderTrajectoryChart(
        'render-postprocessed-chart',
        REPORT_DATA.postprocessed.series,
        {{
          title: 'postprocessed trajectory',
          emptyMessage: REPORT_DATA.postprocessed.empty_message,
          breakFrameGap: REPORT_DATA.postprocessed.gap_break_frames || 2
        }}
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
        <a href="./performance_report.html">성능 KPI 리포트</a>
        <a href="./trajectory_replay.html">움직임 재생</a>
        <a href="./processed_report.html">processed 리포트</a>
        <a href="./event_report.html">event 리포트</a>
        <a href="../../../docs/log_guides/render_frames_guide.html">render 가이드</a>
        <a href="../index.html">전체 비교 대시보드</a>
      </nav>
    </header>

    <section class="card">
      <h2>핵심 지표</h2>
      <p class="subtle">세션 상태: {_pill(health["label"], health["tone"])} | transport: {_pill(transport.get("label", "n/a"), transport.get("tone", "brand"))} | {transport.get("suitability", "n/a")}</p>
      <div class="grid">
        <div class="metric"><div class="label">Render Frame Count</div><div class="value">{render.get("frame_count", 0)}</div></div>
        <div class="metric"><div class="label">Invalid Rate</div><div class="value">{_fmt_pct(render.get("invalid_rate"))}</div></div>
        <div class="metric"><div class="label">Display Track Mean</div><div class="value">{_fmt(render.get("display_track_count", {}).get("mean"))}</div></div>
        <div class="metric"><div class="label">Held Display Mean</div><div class="value">{_fmt(render.get("display_held_track_count", {}).get("mean"))}</div></div>
        <div class="metric"><div class="label">Render P95</div><div class="value">{_fmt(render.get("capture_to_render_ms", {}).get("p95"), suffix=" ms")}</div></div>
        <div class="metric"><div class="label">Multi Display Success</div><div class="value">{_fmt_pct(render.get("multi_display_success_rate"))}</div></div>
      </div>
      <p class="note" style="margin-top:16px;"><strong>읽는 법:</strong> candidate는 많은데 display track이 거의 0이면 내부 후보는 있지만 화면에 남는 결과는 거의 없다는 뜻입니다. Held Display가 높으면 display hysteresis가 화면 끊김을 완충한 것이므로 processed 기준 track 품질과 함께 봐야 합니다.</p>
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
      <p class="subtle" style="margin-top:12px;">발자취가 과장되거나 끊겨 보이면 <a href="./trajectory_replay.html">trajectory replay</a>에서 시간축 기준으로 실제 진행 순서를 재생해 봅니다.</p>
    </section>

    <section class="card">
      <h2>로그 기반 후처리 궤적</h2>
      <p class="subtle">{_postprocess_summary_text(postprocessed)}</p>
      <div id="render-postprocessed-chart" class="chart"></div>
      <p class="subtle" style="margin-top:12px;">이 경로는 세션 로그를 읽어 lead trajectory를 다시 구성한 offline 디버그용 결과입니다. 원본 render path와 얼마나 다른지 비교하면 display filter와 representative point drift를 분리해서 보기 쉽습니다.</p>
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
        <a href="./performance_report.html">성능 KPI 리포트</a>
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
                "transport_quality": _transport_quality(summary),
                "links": {
                    "index": f"./{session_dir.name}/index.html",
                    "ops": f"./{session_dir.name}/ops_report.html",
                    "performance": f"./{session_dir.name}/performance_report.html",
                    "replay": f"./{session_dir.name}/trajectory_replay.html",
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
          <td><strong>{row['summary'].get('assessment', {}).get('overall', {}).get('score', 'n/a')}</strong><br />{_pill(row['summary'].get('assessment', {}).get('overall', {}).get('grade', 'n/a'), row['summary'].get('assessment', {}).get('overall', {}).get('tone', 'brand'))}</td>
          <td><strong>{_fmt(row['summary'].get('performance', {}).get('scoring', {}).get('overall_score_100'))}</strong><br /><span class="subtle">{_fmt(row['summary'].get('performance', {}).get('scoring', {}).get('overall_score_10'))}/10</span></td>
          <td>{_pill(row['health']['label'], row['health']['tone'])}</td>
          <td>{_pill(row['transport_quality'].get('label', 'n/a'), row['transport_quality'].get('tone', 'brand'))}<br /><span class="subtle">{row['transport_quality'].get('suitability', 'n/a')}</span></td>
          <td>{_fmt_pct(row['summary']['render']['invalid_rate'])}</td>
          <td>{_fmt_pct(row['summary'].get('performance', {}).get('throughput', {}).get('render_vs_expected_ratio'))}</td>
          <td>{_fmt((row['summary'].get('performance', {}).get('continuity', {}).get('lead_confirmed') or {}).get('switch_count'), digits=0)}</td>
          <td>{_fmt(row['summary']['render']['capture_to_render_ms']['p95'], suffix=' ms')}</td>
          <td>
            <a href="{row['links']['index']}">개요</a> |
            <a href="{row['links']['ops']}">ops</a> |
            <a href="{row['links']['performance']}">perf</a> |
            <a href="{row['links']['replay']}">replay</a> |
            <a href="{row['links']['processed']}">processed</a> |
            <a href="{row['links']['render']}">render</a> |
            <a href="{row['links']['event']}">event</a>
          </td>
        </tr>
        """
        for row in session_rows
    ) or '<tr><td colspan="13">표시할 세션이 없습니다.</td></tr>'

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

      function formatMetricValue(row, value) {{
        if (!Number.isFinite(value)) return 'n/a';
        const percentKeys = new Set([
          'performance.throughput.render_vs_expected_ratio',
          'performance.compute.compute_utilization_p95_ratio',
          'processed.invalid_rate',
          'render.invalid_rate',
          'processed.multi_confirmed_success_rate',
          'render.multi_display_success_rate'
        ]);
        if (percentKeys.has(row.key)) return `${{(value * 100).toFixed(1)}}%`;
        if (String(row.key || '').includes('ms')) return `${{fmt(value, 1)}} ms`;
        if (String(row.key || '').includes('switch_count')) return fmt(value, 0);
        return fmt(value, 3);
      }}

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
            <div class="value">${{formatMetricValue(row, row.after)}}</div>
            <div class="hint">
              before=${{formatMetricValue(row, row.before)}} |
              delta=${{row.delta === null ? 'n/a' : formatMetricValue(row, row.delta)}} |
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
                  <span>before=${{formatMetricValue(row, row.before)}}</span>
                  <span>after=${{formatMetricValue(row, row.after)}}</span>
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
          emptyMessage: beforeTrajectory.empty_message,
          breakFrameGap: beforeTrajectory.gap_break_frames || 2
        }});
        renderTrajectoryChart('after-trajectory-chart', afterTrajectory.series, {{
          title: 'after trajectory',
          emptyMessage: afterTrajectory.empty_message,
          breakFrameGap: afterTrajectory.gap_break_frames || 2
        }});
      }}

      function initDashboard() {{
        const beforeSelect = document.getElementById('before-session');
        const afterSelect = document.getElementById('after-session');
        beforeSelect.innerHTML = '';
        afterSelect.innerHTML = '';
        DASHBOARD.sessions.forEach((session) => {{
          const score = nestedGet(session.summary, 'assessment.overall.score');
          const perfScore = nestedGet(session.summary, 'performance.scoring.overall_score_100');
          const transportLabel = ((session.transport_quality || {{}}).label) || 'n/a';
          const beforeOption = document.createElement('option');
          beforeOption.value = session.session_id;
          beforeOption.textContent = `${{session.session_id}} | ops=${{score ?? 'n/a'}} | perf=${{perfScore ?? 'n/a'}} | transport=${{transportLabel}} | ${{session.variant || 'n/a'}}`;
          beforeSelect.appendChild(beforeOption);

          const afterOption = document.createElement('option');
          afterOption.value = session.session_id;
          afterOption.textContent = `${{session.session_id}} | ops=${{score ?? 'n/a'}} | perf=${{perfScore ?? 'n/a'}} | transport=${{transportLabel}} | ${{session.variant || 'n/a'}}`;
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
            <th>세션</th><th>생성 시각</th><th>variant</th><th>scenario</th><th>ops 점수</th><th>perf 점수</th><th>상태</th><th>transport</th><th>render invalid</th><th>render target</th><th>lead switch</th><th>render p95</th><th>리포트</th>
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
        ("render.display_held_track_count.mean", "Held Display Mean", "lower"),
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
        ("performance.scoring.overall_score_100", "Performance Score", "number"),
        ("render.invalid_rate", "Render Invalid", "percent"),
        ("render.capture_to_render_ms.p95", "Render P95", "ms"),
        ("render.display_track_count.mean", "Display Track Mean", "number"),
        ("render.display_held_track_count.mean", "Held Display Mean", "number"),
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
        emptyMessage: COMPARISON_DATA.beforeTrajectory.empty_message,
        breakFrameGap: COMPARISON_DATA.beforeTrajectory.gap_break_frames || 2
      }});
      renderTrajectoryChart('static-after-trajectory-chart', COMPARISON_DATA.afterTrajectory.series, {{
        title: 'after trajectory',
        emptyMessage: COMPARISON_DATA.afterTrajectory.empty_message,
        breakFrameGap: COMPARISON_DATA.afterTrajectory.gap_break_frames || 2
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
    _write_text(session_dir / "performance_report.html", _build_performance_html(session_dir, summary))
    _write_text(session_dir / "trajectory_replay.html", _build_trajectory_replay_html(session_dir, render_records, processed_records))
    if str(summary.get("session_meta", {}).get("input_mode") or "").strip().lower() == "replay":
        _write_text(session_dir / "replay_report.html", _build_replay_report_html(session_dir, summary, event_summary))
    _write_text(session_dir / "processed_report.html", _build_processed_html(session_dir, summary, processed_records))
    _write_text(session_dir / "render_report.html", _build_render_html(session_dir, summary, render_records))
    _write_text(session_dir / "event_report.html", _build_event_html(session_dir, events, event_summary))

    return {
        "summary_path": session_dir / "summary.json",
        "index_path": session_dir / "index.html",
        "ops_report_path": session_dir / "ops_report.html",
        "performance_report_path": session_dir / "performance_report.html",
        "trajectory_replay_path": session_dir / "trajectory_replay.html",
        "replay_report_path": (
            session_dir / "replay_report.html"
            if str(summary.get("session_meta", {}).get("input_mode") or "").strip().lower() == "replay"
            else None
        ),
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
