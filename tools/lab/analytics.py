from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from tools.lab import registry, stage_cache


METRIC_DEFINITIONS = {
    "performance_score": {
        "label": "Performance Score",
        "direction": "higher",
        "unit": "/100",
        "meaning": "실시간 처리/연속성/경로 품질을 합친 전체 성능 점수입니다.",
    },
    "path_cleanliness_score_10": {
        "label": "Path Cleanliness",
        "direction": "higher",
        "unit": "/10",
        "meaning": "경로 gap, local residual, jump ratio를 합친 경로 품질 점수입니다.",
    },
    "path_local_residual_rms_m": {
        "label": "Path Local Residual RMS",
        "direction": "lower",
        "unit": "m",
        "meaning": "근처 프레임 이동 패턴에서 벗어난 흔들림 크기입니다.",
    },
    "path_jump_ratio": {
        "label": "Path Jump Ratio",
        "direction": "lower",
        "unit": "",
        "meaning": "사람 이동으로 보기 어려운 급격한 위치 점프 비율입니다.",
    },
    "lead_confirmed_switch_rate": {
        "label": "Lead Switch Rate",
        "direction": "lower",
        "unit": "",
        "meaning": "화면 기준 대표 confirmed track ID가 바뀌는 비율입니다.",
    },
    "candidate_to_confirmed_ratio": {
        "label": "Candidate / Confirmed",
        "direction": "lower",
        "unit": "",
        "meaning": "후보가 confirmed track보다 얼마나 많이 생기는지 보는 분해/과검출 지표입니다.",
    },
    "display_to_confirmed_ratio": {
        "label": "Display / Confirmed",
        "direction": "higher",
        "unit": "",
        "meaning": "내부 confirmed track 중 실제 화면에 남는 비율입니다.",
    },
    "render_latency_p95_ms": {
        "label": "Render Latency P95",
        "direction": "lower",
        "unit": "ms",
        "meaning": "렌더 기준 p95 지연입니다. UI에서 늦게 보이는 문제를 확인합니다.",
    },
    "compute_utilization_p95": {
        "label": "Compute Utilization P95",
        "direction": "lower",
        "unit": "",
        "meaning": "처리 시간이 프레임 budget 대비 얼마나 큰지 보는 지표입니다.",
    },
}


TARGETS = [
    {
        "metric": "render_latency_p95_ms",
        "target": 180.0,
        "direction": "lower",
        "why": "실시간 표시가 답답해지기 전의 1차 경고선입니다.",
    },
    {
        "metric": "compute_utilization_p95",
        "target": 0.85,
        "direction": "lower",
        "why": "계산이 프레임 budget을 거의 다 쓰기 시작하는 구간입니다.",
    },
    {
        "metric": "candidate_to_confirmed_ratio",
        "target": 1.5,
        "direction": "lower",
        "why": "한 사람을 여러 후보/track으로 쪼개는지 보는 기준입니다.",
    },
    {
        "metric": "lead_confirmed_switch_rate",
        "target": 0.05,
        "direction": "lower",
        "why": "대표 ID가 너무 자주 바뀌면 경로가 사람 눈에 끊겨 보입니다.",
    },
    {
        "metric": "path_local_residual_rms_m",
        "target": 0.12,
        "direction": "lower",
        "why": "몸 중심 대표점이 프레임마다 흔들리는지 보는 기준입니다.",
    },
    {
        "metric": "path_jump_ratio",
        "target": 0.05,
        "direction": "lower",
        "why": "순간 점프가 많으면 직선/사각형 운동도 곡선처럼 보입니다.",
    },
    {
        "metric": "path_cleanliness_score_10",
        "target": 8.0,
        "direction": "higher",
        "why": "논문/발표용 성공 사례로 쓰기 좋은 경로 품질 기준입니다.",
    },
]


BOTTLENECK_PARAMETER_RECOMMENDATIONS = {
    "compute_latency": [
        {
            "param_group": "detection",
            "param_key": "detection.max_targets",
            "intent": "후보 상한을 낮춰 detection/tracking 계산량을 제한",
            "tuning_hint": "한 사람 baseline이면 4-6 사이에서 재생 비교",
        },
        {
            "param_group": "detection.algorithm",
            "param_key": "detection.algorithm.cfar_scale",
            "intent": "CFAR 후보 수를 줄여 downstream stage 부하 완화",
            "tuning_hint": "후보가 과다하면 소폭 상향, 미검출이면 원복",
        },
        {
            "param_group": "pipeline",
            "param_key": "pipeline.queue_size",
            "intent": "프레임 burst를 흡수하되 지연 누적을 관찰",
            "tuning_hint": "queue 증가 전후 render p95와 dropped frame을 같이 확인",
        },
    ],
    "detection_over_split": [
        {
            "param_group": "detection.algorithm",
            "param_key": "detection.algorithm.min_cartesian_separation_m",
            "intent": "한 사람의 여러 peak가 별도 후보로 갈라지는 현상 완화",
            "tuning_hint": "0.05 m 단위로 올리며 candidate/confirmed와 path residual 확인",
        },
        {
            "param_group": "detection",
            "param_key": "detection.dbscan_adaptive_eps_bands",
            "intent": "거리별 cluster eps를 조정해 body 후보를 더 안정적으로 병합",
            "tuning_hint": "가까운 거리 eps부터 작은 폭으로 sweep",
        },
        {
            "param_group": "detection",
            "param_key": "detection.cluster_min_samples",
            "intent": "단발성 약한 cluster가 track 후보가 되는 것을 줄임",
            "tuning_hint": "두 사람/약한 반사 세션에서는 과도한 상향 주의",
        },
    ],
    "tracking_association_failure": [
        {
            "param_group": "tracking",
            "param_key": "tracking.association_gate",
            "intent": "예측 track과 측정 후보 매칭 허용 범위 조정",
            "tuning_hint": "ID switch가 많으면 gate와 doppler cost를 함께 비교",
        },
        {
            "param_group": "tracking",
            "param_key": "tracking.doppler_cost_weight",
            "intent": "속도 일관성을 association cost에 더 강하게 반영",
            "tuning_hint": "원/사각형 운동에서는 너무 높이면 코너에서 miss 가능",
        },
        {
            "param_group": "tracking",
            "param_key": "tracking.max_misses",
            "intent": "일시적 미검출에도 기존 track을 더 오래 유지",
            "tuning_hint": "ghost track 지속 시간과 lead switch rate를 같이 확인",
        },
    ],
    "representative_point_jump": [
        {
            "param_group": "tracking",
            "param_key": "tracking.measurement_var",
            "intent": "측정 위치 흔들림을 Kalman update에서 덜 민감하게 반영",
            "tuning_hint": "path residual은 낮추되 반응 지연이 생기는지 확인",
        },
        {
            "param_group": "tracking",
            "param_key": "tracking.range_measurement_scale",
            "intent": "range 방향 측정 신뢰도를 조정해 앞뒤 튐 완화",
            "tuning_hint": "직선 왕복 raw에서 before/after 궤적을 비교",
        },
        {
            "param_group": "detection.algorithm",
            "param_key": "detection.algorithm.min_cartesian_separation_m",
            "intent": "강한 반사점 중심으로 대표점이 바뀌는 현상 완화",
            "tuning_hint": "body-center stage output과 tracker input을 같이 확인",
        },
    ],
    "path_jump": [
        {
            "param_group": "tracking",
            "param_key": "tracking.association_gate",
            "intent": "멀리 떨어진 후보를 같은 track으로 붙이는지 제어",
            "tuning_hint": "jump ratio가 높으면 gate를 좁히고 miss 증가 여부 확인",
        },
        {
            "param_group": "tracking",
            "param_key": "tracking.process_var",
            "intent": "예측 모델이 급격한 이동을 얼마나 허용할지 조정",
            "tuning_hint": "원운동/사각형 코너 raw에서 과소 추종 여부 확인",
        },
        {
            "param_group": "tracking",
            "param_key": "tracking.doppler_gate_bins",
            "intent": "속도 불일치 후보가 association되는 범위 제한",
            "tuning_hint": "doppler zero 주변 guard와 함께 비교",
        },
    ],
    "render_latency": [
        {
            "param_group": "visualization",
            "param_key": "visualization.show_tentative_tracks",
            "intent": "화면에 그리는 track 수와 redraw 비용을 줄임",
            "tuning_hint": "false 설정 후 display/confirmed 비율과 UX를 같이 확인",
        },
        {
            "param_group": "visualization",
            "param_key": "visualization.tentative_min_confidence",
            "intent": "낮은 confidence tentative 표시를 줄여 render 부담 완화",
            "tuning_hint": "0.05 단위로 올리며 놓친 후보가 없는지 확인",
        },
        {
            "param_group": "pipeline",
            "param_key": "pipeline.queue_size",
            "intent": "처리 backlog가 화면 지연으로 보이는지 분리",
            "tuning_hint": "render p95와 compute utilization p95를 함께 비교",
        },
    ],
    "transport_issue": [
        {
            "param_group": "pipeline.invalid_policy",
            "param_key": "pipeline.invalid_policy.drop_gap_threshold",
            "intent": "UDP gap이 큰 프레임을 처리에서 제외할 기준 조정",
            "tuning_hint": "알고리즘 KPI 비교보다 capture 품질 분리를 우선",
        },
        {
            "param_group": "pipeline.invalid_policy",
            "param_key": "pipeline.block_track_birth_on_invalid",
            "intent": "깨진 프레임에서 새 track이 태어나는 것을 차단",
            "tuning_hint": "noisy raw robustness 용도로만 비교",
        },
    ],
    "display_or_confirmation_loss": [
        {
            "param_group": "tracking",
            "param_key": "tracking.confirm_hits",
            "intent": "confirmed 전환까지 필요한 hit 수를 조정",
            "tuning_hint": "낮추면 빠르게 보이지만 ghost confirmed 가능성 확인",
        },
        {
            "param_group": "visualization",
            "param_key": "visualization.tentative_min_hits",
            "intent": "tentative 표시 시작 기준을 조정",
            "tuning_hint": "display/confirmed와 visual noise를 같이 확인",
        },
    ],
    "path_quality_low": [
        {
            "param_group": "tracking",
            "param_key": "tracking.measurement_var",
            "intent": "경로 흔들림과 반응 지연의 균형 조정",
            "tuning_hint": "path cleanliness, residual, jump ratio를 함께 비교",
        },
        {
            "param_group": "detection",
            "param_key": "detection.dbscan_adaptive_eps_bands",
            "intent": "경로 품질 저하가 후보 병합 실패에서 오는지 확인",
            "tuning_hint": "같은 raw capture replay로 eps band sweep",
        },
        {
            "param_group": "tracking",
            "param_key": "tracking.association_gate",
            "intent": "경로 끊김과 잘못된 매칭 사이의 균형 조정",
            "tuning_hint": "lead switch rate와 max gap을 같이 확인",
        },
    ],
    "stage_hotspot": [
        {
            "param_group": "detection",
            "param_key": "detection.max_targets",
            "intent": "hot stage 뒤로 넘어가는 후보 수 제한",
            "tuning_hint": "slowest stage p95가 내려가는지 확인",
        },
        {
            "param_group": "processing",
            "param_key": "processing.doppler_guard_bins",
            "intent": "static/zero doppler 주변 처리량과 검출 품질 균형 확인",
            "tuning_hint": "RDI/RAI stage cache와 함께 비교",
        },
    ],
}


DEFAULT_PARAMETER_RECOMMENDATIONS = [
    {
        "param_group": "detection",
        "param_key": "detection.max_targets",
        "intent": "후보 수와 추적 안정성의 기본 균형점 확인",
        "tuning_hint": "같은 raw replay에서 KPI diff를 먼저 확인",
    },
    {
        "param_group": "tracking",
        "param_key": "tracking.association_gate",
        "intent": "track continuity와 잘못된 매칭의 기본 균형점 확인",
        "tuning_hint": "lead switch rate, jump ratio를 함께 확인",
    },
]


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _pct(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value) * 100.0, 2)


def _push_issue(
    issues: list[dict],
    *,
    label: str,
    severity: float,
    evidence: str,
    action: str,
) -> None:
    issues.append(
        {
            "label": label,
            "severity_score_10": round(max(0.0, min(10.0, float(severity))), 2),
            "evidence": evidence,
            "recommended_action": action,
        }
    )


def recommended_parameters_for_bottleneck(label: str | None) -> list[dict]:
    key = str(label or "").strip()
    recommendations = BOTTLENECK_PARAMETER_RECOMMENDATIONS.get(key, DEFAULT_PARAMETER_RECOMMENDATIONS)
    return [dict(item) for item in recommendations]


def diagnose_run(row: dict) -> dict:
    issues: list[dict] = []
    transport = row.get("transport_category") or "unknown"
    compute_util = _num(row.get("compute_utilization_p95"))
    render_p95 = _num(row.get("render_latency_p95_ms"))
    candidate_ratio = _num(row.get("candidate_to_confirmed_ratio"))
    display_ratio = _num(row.get("display_to_confirmed_ratio"))
    lead_switch = _num(row.get("lead_confirmed_switch_rate"))
    path_clean = _num(row.get("path_cleanliness_score_10"))
    path_residual = _num(row.get("path_local_residual_rms_m"))
    jump_ratio = _num(row.get("path_jump_ratio"))
    slowest_stage = row.get("slowest_stage_name")
    slowest_stage_p95 = _num(row.get("slowest_stage_p95_ms"))

    if transport == "unusable":
        _push_issue(
            issues,
            label="transport_issue",
            severity=10,
            evidence="transport quality가 unusable입니다.",
            action="알고리즘 비교에서 제외하고 UDP/raw 수신 조건부터 다시 확인합니다.",
        )
    elif transport == "noisy":
        _push_issue(
            issues,
            label="transport_issue",
            severity=7,
            evidence="transport quality가 noisy입니다.",
            action="robustness 확인용으로만 쓰고 baseline 튜닝 기준에서는 낮은 가중치로 봅니다.",
        )

    if compute_util is not None and compute_util >= 1.0:
        _push_issue(
            issues,
            label="compute_latency",
            severity=8.5,
            evidence=f"compute utilization p95={compute_util:.2f}로 frame budget을 넘습니다.",
            action="slowest stage와 stage timing을 먼저 보고 FFT/RAI/detection 계산량을 줄입니다.",
        )
    elif compute_util is not None and compute_util >= 0.85:
        _push_issue(
            issues,
            label="compute_latency",
            severity=6,
            evidence=f"compute utilization p95={compute_util:.2f}로 여유가 작습니다.",
            action="실시간 측정 중 queue 밀림이 생기는지 Stage Debug와 status log를 함께 봅니다.",
        )

    if render_p95 is not None and render_p95 >= 250:
        _push_issue(
            issues,
            label="render_latency",
            severity=7.5,
            evidence=f"render latency p95={render_p95:.1f} ms입니다.",
            action="화면 갱신/로그 쓰기와 처리 루프를 분리해서 체감 지연을 줄입니다.",
        )
    elif render_p95 is not None and render_p95 >= 180:
        _push_issue(
            issues,
            label="render_latency",
            severity=5,
            evidence=f"render latency p95={render_p95:.1f} ms로 경고선에 가깝습니다.",
            action="성능 회귀는 아니더라도 장시간 측정에서 p95/p99 지연을 같이 봅니다.",
        )

    if candidate_ratio is not None and candidate_ratio >= 2.0:
        _push_issue(
            issues,
            label="detection_over_split",
            severity=8,
            evidence=f"candidate/confirmed={candidate_ratio:.2f}로 후보가 과도하게 많습니다.",
            action="body-center merge, adaptive eps, min separation 기준을 먼저 점검합니다.",
        )
    elif candidate_ratio is not None and candidate_ratio >= 1.5:
        _push_issue(
            issues,
            label="detection_over_split",
            severity=5.5,
            evidence=f"candidate/confirmed={candidate_ratio:.2f}로 분해 경향이 있습니다.",
            action="한 사람 세션이면 후보 merge를 조금 더 강하게, 두 사람 세션이면 분리 유지 여부를 같이 봅니다.",
        )

    if display_ratio is not None and display_ratio <= 0.45:
        _push_issue(
            issues,
            label="display_or_confirmation_loss",
            severity=6,
            evidence=f"display/confirmed={display_ratio:.2f}로 화면까지 남는 track이 적습니다.",
            action="confirm_hits, display hysteresis, tentative 표시 정책을 분리해서 봅니다.",
        )

    if lead_switch is not None and lead_switch >= 0.12:
        _push_issue(
            issues,
            label="tracking_association_failure",
            severity=8.5,
            evidence=f"lead switch rate={lead_switch:.2%}로 대표 ID가 자주 바뀝니다.",
            action="association cost에 heading/velocity consistency와 primary hold 정책을 추가로 검토합니다.",
        )
    elif lead_switch is not None and lead_switch >= 0.05:
        _push_issue(
            issues,
            label="tracking_association_failure",
            severity=6,
            evidence=f"lead switch rate={lead_switch:.2%}로 ID continuity가 흔들립니다.",
            action="코너/원운동 세션에서 tracker input detection의 위치 점프를 먼저 확인합니다.",
        )

    if path_residual is not None and path_residual >= 0.18:
        _push_issue(
            issues,
            label="representative_point_jump",
            severity=8,
            evidence=f"local residual RMS={path_residual:.3f} m로 경로 흔들림이 큽니다.",
            action="detection 대표점을 강한 반사점이 아니라 body-center로 안정화하는 쪽을 우선 봅니다.",
        )
    elif path_residual is not None and path_residual >= 0.12:
        _push_issue(
            issues,
            label="representative_point_jump",
            severity=6,
            evidence=f"local residual RMS={path_residual:.3f} m로 기준보다 큽니다.",
            action="Stage Debug에서 RAI peak와 merged detection 위치가 같이 흔들리는지 확인합니다.",
        )

    if jump_ratio is not None and jump_ratio >= 0.08:
        _push_issue(
            issues,
            label="path_jump",
            severity=8,
            evidence=f"jump ratio={jump_ratio:.2%}로 비정상 점프가 많습니다.",
            action="tracker association gate와 measurement soft gate가 점프를 허용하는지 확인합니다.",
        )
    elif jump_ratio is not None and jump_ratio >= 0.05:
        _push_issue(
            issues,
            label="path_jump",
            severity=6,
            evidence=f"jump ratio={jump_ratio:.2%}로 점프 경향이 있습니다.",
            action="동일 raw replay로 튜닝 전후 jump CDF를 비교합니다.",
        )

    if path_clean is not None and path_clean < 5.0:
        _push_issue(
            issues,
            label="path_quality_low",
            severity=8.5,
            evidence=f"path cleanliness={path_clean:.2f}/10입니다.",
            action="raw가 clean이면 detection/tracking 병목이고, raw가 noisy면 먼저 capture 품질을 분리합니다.",
        )
    elif path_clean is not None and path_clean < 7.0:
        _push_issue(
            issues,
            label="path_quality_low",
            severity=5.5,
            evidence=f"path cleanliness={path_clean:.2f}/10으로 발표용 성공 사례 기준에는 부족합니다.",
            action="대표점 흔들림인지 ID switch인지 stage별로 나눠서 확인합니다.",
        )

    if slowest_stage and slowest_stage_p95 is not None and slowest_stage_p95 >= 35:
        _push_issue(
            issues,
            label="stage_hotspot",
            severity=5.5,
            evidence=f"slowest stage={slowest_stage}, p95={slowest_stage_p95:.2f} ms입니다.",
            action=f"`{slowest_stage}` 계산량을 줄이거나 cache/reuse가 가능한지 봅니다.",
        )

    if not issues:
        if transport == "clean" and path_clean is not None and path_clean >= 8.0:
            _push_issue(
                issues,
                label="baseline_candidate",
                severity=1,
                evidence="transport가 clean이고 path cleanliness가 8/10 이상입니다.",
                action="benchmark/baseline 후보로 태깅하고 알고리즘 비교 기준으로 사용합니다.",
            )
        else:
            _push_issue(
                issues,
                label="needs_more_context",
                severity=3,
                evidence="강한 단일 병목은 보이지 않지만 annotation이나 raw link가 더 필요합니다.",
                action="motion pattern, people count, benchmark flag를 먼저 태깅합니다.",
            )

    issues.sort(key=lambda item: item["severity_score_10"], reverse=True)
    primary = issues[0]
    return {
        "primary_bottleneck": primary["label"],
        "severity_score_10": primary["severity_score_10"],
        "primary_evidence": primary["evidence"],
        "recommended_action": primary["recommended_action"],
        "issue_count": len(issues),
        "issues": issues,
    }


def build_diagnosed_run_rows(runs: list[dict]) -> list[dict]:
    diagnosed: list[dict] = []
    for row in runs:
        diagnosis = diagnose_run(row)
        enriched = dict(row)
        enriched.update(
            {
                "primary_bottleneck": diagnosis["primary_bottleneck"],
                "severity_score_10": diagnosis["severity_score_10"],
                "primary_evidence": diagnosis["primary_evidence"],
                "recommended_action": diagnosis["recommended_action"],
                "issue_count": diagnosis["issue_count"],
                "issues": diagnosis["issues"],
            }
        )
        diagnosed.append(enriched)
    diagnosed.sort(
        key=lambda item: (
            float(item.get("severity_score_10") or 0.0),
            str(item.get("created_at") or item.get("session_id") or ""),
        ),
        reverse=True,
    )
    return diagnosed


def _values(rows: list[dict], metric: str) -> list[float]:
    values = []
    for row in rows:
        value = _num(row.get(metric))
        if value is not None:
            values.append(value)
    return values


def metric_summary(rows: list[dict]) -> list[dict]:
    summaries: list[dict] = []
    for key, definition in METRIC_DEFINITIONS.items():
        values = _values(rows, key)
        if not values:
            summaries.append(
                {
                    "metric": definition["label"],
                    "count": 0,
                    "mean": None,
                    "p50": None,
                    "p90": None,
                    "p95": None,
                    "min": None,
                    "max": None,
                    "meaning": definition["meaning"],
                }
            )
            continue
        array = np.asarray(values, dtype=float)
        summaries.append(
            {
                "metric": definition["label"],
                "count": int(array.size),
                "mean": _round(float(mean(values))),
                "p50": _round(float(np.percentile(array, 50))),
                "p90": _round(float(np.percentile(array, 90))),
                "p95": _round(float(np.percentile(array, 95))),
                "min": _round(float(np.min(array))),
                "max": _round(float(np.max(array))),
                "meaning": definition["meaning"],
            }
        )
    return summaries


def _bin_label(value: float | None, bins: list[tuple[str, float | None, float | None]]) -> str:
    if value is None:
        return "n/a"
    for label, lower, upper in bins:
        if lower is not None and value < lower:
            continue
        if upper is not None and value >= upper:
            continue
        return label
    return "n/a"


def pmf_rows(rows: list[dict]) -> list[dict]:
    specs = [
        (
            "transport_category",
            "Transport Quality",
            None,
        ),
        (
            "primary_bottleneck",
            "Primary Bottleneck",
            None,
        ),
        (
            "candidate_to_confirmed_ratio",
            "Candidate / Confirmed",
            [
                ("<=1.2 stable", None, 1.2),
                ("1.2-1.5 mild split", 1.2, 1.5),
                ("1.5-2.0 split", 1.5, 2.0),
                (">=2.0 heavy split", 2.0, None),
            ],
        ),
        (
            "path_cleanliness_score_10",
            "Path Cleanliness",
            [
                ("<4 bad", None, 4.0),
                ("4-6 weak", 4.0, 6.0),
                ("6-8 usable", 6.0, 8.0),
                (">=8 good", 8.0, None),
            ],
        ),
        (
            "lead_confirmed_switch_rate",
            "Lead Switch Rate",
            [
                ("<=3% stable", None, 0.03),
                ("3-8% watch", 0.03, 0.08),
                ("8-15% unstable", 0.08, 0.15),
                (">=15% severe", 0.15, None),
            ],
        ),
    ]
    result: list[dict] = []
    total = max(len(rows), 1)
    for key, family, bins in specs:
        counts: dict[str, int] = {}
        for row in rows:
            if bins is None:
                label = str(row.get(key) or "unknown")
            else:
                label = _bin_label(_num(row.get(key)), bins)
            counts[label] = counts.get(label, 0) + 1
        for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            result.append(
                {
                    "family": family,
                    "bucket": label,
                    "count": count,
                    "probability": round(count / total, 4),
                    "percent": round((count / total) * 100.0, 2),
                }
            )
    return result


def ecdf_target_rows(rows: list[dict]) -> list[dict]:
    result: list[dict] = []
    for target in TARGETS:
        metric = target["metric"]
        values = _values(rows, metric)
        if not values:
            hit_count = 0
            pass_rate = None
        elif target["direction"] == "lower":
            hit_count = sum(1 for value in values if value <= float(target["target"]))
            pass_rate = hit_count / len(values)
        else:
            hit_count = sum(1 for value in values if value >= float(target["target"]))
            pass_rate = hit_count / len(values)
        definition = METRIC_DEFINITIONS[metric]
        result.append(
            {
                "metric": definition["label"],
                "target": target["target"],
                "direction": target["direction"],
                "valid_sessions": len(values),
                "pass_count": hit_count,
                "pass_rate_percent": _pct(pass_rate),
                "meaning": definition["meaning"],
                "why_this_target": target["why"],
            }
        )
    return result


def bottleneck_counts(rows: list[dict]) -> list[dict]:
    total = max(len(rows), 1)
    counts: dict[str, int] = {}
    severity_sum: dict[str, float] = {}
    for row in rows:
        label = str(row.get("primary_bottleneck") or "unknown")
        severity = float(row.get("severity_score_10") or 0.0)
        counts[label] = counts.get(label, 0) + 1
        severity_sum[label] = severity_sum.get(label, 0.0) + severity
    result = []
    for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        result.append(
            {
                "primary_bottleneck": label,
                "count": count,
                "probability": round(count / total, 4),
                "avg_severity_10": round(severity_sum[label] / count, 2),
            }
        )
    return result


def _top_count_label(values: list[str]) -> tuple[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        label = str(value or "unknown")
        counts[label] = counts.get(label, 0) + 1
    if not counts:
        return "unknown", 0
    label, count = max(counts.items(), key=lambda item: (item[1], item[0]))
    return label, count


def parameter_impact_rows(
    project_root: Path,
    runs: list[dict],
    metric: str = "performance_score",
    *,
    varying_only: bool = True,
    min_runs_per_value: int = 1,
) -> list[dict]:
    session_ids = [str(row.get("session_id")) for row in runs if row.get("session_id")]
    if not session_ids:
        return []

    run_by_session = {str(row.get("session_id")): row for row in runs if row.get("session_id")}
    parameter_rows = registry.fetch_parameter_values(project_root, session_ids=session_ids)
    if not parameter_rows:
        return []

    values_by_key: dict[str, set[str]] = {}
    grouped: dict[tuple[str, str], dict] = {}
    for param in parameter_rows:
        session_id = str(param.get("session_id") or "")
        run = run_by_session.get(session_id)
        if run is None:
            continue
        param_key = str(param.get("param_key") or "")
        param_value = str(param.get("param_value") or "")
        if not param_key:
            continue
        values_by_key.setdefault(param_key, set()).add(param_value)
        bucket = grouped.setdefault(
            (param_key, param_value),
            {
                "param_group": param.get("param_group") or "",
                "param_key": param_key,
                "param_value": param_value,
                "sessions": [],
                "metric_values": [],
                "bottlenecks": [],
                "severities": [],
            },
        )
        bucket["sessions"].append(session_id)
        metric_value = _num(run.get(metric))
        if metric_value is not None:
            bucket["metric_values"].append(metric_value)
        bottleneck = run.get("primary_bottleneck")
        severity = _num(run.get("severity_score_10"))
        if bottleneck is None or severity is None:
            diagnosis = diagnose_run(run)
            bottleneck = diagnosis["primary_bottleneck"]
            severity = diagnosis["severity_score_10"]
        bucket["bottlenecks"].append(str(bottleneck or "unknown"))
        if severity is not None:
            bucket["severities"].append(severity)

    metric_label = METRIC_DEFINITIONS.get(metric, {}).get("label", metric)
    result: list[dict] = []
    for (param_key, _param_value), bucket in grouped.items():
        distinct_values = len(values_by_key.get(param_key, set()))
        if varying_only and distinct_values <= 1:
            continue
        run_count = len(set(bucket["sessions"]))
        if run_count < int(min_runs_per_value):
            continue
        metric_values = bucket["metric_values"]
        if metric_values:
            array = np.asarray(metric_values, dtype=float)
            metric_mean = _round(float(mean(metric_values)))
            metric_p50 = _round(float(np.percentile(array, 50)))
            metric_min = _round(float(np.min(array)))
            metric_max = _round(float(np.max(array)))
        else:
            metric_mean = None
            metric_p50 = None
            metric_min = None
            metric_max = None
        top_bottleneck, top_bottleneck_count = _top_count_label(bucket["bottlenecks"])
        severities = bucket["severities"]
        result.append(
            {
                "param_group": bucket["param_group"],
                "param_key": param_key,
                "param_value": bucket["param_value"],
                "run_count": run_count,
                "distinct_values": distinct_values,
                "metric": metric_label,
                "metric_count": len(metric_values),
                "metric_mean": metric_mean,
                "metric_p50": metric_p50,
                "metric_min": metric_min,
                "metric_max": metric_max,
                "top_bottleneck": top_bottleneck,
                "top_bottleneck_count": top_bottleneck_count,
                "avg_severity_10": _round(float(mean(severities)), 2) if severities else None,
                "sessions": ", ".join(sorted(set(bucket["sessions"]))[:8]),
            }
        )

    result.sort(key=lambda item: (-int(item["distinct_values"]), str(item["param_key"]), -int(item["run_count"]), str(item["param_value"])))
    return result


def build_snapshot(project_root: Path, runs: list[dict] | None = None) -> dict:
    project_root = Path(project_root)
    if runs is None:
        runs = registry.fetch_runs(project_root)
    diagnosed = build_diagnosed_run_rows(runs)
    run_exports = []
    for row in diagnosed:
        session_id = row.get("session_id")
        feature_summary = stage_cache.load_stage_feature_summary(project_root, session_id) if session_id else None
        run_exports.append(
            {
                "session_id": row.get("session_id"),
                "input_mode": row.get("input_mode"),
                "transport_category": row.get("transport_category"),
                "annotation_label": row.get("annotation_label"),
                "annotation_people_count": row.get("annotation_people_count"),
                "annotation_motion_pattern": row.get("annotation_motion_pattern"),
                "capture_id": row.get("capture_id"),
                "performance_score": _round(_num(row.get("performance_score"))),
                "path_cleanliness_score_10": _round(_num(row.get("path_cleanliness_score_10"))),
                "path_local_residual_rms_m": _round(_num(row.get("path_local_residual_rms_m"))),
                "path_jump_ratio": _round(_num(row.get("path_jump_ratio"))),
                "lead_confirmed_switch_rate": _round(_num(row.get("lead_confirmed_switch_rate"))),
                "candidate_to_confirmed_ratio": _round(_num(row.get("candidate_to_confirmed_ratio"))),
                "display_to_confirmed_ratio": _round(_num(row.get("display_to_confirmed_ratio"))),
                "render_latency_p95_ms": _round(_num(row.get("render_latency_p95_ms"))),
                "compute_utilization_p95": _round(_num(row.get("compute_utilization_p95"))),
                "slowest_stage_name": row.get("slowest_stage_name"),
                "slowest_stage_p95_ms": _round(_num(row.get("slowest_stage_p95_ms"))),
                "primary_bottleneck": row.get("primary_bottleneck"),
                "severity_score_10": row.get("severity_score_10"),
                "primary_evidence": row.get("primary_evidence"),
                "recommended_action": row.get("recommended_action"),
                "recommended_parameters": recommended_parameters_for_bottleneck(row.get("primary_bottleneck")),
                "stage_feature_summary": feature_summary,
                "issues": row.get("issues"),
            }
        )

    return {
        "schema_version": 1,
        "generated_at": _now(),
        "project_root": str(project_root.resolve()),
        "run_count": len(diagnosed),
        "bottleneck_counts": bottleneck_counts(diagnosed),
        "metric_summary": metric_summary(diagnosed),
        "pmf": pmf_rows(diagnosed),
        "ecdf_targets": ecdf_target_rows(diagnosed),
        "bottleneck_parameter_recommendations": BOTTLENECK_PARAMETER_RECOMMENDATIONS,
        "runs": run_exports,
    }


def write_snapshot(project_root: Path, output_path: Path | None = None) -> Path:
    project_root = Path(project_root).resolve()
    snapshot = build_snapshot(project_root)
    if output_path is None:
        output_path = project_root / "lab_data" / "analytics" / "run_triage_snapshot.json"
    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = project_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Radar Lab analytics and bottleneck triage snapshot.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", default=None, help="Optional output JSON path.")
    parser.add_argument("--refresh", action="store_true", help="Refresh registry before analysis.")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    if args.refresh:
        registry.refresh_registry(project_root)
    if args.out:
        output_path = write_snapshot(project_root, Path(args.out))
        print(json.dumps({"written": str(output_path)}, ensure_ascii=False, indent=2))
    else:
        snapshot = build_snapshot(project_root)
        print(
            json.dumps(
                {
                    "run_count": snapshot["run_count"],
                    "bottleneck_counts": snapshot["bottleneck_counts"],
                    "ecdf_targets": snapshot["ecdf_targets"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
