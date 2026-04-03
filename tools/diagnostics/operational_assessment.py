from __future__ import annotations

from datetime import datetime


GRADE_BANDS = (
    (85, "A", "현장 투입 가능", "good"),
    (75, "B", "제한적 현장 파일럿 가능", "good"),
    (65, "C", "PoC 이상, 현장 전 고도화 필요", "brand"),
    (55, "D", "랩 프로토타입", "warn"),
    (0, "F", "디버그 단계", "danger"),
)


def round_or_none(value, digits=3):
    if value is None:
        return None
    return round(float(value), digits)


def build_event_summary(events: list[dict]):
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

    def parse_wall_time(event: dict | None):
        if not event:
            return None
        wall_time = event.get("wall_time")
        if not wall_time:
            return None
        try:
            return datetime.fromisoformat(wall_time)
        except ValueError:
            return None

    radar_open_start = first("radar_open_start")
    shutdown_start = first("shutdown_start")
    first_render = first("first_rendered_frame")
    session_error = first("session_error")

    session_duration_s = None
    radar_open_start_ts = parse_wall_time(radar_open_start)
    shutdown_start_ts = parse_wall_time(shutdown_start)
    if radar_open_start_ts and shutdown_start_ts:
        session_duration_s = max(
            (shutdown_start_ts - radar_open_start_ts).total_seconds(),
            0.0,
        )

    return {
        "event_count": len(events),
        "dca_config_complete": first("dca_config_complete") is not None,
        "radar_open_complete": first("radar_open_complete") is not None,
        "first_rendered_frame": first_render is not None,
        "first_render_elapsed_s": round_or_none(
            None if first_render is None else first_render.get("elapsed_since_stream_start_s"),
            digits=4,
        ),
        "session_duration_s": round_or_none(session_duration_s, digits=3),
        "opengl_unavailable": first("opengl_unavailable") is not None,
        "session_error": session_error is not None,
        "session_error_repr": None if session_error is None else session_error.get("error"),
    }


def _safe_ratio(numerator, denominator):
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def _score_lower(value, thresholds: list[tuple[float, int]]):
    if value is None:
        return 0
    numeric = float(value)
    for limit, score in thresholds:
        if numeric <= limit:
            return int(score)
    return 0


def _score_higher(value, thresholds: list[tuple[float, int]]):
    if value is None:
        return 0
    numeric = float(value)
    for limit, score in thresholds:
        if numeric >= limit:
            return int(score)
    return 0


def _tone_for_ratio(ratio: float):
    if ratio >= 0.85:
        return "good"
    if ratio >= 0.7:
        return "brand"
    if ratio >= 0.5:
        return "warn"
    return "danger"


def _grade_for_score(score: int):
    for minimum, grade, label, tone in GRADE_BANDS:
        if score >= minimum:
            return {
                "grade": grade,
                "label": label,
                "tone": tone,
            }
    return {
        "grade": "F",
        "label": "디버그 단계",
        "tone": "danger",
    }


def _top_reason(reason_counts: dict):
    if not reason_counts:
        return None, 0
    top_reason = max(reason_counts.items(), key=lambda item: item[1])
    return top_reason[0], int(top_reason[1])


def build_operational_assessment(summary: dict, event_summary: dict):
    processed = summary.get("processed", {})
    render = summary.get("render", {})

    render_p95 = render.get("capture_to_render_ms", {}).get("p95")
    render_mean = render.get("capture_to_render_ms", {}).get("mean")
    process_to_render_p95 = render.get("process_to_render_ms", {}).get("p95")
    invalid_rate = render.get("invalid_rate")
    birth_block_rate = processed.get("birth_block_rate")
    skipped_mean = render.get("skipped_render_frames", {}).get("mean_per_render")
    processed_multi_success = processed.get("multi_confirmed_success_rate")
    render_multi_success = render.get("multi_display_success_rate")
    confirmed_mean = processed.get("confirmed_track_count", {}).get("mean")
    display_mean = render.get("display_track_count", {}).get("mean")
    display_to_confirmed_ratio = _safe_ratio(display_mean, confirmed_mean)
    first_render_elapsed_s = event_summary.get("first_render_elapsed_s")
    session_duration_s = event_summary.get("session_duration_s")

    latency_score = (
        _score_lower(render_p95, [(160, 20), (190, 16), (220, 12), (250, 8), (280, 4)])
        + _score_lower(render_mean, [(140, 10), (170, 8), (200, 6), (230, 4), (260, 2)])
        + _score_lower(process_to_render_p95, [(20, 5), (30, 4), (40, 2), (50, 1)])
    )
    integrity_score = (
        _score_lower(invalid_rate, [(0.005, 16), (0.01, 14), (0.02, 12), (0.05, 9), (0.10, 5)])
        + _score_lower(birth_block_rate, [(0.001, 8), (0.005, 7), (0.02, 6), (0.05, 4), (0.10, 2)])
        + _score_lower(skipped_mean, [(0.0, 6), (0.05, 4), (0.2, 2)])
    )
    visibility_score = (
        _score_higher(processed_multi_success, [(0.9, 6), (0.75, 5), (0.5, 3), (0.25, 1)])
        + _score_higher(render_multi_success, [(0.7, 8), (0.5, 6), (0.35, 4), (0.2, 2), (0.05, 1)])
        + _score_higher(display_to_confirmed_ratio, [(0.85, 6), (0.65, 5), (0.5, 4), (0.35, 2), (0.15, 1)])
    )
    readiness_score = (
        _score_lower(first_render_elapsed_s, [(0.5, 4), (1.0, 3), (2.0, 2)])
        + (4 if not event_summary.get("session_error") else 0)
        + (3 if not event_summary.get("opengl_unavailable") else 1)
        + _score_higher(session_duration_s, [(20, 2), (10, 1)])
        + (
            2
            if (
                event_summary.get("dca_config_complete")
                and event_summary.get("radar_open_complete")
                and event_summary.get("first_rendered_frame")
            )
            else 0
        )
    )

    category_scores = {
        "latency": {
            "label": "지연",
            "score": latency_score,
            "max_score": 35,
            "metrics": {
                "render_p95_ms": render_p95,
                "render_mean_ms": render_mean,
                "process_to_render_p95_ms": process_to_render_p95,
            },
        },
        "integrity": {
            "label": "수집 무결성",
            "score": integrity_score,
            "max_score": 30,
            "metrics": {
                "render_invalid_rate": invalid_rate,
                "birth_block_rate": birth_block_rate,
                "skipped_per_render": skipped_mean,
            },
        },
        "visibility": {
            "label": "표시/추적 가시성",
            "score": visibility_score,
            "max_score": 20,
            "metrics": {
                "processed_multi_success_rate": processed_multi_success,
                "render_multi_success_rate": render_multi_success,
                "display_to_confirmed_ratio": round_or_none(display_to_confirmed_ratio, digits=3),
            },
        },
        "readiness": {
            "label": "운영 준비도",
            "score": readiness_score,
            "max_score": 15,
            "metrics": {
                "first_render_elapsed_s": first_render_elapsed_s,
                "session_duration_s": session_duration_s,
                "session_error": bool(event_summary.get("session_error")),
                "opengl_unavailable": bool(event_summary.get("opengl_unavailable")),
            },
        },
    }

    for category in category_scores.values():
        ratio = category["score"] / category["max_score"] if category["max_score"] else 0.0
        category["ratio"] = round_or_none(ratio, digits=3)
        category["tone"] = _tone_for_ratio(ratio)

    total_score = int(sum(item["score"] for item in category_scores.values()))
    grade = _grade_for_score(total_score)

    processed_top_reason, processed_top_reason_count = _top_reason(
        processed.get("invalid_reason_counts", {})
    )
    render_top_reason, render_top_reason_count = _top_reason(
        render.get("invalid_reason_counts", {})
    )

    strengths = []
    issues = []
    recommendations = []

    if skipped_mean == 0:
        strengths.append("렌더 큐 기준 프레임 스킵이 관측되지 않았습니다.")
    if first_render_elapsed_s is not None and first_render_elapsed_s <= 0.5:
        strengths.append(
            f"첫 화면 표시가 {first_render_elapsed_s:.3f}s로 빨라 초기 기동 응답성은 양호합니다."
        )
    if processed_multi_success is not None and processed_multi_success >= 0.75:
        strengths.append(
            f"내부 추적 단계의 다중 타깃 유지율이 {processed_multi_success * 100:.1f}%로 높습니다."
        )
    if not event_summary.get("session_error"):
        strengths.append("세션이 치명적 예외 없이 종료되었습니다.")

    if render_p95 is not None and render_p95 > 220:
        issues.append(
            {
                "severity": "high",
                "title": "엔드투엔드 지연이 현업 체감 기준을 넘습니다.",
                "detail": (
                    f"render p95가 {render_p95:.1f}ms, 평균이 {render_mean:.1f}ms입니다. "
                    "지연의 대부분이 수집 이후 처리 구간에 쌓이고 있습니다."
                ),
            }
        )
        recommendations.append(
            "DSP 처리 시간을 줄이거나 프레임 주기를 낮춰 render p95를 220ms 아래로 내리세요."
        )

    if invalid_rate is not None and invalid_rate >= 0.01:
        reason_text = render_top_reason or processed_top_reason or "unknown"
        reason_count = render_top_reason_count or processed_top_reason_count
        issues.append(
            {
                "severity": "high",
                "title": "간헐적 UDP 무결성 문제가 남아 있습니다.",
                "detail": (
                    f"render invalid rate가 {invalid_rate * 100:.2f}%이며 "
                    f"주된 원인은 {reason_text} ({reason_count}건) 입니다."
                ),
            }
        )
        recommendations.append(
            "NIC 고정 IP, 방화벽, 케이블 상태, DCA1000 링크 품질, SO_RCVBUF 동작을 점검하세요."
        )

    if (
        processed_multi_success is not None
        and render_multi_success is not None
        and processed_multi_success >= 0.7
        and render_multi_success <= 0.25
    ):
        issues.append(
            {
                "severity": "medium",
                "title": "내부 추적 결과에 비해 UI 다중 타깃 표시력이 낮습니다.",
                "detail": (
                    f"processed multi-target success는 {processed_multi_success * 100:.1f}%인데 "
                    f"render multi-target success는 {render_multi_success * 100:.1f}%에 그칩니다."
                ),
            }
        )
        recommendations.append(
            "display threshold와 report miss tolerance를 다시 조정해 화면에서 사라지는 트랙을 줄이세요."
        )

    if event_summary.get("opengl_unavailable"):
        issues.append(
            {
                "severity": "medium",
                "title": "OpenGL 환경 누락으로 3D 시각화가 비활성화되었습니다.",
                "detail": "PyOpenGL 또는 관련 런타임이 없어 spatial view가 동작하지 않았습니다.",
            }
        )
        recommendations.append(
            "배포 환경에 PyOpenGL을 포함해 현장 디버깅 시 3D spatial view를 사용할 수 있게 하세요."
        )

    if not issues:
        issues.append(
            {
                "severity": "low",
                "title": "치명적 이슈는 관측되지 않았습니다.",
                "detail": "현재 세션은 주요 기준을 만족했으며 추세 관찰 단계에 가깝습니다.",
            }
        )

    if not recommendations:
        recommendations.append("현재 설정으로 반복 측정을 이어가며 장기 추세를 확보하세요.")

    overall_summary = (
        f"{grade['label']} 수준입니다. "
        f"지연 점수 {latency_score}/35, 무결성 {integrity_score}/30, "
        f"표시/추적 {visibility_score}/20, 운영 준비도 {readiness_score}/15입니다."
    )

    return {
        "rubric_version": 1,
        "overall": {
            "score": total_score,
            "max_score": 100,
            "grade": grade["grade"],
            "label": grade["label"],
            "tone": grade["tone"],
            "summary": overall_summary,
        },
        "category_scores": category_scores,
        "strengths": strengths,
        "issues": issues,
        "recommendations": recommendations,
        "derived_metrics": {
            "display_to_confirmed_ratio": round_or_none(display_to_confirmed_ratio, digits=3),
            "processed_top_invalid_reason": processed_top_reason,
            "render_top_invalid_reason": render_top_reason,
        },
    }
