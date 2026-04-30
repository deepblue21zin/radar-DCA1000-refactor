from math import atan2, hypot

import numpy as np

from .dbscan_cluster import cluster_points
from .detection_core.cfar import cfar_threshold_2d, local_maxima_mask as _local_maxima_mask
from .detection_core.refinement import (
    body_center_patch_for_range as _body_center_patch_for_range,
    refine_body_center_from_patch as _refine_body_center_from_patch,
)
from .detection_core.trace import (
    trace_candidate as _trace_candidate,
    trace_candidates as _trace_candidates,
    trace_reject as _trace_reject,
)
from .detection_core.types import DetectionCandidate, DetectionRegion


def _angle_roi_mask(range_m, angle_axis_rad, detection_region):
    x_axis = range_m * np.sin(angle_axis_rad)
    y_axis = range_m * np.cos(angle_axis_rad)
    return (
        (np.abs(x_axis) <= detection_region.lateral_limit_m)
        & (y_axis >= detection_region.min_forward_m)
        & (y_axis <= detection_region.forward_limit_m)
    )


def _angle_is_local_peak(angle_profile, angle_bin):
    left_index = max(angle_bin - 1, 0)
    right_index = min(angle_bin + 1, angle_profile.shape[0] - 1)
    return (
        angle_profile[angle_bin] >= angle_profile[left_index]
        and angle_profile[angle_bin] >= angle_profile[right_index]
    )


def _nearest_axis_bin(axis_values, value):
    return int(np.argmin(np.abs(np.asarray(axis_values) - value)))


def _angle_centroid_radius_for_range(range_m, radius_bands, default_radius=1):
    if not radius_bands:
        return int(default_radius)

    for band in radius_bands:
        try:
            r_min = float(band.get("r_min", 0.0))
            r_max = band.get("r_max")
            radius = int(band.get("radius", default_radius))
        except (TypeError, ValueError, AttributeError):
            continue

        if radius < 1:
            radius = int(default_radius)
        if range_m < r_min:
            continue
        if r_max is None or float(range_m) < float(r_max):
            return radius

    return int(default_radius)


def _candidate_merge_window_for_range(
    range_m,
    merge_bands,
    default_merge_radius_m=0.40,
    default_range_bin_radius=1,
    default_doppler_bin_radius=2,
):
    merge_radius_m = float(default_merge_radius_m)
    range_bin_radius = int(default_range_bin_radius)
    doppler_bin_radius = int(default_doppler_bin_radius)

    if not merge_bands:
        return merge_radius_m, range_bin_radius, doppler_bin_radius

    for band in merge_bands:
        try:
            r_min = float(band.get("r_min", 0.0))
            r_max = band.get("r_max")
        except (TypeError, ValueError, AttributeError):
            continue

        if range_m < r_min:
            continue
        if r_max is not None and float(range_m) >= float(r_max):
            continue

        try:
            merge_radius_m = max(0.05, float(band.get("merge_radius_m", merge_radius_m)))
            range_bin_radius = max(0, int(band.get("range_bin_radius", range_bin_radius)))
            doppler_bin_radius = max(0, int(band.get("doppler_bin_radius", doppler_bin_radius)))
        except (TypeError, ValueError, AttributeError):
            return (
                float(default_merge_radius_m),
                int(default_range_bin_radius),
                int(default_doppler_bin_radius),
            )
        break

    return merge_radius_m, range_bin_radius, doppler_bin_radius


def _refine_angle_centroid(
    angle_profile,
    angle_axis_rad,
    peak_bin,
    angle_floor,
    angle_mask,
    radius=1,
):
    lower = max(int(peak_bin) - int(radius), 0)
    upper = min(int(peak_bin) + int(radius) + 1, angle_profile.shape[0])
    local_bins = np.arange(lower, upper)
    local_bins = local_bins[np.asarray(angle_mask[lower:upper], dtype=bool)]
    if local_bins.size == 0:
        return int(peak_bin), float(angle_axis_rad[int(peak_bin)])

    local_values = np.asarray(angle_profile[local_bins], dtype=np.float64)
    weights = np.maximum(local_values - max(float(angle_floor), 0.0), 0.0)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 1e-9:
        weights = np.maximum(local_values, 0.0)
        weight_sum = float(np.sum(weights))
    if weight_sum <= 1e-9:
        return int(peak_bin), float(angle_axis_rad[int(peak_bin)])

    refined_angle_rad = float(
        np.sum(np.asarray(angle_axis_rad[local_bins], dtype=np.float64) * weights)
        / weight_sum
    )
    refined_angle_bin = _nearest_axis_bin(angle_axis_rad, refined_angle_rad)
    return refined_angle_bin, refined_angle_rad


def _doppler_bin_distance(left_bin, right_bin, fft_size):
    delta = abs(int(left_bin) - int(right_bin))
    try:
        fft_size = int(fft_size)
    except (TypeError, ValueError):
        return delta
    if fft_size <= 0:
        return delta
    return min(delta, max(fft_size - delta, 0))


def _merge_candidate_pool(
    candidate_pool,
    runtime_config,
    merge_bands=None,
    default_merge_radius_m=0.40,
    default_range_bin_radius=1,
    default_doppler_bin_radius=2,
):
    if len(candidate_pool) <= 1:
        return list(candidate_pool)

    def _create_group(candidate):
        weight = max(float(candidate.score), 1e-3)
        return {
            "weight_sum": weight,
            "x_sum": float(candidate.x_m) * weight,
            "y_sum": float(candidate.y_m) * weight,
            "doppler_sum": float(candidate.doppler_bin) * weight,
            "score_max": float(candidate.score),
            "rdi_peak_max": float(candidate.rdi_peak),
            "rai_peak_max": float(candidate.rai_peak),
            "member_count": 1,
            "x_m": float(candidate.x_m),
            "y_m": float(candidate.y_m),
            "range_m": float(candidate.range_m),
            "range_bin": int(candidate.range_bin),
            "angle_bin": int(candidate.angle_bin),
            "angle_deg": float(candidate.angle_deg),
            "doppler_bin": int(candidate.doppler_bin),
        }

    def _recompute_group(group):
        weight_sum = max(float(group["weight_sum"]), 1e-6)
        x_m = float(group["x_sum"] / weight_sum)
        y_m = float(group["y_sum"] / weight_sum)
        range_m = float(hypot(x_m, y_m))
        angle_rad = float(atan2(x_m, max(y_m, 1e-6)))
        group["x_m"] = x_m
        group["y_m"] = y_m
        group["range_m"] = range_m
        group["range_bin"] = _nearest_axis_bin(runtime_config.range_axis_m, range_m)
        group["angle_bin"] = _nearest_axis_bin(runtime_config.angle_axis_rad, angle_rad)
        group["angle_deg"] = float(np.degrees(angle_rad))
        group["doppler_bin"] = int(round(group["doppler_sum"] / weight_sum))

    groups = []
    for candidate in candidate_pool:
        best_group_index = None
        best_distance = None

        for group_index, group in enumerate(groups):
            reference_range_m = max(float(candidate.range_m), float(group["range_m"]))
            merge_radius_m, range_bin_radius, doppler_bin_radius = _candidate_merge_window_for_range(
                reference_range_m,
                merge_bands,
                default_merge_radius_m=default_merge_radius_m,
                default_range_bin_radius=default_range_bin_radius,
                default_doppler_bin_radius=default_doppler_bin_radius,
            )
            cart_distance = float(hypot(candidate.x_m - group["x_m"], candidate.y_m - group["y_m"]))
            if cart_distance > merge_radius_m:
                continue
            if abs(int(candidate.range_bin) - int(group["range_bin"])) > range_bin_radius:
                continue
            if _doppler_bin_distance(candidate.doppler_bin, group["doppler_bin"], runtime_config.doppler_fft_size) > doppler_bin_radius:
                continue
            if best_distance is None or cart_distance < best_distance:
                best_distance = cart_distance
                best_group_index = group_index

        if best_group_index is None:
            groups.append(_create_group(candidate))
            continue

        group = groups[best_group_index]
        weight = max(float(candidate.score), 1e-3)
        group["weight_sum"] += weight
        group["x_sum"] += float(candidate.x_m) * weight
        group["y_sum"] += float(candidate.y_m) * weight
        group["doppler_sum"] += float(candidate.doppler_bin) * weight
        group["score_max"] = max(float(group["score_max"]), float(candidate.score))
        group["rdi_peak_max"] = max(float(group["rdi_peak_max"]), float(candidate.rdi_peak))
        group["rai_peak_max"] = max(float(group["rai_peak_max"]), float(candidate.rai_peak))
        group["member_count"] = int(group["member_count"]) + 1
        _recompute_group(group)

    merged_candidates = []
    for group in groups:
        score_scale = min(1.20, 1.0 + 0.05 * max(int(group["member_count"]) - 1, 0))
        merged_candidates.append(
            DetectionCandidate(
                range_bin=int(group["range_bin"]),
                doppler_bin=int(group["doppler_bin"]),
                angle_bin=int(group["angle_bin"]),
                range_m=float(group["range_m"]),
                angle_deg=float(group["angle_deg"]),
                x_m=float(group["x_m"]),
                y_m=float(group["y_m"]),
                rdi_peak=float(group["rdi_peak_max"]),
                rai_peak=float(group["rai_peak_max"]),
                score=float(group["score_max"]) * score_scale,
            )
        )

    merged_candidates.sort(
        key=lambda candidate: (candidate.score, candidate.rdi_peak, candidate.rai_peak),
        reverse=True,
    )
    return merged_candidates


def _select_cluster_representative(members, cluster):
    """Pick an actual candidate so DBSCAN does not invent an off-path centroid."""
    if len(members) <= 1:
        return members[0]

    xs = np.asarray([float(member.x_m) for member in members], dtype=np.float64)
    ys = np.asarray([float(member.y_m) for member in members], dtype=np.float64)
    median_x = float(np.median(xs))
    median_y = float(np.median(ys))
    eps_used = max(float(cluster.get("eps_used", 0.0)), 1e-6)
    max_score = max(max(float(member.score) for member in members), 1e-6)
    max_rdi = max(max(float(member.rdi_peak) for member in members), 1e-6)
    max_rai = max(max(float(member.rai_peak) for member in members), 1e-6)

    def _rank(member):
        distance_m = float(hypot(float(member.x_m) - median_x, float(member.y_m) - median_y))
        distance_penalty = min(distance_m / eps_used, 2.0)
        score_norm = float(member.score) / max_score
        rdi_norm = float(member.rdi_peak) / max_rdi
        rai_norm = float(member.rai_peak) / max_rai
        representative_score = (
            score_norm
            + (0.10 * rdi_norm)
            + (0.10 * rai_norm)
            - (0.25 * distance_penalty)
        )
        return (
            representative_score,
            float(member.score),
            float(member.rdi_peak),
            float(member.rai_peak),
            -distance_m,
        )

    return max(members, key=_rank)


def _cluster_detection_candidates(
    candidate_pool,
    runtime_config,
    detection_region,
    min_cartesian_separation_m,
):
    point_cloud = []
    for candidate_index, candidate in enumerate(candidate_pool):
        point_cloud.append(
            {
                'cluster_index': candidate_index,
                'x': candidate.x_m,
                'y': candidate.y_m,
                'v': float(candidate.doppler_bin),
                'range': candidate.range_m,
                'score': candidate.score,
            }
        )

    clusters = cluster_points(
        point_cloud,
        eps=min_cartesian_separation_m,
        min_samples=detection_region.cluster_min_samples,
        use_velocity_feature=detection_region.cluster_velocity_weight > 0.0,
        velocity_weight=detection_region.cluster_velocity_weight,
        adaptive_eps_bands=detection_region.adaptive_eps_bands,
    )
    if not clusters:
        if candidate_pool:
            fallback = max(
                candidate_pool,
                key=lambda candidate: (candidate.score, candidate.rdi_peak, candidate.rai_peak),
            )
            return [fallback]
        return []

    detections = []
    for cluster in clusters:
        member_points = cluster.get("member_points") or []
        member_indices = [
            int(member["cluster_index"])
            for member in member_points
            if "cluster_index" in member
        ]
        if not member_indices:
            continue

        members = [candidate_pool[index] for index in member_indices]
        representative = _select_cluster_representative(members, cluster)
        x_m = float(representative.x_m)
        y_m = float(representative.y_m)
        range_m = float(representative.range_m)
        angle_rad = float(atan2(x_m, max(y_m, 1e-6)))
        range_bin = int(representative.range_bin)
        angle_bin = int(representative.angle_bin)
        doppler_bin = int(
            round(
                sum(member.doppler_bin * member.score for member in members)
                / max(sum(member.score for member in members), 1e-6)
            )
        )
        detections.append(
            DetectionCandidate(
                range_bin=range_bin,
                doppler_bin=doppler_bin,
                angle_bin=angle_bin,
                range_m=range_m,
                angle_deg=float(np.degrees(angle_rad)),
                x_m=x_m,
                y_m=y_m,
                rdi_peak=max(member.rdi_peak for member in members),
                rai_peak=max(member.rai_peak for member in members),
                score=float(
                    max(cluster.get("peak_score", 0.0), representative.score)
                    * max(cluster.get("confidence", 0.0), 0.5)
                ),
            )
        )

    detections.sort(
        key=lambda candidate: (candidate.score, candidate.rdi_peak, candidate.rai_peak),
        reverse=True,
    )
    return detections


def _suppress_duplicate_candidates(
    candidates,
    runtime_config,
    enabled=True,
    radius_m=0.55,
    range_scale=0.0,
    doppler_bins=6,
    score_ratio=0.82,
):
    if not enabled or len(candidates) <= 1:
        return list(candidates), []

    radius_m = max(0.0, float(radius_m))
    range_scale = max(0.0, float(range_scale))
    doppler_bins = max(0, int(doppler_bins))
    score_ratio = float(np.clip(score_ratio, 0.0, 1.0))
    if radius_m <= 0.0 or score_ratio <= 0.0:
        return list(candidates), []

    ordered = sorted(
        candidates,
        key=lambda candidate: (candidate.score, candidate.rdi_peak, candidate.rai_peak),
        reverse=True,
    )
    kept = []
    suppressed = []
    for candidate in ordered:
        duplicate_of = None
        duplicate_distance_m = None
        duplicate_doppler_bins = None
        duplicate_score_ratio = None
        for reference in kept:
            effective_radius_m = radius_m + (
                range_scale * max(float(candidate.range_m), float(reference.range_m))
            )
            distance_m = float(hypot(candidate.x_m - reference.x_m, candidate.y_m - reference.y_m))
            if distance_m > effective_radius_m:
                continue

            doppler_distance = _doppler_bin_distance(
                candidate.doppler_bin,
                reference.doppler_bin,
                runtime_config.doppler_fft_size,
            )
            if doppler_distance > doppler_bins:
                continue

            relative_score = float(candidate.score) / max(float(reference.score), 1e-6)
            if relative_score > score_ratio:
                continue

            duplicate_of = reference
            duplicate_distance_m = distance_m
            duplicate_doppler_bins = doppler_distance
            duplicate_score_ratio = relative_score
            break

        if duplicate_of is None:
            kept.append(candidate)
            continue

        suppressed.append(
            {
                "candidate": _trace_candidate(candidate),
                "duplicate_of": _trace_candidate(duplicate_of),
                "distance_m": round(float(duplicate_distance_m), 4),
                "doppler_bins": int(duplicate_doppler_bins),
                "score_ratio": round(float(duplicate_score_ratio), 4),
            }
        )

    return kept, suppressed


def detect_targets(
    rdi_map,
    rai_map,
    runtime_config,
    min_range_bin,
    max_range_bin,
    detection_region,
    cfar_training_cells=(6, 6),
    cfar_guard_cells=(1, 1),
    cfar_scale=5.0,
    global_quantile=0.985,
    angle_quantile=0.75,
    angle_contrast_scale=1.35,
    min_cartesian_separation_m=0.45,
    angle_centroid_radius_bands=None,
    body_center_patch_bands=None,
    candidate_merge_bands=None,
    duplicate_suppression_enabled=True,
    duplicate_suppression_radius_m=0.55,
    duplicate_suppression_range_scale=0.03,
    duplicate_suppression_doppler_bins=6,
    duplicate_suppression_score_ratio=0.82,
    trace=None,
):
    trace_enabled = trace is not None
    if trace_enabled:
        trace.clear()
        trace.update(
            {
                "trace_version": 1,
                "roi": {
                    "min_range_bin": int(min_range_bin),
                    "max_range_bin": int(max_range_bin),
                    "max_targets": int(detection_region.max_targets),
                    "lateral_limit_m": float(detection_region.lateral_limit_m),
                    "forward_limit_m": float(detection_region.forward_limit_m),
                    "min_forward_m": float(detection_region.min_forward_m),
                },
                "reject_reasons": {},
            }
        )
    rdi_roi = np.asarray(rdi_map[min_range_bin:max_range_bin], dtype=np.float64)
    if rdi_roi.size == 0:
        if trace_enabled:
            trace["early_exit"] = "empty_rdi_roi"
        return []

    rdi_work = np.array(rdi_roi, copy=True)
    center_bin = runtime_config.doppler_fft_size // 2
    guard_bins = runtime_config.doppler_guard_bins
    lower = max(center_bin - guard_bins, 0)
    upper = min(center_bin + guard_bins + 1, runtime_config.doppler_fft_size)
    rdi_work[:, lower:upper] = 0

    # Suppress broad horizontal bands so compact moving peaks stand out.
    rdi_work = np.maximum(
        rdi_work - np.median(rdi_work, axis=1, keepdims=True),
        0,
    )
    power_map = np.square(rdi_work)
    if np.max(power_map) <= 0:
        if trace_enabled:
            trace["early_exit"] = "zero_power_map"
        return []

    cfar_noise = cfar_threshold_2d(
        power_map,
        training_cells=tuple(cfar_training_cells),
        guard_cells=tuple(cfar_guard_cells),
    )
    threshold_floor = np.quantile(power_map, global_quantile)
    threshold_map = np.maximum(cfar_noise * cfar_scale, threshold_floor)
    peak_mask = (power_map > threshold_map) & _local_maxima_mask(power_map)
    candidate_indices = np.argwhere(peak_mask)

    if candidate_indices.size == 0 and detection_region.allow_strongest_fallback:
        strongest_index = np.unravel_index(np.argmax(power_map), power_map.shape)
        candidate_indices = np.array([strongest_index])

    if candidate_indices.size == 0:
        if trace_enabled:
            trace["cfar"] = {
                "candidate_count": 0,
                "threshold_floor": round(float(threshold_floor), 4),
                "power_max": round(float(np.max(power_map)), 4),
                "fallback_used": False,
                "top_candidates": [],
            }
            trace["early_exit"] = "no_cfar_candidates"
        return []

    candidate_scores = power_map[candidate_indices[:, 0], candidate_indices[:, 1]]
    ordered_indices = candidate_indices[np.argsort(candidate_scores)[::-1]]
    if trace_enabled:
        top_cfar = []
        for range_bin_rel, doppler_bin in ordered_indices[:24]:
            top_cfar.append(
                {
                    "range_bin": int(range_bin_rel + min_range_bin),
                    "doppler_bin": int(doppler_bin),
                    "range_m": round(float(runtime_config.range_axis_m[int(range_bin_rel + min_range_bin)]), 4),
                    "power": round(float(power_map[int(range_bin_rel), int(doppler_bin)]), 4),
                }
            )
        trace["cfar"] = {
            "candidate_count": int(candidate_indices.shape[0]),
            "threshold_floor": round(float(threshold_floor), 4),
            "power_max": round(float(np.max(power_map)), 4),
            "fallback_used": bool(candidate_indices.shape[0] == 1 and not bool(np.any(peak_mask))),
            "top_candidates": top_cfar,
        }
    coarse_candidate_pool = []
    rdi_peak_ceiling = float(np.max(power_map))
    reject_reasons = trace["reject_reasons"] if trace_enabled else {}

    for range_bin_rel, doppler_bin in ordered_indices:
        range_bin = int(range_bin_rel + min_range_bin)
        range_m = float(runtime_config.range_axis_m[range_bin])
        angle_mask = _angle_roi_mask(
            range_m,
            runtime_config.angle_axis_rad,
            detection_region,
        )
        if not np.any(angle_mask):
            if trace_enabled:
                _trace_reject(reject_reasons, "angle_roi_empty")
            continue

        angle_profile = np.asarray(rai_map[range_bin], dtype=np.float64)
        masked_angle_profile = np.where(angle_mask, angle_profile, 0)
        peak_angle_bin = int(np.argmax(masked_angle_profile))
        rai_peak = float(masked_angle_profile[peak_angle_bin])
        if rai_peak <= 0:
            if trace_enabled:
                _trace_reject(reject_reasons, "rai_peak_non_positive")
            continue

        roi_angle_values = masked_angle_profile[angle_mask]
        if roi_angle_values.size == 0:
            if trace_enabled:
                _trace_reject(reject_reasons, "roi_angle_values_empty")
            continue

        angle_floor = float(np.quantile(roi_angle_values, angle_quantile))
        angle_contrast = rai_peak / max(angle_floor, 1e-6)
        if angle_contrast < angle_contrast_scale:
            if trace_enabled:
                _trace_reject(reject_reasons, "angle_contrast_low")
            continue

        if not _angle_is_local_peak(masked_angle_profile, peak_angle_bin):
            if trace_enabled:
                _trace_reject(reject_reasons, "angle_not_local_peak")
            continue

        centroid_radius = _angle_centroid_radius_for_range(
            range_m,
            angle_centroid_radius_bands,
            default_radius=1,
        )
        angle_bin, angle_rad = _refine_angle_centroid(
            masked_angle_profile,
            runtime_config.angle_axis_rad,
            peak_angle_bin,
            angle_floor,
            angle_mask,
            radius=centroid_radius,
        )
        x_m = float(range_m * np.sin(angle_rad))
        y_m = float(range_m * np.cos(angle_rad))
        rdi_peak = float(rdi_map[range_bin, int(doppler_bin)])
        normalized_rdi = float(power_map[range_bin_rel, int(doppler_bin)] / max(rdi_peak_ceiling, 1e-6))
        candidate_score = normalized_rdi * min(angle_contrast, 3.0)

        coarse_candidate_pool.append(
            DetectionCandidate(
                range_bin=range_bin,
                doppler_bin=int(doppler_bin),
                angle_bin=angle_bin,
                range_m=range_m,
                angle_deg=float(np.degrees(angle_rad)),
                x_m=x_m,
                y_m=y_m,
                rdi_peak=rdi_peak,
                rai_peak=rai_peak,
                score=candidate_score,
            )
        )

    coarse_candidate_pool.sort(
        key=lambda candidate: (candidate.score, candidate.rdi_peak, candidate.rai_peak),
        reverse=True,
    )
    if trace_enabled:
        trace["angle_validation"] = {
            "input_count": int(candidate_indices.shape[0]),
            "passed_count": int(len(coarse_candidate_pool)),
            "rejected_count": int(candidate_indices.shape[0] - len(coarse_candidate_pool)),
            "reject_reasons": dict(reject_reasons),
            "top_candidates": _trace_candidates(coarse_candidate_pool),
        }
        pre_merge_coarse = list(coarse_candidate_pool)
    coarse_candidate_pool = _merge_candidate_pool(
        coarse_candidate_pool,
        runtime_config,
        merge_bands=candidate_merge_bands,
        default_merge_radius_m=max(min_cartesian_separation_m * 0.75, 0.25),
        default_range_bin_radius=1,
        default_doppler_bin_radius=max(2, int(runtime_config.doppler_guard_bins)),
    )
    if trace_enabled:
        trace["candidate_merge_coarse"] = {
            "before_count": int(len(pre_merge_coarse)),
            "after_count": int(len(coarse_candidate_pool)),
            "before_top": _trace_candidates(pre_merge_coarse),
            "after_top": _trace_candidates(coarse_candidate_pool),
        }
    if not coarse_candidate_pool:
        if trace_enabled:
            trace["early_exit"] = "coarse_merge_empty"
        return []

    refined_candidate_pool = []
    body_center_pairs = []
    for coarse_candidate in coarse_candidate_pool:
        range_bin = int(np.clip(coarse_candidate.range_bin, 0, rai_map.shape[0] - 1))
        range_m = float(runtime_config.range_axis_m[range_bin])
        angle_mask = _angle_roi_mask(
            range_m,
            runtime_config.angle_axis_rad,
            detection_region,
        )
        if not np.any(angle_mask):
            if trace_enabled:
                _trace_reject(reject_reasons, "refine_angle_roi_empty")
            continue

        angle_profile = np.asarray(rai_map[range_bin], dtype=np.float64)
        masked_angle_profile = np.where(angle_mask, angle_profile, 0.0)
        roi_angle_values = masked_angle_profile[angle_mask]
        if roi_angle_values.size == 0:
            if trace_enabled:
                _trace_reject(reject_reasons, "refine_roi_angle_values_empty")
            continue

        peak_angle_bin = int(np.clip(coarse_candidate.angle_bin, 0, masked_angle_profile.shape[0] - 1))
        if (not bool(angle_mask[peak_angle_bin])) or float(masked_angle_profile[peak_angle_bin]) <= 0.0:
            peak_angle_bin = int(np.argmax(masked_angle_profile))
        rai_peak = float(masked_angle_profile[peak_angle_bin])
        if rai_peak <= 0.0:
            if trace_enabled:
                _trace_reject(reject_reasons, "refine_rai_peak_non_positive")
            continue

        angle_floor = float(np.quantile(roi_angle_values, angle_quantile))
        centroid_radius = _angle_centroid_radius_for_range(
            range_m,
            angle_centroid_radius_bands,
            default_radius=1,
        )
        angle_bin, angle_rad = _refine_angle_centroid(
            masked_angle_profile,
            runtime_config.angle_axis_rad,
            peak_angle_bin,
            angle_floor,
            angle_mask,
            radius=centroid_radius,
        )
        patch_range_radius, patch_angle_radius, patch_relative_floor = _body_center_patch_for_range(
            range_m,
            body_center_patch_bands,
            default_range_radius_bins=1,
            default_angle_radius_bins=max(2, centroid_radius + 1),
            default_relative_floor=0.55,
        )
        (
            refined_range_bin,
            refined_angle_bin,
            refined_range_m,
            refined_angle_rad,
            refined_x_m,
            refined_y_m,
        ) = _refine_body_center_from_patch(
            rai_map,
            runtime_config,
            range_bin,
            angle_bin,
            angle_mask,
            angle_floor=angle_floor,
            range_radius_bins=patch_range_radius,
            angle_radius_bins=patch_angle_radius,
            relative_floor=patch_relative_floor,
        )
        refined_candidate = DetectionCandidate(
            range_bin=refined_range_bin,
            doppler_bin=int(coarse_candidate.doppler_bin),
            angle_bin=refined_angle_bin,
            range_m=refined_range_m,
            angle_deg=float(np.degrees(refined_angle_rad)),
            x_m=refined_x_m,
            y_m=refined_y_m,
            rdi_peak=float(coarse_candidate.rdi_peak),
            rai_peak=max(float(coarse_candidate.rai_peak), rai_peak),
            score=float(coarse_candidate.score),
        )
        refined_candidate_pool.append(refined_candidate)
        if trace_enabled and len(body_center_pairs) < 12:
            body_center_pairs.append(
                {
                    "before": _trace_candidate(coarse_candidate),
                    "after": _trace_candidate(refined_candidate),
                    "shift_m": round(float(hypot(coarse_candidate.x_m - refined_candidate.x_m, coarse_candidate.y_m - refined_candidate.y_m)), 4),
                }
            )

    candidate_pool = refined_candidate_pool or coarse_candidate_pool
    if trace_enabled:
        trace["body_center_refinement"] = {
            "input_count": int(len(coarse_candidate_pool)),
            "refined_count": int(len(refined_candidate_pool)),
            "fallback_to_coarse": bool(not refined_candidate_pool),
            "pairs": body_center_pairs,
        }
        pre_merge_final = list(candidate_pool)
    candidate_pool = _merge_candidate_pool(
        candidate_pool,
        runtime_config,
        merge_bands=candidate_merge_bands,
        default_merge_radius_m=max(min_cartesian_separation_m * 0.75, 0.25),
        default_range_bin_radius=1,
        default_doppler_bin_radius=max(2, int(runtime_config.doppler_guard_bins)),
    )
    if trace_enabled:
        trace["candidate_merge_final"] = {
            "before_count": int(len(pre_merge_final)),
            "after_count": int(len(candidate_pool)),
            "before_top": _trace_candidates(pre_merge_final),
            "after_top": _trace_candidates(candidate_pool),
        }
        trace["dbscan"] = {
            "input_count": int(len(candidate_pool)),
            "eps_base": float(min_cartesian_separation_m),
            "cluster_min_samples": int(detection_region.cluster_min_samples),
            "velocity_weight": float(detection_region.cluster_velocity_weight),
            "adaptive_eps_bands": detection_region.adaptive_eps_bands,
            "input_top": _trace_candidates(candidate_pool),
        }
    clustered_detections = _cluster_detection_candidates(
        candidate_pool,
        runtime_config,
        detection_region,
        min_cartesian_separation_m,
    )
    if not clustered_detections:
        if trace_enabled:
            trace["dbscan"]["output_count"] = 0
            trace["dbscan"]["output_top"] = []
            trace["early_exit"] = "dbscan_empty"
        return []
    pre_duplicate_suppression = list(clustered_detections)
    clustered_detections, suppressed_duplicates = _suppress_duplicate_candidates(
        clustered_detections,
        runtime_config,
        enabled=duplicate_suppression_enabled,
        radius_m=duplicate_suppression_radius_m,
        range_scale=duplicate_suppression_range_scale,
        doppler_bins=duplicate_suppression_doppler_bins,
        score_ratio=duplicate_suppression_score_ratio,
    )
    if trace_enabled:
        trace["duplicate_suppression"] = {
            "enabled": bool(duplicate_suppression_enabled),
            "before_count": int(len(pre_duplicate_suppression)),
            "after_count": int(len(clustered_detections)),
            "suppressed_count": int(len(suppressed_duplicates)),
            "radius_m": round(float(duplicate_suppression_radius_m), 4),
            "range_scale": round(float(duplicate_suppression_range_scale), 4),
            "doppler_bins": int(duplicate_suppression_doppler_bins),
            "score_ratio": round(float(duplicate_suppression_score_ratio), 4),
            "suppressed": suppressed_duplicates[:12],
        }
    if not clustered_detections:
        if trace_enabled:
            trace["early_exit"] = "duplicate_suppression_empty"
        return []
    output = clustered_detections[:detection_region.max_targets]
    if trace_enabled:
        trace["dbscan"]["output_count"] = int(len(clustered_detections))
        trace["dbscan"]["output_top"] = _trace_candidates(clustered_detections)
        trace["final_output"] = {
            "output_count": int(len(output)),
            "truncated_from": int(len(clustered_detections)),
            "top_detections": _trace_candidates(output),
        }
    return output
