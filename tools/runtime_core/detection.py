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


def _weighted_quantile(values, weights, quantile):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if values.size == 0 or weights.size != values.size:
        return None

    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return None

    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    total = float(cumulative[-1])
    if total <= 1e-9:
        return None
    target = float(np.clip(quantile, 0.0, 1.0)) * total
    return float(values[int(np.searchsorted(cumulative, target, side="left"))])


def _connected_component_mask_3d(binary_mask, seed_depth, seed_row, seed_col):
    depth_count, row_count, col_count = binary_mask.shape
    if depth_count == 0 or row_count == 0 or col_count == 0:
        return np.zeros_like(binary_mask, dtype=bool)

    seed_depth = int(np.clip(seed_depth, 0, depth_count - 1))
    seed_row = int(np.clip(seed_row, 0, row_count - 1))
    seed_col = int(np.clip(seed_col, 0, col_count - 1))
    if not bool(binary_mask[seed_depth, seed_row, seed_col]):
        return np.zeros_like(binary_mask, dtype=bool)

    component = np.zeros_like(binary_mask, dtype=bool)
    stack = [(seed_depth, seed_row, seed_col)]
    component[seed_depth, seed_row, seed_col] = True

    while stack:
        depth_index, row_index, col_index = stack.pop()
        for depth_offset in (-1, 0, 1):
            for row_offset in (-1, 0, 1):
                for col_offset in (-1, 0, 1):
                    if depth_offset == 0 and row_offset == 0 and col_offset == 0:
                        continue
                    next_depth = depth_index + depth_offset
                    next_row = row_index + row_offset
                    next_col = col_index + col_offset
                    if not (
                        0 <= next_depth < depth_count
                        and 0 <= next_row < row_count
                        and 0 <= next_col < col_count
                    ):
                        continue
                    if component[next_depth, next_row, next_col]:
                        continue
                    if not bool(binary_mask[next_depth, next_row, next_col]):
                        continue
                    component[next_depth, next_row, next_col] = True
                    stack.append((next_depth, next_row, next_col))

    return component


def _candidate_strength(candidate):
    score = float(getattr(candidate, "score", 0.0) or 0.0)
    if score > 0.0:
        return score
    rdi_peak = float(getattr(candidate, "rdi_peak", 0.0) or 0.0)
    if rdi_peak > 0.0:
        return rdi_peak
    return float(getattr(candidate, "rai_peak", 0.0) or 0.0)


def _candidate_lateral_side(candidate, deadband_m=0.12):
    x_m = float(getattr(candidate, "x_m", 0.0) or 0.0)
    deadband_m = max(0.0, float(deadband_m))
    if abs(x_m) <= deadband_m:
        return 0
    return -1 if x_m < 0.0 else 1


def _single_target_blob_members(
    usable_candidates,
    *,
    min_score_ratio=0.12,
    range_window_m=1.05,
    side_deadband_m=0.15,
):
    ordered = sorted(list(usable_candidates or []), key=_candidate_strength, reverse=True)
    if not ordered:
        return []

    strongest = ordered[0]
    strongest_score = max(_candidate_strength(strongest), 1e-9)
    min_score_ratio = float(np.clip(float(min_score_ratio), 0.0, 1.0))
    range_window_m = max(0.05, float(range_window_m))
    side_deadband_m = max(0.0, float(side_deadband_m))
    dominant_side = _candidate_lateral_side(strongest, side_deadband_m)
    dominant_range_m = float(getattr(strongest, "range_m", 0.0) or 0.0)

    members = []
    for candidate in ordered:
        strength = _candidate_strength(candidate)
        if strength < strongest_score * min_score_ratio:
            continue

        candidate_side = _candidate_lateral_side(candidate, side_deadband_m)
        if dominant_side and candidate_side and candidate_side != dominant_side:
            continue

        candidate_range_m = float(getattr(candidate, "range_m", 0.0) or 0.0)
        dynamic_window_m = range_window_m + max(dominant_range_m, candidate_range_m, 0.0) * 0.08
        if abs(candidate_range_m - dominant_range_m) > dynamic_window_m:
            continue

        members.append(candidate)

    return members or [strongest]


def _estimate_object_count_from_candidates(
    candidates,
    runtime_config,
    *,
    enabled=True,
    max_objects=3,
    min_separation_m=0.65,
    min_doppler_bins=5,
    min_score_ratio=0.05,
):
    if not enabled:
        return {
            "enabled": False,
            "input_count": int(len(candidates or [])),
            "estimated_count": 0,
            "selected": [],
        }

    candidate_list = list(candidates or [])
    if not candidate_list:
        return {
            "enabled": True,
            "input_count": 0,
            "estimated_count": 0,
            "selected": [],
        }

    max_objects = max(1, int(max_objects))
    min_separation_m = max(0.0, float(min_separation_m))
    min_doppler_bins = max(0, int(min_doppler_bins))
    min_score_ratio = float(np.clip(float(min_score_ratio), 0.0, 1.0))
    ordered = sorted(candidate_list, key=_candidate_strength, reverse=True)
    strongest = max(_candidate_strength(ordered[0]), 1e-9)
    selected = []

    for candidate in ordered:
        if _candidate_strength(candidate) < strongest * min_score_ratio:
            continue
        independent = True
        for reference in selected:
            cart_distance = float(hypot(candidate.x_m - reference.x_m, candidate.y_m - reference.y_m))
            doppler_distance = _doppler_bin_distance(
                candidate.doppler_bin,
                reference.doppler_bin,
                runtime_config.doppler_fft_size,
            )
            if cart_distance < min_separation_m and doppler_distance < min_doppler_bins:
                independent = False
                break
        if independent:
            selected.append(candidate)
            if len(selected) >= max_objects:
                break

    return {
        "enabled": True,
        "input_count": int(len(candidate_list)),
        "estimated_count": int(len(selected)),
        "max_objects": int(max_objects),
        "min_separation_m": round(float(min_separation_m), 4),
        "min_doppler_bins": int(min_doppler_bins),
        "min_score_ratio": round(float(min_score_ratio), 4),
        "selected": _trace_candidates(selected),
    }


def _blob_weight(candidate):
    # Use a compressed weight so one very strong limb/multipath peak does not own the center.
    return float(np.sqrt(max(_candidate_strength(candidate), 1e-9)))


def _blob_radius_for_range(range_m, radius_bands, default_radius_m=0.65, range_scale=0.04):
    radius_m = float(default_radius_m) + max(float(range_m), 0.0) * float(range_scale)
    if not radius_bands:
        return max(radius_m, 0.05)

    for band in radius_bands:
        try:
            r_min = float(band.get("r_min", 0.0))
            r_max = band.get("r_max")
        except (TypeError, ValueError, AttributeError):
            continue

        if float(range_m) < r_min:
            continue
        if r_max is not None and float(range_m) >= float(r_max):
            continue

        try:
            radius_m = float(band.get("radius_m", band.get("merge_radius_m", radius_m)))
        except (TypeError, ValueError):
            pass
        break

    return max(float(radius_m), 0.05)


def _candidate_blob_center(members, runtime_config, *, center_method="weighted_median"):
    member_list = list(members or [])
    if not member_list:
        return None

    weights = np.asarray([_blob_weight(candidate) for candidate in member_list], dtype=np.float64)
    xs = np.asarray([float(candidate.x_m) for candidate in member_list], dtype=np.float64)
    ys = np.asarray([float(candidate.y_m) for candidate in member_list], dtype=np.float64)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 1e-9:
        return None

    method = str(center_method or "weighted_median").strip().lower()
    if method in {"median", "weighted_median", "robust"}:
        x_center = _weighted_quantile(xs, weights, 0.5)
        y_center = _weighted_quantile(ys, weights, 0.5)
        if x_center is None or y_center is None:
            x_center = float(np.sum(xs * weights) / weight_sum)
            y_center = float(np.sum(ys * weights) / weight_sum)
            method = "weighted_mean_fallback"
    else:
        x_center = float(np.sum(xs * weights) / weight_sum)
        y_center = float(np.sum(ys * weights) / weight_sum)
        method = "weighted_mean"

    range_m = float(hypot(float(x_center), float(y_center)))
    angle_rad = float(atan2(float(x_center), max(float(y_center), 1e-6)))
    dopplers = np.asarray([float(candidate.doppler_bin) for candidate in member_list], dtype=np.float64)
    doppler_center = _weighted_quantile(dopplers, weights, 0.5)
    if doppler_center is None:
        doppler_center = float(np.sum(dopplers * weights) / weight_sum)

    strongest = max(
        member_list,
        key=lambda candidate: (
            _candidate_strength(candidate),
            float(candidate.rdi_peak),
            float(candidate.rai_peak),
        ),
    )
    score_sum = float(sum(_candidate_strength(candidate) for candidate in member_list))
    score_scale = min(1.35, 1.0 + 0.06 * max(len(member_list) - 1, 0))
    return (
        DetectionCandidate(
            range_bin=_nearest_axis_bin(runtime_config.range_axis_m, range_m),
            doppler_bin=int(np.clip(round(float(doppler_center)), 0, runtime_config.doppler_fft_size - 1)),
            angle_bin=_nearest_axis_bin(runtime_config.angle_axis_rad, angle_rad),
            range_m=range_m,
            angle_deg=float(np.degrees(angle_rad)),
            x_m=float(x_center),
            y_m=float(y_center),
            rdi_peak=float(max(candidate.rdi_peak for candidate in member_list)),
            rai_peak=float(max(candidate.rai_peak for candidate in member_list)),
            score=max(float(strongest.score) * score_scale, score_sum / max(len(member_list), 1)),
        ),
        {
            "method": method,
            "member_count": int(len(member_list)),
            "weight_sum": round(weight_sum, 4),
            "score_sum": round(score_sum, 4),
            "strongest": _trace_candidate(strongest),
        },
    )


def _cube_blob_center_for_members(
    rai_cube,
    members,
    runtime_config,
    detection_region,
    *,
    body_center_patch_bands=None,
    doppler_radius_bins=2,
    floor_quantile=0.65,
    min_points=4,
    center_method="weighted_median",
    peak_blend=0.0,
    cube_range_radius_m=None,
    cube_angle_radius_deg=None,
    cube_relative_floor=None,
):
    if rai_cube is None:
        return None

    member_list = list(members or [])
    if not member_list:
        return None

    cube = np.asarray(rai_cube, dtype=np.float64)
    if cube.ndim != 3 or cube.size == 0:
        return None

    doppler_count, range_count, angle_count = cube.shape
    doppler_radius_bins = max(0, int(doppler_radius_bins))
    floor_quantile = float(np.clip(floor_quantile, 0.0, 0.95))
    min_points = max(1, int(min_points))
    peak_blend = float(np.clip(peak_blend, 0.0, 0.75))
    point_map = {}

    for seed in member_list:
        seed_doppler_bin = int(np.clip(seed.doppler_bin, 0, doppler_count - 1))
        seed_range_bin = int(np.clip(seed.range_bin, 0, range_count - 1))
        seed_angle_bin = int(np.clip(seed.angle_bin, 0, angle_count - 1))
        seed_range_m = float(runtime_config.range_axis_m[seed_range_bin])
        angle_mask = _angle_roi_mask(
            seed_range_m,
            runtime_config.angle_axis_rad,
            detection_region,
        )
        if not np.any(angle_mask):
            continue

        range_radius_bins, angle_radius_bins, relative_floor = _body_center_patch_for_range(
            seed_range_m,
            body_center_patch_bands,
            default_range_radius_bins=2,
            default_angle_radius_bins=3,
            default_relative_floor=0.45,
        )
        range_radius_bins = max(1, int(range_radius_bins))
        angle_radius_bins = max(1, int(angle_radius_bins))
        relative_floor = float(np.clip(relative_floor, 0.0, 0.95))

        if cube_range_radius_m is not None:
            range_axis = np.asarray(runtime_config.range_axis_m, dtype=np.float64)
            range_steps = np.diff(range_axis)
            range_steps = range_steps[np.isfinite(range_steps) & (range_steps > 0)]
            if range_steps.size:
                range_step_m = float(np.median(range_steps))
                expanded_range_bins = int(np.ceil(float(cube_range_radius_m) / max(range_step_m, 1e-6)))
                range_radius_bins = max(range_radius_bins, expanded_range_bins)

        if cube_angle_radius_deg is not None:
            angle_axis = np.asarray(runtime_config.angle_axis_rad, dtype=np.float64)
            angle_steps = np.diff(angle_axis)
            angle_steps = np.abs(angle_steps[np.isfinite(angle_steps) & (np.abs(angle_steps) > 0)])
            if angle_steps.size:
                angle_step_rad = float(np.median(angle_steps))
                expanded_angle_bins = int(
                    np.ceil(np.radians(float(cube_angle_radius_deg)) / max(angle_step_rad, 1e-6))
                )
                angle_radius_bins = max(angle_radius_bins, expanded_angle_bins)

        if cube_relative_floor is not None:
            relative_floor = float(np.clip(float(cube_relative_floor), 0.0, 0.95))

        doppler_lower = max(seed_doppler_bin - doppler_radius_bins, 0)
        doppler_upper = min(seed_doppler_bin + doppler_radius_bins + 1, doppler_count)
        range_lower = max(seed_range_bin - range_radius_bins, 0)
        range_upper = min(seed_range_bin + range_radius_bins + 1, range_count)
        angle_lower = max(seed_angle_bin - angle_radius_bins, 0)
        angle_upper = min(seed_angle_bin + angle_radius_bins + 1, angle_count)
        patch = cube[doppler_lower:doppler_upper, range_lower:range_upper, angle_lower:angle_upper]
        if patch.size == 0:
            continue

        local_angle_mask = np.asarray(angle_mask[angle_lower:angle_upper], dtype=bool)
        if not np.any(local_angle_mask):
            continue

        local_range_axis = np.asarray(runtime_config.range_axis_m[range_lower:range_upper], dtype=np.float64)
        local_angle_axis = np.asarray(runtime_config.angle_axis_rad[angle_lower:angle_upper], dtype=np.float64)
        range_grid = local_range_axis[np.newaxis, :, np.newaxis]
        angle_grid = local_angle_axis[np.newaxis, np.newaxis, :]
        x_grid = range_grid * np.sin(angle_grid)
        y_grid = range_grid * np.cos(angle_grid)
        roi_mask = (
            (np.abs(x_grid) <= float(detection_region.lateral_limit_m))
            & (y_grid >= float(detection_region.min_forward_m))
            & (y_grid <= float(detection_region.forward_limit_m))
        )
        spatial_mask = (
            np.broadcast_to(roi_mask, patch.shape)
            & np.broadcast_to(local_angle_mask[np.newaxis, np.newaxis, :], patch.shape)
        )
        valid_values = patch[spatial_mask]
        if valid_values.size == 0:
            continue

        seed_depth = seed_doppler_bin - doppler_lower
        seed_row = seed_range_bin - range_lower
        seed_col = seed_angle_bin - angle_lower
        seed_value = float(patch[seed_depth, seed_row, seed_col])
        if seed_value <= 0.0:
            seed_value = float(np.max(valid_values))
        if seed_value <= 0.0:
            continue

        quantile_floor = float(np.quantile(valid_values, floor_quantile))
        component_floor = max(seed_value * relative_floor, quantile_floor)
        threshold_mask = spatial_mask & (patch >= component_floor)
        if spatial_mask[seed_depth, seed_row, seed_col]:
            threshold_mask = np.array(threshold_mask, copy=True)
            threshold_mask[seed_depth, seed_row, seed_col] = True
        component_mask = _connected_component_mask_3d(
            threshold_mask,
            seed_depth,
            seed_row,
            seed_col,
        )
        if int(np.count_nonzero(component_mask)) < min_points:
            relaxed_floor = max(seed_value * max(relative_floor * 0.65, 0.18), quantile_floor * 0.65)
            relaxed_mask = spatial_mask & (patch >= relaxed_floor)
            if spatial_mask[seed_depth, seed_row, seed_col]:
                relaxed_mask = np.array(relaxed_mask, copy=True)
                relaxed_mask[seed_depth, seed_row, seed_col] = True
            relaxed_component = _connected_component_mask_3d(
                relaxed_mask,
                seed_depth,
                seed_row,
                seed_col,
            )
            if int(np.count_nonzero(relaxed_component)) > int(np.count_nonzero(component_mask)):
                component_mask = relaxed_component
                component_floor = relaxed_floor

        if int(np.count_nonzero(component_mask)) <= 0:
            continue

        local_indices = np.argwhere(component_mask)
        for depth_index, row_index, col_index in local_indices:
            global_key = (
                int(depth_index + doppler_lower),
                int(row_index + range_lower),
                int(col_index + angle_lower),
            )
            value = float(patch[int(depth_index), int(row_index), int(col_index)])
            weight = float(np.sqrt(max(value - component_floor, value * 0.10, 1e-9)))
            previous = point_map.get(global_key)
            if previous is None or weight > float(previous.get("weight", 0.0)):
                point_map[global_key] = {
                    "weight": weight,
                    "power": value,
                }

    if len(point_map) < min_points:
        return None

    doppler_indices = []
    x_values = []
    y_values = []
    weights = []
    component_points = []
    for (doppler_bin, range_bin, angle_bin), point_record in sorted(
        point_map.items(),
        key=lambda item: float(item[1].get("weight", 0.0)),
        reverse=True,
    ):
        weight = float(point_record.get("weight", 0.0))
        range_m = float(runtime_config.range_axis_m[int(range_bin)])
        angle_rad = float(runtime_config.angle_axis_rad[int(angle_bin)])
        doppler_indices.append(float(doppler_bin))
        x_values.append(float(range_m * np.sin(angle_rad)))
        y_values.append(float(range_m * np.cos(angle_rad)))
        weights.append(float(weight))
        if len(component_points) < 96:
            component_points.append(
                _trace_rda_cube_point(
                    runtime_config,
                    doppler_bin=doppler_bin,
                    range_bin=range_bin,
                    angle_bin=angle_bin,
                    power=float(point_record.get("power", 0.0)),
                    weight=weight,
                )
            )

    x_values = np.asarray(x_values, dtype=np.float64)
    y_values = np.asarray(y_values, dtype=np.float64)
    doppler_indices = np.asarray(doppler_indices, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 1e-9:
        return None

    method = str(center_method or "weighted_median").strip().lower()
    if method in {"median", "weighted_median", "robust"}:
        x_center = _weighted_quantile(x_values, weights, 0.5)
        y_center = _weighted_quantile(y_values, weights, 0.5)
        if x_center is None or y_center is None:
            x_center = float(np.sum(x_values * weights) / weight_sum)
            y_center = float(np.sum(y_values * weights) / weight_sum)
            method = "weighted_mean_fallback"
    else:
        x_center = float(np.sum(x_values * weights) / weight_sum)
        y_center = float(np.sum(y_values * weights) / weight_sum)
        method = "weighted_mean"

    candidate_center, candidate_summary = _candidate_blob_center(
        member_list,
        runtime_config,
        center_method=center_method,
    )
    if candidate_center is not None and peak_blend > 0.0:
        x_center = float((1.0 - peak_blend) * x_center + peak_blend * candidate_center.x_m)
        y_center = float((1.0 - peak_blend) * y_center + peak_blend * candidate_center.y_m)

    range_m = float(hypot(float(x_center), float(y_center)))
    angle_rad = float(atan2(float(x_center), max(float(y_center), 1e-6)))
    doppler_center = _weighted_quantile(doppler_indices, weights, 0.5)
    if doppler_center is None:
        doppler_center = float(np.sum(doppler_indices * weights) / weight_sum)

    strongest = max(member_list, key=_candidate_strength)
    score_sum = float(sum(_candidate_strength(candidate) for candidate in member_list))
    return (
        DetectionCandidate(
            range_bin=_nearest_axis_bin(runtime_config.range_axis_m, range_m),
            doppler_bin=int(np.clip(round(float(doppler_center)), 0, runtime_config.doppler_fft_size - 1)),
            angle_bin=_nearest_axis_bin(runtime_config.angle_axis_rad, angle_rad),
            range_m=range_m,
            angle_deg=float(np.degrees(angle_rad)),
            x_m=float(x_center),
            y_m=float(y_center),
            rdi_peak=float(max(candidate.rdi_peak for candidate in member_list)),
            rai_peak=float(max(candidate.rai_peak for candidate in member_list)),
            score=max(float(strongest.score), score_sum / max(len(member_list), 1)),
        ),
        {
            "method": method,
            "source": "rai_cube_patch_union",
            "member_count": int(len(member_list)),
            "point_count": int(len(point_map)),
            "component_point_count": int(len(point_map)),
            "component_points": component_points,
            "weight_sum": round(weight_sum, 4),
            "candidate_center": _trace_candidate(candidate_center) if candidate_center is not None else None,
            "candidate_summary": candidate_summary,
            "cube_range_radius_m": None if cube_range_radius_m is None else round(float(cube_range_radius_m), 4),
            "cube_angle_radius_deg": None if cube_angle_radius_deg is None else round(float(cube_angle_radius_deg), 4),
            "cube_relative_floor": None if cube_relative_floor is None else round(float(cube_relative_floor), 4),
        },
    )


def _component_center_from_rda_cells(
    cells,
    cube_roi,
    runtime_config,
    range_lower,
    *,
    threshold,
    max_power,
    center_method="weighted_median",
):
    if not cells:
        return None

    x_values = []
    y_values = []
    doppler_values = []
    range_bins = []
    angle_bins = []
    weights = []
    component_points = []
    power_sum = 0.0
    power_max = 0.0

    for doppler_bin, range_rel, angle_bin in cells:
        doppler_bin = int(doppler_bin)
        range_rel = int(range_rel)
        angle_bin = int(angle_bin)
        range_bin = int(range_rel + range_lower)
        power = float(cube_roi[doppler_bin, range_rel, angle_bin])
        power_sum += power
        power_max = max(power_max, power)
        weight = float(np.sqrt(max(power - float(threshold), power * 0.08, 1e-9)))

        range_m = float(runtime_config.range_axis_m[range_bin])
        angle_rad = float(runtime_config.angle_axis_rad[angle_bin])
        x_values.append(float(range_m * np.sin(angle_rad)))
        y_values.append(float(range_m * np.cos(angle_rad)))
        doppler_values.append(float(doppler_bin))
        range_bins.append(float(range_bin))
        angle_bins.append(float(angle_bin))
        weights.append(weight)
        if len(component_points) < 120:
            component_points.append(
                _trace_rda_cube_point(
                    runtime_config,
                    doppler_bin=doppler_bin,
                    range_bin=range_bin,
                    angle_bin=angle_bin,
                    power=power,
                    weight=weight,
                    power_max=max_power,
                )
            )

    weights = np.asarray(weights, dtype=np.float64)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 1e-9:
        return None

    x_values = np.asarray(x_values, dtype=np.float64)
    y_values = np.asarray(y_values, dtype=np.float64)
    doppler_values = np.asarray(doppler_values, dtype=np.float64)
    range_bins = np.asarray(range_bins, dtype=np.float64)
    angle_bins = np.asarray(angle_bins, dtype=np.float64)
    method = str(center_method or "weighted_median").strip().lower()

    x_median = _weighted_quantile(x_values, weights, 0.5)
    y_median = _weighted_quantile(y_values, weights, 0.5)
    if x_median is None or y_median is None:
        x_median = float(np.sum(x_values * weights) / weight_sum)
        y_median = float(np.sum(y_values * weights) / weight_sum)

    if method in {"trimmed_mean", "weighted_trimmed_mean", "weighted_median_trimmed", "robust_trimmed"}:
        base_range = float(hypot(float(x_median), float(y_median)))
        trim_radius_m = 0.30 + 0.07 * max(base_range, 0.0)
        distances = np.hypot(x_values - float(x_median), y_values - float(y_median))
        keep = distances <= trim_radius_m
        if np.count_nonzero(keep) >= max(3, int(len(x_values) * 0.35)):
            trim_weights = weights[keep]
            trim_sum = float(np.sum(trim_weights))
            if trim_sum > 1e-9:
                x_center = float(np.sum(x_values[keep] * trim_weights) / trim_sum)
                y_center = float(np.sum(y_values[keep] * trim_weights) / trim_sum)
                center_method_used = "weighted_median_trimmed"
            else:
                x_center = float(x_median)
                y_center = float(y_median)
                center_method_used = "weighted_median"
        else:
            x_center = float(x_median)
            y_center = float(y_median)
            center_method_used = "weighted_median"
    elif method in {"mean", "weighted_mean"}:
        x_center = float(np.sum(x_values * weights) / weight_sum)
        y_center = float(np.sum(y_values * weights) / weight_sum)
        center_method_used = "weighted_mean"
    else:
        x_center = float(x_median)
        y_center = float(y_median)
        center_method_used = "weighted_median"

    range_m = float(hypot(x_center, y_center))
    angle_rad = float(atan2(x_center, max(y_center, 1e-6)))
    doppler_center = _weighted_quantile(doppler_values, weights, 0.5)
    if doppler_center is None:
        doppler_center = float(np.sum(doppler_values * weights) / weight_sum)
    score = max(float(power_max) / max(float(max_power), 1e-9), 1e-6)
    support_score = min(1.0, float(len(cells)) / 32.0) * 0.30
    candidate = DetectionCandidate(
        range_bin=_nearest_axis_bin(runtime_config.range_axis_m, range_m),
        doppler_bin=int(np.clip(round(float(doppler_center)), 0, runtime_config.doppler_fft_size - 1)),
        angle_bin=_nearest_axis_bin(runtime_config.angle_axis_rad, angle_rad),
        range_m=range_m,
        angle_deg=float(np.degrees(angle_rad)),
        x_m=float(x_center),
        y_m=float(y_center),
        rdi_peak=float(power_max),
        rai_peak=float(power_max),
        score=float(score + support_score),
    )
    summary = {
        "source": "rda_dense_connected_component",
        "method": center_method_used,
        "point_count": int(len(cells)),
        "component_point_count": int(len(cells)),
        "component_points": component_points,
        "power_sum": round(float(power_sum), 4),
        "power_max": round(float(power_max), 4),
        "weight_sum": round(float(weight_sum), 4),
        "range_bin_bounds": [int(np.min(range_bins)), int(np.max(range_bins))],
        "angle_bin_bounds": [int(np.min(angle_bins)), int(np.max(angle_bins))],
        "doppler_bin_bounds": [int(np.min(doppler_values)), int(np.max(doppler_values))],
        "threshold": round(float(threshold), 4),
    }
    return candidate, summary


def _dense_blob_centers_from_rda_cube(
    rai_cube,
    runtime_config,
    detection_region,
    min_range_bin,
    max_range_bin,
    *,
    anchor_candidates=None,
    doppler_guard_bins=0,
    quantile=0.995,
    min_normalized_power=0.08,
    max_points=2400,
    min_points=6,
    max_blobs=3,
    center_method="weighted_median_trimmed",
    anchor_range_bins=8,
    anchor_doppler_bins=10,
):
    if rai_cube is None:
        return [], {"enabled": False, "reason": "missing_rai_cube"}

    cube = np.asarray(rai_cube, dtype=np.float64)
    if cube.ndim != 3 or cube.size == 0:
        return [], {"enabled": False, "reason": "invalid_rai_cube"}

    doppler_count, range_count, angle_count = cube.shape
    range_lower = max(0, int(min_range_bin))
    range_upper = min(int(max_range_bin), range_count)
    if range_upper <= range_lower:
        return [], {"enabled": True, "reason": "empty_range_roi"}

    cube_roi = np.maximum(cube[:, range_lower:range_upper, :], 0.0)
    if cube_roi.size == 0 or float(np.max(cube_roi)) <= 0.0:
        return [], {"enabled": True, "reason": "zero_power_roi"}

    range_axis = np.asarray(runtime_config.range_axis_m[range_lower:range_upper], dtype=np.float64)
    angle_axis = np.asarray(runtime_config.angle_axis_rad, dtype=np.float64)
    range_grid = range_axis[:, np.newaxis]
    angle_grid = angle_axis[np.newaxis, :]
    x_grid = range_grid * np.sin(angle_grid)
    y_grid = range_grid * np.cos(angle_grid)
    spatial_mask_2d = (
        (np.abs(x_grid) <= float(detection_region.lateral_limit_m))
        & (y_grid >= float(detection_region.min_forward_m))
        & (y_grid <= float(detection_region.forward_limit_m))
    )
    valid_mask = np.broadcast_to(spatial_mask_2d[np.newaxis, :, :], cube_roi.shape).copy()

    center_bin = int(getattr(runtime_config, "doppler_fft_size", doppler_count)) // 2
    guard_bins = max(0, int(doppler_guard_bins))
    if guard_bins > 0:
        lower = max(center_bin - guard_bins, 0)
        upper = min(center_bin + guard_bins + 1, doppler_count)
        valid_mask[lower:upper, :, :] = False

    valid_values = cube_roi[valid_mask]
    valid_values = valid_values[np.isfinite(valid_values) & (valid_values > 0.0)]
    if valid_values.size == 0:
        return [], {"enabled": True, "reason": "empty_valid_values"}

    quantile = float(np.clip(quantile, 0.0, 0.9999))
    min_normalized_power = float(np.clip(min_normalized_power, 0.0, 1.0))
    max_power = float(np.max(valid_values))
    threshold = max(float(np.quantile(valid_values, quantile)), max_power * min_normalized_power)
    candidate_mask = valid_mask & (cube_roi >= threshold)
    candidate_indices = np.argwhere(candidate_mask)
    max_points = max(1, int(max_points))
    if candidate_indices.shape[0] > max_points:
        candidate_values = cube_roi[candidate_mask]
        threshold = max(threshold, float(np.partition(candidate_values, -max_points)[-max_points]))
        candidate_mask = valid_mask & (cube_roi >= threshold)
        candidate_indices = np.argwhere(candidate_mask)

    if candidate_indices.size == 0:
        return [], {
            "enabled": True,
            "reason": "empty_candidate_mask",
            "quantile": round(float(quantile), 4),
            "threshold": round(float(threshold), 4),
        }

    cells = [tuple(int(value) for value in row) for row in candidate_indices]
    ordered_cells = sorted(cells, key=lambda idx: float(cube_roi[idx]), reverse=True)
    remaining = set(cells)
    components = []
    for seed in ordered_cells:
        if seed not in remaining:
            continue
        stack = [seed]
        remaining.remove(seed)
        component = []
        while stack:
            doppler_bin, range_rel, angle_bin = stack.pop()
            component.append((doppler_bin, range_rel, angle_bin))
            for doppler_offset in (-1, 0, 1):
                for range_offset in (-1, 0, 1):
                    for angle_offset in (-1, 0, 1):
                        if doppler_offset == 0 and range_offset == 0 and angle_offset == 0:
                            continue
                        neighbor = (
                            doppler_bin + doppler_offset,
                            range_rel + range_offset,
                            angle_bin + angle_offset,
                        )
                        if neighbor not in remaining:
                            continue
                        remaining.remove(neighbor)
                        stack.append(neighbor)
        if len(component) >= int(min_points):
            components.append(component)

    if not components:
        return [], {
            "enabled": True,
            "reason": "no_component_above_min_points",
            "candidate_count": int(candidate_indices.shape[0]),
            "min_points": int(min_points),
            "threshold": round(float(threshold), 4),
        }

    anchors = list(anchor_candidates or [])
    anchor_range_bins = max(0, int(anchor_range_bins))
    anchor_doppler_bins = max(0, int(anchor_doppler_bins))

    def _component_is_anchored(component):
        if not anchors:
            return True
        ranges = [int(row + range_lower) for _, row, _ in component]
        dopplers = [int(depth) for depth, _, _ in component]
        range_min = min(ranges) - anchor_range_bins
        range_max = max(ranges) + anchor_range_bins
        for anchor in anchors:
            anchor_range = int(getattr(anchor, "range_bin", -9999))
            if not (range_min <= anchor_range <= range_max):
                continue
            anchor_doppler = int(getattr(anchor, "doppler_bin", -9999))
            for doppler_bin in dopplers:
                if _doppler_bin_distance(anchor_doppler, doppler_bin, runtime_config.doppler_fft_size) <= anchor_doppler_bins:
                    return True
        return False

    blob_candidates = []
    for component in components:
        if not _component_is_anchored(component):
            continue
        center_result = _component_center_from_rda_cells(
            component,
            cube_roi,
            runtime_config,
            range_lower,
            threshold=threshold,
            max_power=max_power,
            center_method=center_method,
        )
        if center_result is None:
            continue
        center, summary = center_result
        strength = float(summary.get("power_sum", 0.0))
        blob_candidates.append((center, strength, component, summary))

    if not blob_candidates:
        return [], {
            "enabled": True,
            "reason": "no_anchored_dense_components",
            "candidate_count": int(candidate_indices.shape[0]),
            "component_count": int(len(components)),
            "anchor_count": int(len(anchors)),
            "threshold": round(float(threshold), 4),
        }

    blob_candidates.sort(
        key=lambda item: (
            item[1],
            item[3].get("point_count", 0),
            float(item[0].score),
        ),
        reverse=True,
    )
    selected = blob_candidates[:max(1, int(max_blobs))]
    refined = [item[0] for item in selected]
    groups = []
    for center, strength, component, summary in selected[:12]:
        groups.append(
            {
                "center": _trace_candidate(center),
                "member_count": int(summary.get("point_count", len(component))),
                "strength": round(float(strength), 4),
                "summary": summary,
                "members": [],
            }
        )

    return refined, {
        "enabled": True,
        "source": "rda_dense_connected_components",
        "input_count": int(len(anchors)),
        "candidate_count": int(candidate_indices.shape[0]),
        "component_count": int(len(components)),
        "anchored_component_count": int(len(blob_candidates)),
        "output_count": int(len(refined)),
        "max_blobs": int(max_blobs),
        "min_points": int(min_points),
        "quantile": round(float(quantile), 4),
        "threshold": round(float(threshold), 4),
        "max_power": round(float(max_power), 4),
        "min_normalized_power": round(float(min_normalized_power), 4),
        "center_method": str(center_method),
        "groups": groups,
    }


def _refine_blob_centers_from_candidates(
    candidate_pool,
    runtime_config,
    detection_region,
    *,
    rai_cube=None,
    body_center_patch_bands=None,
    enabled=True,
    max_blobs=None,
    max_candidates=36,
    min_points=3,
    min_score_ratio=0.04,
    cluster_radius_m=0.65,
    cluster_radius_range_scale=0.04,
    cluster_radius_bands=None,
    doppler_radius_bins=10,
    center_method="weighted_median",
    trim_radius_m=0.85,
    floor_quantile=0.65,
    peak_blend=0.0,
    single_min_score_ratio=0.12,
    single_range_window_m=1.05,
    single_side_deadband_m=0.15,
    cube_range_radius_m=None,
    cube_angle_radius_deg=None,
    cube_relative_floor=None,
):
    if not enabled:
        return list(candidate_pool or []), {"enabled": False}

    candidates = list(candidate_pool or [])
    max_candidates = max(1, int(max_candidates))
    min_points = max(1, int(min_points))
    min_seed_count = 1 if rai_cube is not None else min_points
    min_cube_points = max(4, min_points)
    min_score_ratio = float(np.clip(float(min_score_ratio), 0.0, 1.0))
    doppler_radius_bins = max(0, int(doppler_radius_bins))
    max_blobs = int(max_blobs if max_blobs is not None else detection_region.max_targets)
    max_blobs = max(1, max_blobs)

    if len(candidates) <= 1:
        if candidates and rai_cube is not None:
            center_result = _cube_blob_center_for_members(
                rai_cube,
                candidates,
                runtime_config,
                detection_region,
                body_center_patch_bands=body_center_patch_bands,
                doppler_radius_bins=doppler_radius_bins,
                floor_quantile=floor_quantile,
                min_points=min_cube_points,
                center_method=center_method,
                peak_blend=peak_blend,
                cube_range_radius_m=cube_range_radius_m,
                cube_angle_radius_deg=cube_angle_radius_deg,
                cube_relative_floor=cube_relative_floor,
            )
            if center_result is not None:
                center, summary = center_result
                return [center], {
                    "enabled": True,
                    "input_count": int(len(candidates)),
                    "usable_count": int(len(candidates)),
                    "output_count": 1,
                    "single_seed_cube_refined": True,
                    "groups": [
                        {
                            "center": _trace_candidate(center),
                            "member_count": 1,
                            "strength": round(float(_candidate_strength(candidates[0])), 4),
                            "summary": summary,
                            "members": _trace_candidates(candidates),
                        }
                    ],
                }
        return candidates, {
            "enabled": True,
            "input_count": int(len(candidates)),
            "output_count": int(len(candidates)),
            "reason": "not_enough_candidates",
        }

    ordered = sorted(candidates, key=_candidate_strength, reverse=True)
    strongest = max(_candidate_strength(ordered[0]), 1e-9)
    usable = [
        candidate
        for candidate in ordered[:max_candidates]
        if _candidate_strength(candidate) >= strongest * min_score_ratio
    ]
    if len(usable) < min_seed_count:
        return candidates, {
            "enabled": True,
            "input_count": int(len(candidates)),
            "usable_count": int(len(usable)),
            "output_count": int(len(candidates)),
            "reason": "not_enough_usable_candidates",
        }

    if max_blobs == 1:
        single_members = _single_target_blob_members(
            usable,
            min_score_ratio=max(float(min_score_ratio), float(single_min_score_ratio)),
            range_window_m=single_range_window_m,
            side_deadband_m=single_side_deadband_m,
        )
        if len(single_members) >= min_seed_count:
            center_result = _cube_blob_center_for_members(
                rai_cube,
                single_members,
                runtime_config,
                detection_region,
                body_center_patch_bands=body_center_patch_bands,
                doppler_radius_bins=doppler_radius_bins,
                floor_quantile=floor_quantile,
                min_points=min_cube_points,
                center_method=center_method,
                peak_blend=peak_blend,
                cube_range_radius_m=cube_range_radius_m,
                cube_angle_radius_deg=cube_angle_radius_deg,
                cube_relative_floor=cube_relative_floor,
            )
            if center_result is None:
                center_result = _candidate_blob_center(
                    single_members,
                    runtime_config,
                    center_method=center_method,
                )
            if center_result is not None:
                center, summary = center_result
                strength = float(sum(_candidate_strength(candidate) for candidate in single_members))
                return [center], {
                    "enabled": True,
                    "input_count": int(len(candidates)),
                    "usable_count": int(len(usable)),
                    "output_count": 1,
                    "max_blobs": int(max_blobs),
                    "max_candidates": int(max_candidates),
                    "min_points": int(min_points),
                    "min_score_ratio": round(float(min_score_ratio), 4),
                    "single_target_dominant_blob": True,
                    "single_min_score_ratio": round(float(single_min_score_ratio), 4),
                    "single_range_window_m": round(float(single_range_window_m), 4),
                    "single_side_deadband_m": round(float(single_side_deadband_m), 4),
                    "center_method": str(center_method),
                    "floor_quantile": round(float(floor_quantile), 4),
                    "peak_blend": round(float(peak_blend), 4),
                    "groups": [
                        {
                            "center": _trace_candidate(center),
                            "member_count": int(len(single_members)),
                            "strength": round(float(strength), 4),
                            "summary": summary,
                            "members": _trace_candidates(single_members[:12]),
                        }
                    ],
                }

    def _member_distance(left, right):
        return float(hypot(float(left.x_m) - float(right.x_m), float(left.y_m) - float(right.y_m)))

    groups = []
    for candidate in usable:
        best_group = None
        best_distance = None
        for group in groups:
            reference = group["center"]
            reference_range = max(float(candidate.range_m), float(reference.range_m))
            radius_m = _blob_radius_for_range(
                reference_range,
                cluster_radius_bands,
                default_radius_m=cluster_radius_m,
                range_scale=cluster_radius_range_scale,
            )
            distance_m = _member_distance(candidate, reference)
            doppler_distance = _doppler_bin_distance(
                candidate.doppler_bin,
                reference.doppler_bin,
                runtime_config.doppler_fft_size,
            )
            if distance_m > radius_m or doppler_distance > doppler_radius_bins:
                continue
            if best_distance is None or distance_m < best_distance:
                best_distance = distance_m
                best_group = group

        if best_group is None:
            best_group = {"members": [candidate], "center": candidate}
            groups.append(best_group)
            continue

        best_group["members"].append(candidate)
        center_result = _candidate_blob_center(
            best_group["members"],
            runtime_config,
            center_method=center_method,
        )
        if center_result is not None:
            best_group["center"] = center_result[0]

    blob_candidates = []
    group_traces = []
    for group in groups:
        members = list(group["members"])
        if len(members) < min_seed_count:
            continue

        center_result = _cube_blob_center_for_members(
            rai_cube,
            members,
            runtime_config,
            detection_region,
            body_center_patch_bands=body_center_patch_bands,
            doppler_radius_bins=doppler_radius_bins,
            floor_quantile=floor_quantile,
            min_points=min_cube_points,
            center_method=center_method,
            peak_blend=peak_blend,
            cube_range_radius_m=cube_range_radius_m,
            cube_angle_radius_deg=cube_angle_radius_deg,
            cube_relative_floor=cube_relative_floor,
        )
        if center_result is None:
            center_result = _candidate_blob_center(members, runtime_config, center_method=center_method)
        if center_result is None:
            continue
        center, summary = center_result

        if (
            summary.get("source") != "rai_cube_patch_union"
            and trim_radius_m
            and float(trim_radius_m) > 0.0
            and len(members) > min_points
        ):
            trim_radius = float(trim_radius_m) + max(float(center.range_m), 0.0) * float(cluster_radius_range_scale)
            trimmed_members = [
                candidate
                for candidate in members
                if float(hypot(candidate.x_m - center.x_m, candidate.y_m - center.y_m)) <= trim_radius
            ]
            if len(trimmed_members) >= min_points and len(trimmed_members) < len(members):
                trimmed_result = _candidate_blob_center(
                    trimmed_members,
                    runtime_config,
                    center_method=center_method,
                )
                if trimmed_result is not None:
                    center, summary = trimmed_result
                    members = trimmed_members
                    summary["trimmed"] = True
                    summary["trim_radius_m"] = round(float(trim_radius), 4)

        strength = float(sum(_candidate_strength(candidate) for candidate in members))
        blob_candidates.append((center, strength, members, summary))

    if not blob_candidates:
        return candidates, {
            "enabled": True,
            "input_count": int(len(candidates)),
            "usable_count": int(len(usable)),
            "output_count": int(len(candidates)),
            "reason": "no_valid_blob_groups",
        }

    blob_candidates.sort(
        key=lambda item: (
            item[1],
            len(item[2]),
            float(item[0].score),
            float(item[0].rdi_peak),
        ),
        reverse=True,
    )
    refined = [item[0] for item in blob_candidates[:max_blobs]]
    for center, strength, members, summary in blob_candidates[:12]:
        group_traces.append(
            {
                "center": _trace_candidate(center),
                "member_count": int(len(members)),
                "strength": round(float(strength), 4),
                "summary": summary,
                "members": _trace_candidates(sorted(members, key=_candidate_strength, reverse=True)[:8]),
            }
        )

    return refined, {
        "enabled": True,
        "input_count": int(len(candidates)),
        "usable_count": int(len(usable)),
        "output_count": int(len(refined)),
        "max_blobs": int(max_blobs),
        "max_candidates": int(max_candidates),
        "min_points": int(min_points),
        "min_score_ratio": round(float(min_score_ratio), 4),
        "cluster_radius_m": round(float(cluster_radius_m), 4),
        "cluster_radius_range_scale": round(float(cluster_radius_range_scale), 4),
        "doppler_radius_bins": int(doppler_radius_bins),
        "center_method": str(center_method),
        "floor_quantile": round(float(floor_quantile), 4),
        "peak_blend": round(float(peak_blend), 4),
        "groups": group_traces,
    }


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


def _refine_person_blob_from_cube(
    rai_cube,
    runtime_config,
    detection_region,
    seed_range_bin,
    seed_angle_bin,
    seed_doppler_bin,
    angle_mask,
    *,
    range_radius_bins=2,
    angle_radius_bins=3,
    doppler_radius_bins=2,
    relative_floor=0.55,
    floor_quantile=0.65,
    min_points=4,
    center_method="weighted_median",
    peak_blend=0.10,
):
    if rai_cube is None:
        return None

    cube = np.asarray(rai_cube, dtype=np.float64)
    if cube.ndim != 3 or cube.size == 0:
        return None

    doppler_count, range_count, angle_count = cube.shape
    seed_doppler_bin = int(np.clip(seed_doppler_bin, 0, doppler_count - 1))
    seed_range_bin = int(np.clip(seed_range_bin, 0, range_count - 1))
    seed_angle_bin = int(np.clip(seed_angle_bin, 0, angle_count - 1))
    range_radius_bins = max(1, int(range_radius_bins))
    angle_radius_bins = max(1, int(angle_radius_bins))
    doppler_radius_bins = max(0, int(doppler_radius_bins))
    relative_floor = float(np.clip(relative_floor, 0.0, 0.95))
    floor_quantile = float(np.clip(floor_quantile, 0.0, 0.95))
    min_points = max(1, int(min_points))
    peak_blend = float(np.clip(peak_blend, 0.0, 0.75))

    doppler_lower = max(seed_doppler_bin - doppler_radius_bins, 0)
    doppler_upper = min(seed_doppler_bin + doppler_radius_bins + 1, doppler_count)
    range_lower = max(seed_range_bin - range_radius_bins, 0)
    range_upper = min(seed_range_bin + range_radius_bins + 1, range_count)
    angle_lower = max(seed_angle_bin - angle_radius_bins, 0)
    angle_upper = min(seed_angle_bin + angle_radius_bins + 1, angle_count)

    patch = np.asarray(
        cube[doppler_lower:doppler_upper, range_lower:range_upper, angle_lower:angle_upper],
        dtype=np.float64,
    )
    if patch.size == 0:
        return None

    local_angle_mask = np.asarray(angle_mask[angle_lower:angle_upper], dtype=bool)
    if not np.any(local_angle_mask):
        return None

    local_range_axis = np.asarray(runtime_config.range_axis_m[range_lower:range_upper], dtype=np.float64)
    local_angle_axis = np.asarray(runtime_config.angle_axis_rad[angle_lower:angle_upper], dtype=np.float64)
    range_grid = local_range_axis[np.newaxis, :, np.newaxis]
    angle_grid = local_angle_axis[np.newaxis, np.newaxis, :]
    x_grid = range_grid * np.sin(angle_grid)
    y_grid = range_grid * np.cos(angle_grid)
    roi_mask = (
        (np.abs(x_grid) <= float(detection_region.lateral_limit_m))
        & (y_grid >= float(detection_region.min_forward_m))
        & (y_grid <= float(detection_region.forward_limit_m))
    )
    angle_roi_mask = np.broadcast_to(local_angle_mask[np.newaxis, np.newaxis, :], patch.shape)
    spatial_mask = np.broadcast_to(roi_mask, patch.shape) & angle_roi_mask
    valid_values = patch[spatial_mask]
    if valid_values.size == 0:
        return None

    seed_depth = seed_doppler_bin - doppler_lower
    seed_row = seed_range_bin - range_lower
    seed_col = seed_angle_bin - angle_lower
    seed_value = float(patch[seed_depth, seed_row, seed_col])
    if seed_value <= 0.0:
        seed_value = float(np.max(valid_values))
    if seed_value <= 0.0:
        return None

    quantile_floor = float(np.quantile(valid_values, floor_quantile)) if valid_values.size else 0.0
    component_floor = max(seed_value * relative_floor, quantile_floor)
    threshold_mask = spatial_mask & (patch >= component_floor)
    if spatial_mask[seed_depth, seed_row, seed_col]:
        threshold_mask = np.array(threshold_mask, copy=True)
        threshold_mask[seed_depth, seed_row, seed_col] = True

    component_mask = _connected_component_mask_3d(threshold_mask, seed_depth, seed_row, seed_col)
    if int(np.count_nonzero(component_mask)) < min_points:
        relaxed_floor = max(seed_value * max(relative_floor * 0.75, 0.20), quantile_floor * 0.75)
        relaxed_mask = spatial_mask & (patch >= relaxed_floor)
        if spatial_mask[seed_depth, seed_row, seed_col]:
            relaxed_mask = np.array(relaxed_mask, copy=True)
            relaxed_mask[seed_depth, seed_row, seed_col] = True
        component_mask = _connected_component_mask_3d(relaxed_mask, seed_depth, seed_row, seed_col)
        component_floor = relaxed_floor

    if int(np.count_nonzero(component_mask)) < min_points:
        return None

    weights = np.where(component_mask, np.maximum(patch - component_floor, 0.0), 0.0)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 1e-9:
        weights = np.where(component_mask, np.maximum(patch, 0.0), 0.0)
        weight_sum = float(np.sum(weights))
    if weight_sum <= 1e-9:
        return None

    x_values = np.broadcast_to(x_grid, patch.shape)[component_mask]
    y_values = np.broadcast_to(y_grid, patch.shape)[component_mask]
    weight_values = weights[component_mask]
    method = str(center_method or "weighted_median").strip().lower()
    if method in {"median", "weighted_median", "robust"}:
        x_center = _weighted_quantile(x_values, weight_values, 0.5)
        y_center = _weighted_quantile(y_values, weight_values, 0.5)
        if x_center is None or y_center is None:
            x_center = float(np.sum(x_values * weight_values) / weight_sum)
            y_center = float(np.sum(y_values * weight_values) / weight_sum)
            method = "weighted_mean_fallback"
    else:
        x_center = float(np.sum(x_values * weight_values) / weight_sum)
        y_center = float(np.sum(y_values * weight_values) / weight_sum)
        method = "weighted_mean"

    seed_range_m = float(runtime_config.range_axis_m[seed_range_bin])
    seed_angle_rad = float(runtime_config.angle_axis_rad[seed_angle_bin])
    seed_x_m = float(seed_range_m * np.sin(seed_angle_rad))
    seed_y_m = float(seed_range_m * np.cos(seed_angle_rad))
    x_center = float((1.0 - peak_blend) * x_center + peak_blend * seed_x_m)
    y_center = float((1.0 - peak_blend) * y_center + peak_blend * seed_y_m)

    range_m = float(hypot(x_center, y_center))
    angle_rad = float(atan2(x_center, max(y_center, 1e-6)))
    doppler_indices = np.arange(doppler_lower, doppler_upper, dtype=np.float64)[:, np.newaxis, np.newaxis]
    doppler_bin = int(round(float(np.sum(doppler_indices * weights) / weight_sum)))
    return {
        "range_bin": _nearest_axis_bin(runtime_config.range_axis_m, range_m),
        "angle_bin": _nearest_axis_bin(runtime_config.angle_axis_rad, angle_rad),
        "doppler_bin": int(np.clip(doppler_bin, 0, doppler_count - 1)),
        "range_m": range_m,
        "angle_rad": angle_rad,
        "x_m": x_center,
        "y_m": y_center,
        "point_count": int(np.count_nonzero(component_mask)),
        "weight_sum": round(float(weight_sum), 4),
        "floor": round(float(component_floor), 4),
        "method": method,
        "bounds": {
            "doppler": [int(doppler_lower), int(doppler_upper - 1)],
            "range": [int(range_lower), int(range_upper - 1)],
            "angle": [int(angle_lower), int(angle_upper - 1)],
        },
    }


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


def _candidate_angle_map(rai_map, rai_cube, doppler_bin, angle_source):
    source = str(angle_source or "collapsed_rai").strip().lower()
    if source in {"doppler_slice_rai", "doppler_slice", "rda_slice"} and rai_cube is not None:
        cube = np.asarray(rai_cube)
        if cube.ndim == 3 and 0 <= int(doppler_bin) < cube.shape[0]:
            return np.asarray(cube[int(doppler_bin)], dtype=np.float64), "doppler_slice_rai"
    return np.asarray(rai_map, dtype=np.float64), "collapsed_rai"


def _trace_rda_cube_point(
    runtime_config,
    *,
    doppler_bin,
    range_bin,
    angle_bin,
    power,
    weight=None,
    power_max=None,
):
    range_bin = int(range_bin)
    angle_bin = int(angle_bin)
    range_m = float(runtime_config.range_axis_m[range_bin])
    angle_rad = float(runtime_config.angle_axis_rad[angle_bin])
    x_m = float(range_m * np.sin(angle_rad))
    y_m = float(range_m * np.cos(angle_rad))
    point = {
        "range_bin": int(range_bin),
        "doppler_bin": int(doppler_bin),
        "angle_bin": int(angle_bin),
        "range_m": round(float(range_m), 4),
        "angle_deg": round(float(np.degrees(angle_rad)), 3),
        "x_m": round(float(x_m), 4),
        "y_m": round(float(y_m), 4),
        "power": round(float(power), 4),
        "score": round(float(power), 4),
    }
    if weight is not None:
        point["weight"] = round(float(weight), 4)
    if power_max is not None and float(power_max) > 1e-9:
        point["normalized_power"] = round(float(power) / float(power_max), 4)
    return point


def _trace_rda_dense_points(
    rai_cube,
    runtime_config,
    detection_region,
    min_range_bin,
    max_range_bin,
    *,
    doppler_guard_bins=0,
    quantile=0.995,
    max_points=180,
):
    if rai_cube is None:
        return {
            "enabled": False,
            "reason": "missing_rai_cube",
            "candidate_count": 0,
            "top_points": [],
        }

    cube = np.asarray(rai_cube, dtype=np.float64)
    if cube.ndim != 3 or cube.size == 0:
        return {
            "enabled": False,
            "reason": "invalid_rai_cube",
            "candidate_count": 0,
            "top_points": [],
        }

    doppler_count, range_count, angle_count = cube.shape
    range_lower = max(0, int(min_range_bin))
    range_upper = min(int(max_range_bin), range_count)
    if range_upper <= range_lower:
        return {
            "enabled": True,
            "reason": "empty_range_roi",
            "candidate_count": 0,
            "top_points": [],
        }

    cube_roi = np.asarray(cube[:, range_lower:range_upper, :], dtype=np.float64)
    cube_roi = np.maximum(cube_roi, 0.0)
    if cube_roi.size == 0 or float(np.max(cube_roi)) <= 0.0:
        return {
            "enabled": True,
            "reason": "zero_power_roi",
            "candidate_count": 0,
            "top_points": [],
        }

    range_axis = np.asarray(runtime_config.range_axis_m[range_lower:range_upper], dtype=np.float64)
    angle_axis = np.asarray(runtime_config.angle_axis_rad, dtype=np.float64)
    range_grid = range_axis[:, np.newaxis]
    angle_grid = angle_axis[np.newaxis, :]
    x_grid = range_grid * np.sin(angle_grid)
    y_grid = range_grid * np.cos(angle_grid)
    spatial_mask_2d = (
        (np.abs(x_grid) <= float(detection_region.lateral_limit_m))
        & (y_grid >= float(detection_region.min_forward_m))
        & (y_grid <= float(detection_region.forward_limit_m))
    )
    valid_mask = np.broadcast_to(spatial_mask_2d[np.newaxis, :, :], cube_roi.shape).copy()

    center_bin = int(getattr(runtime_config, "doppler_fft_size", doppler_count)) // 2
    guard_bins = max(0, int(doppler_guard_bins))
    if guard_bins > 0:
        lower = max(center_bin - guard_bins, 0)
        upper = min(center_bin + guard_bins + 1, doppler_count)
        valid_mask[lower:upper, :, :] = False

    valid_values = cube_roi[valid_mask]
    valid_values = valid_values[np.isfinite(valid_values) & (valid_values > 0.0)]
    if valid_values.size == 0:
        return {
            "enabled": True,
            "reason": "empty_valid_values",
            "candidate_count": 0,
            "top_points": [],
        }

    quantile = float(np.clip(quantile, 0.0, 0.9999))
    threshold = float(np.quantile(valid_values, quantile))
    candidate_mask = valid_mask & (cube_roi >= threshold)
    candidate_indices = np.argwhere(candidate_mask)
    if candidate_indices.size == 0:
        return {
            "enabled": True,
            "quantile": round(float(quantile), 4),
            "threshold": round(float(threshold), 4),
            "candidate_count": 0,
            "top_points": [],
        }

    powers = cube_roi[candidate_indices[:, 0], candidate_indices[:, 1], candidate_indices[:, 2]]
    order = np.argsort(powers)[::-1]
    top_points = []
    power_max = float(np.max(valid_values))
    for local_index in order[: max(1, int(max_points))]:
        doppler_bin = int(candidate_indices[local_index, 0])
        range_bin = int(candidate_indices[local_index, 1] + range_lower)
        angle_bin = int(candidate_indices[local_index, 2])
        power = float(powers[local_index])
        top_points.append(
            _trace_rda_cube_point(
                runtime_config,
                doppler_bin=doppler_bin,
                range_bin=range_bin,
                angle_bin=angle_bin,
                power=power,
                power_max=power_max,
            )
        )

    return {
        "enabled": True,
        "source": "range_doppler_angle_cube",
        "quantile": round(float(quantile), 4),
        "threshold": round(float(threshold), 4),
        "valid_value_count": int(valid_values.size),
        "candidate_count": int(candidate_indices.shape[0]),
        "stored_count": int(len(top_points)),
        "top_points": top_points,
    }


def _trace_projected_cfar_seeds(
    ordered_indices,
    power_map,
    rai_map,
    rai_cube,
    runtime_config,
    detection_region,
    min_range_bin,
    *,
    angle_source,
    max_points=48,
):
    seeds = []
    if ordered_indices is None:
        iterable_indices = []
    else:
        iterable_indices = list(ordered_indices)
    for range_bin_rel, doppler_bin in iterable_indices[: max(1, int(max_points))]:
        range_bin = int(range_bin_rel + min_range_bin)
        if not (0 <= range_bin < len(runtime_config.range_axis_m)):
            continue
        range_m = float(runtime_config.range_axis_m[range_bin])
        angle_mask = _angle_roi_mask(
            range_m,
            runtime_config.angle_axis_rad,
            detection_region,
        )
        if not np.any(angle_mask):
            continue
        candidate_rai_map, resolved_angle_source = _candidate_angle_map(
            rai_map,
            rai_cube,
            int(doppler_bin),
            angle_source,
        )
        if range_bin >= candidate_rai_map.shape[0]:
            continue
        angle_profile = np.asarray(candidate_rai_map[range_bin], dtype=np.float64)
        masked_angle_profile = np.where(angle_mask, angle_profile, 0.0)
        angle_bin = int(np.argmax(masked_angle_profile))
        angle_power = float(masked_angle_profile[angle_bin])
        if angle_power <= 0.0:
            continue
        angle_rad = float(runtime_config.angle_axis_rad[angle_bin])
        power = float(power_map[int(range_bin_rel), int(doppler_bin)])
        seeds.append(
            {
                "range_bin": int(range_bin),
                "doppler_bin": int(doppler_bin),
                "angle_bin": int(angle_bin),
                "range_m": round(float(range_m), 4),
                "angle_deg": round(float(np.degrees(angle_rad)), 3),
                "x_m": round(float(range_m * np.sin(angle_rad)), 4),
                "y_m": round(float(range_m * np.cos(angle_rad)), 4),
                "rdi_power": round(float(power), 4),
                "angle_power": round(float(angle_power), 4),
                "score": round(float(power), 4),
                "angle_source": resolved_angle_source,
            }
        )
    return seeds


def detect_targets(
    rdi_map,
    rai_map,
    runtime_config,
    min_range_bin,
    max_range_bin,
    detection_region,
    rai_cube=None,
    angle_source=None,
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
    object_count_estimator_enabled=True,
    object_count_max_objects=3,
    object_count_min_separation_m=0.65,
    object_count_min_doppler_bins=7,
    object_count_min_score_ratio=0.05,
    protect_multi_object_candidates=False,
    limit_output_to_object_count=False,
    min_output_score=0.0,
    person_blob_refinement_enabled=False,
    person_blob_doppler_radius_bins=2,
    person_blob_min_points=4,
    person_blob_floor_quantile=0.65,
    person_blob_center_method="weighted_median",
    person_blob_peak_blend=0.10,
    blob_center_refinement_enabled=False,
    blob_center_max_candidates=36,
    blob_center_min_points=2,
    blob_center_min_score_ratio=0.04,
    blob_center_cluster_radius_m=0.65,
    blob_center_cluster_radius_range_scale=0.04,
    blob_center_cluster_radius_bands=None,
    blob_center_doppler_radius_bins=10,
    blob_center_method="weighted_median",
    blob_center_trim_radius_m=0.85,
    blob_center_floor_quantile=0.65,
    blob_center_peak_blend=0.0,
    blob_center_single_min_score_ratio=0.12,
    blob_center_single_range_window_m=1.05,
    blob_center_single_side_deadband_m=0.15,
    blob_center_cube_range_radius_m=None,
    blob_center_cube_angle_radius_deg=None,
    blob_center_cube_relative_floor=None,
    blob_center_dense_enabled=True,
    blob_center_dense_quantile=0.995,
    blob_center_dense_min_normalized_power=0.08,
    blob_center_dense_max_points=2400,
    blob_center_dense_min_points=6,
    enable_body_center_refinement=True,
    enable_candidate_merge=True,
    enable_dbscan=True,
    trace=None,
):
    angle_source = str(
        angle_source
        or getattr(runtime_config, "angle_source", "collapsed_rai")
        or "collapsed_rai"
    ).strip().lower()
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
                "mode": {
                    "angle_source": angle_source,
                    "body_center_refinement": bool(enable_body_center_refinement),
                    "person_blob_refinement": bool(person_blob_refinement_enabled),
                    "blob_center_refinement": bool(blob_center_refinement_enabled),
                    "candidate_merge": bool(enable_candidate_merge),
                    "dbscan": bool(enable_dbscan),
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
    if trace_enabled:
        trace["rda_dense_points"] = _trace_rda_dense_points(
            rai_cube,
            runtime_config,
            detection_region,
            min_range_bin,
            max_range_bin,
            doppler_guard_bins=runtime_config.doppler_guard_bins,
            quantile=0.995,
            max_points=180,
        )

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
        projected_seeds = _trace_projected_cfar_seeds(
            ordered_indices,
            power_map,
            rai_map,
            rai_cube,
            runtime_config,
            detection_region,
            min_range_bin,
            angle_source=angle_source,
            max_points=48,
        )
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
            "projected_seed_count": int(len(projected_seeds)),
            "projected_seeds": projected_seeds,
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

        candidate_rai_map, resolved_angle_source = _candidate_angle_map(
            rai_map,
            rai_cube,
            int(doppler_bin),
            angle_source,
        )
        angle_profile = np.asarray(candidate_rai_map[range_bin], dtype=np.float64)
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
    if enable_candidate_merge:
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
            "enabled": bool(enable_candidate_merge),
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
    person_blob_pairs = []
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

        candidate_rai_map, _resolved_angle_source = _candidate_angle_map(
            rai_map,
            rai_cube,
            int(coarse_candidate.doppler_bin),
            angle_source,
        )
        angle_profile = np.asarray(candidate_rai_map[range_bin], dtype=np.float64)
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
        person_blob = None
        if person_blob_refinement_enabled:
            person_blob = _refine_person_blob_from_cube(
                rai_cube,
                runtime_config,
                detection_region,
                range_bin,
                angle_bin,
                int(coarse_candidate.doppler_bin),
                angle_mask,
                range_radius_bins=patch_range_radius,
                angle_radius_bins=patch_angle_radius,
                doppler_radius_bins=person_blob_doppler_radius_bins,
                relative_floor=patch_relative_floor,
                floor_quantile=person_blob_floor_quantile,
                min_points=person_blob_min_points,
                center_method=person_blob_center_method,
                peak_blend=person_blob_peak_blend,
            )
        if person_blob is not None:
            refined_range_bin = int(person_blob["range_bin"])
            refined_angle_bin = int(person_blob["angle_bin"])
            refined_range_m = float(person_blob["range_m"])
            refined_angle_rad = float(person_blob["angle_rad"])
            refined_x_m = float(person_blob["x_m"])
            refined_y_m = float(person_blob["y_m"])
            refined_doppler_bin = int(person_blob["doppler_bin"])
        elif enable_body_center_refinement:
            (
                refined_range_bin,
                refined_angle_bin,
                refined_range_m,
                refined_angle_rad,
                refined_x_m,
                refined_y_m,
            ) = _refine_body_center_from_patch(
                candidate_rai_map,
                runtime_config,
                range_bin,
                angle_bin,
                angle_mask,
                angle_floor=angle_floor,
                range_radius_bins=patch_range_radius,
                angle_radius_bins=patch_angle_radius,
                relative_floor=patch_relative_floor,
            )
            refined_doppler_bin = int(coarse_candidate.doppler_bin)
        else:
            refined_range_bin = int(range_bin)
            refined_angle_bin = int(angle_bin)
            refined_range_m = float(range_m)
            refined_angle_rad = float(angle_rad)
            refined_x_m = float(range_m * np.sin(angle_rad))
            refined_y_m = float(range_m * np.cos(angle_rad))
            refined_doppler_bin = int(coarse_candidate.doppler_bin)
        refined_candidate = DetectionCandidate(
            range_bin=refined_range_bin,
            doppler_bin=refined_doppler_bin,
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
        if person_blob is not None and trace_enabled and len(person_blob_pairs) < 12:
            blob_trace = {
                "before": _trace_candidate(coarse_candidate),
                "after": _trace_candidate(refined_candidate),
                "shift_m": round(float(hypot(coarse_candidate.x_m - refined_candidate.x_m, coarse_candidate.y_m - refined_candidate.y_m)), 4),
                "point_count": int(person_blob["point_count"]),
                "weight_sum": person_blob["weight_sum"],
                "floor": person_blob["floor"],
                "method": person_blob["method"],
                "bounds": person_blob["bounds"],
            }
            person_blob_pairs.append(blob_trace)
        if trace_enabled and len(body_center_pairs) < 12:
            body_center_pairs.append(
                {
                    "before": _trace_candidate(coarse_candidate),
                    "after": _trace_candidate(refined_candidate),
                    "shift_m": round(float(hypot(coarse_candidate.x_m - refined_candidate.x_m, coarse_candidate.y_m - refined_candidate.y_m)), 4),
                }
            )

    candidate_pool = refined_candidate_pool or coarse_candidate_pool
    if blob_center_refinement_enabled:
        dense_blob_trace = None
        dense_blob_candidates = []
        if blob_center_dense_enabled and rai_cube is not None:
            dense_blob_candidates, dense_blob_trace = _dense_blob_centers_from_rda_cube(
                rai_cube,
                runtime_config,
                detection_region,
                min_range_bin,
                max_range_bin,
                anchor_candidates=candidate_pool,
                doppler_guard_bins=runtime_config.doppler_guard_bins,
                quantile=blob_center_dense_quantile,
                min_normalized_power=blob_center_dense_min_normalized_power,
                max_points=blob_center_dense_max_points,
                min_points=max(blob_center_dense_min_points, blob_center_min_points),
                max_blobs=int(detection_region.max_targets),
                center_method=blob_center_method,
                anchor_range_bins=max(3, int(round((blob_center_cube_range_radius_m or 0.35) / max(float(np.median(np.diff(runtime_config.range_axis_m))), 1e-6)))) if len(runtime_config.range_axis_m) > 1 else 6,
                anchor_doppler_bins=blob_center_doppler_radius_bins,
            )

        if dense_blob_candidates:
            candidate_pool = dense_blob_candidates
            blob_center_trace = dense_blob_trace or {}
            blob_center_trace["fallback_used"] = False
        else:
            candidate_pool, blob_center_trace = _refine_blob_centers_from_candidates(
                candidate_pool,
                runtime_config,
                detection_region,
                rai_cube=rai_cube,
                body_center_patch_bands=body_center_patch_bands,
                enabled=True,
                max_blobs=int(detection_region.max_targets),
                max_candidates=blob_center_max_candidates,
                min_points=blob_center_min_points,
                min_score_ratio=blob_center_min_score_ratio,
                cluster_radius_m=blob_center_cluster_radius_m,
                cluster_radius_range_scale=blob_center_cluster_radius_range_scale,
                cluster_radius_bands=blob_center_cluster_radius_bands,
                doppler_radius_bins=blob_center_doppler_radius_bins,
                center_method=blob_center_method,
                trim_radius_m=blob_center_trim_radius_m,
                floor_quantile=blob_center_floor_quantile,
                peak_blend=blob_center_peak_blend,
                single_min_score_ratio=blob_center_single_min_score_ratio,
                single_range_window_m=blob_center_single_range_window_m,
                single_side_deadband_m=blob_center_single_side_deadband_m,
                cube_range_radius_m=blob_center_cube_range_radius_m,
                cube_angle_radius_deg=blob_center_cube_angle_radius_deg,
                cube_relative_floor=blob_center_cube_relative_floor,
            )
            if dense_blob_trace is not None:
                blob_center_trace["dense_component_attempt"] = dense_blob_trace
                blob_center_trace["fallback_used"] = True
    else:
        blob_center_trace = {
            "enabled": False,
            "input_count": int(len(candidate_pool)),
            "output_count": int(len(candidate_pool)),
        }
    if trace_enabled:
        trace["body_center_refinement"] = {
            "enabled": bool(enable_body_center_refinement),
            "input_count": int(len(coarse_candidate_pool)),
            "refined_count": int(len(refined_candidate_pool)),
            "fallback_to_coarse": bool(not refined_candidate_pool),
            "pairs": body_center_pairs,
        }
        trace["person_blob_refinement"] = {
            "enabled": bool(person_blob_refinement_enabled),
            "used_count": int(len(person_blob_pairs)),
            "doppler_radius_bins": int(person_blob_doppler_radius_bins),
            "min_points": int(person_blob_min_points),
            "floor_quantile": round(float(person_blob_floor_quantile), 4),
            "center_method": str(person_blob_center_method),
            "peak_blend": round(float(person_blob_peak_blend), 4),
            "pairs": person_blob_pairs,
        }
        trace["blob_center_refinement"] = blob_center_trace
        pre_merge_final = list(candidate_pool)
    if enable_candidate_merge:
        candidate_pool = _merge_candidate_pool(
            candidate_pool,
            runtime_config,
            merge_bands=candidate_merge_bands,
            default_merge_radius_m=max(min_cartesian_separation_m * 0.75, 0.25),
            default_range_bin_radius=1,
            default_doppler_bin_radius=max(2, int(runtime_config.doppler_guard_bins)),
        )
    object_count_estimate = None
    if trace_enabled:
        object_count_estimate = _estimate_object_count_from_candidates(
            candidate_pool,
            runtime_config,
            enabled=object_count_estimator_enabled,
            max_objects=object_count_max_objects,
            min_separation_m=object_count_min_separation_m,
            min_doppler_bins=object_count_min_doppler_bins,
            min_score_ratio=object_count_min_score_ratio,
        )
        trace["candidate_merge_final"] = {
            "enabled": bool(enable_candidate_merge),
            "before_count": int(len(pre_merge_final)),
            "after_count": int(len(candidate_pool)),
            "before_top": _trace_candidates(pre_merge_final),
            "after_top": _trace_candidates(candidate_pool),
        }
        trace["object_count_estimator"] = object_count_estimate
        trace["dbscan"] = {
            "input_count": int(len(candidate_pool)),
            "eps_base": float(min_cartesian_separation_m),
            "cluster_min_samples": int(detection_region.cluster_min_samples),
            "velocity_weight": float(detection_region.cluster_velocity_weight),
            "adaptive_eps_bands": detection_region.adaptive_eps_bands,
            "input_top": _trace_candidates(candidate_pool),
        }
    if object_count_estimate is None:
        object_count_estimate = _estimate_object_count_from_candidates(
            candidate_pool,
            runtime_config,
            enabled=object_count_estimator_enabled,
            max_objects=object_count_max_objects,
            min_separation_m=object_count_min_separation_m,
            min_doppler_bins=object_count_min_doppler_bins,
            min_score_ratio=object_count_min_score_ratio,
        )
    estimated_object_count = int(object_count_estimate.get("estimated_count", 0) or 0)
    protect_multi_object = bool(protect_multi_object_candidates and estimated_object_count >= 2)
    if enable_dbscan and not protect_multi_object:
        clustered_detections = _cluster_detection_candidates(
            candidate_pool,
            runtime_config,
            detection_region,
            min_cartesian_separation_m,
        )
    else:
        clustered_detections = list(candidate_pool)
        if trace_enabled:
            trace["dbscan"]["enabled"] = False
            trace["dbscan"]["skipped_for_multi_object_protection"] = bool(protect_multi_object)
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
    pre_score_filter = list(clustered_detections)
    min_output_score = max(0.0, float(min_output_score))
    if min_output_score > 0.0:
        clustered_detections = [
            candidate for candidate in clustered_detections
            if float(candidate.score) >= min_output_score
        ]
    if trace_enabled:
        trace["output_score_filter"] = {
            "enabled": bool(min_output_score > 0.0),
            "min_output_score": round(float(min_output_score), 4),
            "before_count": int(len(pre_score_filter)),
            "after_count": int(len(clustered_detections)),
            "dropped_count": int(len(pre_score_filter) - len(clustered_detections)),
            "dropped": _trace_candidates([
                candidate for candidate in pre_score_filter
                if candidate not in clustered_detections
            ][:12]),
        }
    if not clustered_detections:
        if trace_enabled:
            trace["early_exit"] = "output_score_filter_empty"
        return []
    output_limit = int(detection_region.max_targets)
    if limit_output_to_object_count and estimated_object_count > 0:
        output_limit = min(output_limit, estimated_object_count)
    output = clustered_detections[:output_limit]
    if trace_enabled:
        trace["dbscan"]["output_count"] = int(len(clustered_detections))
        trace["dbscan"]["output_top"] = _trace_candidates(clustered_detections)
        trace["final_output"] = {
            "output_count": int(len(output)),
            "truncated_from": int(len(clustered_detections)),
            "output_limit": int(output_limit),
            "limit_output_to_object_count": bool(limit_output_to_object_count),
            "top_detections": _trace_candidates(output),
        }
    return output
