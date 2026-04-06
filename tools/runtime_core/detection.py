from dataclasses import dataclass
from math import atan2, hypot

import numpy as np

from .dbscan_cluster import cluster_points


@dataclass(frozen=True)
class DetectionRegion:
    lateral_limit_m: float
    forward_limit_m: float
    min_forward_m: float = 0.0
    max_targets: int = 6
    allow_strongest_fallback: bool = False
    adaptive_eps_bands: object = None
    cluster_min_samples: int = 1
    cluster_velocity_weight: float = 0.0


@dataclass(frozen=True)
class DetectionCandidate:
    range_bin: int
    doppler_bin: int
    angle_bin: int
    range_m: float
    angle_deg: float
    x_m: float
    y_m: float
    rdi_peak: float
    rai_peak: float
    score: float


def _local_maxima_mask(power_map):
    rows, cols = power_map.shape
    padded = np.pad(power_map, 1, mode='constant', constant_values=-np.inf)
    center = padded[1:-1, 1:-1]
    maxima_mask = np.ones((rows, cols), dtype=bool)

    for row_offset in (-1, 0, 1):
        for col_offset in (-1, 0, 1):
            if row_offset == 0 and col_offset == 0:
                continue
            neighbor = padded[
                1 + row_offset:1 + row_offset + rows,
                1 + col_offset:1 + col_offset + cols,
            ]
            maxima_mask &= center >= neighbor

    return maxima_mask


def _build_integral_image(power_map):
    padded = np.pad(power_map, ((1, 0), (1, 0)), mode='constant')
    return padded.cumsum(axis=0).cumsum(axis=1)


def _rect_sum(integral_image, top, left, bottom, right):
    return (
        integral_image[bottom, right]
        - integral_image[top, right]
        - integral_image[bottom, left]
        + integral_image[top, left]
    )


def cfar_threshold_2d(power_map, training_cells=(6, 6), guard_cells=(1, 1)):
    rows, cols = power_map.shape
    train_rows, train_cols = training_cells
    guard_rows, guard_cols = guard_cells
    outer_rows = train_rows + guard_rows
    outer_cols = train_cols + guard_cols
    padded = np.pad(
        power_map,
        ((outer_rows, outer_rows), (outer_cols, outer_cols)),
        mode='edge',
    )
    integral = _build_integral_image(padded)
    thresholds = np.zeros_like(power_map, dtype=np.float64)
    outer_count = (2 * outer_rows + 1) * (2 * outer_cols + 1)
    guard_count = (2 * guard_rows + 1) * (2 * guard_cols + 1)
    training_count = max(outer_count - guard_count, 1)

    for row_index in range(rows):
        padded_row = row_index + outer_rows
        outer_top = padded_row - outer_rows
        outer_bottom = padded_row + outer_rows + 1
        guard_top = padded_row - guard_rows
        guard_bottom = padded_row + guard_rows + 1

        for col_index in range(cols):
            padded_col = col_index + outer_cols
            outer_left = padded_col - outer_cols
            outer_right = padded_col + outer_cols + 1
            guard_left = padded_col - guard_cols
            guard_right = padded_col + guard_cols + 1

            outer_sum = _rect_sum(
                integral,
                outer_top,
                outer_left,
                outer_bottom,
                outer_right,
            )
            guard_sum = _rect_sum(
                integral,
                guard_top,
                guard_left,
                guard_bottom,
                guard_right,
            )
            thresholds[row_index, col_index] = (outer_sum - guard_sum) / training_count

    return thresholds


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


def _body_center_patch_for_range(
    range_m,
    patch_bands,
    default_range_radius_bins=1,
    default_angle_radius_bins=2,
    default_relative_floor=0.55,
):
    range_radius_bins = int(default_range_radius_bins)
    angle_radius_bins = int(default_angle_radius_bins)
    relative_floor = float(default_relative_floor)

    if not patch_bands:
        return range_radius_bins, angle_radius_bins, relative_floor

    for band in patch_bands:
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
            range_radius_bins = max(1, int(band.get("range_radius_bins", range_radius_bins)))
            angle_radius_bins = max(1, int(band.get("angle_radius_bins", angle_radius_bins)))
            relative_floor = float(band.get("relative_floor", relative_floor))
        except (TypeError, ValueError, AttributeError):
            return (
                int(default_range_radius_bins),
                int(default_angle_radius_bins),
                float(default_relative_floor),
            )
        break

    relative_floor = float(np.clip(relative_floor, 0.0, 0.95))
    return range_radius_bins, angle_radius_bins, relative_floor


def _connected_component_mask(binary_mask, seed_row, seed_col):
    rows, cols = binary_mask.shape
    if rows == 0 or cols == 0:
        return np.zeros_like(binary_mask, dtype=bool)

    seed_row = int(np.clip(seed_row, 0, rows - 1))
    seed_col = int(np.clip(seed_col, 0, cols - 1))
    if not bool(binary_mask[seed_row, seed_col]):
        return np.zeros_like(binary_mask, dtype=bool)

    component = np.zeros_like(binary_mask, dtype=bool)
    stack = [(seed_row, seed_col)]
    component[seed_row, seed_col] = True

    while stack:
        row_index, col_index = stack.pop()
        for row_offset in (-1, 0, 1):
            for col_offset in (-1, 0, 1):
                if row_offset == 0 and col_offset == 0:
                    continue
                next_row = row_index + row_offset
                next_col = col_index + col_offset
                if not (0 <= next_row < rows and 0 <= next_col < cols):
                    continue
                if component[next_row, next_col] or not bool(binary_mask[next_row, next_col]):
                    continue
                component[next_row, next_col] = True
                stack.append((next_row, next_col))

    return component


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


def _refine_body_center_from_patch(
    rai_map,
    runtime_config,
    seed_range_bin,
    seed_angle_bin,
    angle_mask,
    angle_floor=0.0,
    range_radius_bins=1,
    angle_radius_bins=2,
    relative_floor=0.55,
):
    range_radius_bins = max(1, int(range_radius_bins))
    angle_radius_bins = max(1, int(angle_radius_bins))
    relative_floor = float(np.clip(relative_floor, 0.0, 0.95))
    range_count, angle_count = rai_map.shape

    range_lower = max(int(seed_range_bin) - range_radius_bins, 0)
    range_upper = min(int(seed_range_bin) + range_radius_bins + 1, range_count)
    angle_lower = max(int(seed_angle_bin) - angle_radius_bins, 0)
    angle_upper = min(int(seed_angle_bin) + angle_radius_bins + 1, angle_count)

    patch = np.asarray(rai_map[range_lower:range_upper, angle_lower:angle_upper], dtype=np.float64)
    if patch.size == 0:
        seed_range_m = float(runtime_config.range_axis_m[int(seed_range_bin)])
        seed_angle_rad = float(runtime_config.angle_axis_rad[int(seed_angle_bin)])
        return (
            int(seed_range_bin),
            int(seed_angle_bin),
            seed_range_m,
            seed_angle_rad,
            float(seed_range_m * np.sin(seed_angle_rad)),
            float(seed_range_m * np.cos(seed_angle_rad)),
        )

    local_angle_mask = np.asarray(angle_mask[angle_lower:angle_upper], dtype=bool)
    if not np.any(local_angle_mask):
        seed_range_m = float(runtime_config.range_axis_m[int(seed_range_bin)])
        seed_angle_rad = float(runtime_config.angle_axis_rad[int(seed_angle_bin)])
        return (
            int(seed_range_bin),
            int(seed_angle_bin),
            seed_range_m,
            seed_angle_rad,
            float(seed_range_m * np.sin(seed_angle_rad)),
            float(seed_range_m * np.cos(seed_angle_rad)),
        )

    patch_mask = np.broadcast_to(local_angle_mask[np.newaxis, :], patch.shape)
    seed_row = int(seed_range_bin) - range_lower
    seed_col = int(seed_angle_bin) - angle_lower
    seed_value = float(patch[seed_row, seed_col]) if patch.size else 0.0
    component_floor = max(float(angle_floor), seed_value * relative_floor)
    threshold_mask = patch_mask & (patch >= component_floor)
    if patch_mask[seed_row, seed_col]:
        threshold_mask = np.array(threshold_mask, copy=True)
        threshold_mask[seed_row, seed_col] = True

    component_mask = _connected_component_mask(threshold_mask, seed_row, seed_col)
    if not np.any(component_mask):
        component_mask = patch_mask & (patch > 0.0)
    if not np.any(component_mask):
        seed_range_m = float(runtime_config.range_axis_m[int(seed_range_bin)])
        seed_angle_rad = float(runtime_config.angle_axis_rad[int(seed_angle_bin)])
        return (
            int(seed_range_bin),
            int(seed_angle_bin),
            seed_range_m,
            seed_angle_rad,
            float(seed_range_m * np.sin(seed_angle_rad)),
            float(seed_range_m * np.cos(seed_angle_rad)),
        )

    weights = np.where(component_mask, np.maximum(patch - component_floor, 0.0), 0.0)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 1e-9:
        weights = np.where(component_mask, np.maximum(patch, 0.0), 0.0)
        weight_sum = float(np.sum(weights))
    if weight_sum <= 1e-9:
        seed_range_m = float(runtime_config.range_axis_m[int(seed_range_bin)])
        seed_angle_rad = float(runtime_config.angle_axis_rad[int(seed_angle_bin)])
        return (
            int(seed_range_bin),
            int(seed_angle_bin),
            seed_range_m,
            seed_angle_rad,
            float(seed_range_m * np.sin(seed_angle_rad)),
            float(seed_range_m * np.cos(seed_angle_rad)),
        )

    local_range_axis = np.asarray(runtime_config.range_axis_m[range_lower:range_upper], dtype=np.float64)
    local_angle_axis = np.asarray(runtime_config.angle_axis_rad[angle_lower:angle_upper], dtype=np.float64)
    range_grid = local_range_axis[:, np.newaxis]
    angle_grid = local_angle_axis[np.newaxis, :]
    x_grid = range_grid * np.sin(angle_grid)
    y_grid = range_grid * np.cos(angle_grid)

    x_m = float(np.sum(x_grid * weights) / weight_sum)
    y_m = float(np.sum(y_grid * weights) / weight_sum)
    range_m = float(hypot(x_m, y_m))
    angle_rad = float(atan2(x_m, max(y_m, 1e-6)))
    range_bin = _nearest_axis_bin(runtime_config.range_axis_m, range_m)
    angle_bin = _nearest_axis_bin(runtime_config.angle_axis_rad, angle_rad)
    return range_bin, angle_bin, range_m, angle_rad, x_m, y_m


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
        seed = max(members, key=lambda candidate: (candidate.score, candidate.rdi_peak, candidate.rai_peak))
        x_m = float(cluster["x"])
        y_m = float(cluster["y"])
        range_m = float(hypot(x_m, y_m))
        angle_rad = float(atan2(x_m, max(y_m, 1e-6)))
        range_bin = _nearest_axis_bin(runtime_config.range_axis_m, range_m)
        angle_bin = _nearest_axis_bin(runtime_config.angle_axis_rad, angle_rad)
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
                score=float(max(cluster.get("peak_score", 0.0), seed.score) * max(cluster.get("confidence", 0.0), 0.5)),
            )
        )

    detections.sort(
        key=lambda candidate: (candidate.score, candidate.rdi_peak, candidate.rai_peak),
        reverse=True,
    )
    return detections


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
):
    rdi_roi = np.asarray(rdi_map[min_range_bin:max_range_bin], dtype=np.float64)
    if rdi_roi.size == 0:
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
        return []

    candidate_scores = power_map[candidate_indices[:, 0], candidate_indices[:, 1]]
    ordered_indices = candidate_indices[np.argsort(candidate_scores)[::-1]]
    candidate_pool = []
    rdi_peak_ceiling = float(np.max(power_map))

    for range_bin_rel, doppler_bin in ordered_indices:
        range_bin = int(range_bin_rel + min_range_bin)
        range_m = float(runtime_config.range_axis_m[range_bin])
        angle_mask = _angle_roi_mask(
            range_m,
            runtime_config.angle_axis_rad,
            detection_region,
        )
        if not np.any(angle_mask):
            continue

        angle_profile = np.asarray(rai_map[range_bin], dtype=np.float64)
        masked_angle_profile = np.where(angle_mask, angle_profile, 0)
        peak_angle_bin = int(np.argmax(masked_angle_profile))
        rai_peak = float(masked_angle_profile[peak_angle_bin])
        if rai_peak <= 0:
            continue

        roi_angle_values = masked_angle_profile[angle_mask]
        if roi_angle_values.size == 0:
            continue

        angle_floor = float(np.quantile(roi_angle_values, angle_quantile))
        angle_contrast = rai_peak / max(angle_floor, 1e-6)
        if angle_contrast < angle_contrast_scale:
            continue

        if not _angle_is_local_peak(masked_angle_profile, peak_angle_bin):
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
        patch_range_radius, patch_angle_radius, patch_relative_floor = _body_center_patch_for_range(
            range_m,
            body_center_patch_bands,
            default_range_radius_bins=1,
            default_angle_radius_bins=max(2, centroid_radius + 1),
            default_relative_floor=0.55,
        )
        (
            range_bin,
            angle_bin,
            range_m,
            angle_rad,
            x_m,
            y_m,
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
        rdi_peak = float(rdi_map[range_bin, int(doppler_bin)])
        normalized_rdi = float(power_map[range_bin_rel, int(doppler_bin)] / max(rdi_peak_ceiling, 1e-6))
        candidate_score = normalized_rdi * min(angle_contrast, 3.0)

        candidate_pool.append(
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

    candidate_pool.sort(
        key=lambda candidate: (candidate.score, candidate.rdi_peak, candidate.rai_peak),
        reverse=True,
    )
    clustered_detections = _cluster_detection_candidates(
        candidate_pool,
        runtime_config,
        detection_region,
        min_cartesian_separation_m,
    )
    if not clustered_detections:
        return []
    return clustered_detections[:detection_region.max_targets]
