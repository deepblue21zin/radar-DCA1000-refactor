from math import atan2, hypot

import numpy as np


def nearest_axis_bin(axis_values, value):
    return int(np.argmin(np.abs(np.asarray(axis_values) - value)))


def body_center_patch_for_range(
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


def connected_component_mask(binary_mask, seed_row, seed_col):
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


def refine_body_center_from_patch(
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

    component_mask = connected_component_mask(threshold_mask, seed_row, seed_col)
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
    range_bin = nearest_axis_bin(runtime_config.range_axis_m, range_m)
    angle_bin = nearest_axis_bin(runtime_config.angle_axis_rad, angle_rad)
    return range_bin, angle_bin, range_m, angle_rad, x_m, y_m

