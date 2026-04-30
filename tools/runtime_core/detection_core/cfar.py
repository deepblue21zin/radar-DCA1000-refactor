import numpy as np


def local_maxima_mask(power_map):
    rows, cols = power_map.shape
    padded = np.pad(power_map, 1, mode="constant", constant_values=-np.inf)
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


def build_integral_image(power_map):
    padded = np.pad(power_map, ((1, 0), (1, 0)), mode="constant")
    return padded.cumsum(axis=0).cumsum(axis=1)


def rect_sum(integral_image, top, left, bottom, right):
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
        mode="edge",
    )
    integral = build_integral_image(padded)
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

            outer_sum = rect_sum(
                integral,
                outer_top,
                outer_left,
                outer_bottom,
                outer_right,
            )
            guard_sum = rect_sum(
                integral,
                guard_top,
                guard_left,
                guard_bottom,
                guard_right,
            )
            thresholds[row_index, col_index] = (outer_sum - guard_sum) / training_count

    return thresholds

