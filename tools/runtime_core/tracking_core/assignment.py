from typing import Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment as _scipy_linear_sum_assignment
except ImportError:
    _scipy_linear_sum_assignment = None


def hungarian_fallback(cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pure-numpy assignment fallback for rectangular cost matrices."""
    cost = np.asarray(cost_matrix, dtype=float)
    if cost.ndim != 2:
        raise ValueError("cost_matrix must be 2-dimensional.")
    if cost.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    transposed = False
    rows, cols = cost.shape
    if rows > cols:
        cost = cost.T
        rows, cols = cost.shape
        transposed = True

    u = np.zeros(rows + 1, dtype=float)
    v = np.zeros(cols + 1, dtype=float)
    p = np.zeros(cols + 1, dtype=int)
    way = np.zeros(cols + 1, dtype=int)

    for row in range(1, rows + 1):
        p[0] = row
        col0 = 0
        minv = np.full(cols + 1, np.inf, dtype=float)
        used = np.zeros(cols + 1, dtype=bool)
        while True:
            used[col0] = True
            row0 = p[col0]
            delta = np.inf
            col1 = 0
            for col in range(1, cols + 1):
                if used[col]:
                    continue
                cur = cost[row0 - 1, col - 1] - u[row0] - v[col]
                if cur < minv[col]:
                    minv[col] = cur
                    way[col] = col0
                if minv[col] < delta:
                    delta = minv[col]
                    col1 = col
            for col in range(cols + 1):
                if used[col]:
                    u[p[col]] += delta
                    v[col] -= delta
                else:
                    minv[col] -= delta
            col0 = col1
            if p[col0] == 0:
                break

        while True:
            col1 = way[col0]
            p[col0] = p[col1]
            col0 = col1
            if col0 == 0:
                break

    row_ind = []
    col_ind = []
    for col in range(1, cols + 1):
        if p[col] != 0:
            row_ind.append(p[col] - 1)
            col_ind.append(col - 1)

    row_ind_array = np.asarray(row_ind, dtype=int)
    col_ind_array = np.asarray(col_ind, dtype=int)
    order = np.argsort(row_ind_array)
    row_ind_array = row_ind_array[order]
    col_ind_array = col_ind_array[order]

    if transposed:
        return col_ind_array, row_ind_array
    return row_ind_array, col_ind_array


def linear_sum_assignment(cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if _scipy_linear_sum_assignment is not None:
        return _scipy_linear_sum_assignment(cost_matrix)
    return hungarian_fallback(cost_matrix)

