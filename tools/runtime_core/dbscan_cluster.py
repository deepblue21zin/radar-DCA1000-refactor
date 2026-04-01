"""Lightweight adaptive DBSCAN for sparse radar detections."""

from __future__ import annotations

import json
import math
from typing import Iterable, List, Mapping, Optional


def _effective_range(point: Mapping[str, object]) -> float:
    range_val = float(point.get("range", 0.0)) # point에서 "range" 필드의 값을 가져와서 float로 변환합니다. 만약 "range" 필드가 없거나 변환할 수 없는 경우에는 0.0으로 간주합니다.
    if math.isfinite(range_val) and range_val > 0.0:  #정상적인 숫자이니 확인(무한대, NaN, 음수는 제외 )
        return range_val
    return math.hypot(float(point["x"]), float(point["y"])) # range가 유효하지 않은 경우, 점의 x와 y 좌표를 사용하여 원점에서의 거리를 계산하여 반환합니다.


def _normalize_band_float(value: object, field_name: str, allow_none: bool = False) -> Optional[float]:
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"Adaptive DBSCAN band field '{field_name}' is required.")

    if isinstance(value, str):
        lowered = value.strip().lower()
        if allow_none and lowered in {"", "none", "null", "inf", "+inf", "infinity", "+infinity"}:
            return None

    try:
        float_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Adaptive DBSCAN band field '{field_name}' must be numeric.") from exc

    if not math.isfinite(float_value):
        if allow_none:
            return None
        raise ValueError(f"Adaptive DBSCAN band field '{field_name}' must be finite.")

    return float_value


def _format_band_desc(r_min: float, r_max: Optional[float]) -> str:
    upper_text = "inf" if r_max is None else f"{r_max:.2f}"
    return f"{r_min:.2f}-{upper_text}m"


def normalize_adaptive_eps_bands(raw_bands: object) -> List[dict]: # adaptive DBSCAN 밴드를 다양한 입력 형식으로 허용하는 함수입니다. 입력이 문자열인 경우, JSON 배열 또는 "r_min:r_max:eps[:min_samples]" 형식의 세미콜론/쉼표 구분 문자열로 파싱합니다. 각 밴드는 r_min, r_max, eps, (선택적) min_samples 필드를 포함하는 딕셔너리로 정규화됩니다. 밴드가 유효한지 확인하고, 정렬되고 겹치지 않도록 합니다. 반환된 밴드 목록은 클러스터링에 사용됩니다.
    if raw_bands in (None, "", []): # adaptive_eps_bands가 None, 빈 문자열 또는 빈 리스트인 경우, 빈 리스트를 반환하여 adaptive DBSCAN 밴드가 없음을 나타냅니다.
        return []

    parsed_bands: object
    if isinstance(raw_bands, str):
        text = raw_bands.strip()
        if not text:
            return []
        if text.startswith("["): # 문자열이 "["로 시작하면 JSON 배열로 파싱을 시도합니다. 그렇지 않으면 "r_min:r_max:eps[:min_samples]" 형식의 세미콜론/쉼표 구분 문자열로 처리합니다.
            try:
                parsed_bands = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError("Invalid JSON for adaptive DBSCAN bands.") from exc
        else:
            parsed_bands = []
            for chunk in text.replace(";", ",").split(","): # 문자열에서 세미콜론을 쉼표로 대체한 후 쉼표로 분할하여 각 밴드 세그먼트를 처리합니다.
                segment = chunk.strip()
                if not segment:
                    continue
                parts = [part.strip() for part in segment.split(":")]
                if len(parts) not in {3, 4}:
                    raise ValueError( # 각 세그먼트는 3개 또는 4개의 부분으로 구성되어야 합니다. 3개는 r_min, r_max, eps를 나타내고, 4개는 min_samples도 포함합니다.
                        "Adaptive DBSCAN bands must use 'r_min:r_max:eps[:min_samples]' segments."
                    )
                band = {
                    "r_min": parts[0],
                    "r_max": parts[1],
                    "eps": parts[2],
                } # 각 부분을 밴드 딕셔너리에 할당합니다. r_min, r_max, eps는 문자열로 저장되며, 나중에 정규화 과정에서 float로 변환됩니다.
                if len(parts) == 4:
                    band["min_samples"] = parts[3]
                parsed_bands.append(band)
    else:
        parsed_bands = raw_bands

    if not isinstance(parsed_bands, (list, tuple)):
        raise ValueError("Adaptive DBSCAN bands must be a list/tuple or a compact string.")

    normalized: List[dict] = []
    previous_upper: Optional[float] = None
    for index, band in enumerate(parsed_bands):
        if not isinstance(band, Mapping):
            raise ValueError(f"Adaptive DBSCAN band #{index + 1} must be an object/dict.")

        r_min = _normalize_band_float(band.get("r_min"), "r_min")
        r_max = _normalize_band_float(band.get("r_max"), "r_max", allow_none=True)
        eps = _normalize_band_float(band.get("eps"), "eps")

        if r_min is None or r_min < 0.0:
            raise ValueError(f"Adaptive DBSCAN band #{index + 1} requires r_min >= 0.0.")
        if eps is None or eps <= 0.0:
            raise ValueError(f"Adaptive DBSCAN band #{index + 1} requires eps > 0.0.")
        if r_max is not None and r_max <= r_min:
            raise ValueError(f"Adaptive DBSCAN band #{index + 1} requires r_max > r_min.")
        if previous_upper is not None and r_min < previous_upper:
            raise ValueError("Adaptive DBSCAN bands must be sorted and non-overlapping.")

        min_samples = band.get("min_samples")
        if min_samples is not None:
            try:
                min_samples = int(min_samples)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Adaptive DBSCAN band #{index + 1} has invalid min_samples."
                ) from exc
            if min_samples < 1:
                raise ValueError(f"Adaptive DBSCAN band #{index + 1} requires min_samples >= 1.")

        normalized_band = {
            "r_min": float(r_min),
            "r_max": None if r_max is None else float(r_max),
            "eps": float(eps),
            "description": _format_band_desc(float(r_min), None if r_max is None else float(r_max)),
        }
        if min_samples is not None:
            normalized_band["min_samples"] = min_samples
        normalized.append(normalized_band)

        previous_upper = None if r_max is None else float(r_max)
        if previous_upper is None and index != len(parsed_bands) - 1:
            raise ValueError("An adaptive DBSCAN band with r_max=None must be the last band.")

    return normalized


def _range_matches_band(range_val: float, band: Mapping[str, object]) -> bool:
    band_min = float(band["r_min"])
    band_max = band.get("r_max")
    if range_val < band_min:
        return False
    if band_max is None:
        return True
    return range_val < float(band_max)      


def _build_feature_matrix(point_list: List[dict], use_velocity_feature: bool, velocity_weight: float): # 점 목록에서 DBSCAN 클러스터링에 사용할 특징 행렬을 구축하는 함수입니다. 각 점의 x, y 좌표를 기본 특징으로 사용하며, use_velocity_feature가 True인 경우에는 v(속도)도 포함합니다. velocity_weight는 속도 특징의 중요도를 조절하는 가중치로 사용됩니다. 반환된 특징 행렬은 DBSCAN 알고리즘에 입력으로 사용됩니다.
    if use_velocity_feature:
        return [
            (
                float(point["x"]),
                float(point["y"]),
                float(point.get("v", 0.0)) * velocity_weight,
            )
            for point in point_list
        ]
    return [
        (
            float(point["x"]),
            float(point["y"]),
        )
        for point in point_list
    ]
# 속도의 가중치를 적용하여 특징 행렬을 구축하는 방법, use_velocity_feature가 False인 경우에는 x와 y 좌표만 포함된 특징 행렬을 반환하는 방법을 보여줍니다.

def _region_query(features, point_index, eps): # DBSCAN의 핵심 부분으로, 주어진 점에서 eps 이내에 있는 모든 점의 인덱스를 반환하는 함수입니다.
    eps_squared = eps * eps
    query_point = features[point_index]
    neighbors = []
    for candidate_index, candidate_point in enumerate(features):
        distance_squared = 0.0
        for dimension, value in enumerate(query_point):
            delta = value - candidate_point[dimension]
            distance_squared += delta * delta
        if distance_squared <= eps_squared:
            neighbors.append(candidate_index)
    return neighbors


def _dbscan_labels(features, eps, min_samples): # DBSCAN 알고리즘을 구현하여 각 점에 대한 클러스터 레이블을 반환하는 함수입니다. -1은 노이즈를 나타냅니다.  
    labels = [-1] * len(features)
    visited = [False] * len(features)
    cluster_id = 0

    for point_index in range(len(features)):
        if visited[point_index]:
            continue

        visited[point_index] = True
        neighbors = _region_query(features, point_index, eps) # 현재 점에서 eps 이내에 있는 모든 점의 인덱스를 가져옵니다.
        if len(neighbors) < min_samples:
            continue

        labels[point_index] = cluster_id # 현재 점에 클러스터 ID를 할당합니다. 이 점은 클러스터의 일부가 됩니다.
        seeds = list(neighbors) # 현재 만들어진 군집을 어디까지 확장할 지 검사하기 위해 쌓아두는 대기열.
        seed_index = 0 # 대기열에서 처리할 다음 점의 인덱스입니다. 초기에는 0으로 설정되어 첫 번째 시드 점부터 시작합니다.(초기화)
        seed_set = set(seeds) # 대기열에 있는 점들의 인덱스를 추적하는 집합. (중복 제거)

        while seed_index < len(seeds):
            candidate_index = seeds[seed_index] # 대기열에 있는 인덱스를 가져옴.
            seed_index += 1 

            if not visited[candidate_index]: # 방문한 점은 건너뛰고, 방문하지 않은 점에 대해서만 처리합니다.
                visited[candidate_index] = True 
                candidate_neighbors = _region_query(features, candidate_index, eps) # 후보 점에서 eps 이내에 있는 모든 점의 인덱스를 가져옵니다.
                if len(candidate_neighbors) >= min_samples:
                    for neighbor_index in candidate_neighbors:   # 후보 점이 충분한 이웃을 가지고 있다면, 그 이웃들을 대기열에 추가하여 클러스터 확장을 계속합니다.
                        if neighbor_index not in seed_set: 
                            seed_set.add(neighbor_index) # 대기열에 추가된 점의 인덱스를 집합에 추가하여 중복을 방지합니다.
                            seeds.append(neighbor_index) # 후보 점의 이웃 인덱스를 대기열에 추가하여 클러스터 확장을 계속합니다.

            if labels[candidate_index] == -1: #노이즈 할당
                labels[candidate_index] = cluster_id # ID할당

        cluster_id += 1

    return labels


def _shared_band_boundary(cluster_a: Mapping[str, object], cluster_b: Mapping[str, object]) -> Optional[float]: # 두 클러스터가 적어도 하나의 공통 밴드 경계를 공유하는 지 확인
    tolerance = 1e-6
    a_min = cluster_a.get("_band_r_min")
    a_max = cluster_a.get("_band_r_max")
    b_min = cluster_b.get("_band_r_min")
    b_max = cluster_b.get("_band_r_max")

    if a_max is not None and b_min is not None and abs(float(a_max) - float(b_min)) <= tolerance:
        return float(a_max)
    if b_max is not None and a_min is not None and abs(float(b_max) - float(a_min)) <= tolerance:
        return float(b_max)
    return None


def _merge_band_description(desc_a: Optional[str], desc_b: Optional[str]) -> Optional[str]: 
    descriptions = [desc for desc in (desc_a, desc_b) if desc]
    if not descriptions:
        return None

    unique_descriptions = []
    for description in descriptions:
        if description not in unique_descriptions:
            unique_descriptions.append(description)
    return "|".join(unique_descriptions)


def _merge_band_upper(a_max: Optional[float], b_max: Optional[float]) -> Optional[float]:
    if a_max is None or b_max is None:
        return None
    return max(float(a_max), float(b_max))

# 클러스터 요약값 만들기. 클러스터에 속한 점들의 x, y, v의 가중 평균을 계산하여 클러스터 중심을 구하고, 클러스터의 크기, 확산 정도, 점수 통계 등을 계산하여 클러스터의 특성을 요약하는 딕셔너리를 반환합니다. 또한 adaptive eps 밴드 정보와 경계 병합 여부도 포함할 수 있습니다.
def _summarize_cluster_points( # 즉, 물체가 하나처럼 보이는 요약 정보를 만든다.
    c_points: List[dict],
    label: int,
    eps: float,
    min_samples: int,
    range_band_desc: Optional[str] = None,
    band_r_min: Optional[float] = None,
    band_r_max: Optional[float] = None,
    boundary_merged: bool = False,
) -> dict:
    size = len(c_points)
    if size == 0:
        raise ValueError("Cannot summarize an empty cluster.")

    weights = [max(float(point.get("score", 0.0)), 0.0) for point in c_points]
    weight_sum = sum(weights)
    if weight_sum <= 0.0:
        weights = [1.0] * size # 점들의 score가 0이하면 weight를 1로 설정
        weight_sum = float(size)
    # 가중평균 계산 
    x_mean = sum(float(point["x"]) * weight for point, weight in zip(c_points, weights)) / weight_sum
    y_mean = sum(float(point["y"]) * weight for point, weight in zip(c_points, weights)) / weight_sum
    v_mean = sum(float(point.get("v", 0.0)) * weight for point, weight in zip(c_points, weights)) / weight_sum
    range_vals = [_effective_range(point) for point in c_points]
    spread_xy = math.sqrt( # 중심으로부터 퍼진 정도를 계산. 작을 수록 점들이 촘촘히 모여 있다는 뜻이다.
        sum(
            (math.hypot(float(point["x"]) - x_mean, float(point["y"]) - y_mean) ** 2)
            for point in c_points
        ) / float(size)
    )
    peak_score = max(float(point.get("score", 0.0)) for point in c_points) # 이 값이 클러스터의 대표 점수, 높을 수록 강한 검출
    mean_score = sum(float(point.get("score", 0.0)) for point in c_points) / float(size)

    size_score = min(1.0, float(size) / max(float(min_samples), 1.0))
    spread_score = max(0.0, 1.0 - (spread_xy / max(eps, 1e-6))) # spread가 eps보다 작을 수록 1에 가까워지는 점수. spread가 eps 이상이면 0이 됩니다.
    score_strength = min(1.0, peak_score / 3.0)
    confidence = max(0.0, min(1.0, (0.45 * size_score) + (0.3 * score_strength) + (0.25 * spread_score))) # 클러스터의 신뢰도 계산, 너무 퍼져있지 않을때 높은 점수 받음

    cluster = { # 사용자에게 보여줄 정보
        "x": float(x_mean),
        "y": float(y_mean),
        "v": float(v_mean),
        "size": size,
        "confidence": float(confidence),
        "label": label,
        "spread_xy": float(spread_xy),
        "mean_score": float(mean_score),
        "peak_score": float(peak_score),
        "eps_used": float(eps),
        "min_samples_used": int(min_samples),
        "member_points": list(c_points),
        "_member_points": list(c_points),
        "_band_r_min": band_r_min,
        "_band_r_max": band_r_max,
        "_point_range_min": float(min(range_vals)),
        "_point_range_max": float(max(range_vals)),
    }
    if range_band_desc is not None:
        cluster["range_band"] = range_band_desc
    if boundary_merged:
        cluster["boundary_merged"] = True
    return cluster


def _cluster_single_batch( # 단일 점 목록에 대해 DBSCAN 클러스터링을 수행하는 함수입니다. 입력으로 점 목록과 DBSCAN 매개변수를 받아서 클러스터링을 수행하고, 각 클러스터에 대한 요약 정보를 포함하는 딕셔너리 목록과 다음 레이블 오프셋을 반환합니다. adaptive eps 밴드 정보도 포함할 수 있습니다.
    point_list: List[dict],
    eps: float,
    min_samples: int,
    use_velocity_feature: bool,
    velocity_weight: float,
    label_offset: int = 0,
    range_band_desc: Optional[str] = None,
    band_r_min: Optional[float] = None,
    band_r_max: Optional[float] = None,
) -> tuple[List[dict], int]:
    if not point_list:
        return [], label_offset

    features = _build_feature_matrix(point_list, use_velocity_feature, velocity_weight)
    labels = _dbscan_labels(features, eps, min_samples)

    clusters: List[dict] = []
    next_label = label_offset
    unique_labels = sorted(set(labels))
    for label in unique_labels:
        if label == -1:
            continue

        member_indices = [index for index, current_label in enumerate(labels) if current_label == label]
        if not member_indices:
            continue

        c_points = [point_list[index] for index in member_indices]
        cluster = _summarize_cluster_points(
            c_points=c_points,
            label=next_label,
            eps=eps,
            min_samples=min_samples,
            range_band_desc=range_band_desc,
            band_r_min=band_r_min,
            band_r_max=band_r_max,
        )
        clusters.append(cluster)
        next_label += 1

    return clusters, next_label


def _merge_adaptive_boundary_clusters(clusters: List[dict]) -> List[dict]: # adaptive eps 밴드 경계를 공유하는 클러스터들을 병합하여 더 큰 클러스터로 만드는 함수입니다. 각 클러스터의 중심 간 거리가 병합 임계값(eps) 이내이고, 클러스터의 점들이 공유된 밴드 경계 근처에 있는 경우에 병합이 수행됩니다. 병합된 클러스터는 점들의 결합된 요약값으로 다시 계산됩니다. 이 과정을 반복하여 더 이상 병합할 수 있는 클러스터 쌍이 없을 때까지 진행합니다.
    if len(clusters) < 2:
        return clusters

    merged_clusters = list(clusters)
    while True:
        best_pair: Optional[tuple[int, int]] = None
        best_distance: Optional[float] = None

        for left_index in range(len(merged_clusters)):
            for right_index in range(left_index + 1, len(merged_clusters)):
                left_cluster = merged_clusters[left_index]
                right_cluster = merged_clusters[right_index]
                shared_boundary = _shared_band_boundary(left_cluster, right_cluster)
                if shared_boundary is None:
                    continue

                merge_threshold = max(float(left_cluster["eps_used"]), float(right_cluster["eps_used"]))
                centroid_distance = math.hypot(
                    float(left_cluster["x"]) - float(right_cluster["x"]),
                    float(left_cluster["y"]) - float(right_cluster["y"]),
                )
                if centroid_distance > merge_threshold:
                    continue

                if float(left_cluster["_point_range_max"]) <= float(right_cluster["_point_range_min"]):
                    lower_cluster = left_cluster
                    upper_cluster = right_cluster
                else:
                    lower_cluster = right_cluster
                    upper_cluster = left_cluster

                lower_near_boundary = float(lower_cluster["_point_range_max"]) >= shared_boundary - merge_threshold
                upper_near_boundary = float(upper_cluster["_point_range_min"]) <= shared_boundary + merge_threshold
                if not (lower_near_boundary and upper_near_boundary):
                    continue

                if best_distance is None or centroid_distance < best_distance:
                    best_distance = centroid_distance
                    best_pair = (left_index, right_index)

        if best_pair is None:
            break

        left_index, right_index = best_pair
        left_cluster = merged_clusters[left_index]
        right_cluster = merged_clusters[right_index]
        combined_points = list(left_cluster["_member_points"]) + list(right_cluster["_member_points"])
        merged_cluster = _summarize_cluster_points(
            c_points=combined_points,
            label=min(int(left_cluster["label"]), int(right_cluster["label"])),
            eps=max(float(left_cluster["eps_used"]), float(right_cluster["eps_used"])),
            min_samples=max(int(left_cluster["min_samples_used"]), int(right_cluster["min_samples_used"])),
            range_band_desc=_merge_band_description(left_cluster.get("range_band"), right_cluster.get("range_band")),
            band_r_min=min(float(left_cluster["_band_r_min"]), float(right_cluster["_band_r_min"])),
            band_r_max=_merge_band_upper(left_cluster.get("_band_r_max"), right_cluster.get("_band_r_max")),
            boundary_merged=True,
        )
        merged_clusters[left_index] = merged_cluster
        del merged_clusters[right_index]

    return merged_clusters


def _strip_internal_cluster_fields(clusters: List[dict]) -> List[dict]:
    public_clusters = []
    for cluster in clusters:
        public_clusters.append({key: value for key, value in cluster.items() if not key.startswith("_")})
    return public_clusters


def cluster_points( # DBSCAN을 사용하여 점들을 클러스터링하는 주요 함수입니다. 입력으로 점들의 이터러블과 DBSCAN 매개변수(eps, min_samples, use_velocity_feature, velocity_weight, adaptive_eps_bands)를 받아서 클러스터링을 수행합니다. adaptive_eps_bands가 제공된 경우, 점들을 해당 밴드에 따라 그룹화하여 각 밴드에 대해 별도의 DBSCAN을 실행합니다. 그런 다음 adaptive 밴드 경계를 공유하는 클러스터들을 병합하여 최종 클러스터 목록을 반환합니다. 반환된 클러스터는 각 클러스터의 중심 좌표, 크기, 신뢰도 등의 요약 정보를 포함하는 딕셔너리로 구성됩니다.
    points: Iterable[dict],
    eps: float = 0.35,
    min_samples: int = 1,
    use_velocity_feature: bool = False,
    velocity_weight: float = 0.25,
    adaptive_eps_bands: object = None,
) -> List[dict]:
    # 불량 입력에 대한 유효성 검사
    if velocity_weight < 0.0:
        raise ValueError("velocity_weight must be >= 0.0")
    if eps <= 0.0:
        raise ValueError("eps must be > 0.0")
    if min_samples < 1:
        raise ValueError("min_samples must be >= 1")

    point_list: List[dict] = []
    for index, point in enumerate(points):
        try:
            x_val = float(point["x"])
            y_val = float(point["y"])
        except (KeyError, TypeError, ValueError):
            continue

        if not math.isfinite(x_val) or not math.isfinite(y_val):
            continue

        clean_point = dict(point)
        clean_point["x"] = x_val
        clean_point["y"] = y_val
        clean_point["cluster_index"] = int(point.get("cluster_index", index))
        for key in ("v", "score", "range"):
            try:
                value = float(clean_point.get(key, 0.0))
            except (TypeError, ValueError):
                value = 0.0
            clean_point[key] = value if math.isfinite(value) else 0.0
        point_list.append(clean_point)

    if not point_list:
        return []

    normalized_bands = normalize_adaptive_eps_bands(adaptive_eps_bands)
    if not normalized_bands: #일반모드
        clusters, _ = _cluster_single_batch(
            point_list=point_list,
            eps=eps,
            min_samples=min_samples,
            use_velocity_feature=use_velocity_feature,
            velocity_weight=velocity_weight,
        )
        return _strip_internal_cluster_fields(clusters)

    band_point_lists: List[List[dict]] = [[] for _ in normalized_bands]
    fallback_points: List[dict] = []
    for point in point_list:
        point_range = _effective_range(point)
        matched = False
        for band_index, band in enumerate(normalized_bands):
            if _range_matches_band(point_range, band):
                band_point_lists[band_index].append(point)
                matched = True
                break
        if not matched:
            fallback_points.append(point)

    clusters: List[dict] = []
    next_label = 0
    for band, band_points in zip(normalized_bands, band_point_lists):
        if not band_points:
            continue

        band_clusters, next_label = _cluster_single_batch(
            point_list=band_points,
            eps=float(band["eps"]),
            min_samples=int(band.get("min_samples", min_samples)),
            use_velocity_feature=use_velocity_feature,
            velocity_weight=velocity_weight,
            label_offset=next_label,
            range_band_desc=str(band["description"]),
            band_r_min=float(band["r_min"]),
            band_r_max=None if band.get("r_max") is None else float(band["r_max"]),
        )
        clusters.extend(band_clusters)

    if fallback_points:
        fallback_clusters, _ = _cluster_single_batch(
            point_list=fallback_points,
            eps=eps,
            min_samples=min_samples,
            use_velocity_feature=use_velocity_feature,
            velocity_weight=velocity_weight,
            label_offset=next_label,
            range_band_desc="fallback",
        )
        clusters.extend(fallback_clusters)

    clusters = _merge_adaptive_boundary_clusters(clusters)
    return _strip_internal_cluster_fields(clusters)
