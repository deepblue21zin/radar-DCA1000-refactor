from dataclasses import dataclass


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

