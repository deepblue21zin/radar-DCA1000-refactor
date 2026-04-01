from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _count_enabled_bits(mask_value):
    return int(mask_value).bit_count()


def _next_power_of_two(value, floor):
    size = max(int(value), int(floor))
    return 1 << (size - 1).bit_length()


@dataclass(frozen=True)
class RadarRuntimeConfig:
    config_path: str
    adc_sample: int
    sample_rate_ksps: int
    freq_slope_mhz_per_us: float
    chirp_loops: int
    tx_num: int
    rx_num: int
    chirp_start_idx: int
    chirp_end_idx: int
    range_fft_size: int
    doppler_fft_size: int
    angle_fft_size: int
    remove_static: bool = True
    doppler_guard_bins: int = 1

    @property
    def virtual_antennas(self):
        return self.tx_num * self.rx_num

    @property
    def frame_length(self):
        return self.adc_sample * self.chirp_loops * self.tx_num * self.rx_num * 2

    @property
    def range_resolution_m(self):
        light_speed = 299792458.0
        sample_rate_hz = self.sample_rate_ksps * 1e3
        slope_hz_per_s = self.freq_slope_mhz_per_us * 1e12
        return light_speed * sample_rate_hz / (2.0 * slope_hz_per_s * self.range_fft_size)

    @property
    def max_range_m(self):
        return self.range_resolution_m * self.range_fft_size

    @property
    def range_axis_m(self):
        return np.arange(self.range_fft_size) * self.range_resolution_m

    @property
    def angle_axis_rad(self):
        centered_bins = np.arange(self.angle_fft_size) - (self.angle_fft_size / 2.0)
        spatial_frequency = (2.0 * centered_bins) / self.angle_fft_size
        spatial_frequency = np.clip(spatial_frequency, -1.0, 1.0)
        return np.arcsin(spatial_frequency)


def parse_runtime_config(config_path, remove_static=True, doppler_guard_bins=1):
    config_path = Path(config_path)
    channel_cfg = None
    profile_cfg = None
    frame_cfg = None

    with config_path.open() as cfg_file:
        for raw_line in cfg_file:
            line = raw_line.strip()
            if not line or line.startswith('%'):
                continue

            parts = line.split()
            command = parts[0]
            if command == 'channelCfg':
                channel_cfg = parts
            elif command == 'profileCfg':
                profile_cfg = parts
            elif command == 'frameCfg':
                frame_cfg = parts

    if channel_cfg is None or profile_cfg is None or frame_cfg is None:
        raise ValueError(
            f'Config file {config_path} is missing one of channelCfg/profileCfg/frameCfg.'
        )

    rx_num = _count_enabled_bits(channel_cfg[1])
    tx_num = _count_enabled_bits(channel_cfg[2])
    adc_sample = int(profile_cfg[10])
    sample_rate_ksps = int(profile_cfg[11])
    freq_slope_mhz_per_us = float(profile_cfg[8])
    chirp_start_idx = int(frame_cfg[1])
    chirp_end_idx = int(frame_cfg[2])
    chirp_loops = int(frame_cfg[3])
    chirps_per_burst = chirp_end_idx - chirp_start_idx + 1

    if chirps_per_burst <= 0:
        raise ValueError(f'Invalid chirp range in {config_path}.')

    if tx_num <= 0 or rx_num <= 0:
        raise ValueError(f'Invalid TX/RX mask in {config_path}.')

    if chirps_per_burst not in (1, tx_num):
        raise ValueError(
            'This pipeline expects one chirp per enabled TX in each burst. '
            f'Found chirp range {chirp_start_idx}-{chirp_end_idx} for {tx_num} TX.'
        )

    return RadarRuntimeConfig(
        config_path=str(config_path),
        adc_sample=adc_sample,
        sample_rate_ksps=sample_rate_ksps,
        freq_slope_mhz_per_us=freq_slope_mhz_per_us,
        chirp_loops=chirp_loops,
        tx_num=tx_num,
        rx_num=rx_num,
        chirp_start_idx=chirp_start_idx,
        chirp_end_idx=chirp_end_idx,
        range_fft_size=_next_power_of_two(adc_sample, 128),
        doppler_fft_size=_next_power_of_two(chirp_loops, 64),
        angle_fft_size=_next_power_of_two(tx_num * rx_num * 2, 32),
        remove_static=remove_static,
        doppler_guard_bins=doppler_guard_bins,
    )


def frame_to_radar_cube(frame_data, runtime_config):
    complex_frame = np.reshape(np.asarray(frame_data), [-1, 4])
    complex_frame = complex_frame[:, 0:2:] + 1j * complex_frame[:, 2::]
    complex_frame = np.reshape(
        complex_frame,
        [
            runtime_config.chirp_loops,
            runtime_config.tx_num,
            runtime_config.rx_num,
            runtime_config.adc_sample,
        ],
    )

    rx_major_channels = []
    for rx_index in range(runtime_config.rx_num):
        channel_view = complex_frame[:, :, rx_index, :]
        channel_view = np.transpose(channel_view, [0, 2, 1])
        rx_major_channels.append(channel_view)

    return np.concatenate(rx_major_channels, axis=2)


def remove_static_clutter(radar_cube):
    static_profile = np.mean(radar_cube, axis=0, keepdims=True)
    return radar_cube - static_profile


def integrate_rdi_channels(rdi_cube):
    if rdi_cube.ndim == 2:
        return rdi_cube
    return np.sum(rdi_cube, axis=2)


def collapse_motion_rai(rai_cube, guard_bins=1):
    if rai_cube.ndim != 3:
        return rai_cube

    motion_cube = np.array(rai_cube, copy=True)
    center_bin = motion_cube.shape[0] // 2
    lower = max(center_bin - guard_bins, 0)
    upper = min(center_bin + guard_bins + 1, motion_cube.shape[0])
    motion_cube[lower:upper, :, :] = 0
    return np.max(motion_cube, axis=0)


def radial_bin_limit(runtime_config, max_distance_m):
    upper_bin = int(np.ceil(max_distance_m / runtime_config.range_resolution_m))
    return int(np.clip(upper_bin, 1, runtime_config.range_fft_size))


def apply_cartesian_roi_to_rai(
    rai_map,
    runtime_config,
    lateral_limit_m,
    forward_limit_m,
    min_forward_m=0.0,
):
    range_axis = runtime_config.range_axis_m[:, np.newaxis]
    angle_axis = runtime_config.angle_axis_rad[np.newaxis, :]
    x_axis = range_axis * np.sin(angle_axis)
    y_axis = range_axis * np.cos(angle_axis)
    roi_mask = (
        (np.abs(x_axis) <= lateral_limit_m)
        & (y_axis >= min_forward_m)
        & (y_axis <= forward_limit_m)
    )
    masked_rai = np.where(roi_mask, rai_map, 0)
    min_radial_bin = radial_bin_limit(runtime_config, min_forward_m) if min_forward_m > 0 else 0
    max_radial_distance = np.sqrt((lateral_limit_m ** 2) + (forward_limit_m ** 2))
    max_radial_bin = radial_bin_limit(runtime_config, max_radial_distance)
    return masked_rai[min_radial_bin:max_radial_bin, :]
