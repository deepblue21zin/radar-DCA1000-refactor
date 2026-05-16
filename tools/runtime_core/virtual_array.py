from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np


ISK_TX_GEOMETRY_ORDER = (1, 2, 4)
ISK_RX_X_LAMBDA = (0.0, 0.5, 1.0, 1.5)
ISK_TX_POSITIONS_LAMBDA = {
    1: (0.0, 0.0, "TX1"),
    2: (1.0, 0.5, "TX2"),
    4: (2.0, 0.0, "TX3"),
}


@dataclass(frozen=True)
class VirtualArrayModel:
    name: str
    raw_order: np.ndarray
    x_lambda: np.ndarray
    z_lambda: np.ndarray
    labels: tuple[str, ...]
    source: str


def parse_chirp_tx_masks(config_path: str | Path) -> list[int]:
    """Return chirp TX masks in chirpCfg order."""
    masks: list[int] = []
    with Path(config_path).open(encoding="utf-8") as cfg_file:
        for raw_line in cfg_file:
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue
            parts = line.split()
            if parts and parts[0] == "chirpCfg":
                masks.append(int(parts[8]))
    return masks


def parse_ant_geometry(config_path: str | Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Parse TI antGeometry0/1 arrays when present.

    TI people-counting configs express geometry in half-wavelength grid units.
    This project internally uses wavelength units, so values are divided by 2.
    """
    geom0 = None
    geom1 = None
    with Path(config_path).open(encoding="utf-8") as cfg_file:
        for raw_line in cfg_file:
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue
            parts = line.split()
            if parts[0] == "antGeometry0":
                geom0 = np.asarray([float(value) for value in parts[1:]], dtype=np.float64) / 2.0
            elif parts[0] == "antGeometry1":
                geom1 = np.asarray([float(value) for value in parts[1:]], dtype=np.float64) / 2.0
    if geom0 is None or geom1 is None or geom0.size != geom1.size:
        return None
    return geom0, geom1


def _raw_channel_for_tx_bit(rx: int, tx_bit: int, chirp_tx_masks: list[int], tx_num: int) -> int | None:
    try:
        tx_slot = chirp_tx_masks.index(int(tx_bit))
    except ValueError:
        return None
    return int(rx) * int(tx_num) + int(tx_slot)


def build_iwr6843isk_virtual_array(config_path: str | Path, *, rx_num: int = 4, tx_num: int = 3) -> VirtualArrayModel:
    """Build IWR6843ISK virtual array order for this repo's raw cube layout.

    frame_to_radar_cube stores channels as RX-major / TX-slot-fast.  TI
    people-counting geometry is TX-major / RX-fast in TX bit order 1, 2, 4.
    This function maps the current cfg chirp TX masks back to the TI geometry
    order, so changing chirpCfg from 1,4,2 to 1,2,4 updates the raw order
    automatically.
    """
    config_path = Path(config_path)
    chirp_tx_masks = parse_chirp_tx_masks(config_path)
    if len(chirp_tx_masks) < int(tx_num):
        raise ValueError(f"Config {config_path} has too few chirpCfg TX masks: {chirp_tx_masks}")
    chirp_tx_masks = chirp_tx_masks[: int(tx_num)]

    raw_order: list[int] = []
    x_lambda: list[float] = []
    z_lambda: list[float] = []
    labels: list[str] = []

    for tx_bit in ISK_TX_GEOMETRY_ORDER:
        tx_position = ISK_TX_POSITIONS_LAMBDA.get(tx_bit)
        if tx_position is None:
            continue
        tx_x, tx_z, tx_label = tx_position
        for rx in range(int(rx_num)):
            raw_channel = _raw_channel_for_tx_bit(rx, tx_bit, chirp_tx_masks, tx_num)
            if raw_channel is None:
                continue
            raw_order.append(raw_channel)
            x_lambda.append(float(tx_x + ISK_RX_X_LAMBDA[rx]))
            z_lambda.append(float(tx_z))
            labels.append(f"{tx_label}-RX{rx + 1}")

    if len(raw_order) != int(rx_num) * int(tx_num):
        raise ValueError(
            "Could not build a complete IWR6843ISK virtual array from "
            f"chirp TX masks {chirp_tx_masks}."
        )

    return VirtualArrayModel(
        name="iwr6843isk_ti_geometry",
        raw_order=np.asarray(raw_order, dtype=np.int64),
        x_lambda=np.asarray(x_lambda, dtype=np.float64),
        z_lambda=np.asarray(z_lambda, dtype=np.float64),
        labels=tuple(labels),
        source="IWR6843ISK TX/RX geometry in TI people-counting TX bit order 1,2,4",
    )


@lru_cache(maxsize=16)
def cached_iwr6843isk_virtual_array(config_path: str, rx_num: int = 4, tx_num: int = 3) -> VirtualArrayModel:
    return build_iwr6843isk_virtual_array(config_path, rx_num=rx_num, tx_num=tx_num)


def apply_tdm_mimo_doppler_phase_compensation(
    range_doppler_fft: np.ndarray,
    *,
    tx_num: int,
    phase_sign: float = 1.0,
    reference_tx_slot: int = 0,
    slot_time_model: str = "uniform_tx_slot",
) -> np.ndarray:
    """Compensate TDM-MIMO TX-slot Doppler phase before AoA projection.

    frame_to_radar_cube stores virtual channels as RX-major / TX-slot-fast, so
    channel % tx_num gives the TX slot used by that raw virtual channel. The
    input Doppler axis is unshifted, matching DSP.shared_range_doppler_fft.
    """
    cube = np.asarray(range_doppler_fft)
    if cube.ndim != 3:
        raise ValueError("Expected range_doppler_fft with shape [doppler, range, channels].")

    tx_num = int(tx_num)
    if tx_num <= 1 or float(phase_sign) == 0.0:
        return cube

    slot_time_model = str(slot_time_model or "uniform_tx_slot").strip().lower()
    if slot_time_model not in {"uniform_tx_slot", "uniform"}:
        raise ValueError(f"Unsupported TDM-MIMO slot_time_model: {slot_time_model}")

    doppler_count, _, channel_count = cube.shape
    if doppler_count <= 1 or channel_count <= 0 or channel_count % tx_num != 0:
        return cube

    reference_tx_slot = int(reference_tx_slot)
    if reference_tx_slot < 0 or reference_tx_slot >= tx_num:
        reference_tx_slot = 0

    signed_doppler_bins = np.fft.fftfreq(doppler_count) * float(doppler_count)
    channel_tx_slots = np.arange(channel_count, dtype=np.float64) % float(tx_num)
    tx_slot_offsets = (channel_tx_slots - float(reference_tx_slot)) / float(tx_num)
    phase = (
        2.0
        * np.pi
        * signed_doppler_bins[:, np.newaxis]
        * tx_slot_offsets[np.newaxis, :]
        / float(doppler_count)
    )
    correction = np.exp(-1j * float(phase_sign) * phase).astype(np.complex64)
    return cube * correction[:, np.newaxis, :]


@lru_cache(maxsize=16)
def _cached_geometry_steering(
    x_lambda: tuple[float, ...],
    z_lambda: tuple[float, ...],
    azimuth_axis_rad: tuple[float, ...],
    elevation_axis_deg: tuple[float, ...],
    phase_sign: float,
) -> np.ndarray:
    x = np.asarray(x_lambda, dtype=np.float64)
    z = np.asarray(z_lambda, dtype=np.float64)
    az_axis = np.asarray(azimuth_axis_rad, dtype=np.float64)
    el_axis = np.radians(np.asarray(elevation_axis_deg, dtype=np.float64))

    steering = np.empty((az_axis.size, x.size, el_axis.size), dtype=np.complex128)
    cos_el = np.cos(el_axis)
    sin_el = np.sin(el_axis)
    for az_index, azimuth_rad in enumerate(az_axis):
        phase = (x[:, np.newaxis] * np.sin(azimuth_rad) * cos_el[np.newaxis, :]) + (
            z[:, np.newaxis] * sin_el[np.newaxis, :]
        )
        steering[az_index] = np.exp(1j * float(phase_sign) * 2.0 * np.pi * phase)
    return steering


def geometry_range_angle_from_fft(
    range_doppler_fft: np.ndarray,
    *,
    raw_order: np.ndarray,
    x_lambda: np.ndarray,
    z_lambda: np.ndarray,
    azimuth_axis_rad: np.ndarray,
    elevation_axis_deg: np.ndarray,
    phase_sign: float = -1.0,
    channel_coefficients=None,
) -> np.ndarray:
    """Project a range-Doppler-channel cube to range-Doppler-azimuth.

    The output layout matches DSP.range_angle_from_fft(..., mode=1):
    [doppler, flipped-range, azimuth] with non-negative magnitudes.
    Elevation is scanned internally and collapsed with a max operation so the
    existing 2D tracking/detection pipeline can keep consuming a 2D RAI map.
    """
    cube = np.asarray(range_doppler_fft)
    if cube.ndim != 3:
        raise ValueError("Expected range_doppler_fft with shape [doppler, range, channels].")

    order = np.asarray(raw_order, dtype=np.int64)
    if order.size == 0 or int(np.max(order)) >= cube.shape[2]:
        raise ValueError("Virtual array raw_order does not match range_doppler_fft channels.")

    reordered = cube[:, :, order]
    channel_count = reordered.shape[2]
    x = np.asarray(x_lambda, dtype=np.float64)
    z = np.asarray(z_lambda, dtype=np.float64)
    if x.size != channel_count or z.size != channel_count:
        raise ValueError("Virtual array geometry does not match reordered channel count.")

    if channel_coefficients:
        coefficients = np.asarray(channel_coefficients, dtype=np.complex64)
        if coefficients.size != channel_count:
            raise ValueError(
                "Channel calibration coefficient count does not match reordered channel count."
            )
        reordered = reordered * coefficients[np.newaxis, np.newaxis, :]

    steering = _cached_geometry_steering(
        tuple(float(v) for v in x.tolist()),
        tuple(float(v) for v in z.tolist()),
        tuple(float(v) for v in np.asarray(azimuth_axis_rad, dtype=np.float64).tolist()),
        tuple(float(v) for v in np.asarray(elevation_axis_deg, dtype=np.float64).tolist()),
        round(float(phase_sign), 6),
    )
    flat = np.asarray(reordered.reshape((-1, channel_count)), dtype=np.complex64)
    steering_matrix = np.asarray(
        np.conj(steering.transpose(1, 0, 2).reshape(channel_count, -1)),
        dtype=np.complex64,
    )
    projected = flat @ steering_matrix
    rai_abs = np.max(
        np.abs(projected.reshape(flat.shape[0], steering.shape[0], steering.shape[2])),
        axis=2,
    )
    rai_abs = rai_abs.reshape((reordered.shape[0], reordered.shape[1], steering.shape[0]))
    return np.flip(rai_abs, axis=1)


def geometry_range_azimuth_elevation_from_fft(
    range_doppler_fft: np.ndarray,
    *,
    raw_order: np.ndarray,
    x_lambda: np.ndarray,
    z_lambda: np.ndarray,
    azimuth_axis_rad: np.ndarray,
    elevation_axis_deg: np.ndarray,
    phase_sign: float = -1.0,
    channel_coefficients=None,
) -> np.ndarray:
    """Project a range-Doppler-channel cube to range-Doppler-azimuth-elevation.

    This diagnostic helper keeps the elevation dimension instead of collapsing
    it. It is intended for offline point-cloud visualization, not the live
    tracker path.
    """
    cube = np.asarray(range_doppler_fft)
    if cube.ndim != 3:
        raise ValueError("Expected range_doppler_fft with shape [doppler, range, channels].")

    order = np.asarray(raw_order, dtype=np.int64)
    if order.size == 0 or int(np.max(order)) >= cube.shape[2]:
        raise ValueError("Virtual array raw_order does not match range_doppler_fft channels.")

    reordered = cube[:, :, order]
    channel_count = reordered.shape[2]
    x = np.asarray(x_lambda, dtype=np.float64)
    z = np.asarray(z_lambda, dtype=np.float64)
    if x.size != channel_count or z.size != channel_count:
        raise ValueError("Virtual array geometry does not match reordered channel count.")

    if channel_coefficients:
        coefficients = np.asarray(channel_coefficients, dtype=np.complex64)
        if coefficients.size != channel_count:
            raise ValueError(
                "Channel calibration coefficient count does not match reordered channel count."
            )
        reordered = reordered * coefficients[np.newaxis, np.newaxis, :]

    steering = _cached_geometry_steering(
        tuple(float(v) for v in x.tolist()),
        tuple(float(v) for v in z.tolist()),
        tuple(float(v) for v in np.asarray(azimuth_axis_rad, dtype=np.float64).tolist()),
        tuple(float(v) for v in np.asarray(elevation_axis_deg, dtype=np.float64).tolist()),
        round(float(phase_sign), 6),
    )
    flat = np.asarray(reordered.reshape((-1, channel_count)), dtype=np.complex64)
    steering_matrix = np.asarray(
        np.conj(steering.transpose(1, 0, 2).reshape(channel_count, -1)),
        dtype=np.complex64,
    )
    projected = flat @ steering_matrix
    response = np.abs(
        projected.reshape(
            flat.shape[0],
            steering.shape[0],
            steering.shape[2],
        )
    )
    response = response.reshape(
        (
            reordered.shape[0],
            reordered.shape[1],
            steering.shape[0],
            steering.shape[2],
        )
    )
    return np.flip(response, axis=1)


def geometry_azimuth_spectrum(
    target_vectors: np.ndarray,
    *,
    x_lambda: np.ndarray,
    z_lambda: np.ndarray | None,
    azimuth_axis_rad: np.ndarray,
    elevation_axis_deg: np.ndarray | None = None,
    phase_sign: float = -1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a geometry-aware azimuth spectrum by scanning elevation too.

    Returns a 1D azimuth spectrum and the elevation angle selected for each
    azimuth bin.  This is a beamscan diagnostic; it does not apply calibration
    coefficients and should be treated as a geometry sanity check.
    """
    vectors = np.asarray(target_vectors)
    x = np.asarray(x_lambda, dtype=np.float64)
    z = np.zeros_like(x) if z_lambda is None else np.asarray(z_lambda, dtype=np.float64)
    az_axis = np.asarray(azimuth_axis_rad, dtype=np.float64)
    if vectors.ndim != 2:
        raise ValueError("target_vectors must have shape [frames, channels].")
    if vectors.shape[1] != x.size or z.size != x.size:
        raise ValueError("Geometry channel count does not match target vectors.")

    if elevation_axis_deg is None:
        elevation_axis_deg = np.linspace(-40.0, 40.0, 41)
    el_axis = np.radians(np.asarray(elevation_axis_deg, dtype=np.float64))

    spectrum = np.zeros(az_axis.size, dtype=np.float64)
    selected_elevation_deg = np.zeros(az_axis.size, dtype=np.float64)

    for az_index, azimuth_rad in enumerate(az_axis):
        best_power = -np.inf
        best_elevation = 0.0
        sin_az = np.sin(azimuth_rad)
        cos_el = np.cos(el_axis)
        sin_el = np.sin(el_axis)
        for el_index, _elevation_rad in enumerate(el_axis):
            phase = (x * sin_az * cos_el[el_index]) + (z * sin_el[el_index])
            steering = np.exp(1j * float(phase_sign) * 2.0 * np.pi * phase)
            projected = vectors @ np.conj(steering)
            power = float(np.mean(np.square(np.abs(projected))))
            if power > best_power:
                best_power = power
                best_elevation = float(np.degrees(el_axis[el_index]))
        spectrum[az_index] = max(best_power, 0.0)
        selected_elevation_deg[az_index] = best_elevation

    return spectrum, selected_elevation_deg
