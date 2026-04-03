# version: 1.0

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=8)
def _cached_range_doppler_window(chirp_count, sample_count, dtype_name):
    dtype = np.dtype(dtype_name)
    chirp_window = np.hanning(chirp_count).astype(dtype, copy=False)
    sample_window = np.hanning(sample_count).astype(dtype, copy=False)
    return chirp_window[:, np.newaxis] * sample_window[np.newaxis, :]


def _range_doppler_window(chirp_count, sample_count, dtype):
    return _cached_range_doppler_window(
        chirp_count,
        sample_count,
        np.dtype(dtype).str,
    )


def shared_range_doppler_fft(data, padding_size=None):
    data = np.asarray(data)
    if data.ndim != 3:
        raise ValueError("Expected data with shape [chirps, samples, channels].")

    window = _range_doppler_window(
        data.shape[0],
        data.shape[1],
        data.real.dtype,
    )
    weighted = data * window[:, :, np.newaxis]
    return np.fft.fft2(weighted, s=padding_size, axes=[0, 1])


def range_doppler_from_fft(range_doppler_fft, mode=0):
    if mode == 0:
        return range_doppler_fft

    if mode == 1:
        rdi_abs = np.transpose(
            np.fft.fftshift(np.abs(range_doppler_fft), axes=0),
            [1, 0, 2],
        )
        rdi_abs = np.flip(rdi_abs, axis=0)
        return rdi_abs

    if mode == 2:
        rdi_abs = np.transpose(
            np.fft.fftshift(np.abs(range_doppler_fft), axes=0),
            [1, 0, 2],
        )
        rdi_abs = np.flip(rdi_abs, axis=0)
        return [range_doppler_fft, rdi_abs]

    print("Error mode")
    raise ValueError


def range_angle_from_fft(range_doppler_fft, mode=0, angle_fft_size=None):
    if angle_fft_size is None:
        angle_fft_size = range_doppler_fft.shape[2]

    if mode == 0:
        rai_raw = np.fft.fft(range_doppler_fft, n=angle_fft_size, axis=2)
        return rai_raw

    if mode == 1:
        rai_abs = np.fft.fft(range_doppler_fft, n=angle_fft_size, axis=2)
        rai_abs = np.fft.fftshift(np.abs(rai_abs), axes=2)
        rai_abs = np.flip(rai_abs, axis=1)
        return rai_abs

    if mode == 2:
        rai_raw = np.fft.fft(range_doppler_fft, n=angle_fft_size, axis=2)
        rai_abs = np.fft.fftshift(np.abs(rai_raw), axes=2)
        rai_abs = np.flip(rai_abs, axis=1)
        return [rai_raw, rai_abs]

    print("Error mode")
    raise ValueError


def Range_Doppler(data, mode=0, padding_size=None):
    """
    :param data: array_like
                    Input array with the shape [chirps, samples, channels]

    :param mode: int, optional
                    Mode of the output Range Doppler Image format, default is 0
                    0: return RDI in raw mode
                    1: return RDI in abs mode with fft shift and flip
                    2: return both mode 0 and 1

    :param padding_size: sequence of ints, optional
                    Shape(length after the transformed), s[0] refers to axis 0, s[1] to axis 1

    :return:complex array
                    Output RDI depends on mode, return a range doppler cube
    """
    range_doppler_fft = shared_range_doppler_fft(data, padding_size=padding_size)
    return range_doppler_from_fft(range_doppler_fft, mode=mode)


def Range_Angle(data, mode=0, padding_size=None):
    """
    :param data: array_like
                    Input array with the shape [chirps, samples, channels]

    :param mode: int, optional
                    Mode of the output Range Doppler Image format, default is 0
                    0: return RAI in raw mode
                    1: return RAI in abs mode with fft shift and flip
                    2: return both mode 0 and 1

    :param padding_size: sequence of ints, optional
                    Shape(length after the transformed), s[0] refers to axis 0, s[1] to axis 1, etc

    :return: complex array
                    Output RAI depends on mode, return a range angle cube
    """
    if padding_size is None:
        padding_size = data.shape
    range_doppler_fft = shared_range_doppler_fft(
        data,
        padding_size=[padding_size[0], padding_size[1]],
    )
    return range_angle_from_fft(
        range_doppler_fft,
        mode=mode,
        angle_fft_size=padding_size[2],
    )
