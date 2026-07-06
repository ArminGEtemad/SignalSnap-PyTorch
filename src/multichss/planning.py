# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemed, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .configurators import DataConfig, SpectrumConfig
from .results import SpectrumResult, SpectrumResultStore
from .utils import ChannelIndex, FrequencyUnits, S3Calcs, TimeUnits, unit_conversion_time_to_freq


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Resolved calculation settings derived from user configuration.

    :class:`SpectrumConfig` and :class:`DataConfig` describe what the user asked for
    :class:`RuntimeConfig` describes what the calculation will actually use after defaults,
    data-size constraints, frequency axes, and device details have been resolved.

    Attributes
    ----------
    active_channels : tuple[int, ...]
        Data-channel indices used by the calculation.
    spectra_channels : tuple[tuple[int, ...], ...]
        Specifies which (multi-channel) spectra will be calculated. Each tuple represents one auto-
        or cross-correlation spectrum. Each tuple entry is a channel index.
    orders : tuple[int, ...]
        Orders at which spectra are computed.
    dt : float
        Sampling interval shared by all selected data channels.
    window_points : int
        Number of samples per window.
    m : int
        Number of windows used per spectral estimate. This may be reduced at runtime if the signal
        is too short. Must be positive.
    n_data_points : int
        Number of samples in each selected data channel.
    freq_band : np.ndarray
        Selected frequency axis.
    freq_unit : Literal["Hz", "kHz", "MHz", "GHz", "THz"]
        Unit of the frequency axis.
    f_min_idx, f_max_idx : int
        Slice indices selecting the configured frequency band.
    use_full_fft : bool
        Whether negative frequencies require full FFT handling.
    real_dtype : torch.dtype
        Sets the dtype of floats.
    complex_dtype : torch.dtype
        Sets the dtype of complex numbers.
    device : torch.device
        Torch device used for calculation.
    s3_calc : Literal["1/4", "1/2"]
        Method used for third-order spectrum calculation.
    spectral_estimates: int
        Number of unshifted spectral estimates processed by the base calculation. If
        ``interlacing=True``, up to the same number of additional shifted estimates are calculated
        when enough data is available.
    interlacing : bool
        Compute additional spectral estimates for windows shifted by half a window size, to
        compensate the low weight of data points produced by the window function near the original
        window edges.
    old_window : bool
        Compatibility option. If set to ``True``, the approximated confined Gaussian window from the
        old API is used as a window function.
    """

    active_channels: tuple[int, ...]
    spectra_channels: tuple[tuple[ChannelIndex, ...], ...]
    orders: tuple[int, ...]
    dt: float
    window_points: int
    m: int
    n_data_points: int
    freq_band: np.ndarray
    freq_unit: FrequencyUnits
    f_min_idx: int
    f_max_idx: int
    use_full_fft: bool
    real_dtype: torch.dtype
    complex_dtype: torch.dtype
    device: torch.device
    s3_calc: S3Calcs
    spectral_estimates: int
    interlacing: bool
    old_window: bool


def _get_and_validate_selected_channels(
    spectrum_config: SpectrumConfig,
    data_config_list: list[DataConfig],
) -> tuple[tuple[int, ...], int, float, TimeUnits]:
    """Resolve selected data-channel indices and validate the corresponding data."""
    if not data_config_list:
        raise ValueError("At least one DataConfig is required.")

    n_data_configs = len(data_config_list)

    if spectrum_config.spectra_channels is None:
        active_channels = list(range(n_data_configs))
    else:
        active_channels = []
        for channels in spectrum_config.spectra_channels:
            for channel in channels:
                if channel < 0 or channel >= n_data_configs:
                    raise ValueError(
                        "Channel indices must be in the range of valid DataConfig list indices. "
                        f"Channel {channel} out of bounds."
                    )
                if channel not in active_channels:
                    active_channels.append(channel)

    first_config = data_config_list[active_channels[0]]

    for channel in active_channels:
        data_config = data_config_list[channel]
        if data_config.data.shape[0] != first_config.data.shape[0]:
            raise ValueError("Imported data must have same length!")
        if data_config.dt != first_config.dt or data_config.t_unit != first_config.t_unit:
            raise ValueError("Selected data channels must use the same dt and t_unit.")

    return tuple(active_channels), first_config.data.shape[0], first_config.dt, first_config.t_unit


def build_runtime_config(
    spectrum_config: SpectrumConfig,
    data_config_list: list[DataConfig],
) -> RuntimeConfig:
    """Resolve user configuration into immutable runtime calculation settings.

    Validates the selected data channels, derives the frequency axis and frequency-band indices,
    checks Nyquist-frequency bounds, resolves the effective window size, and
    selects torch dtypes and device settings used by the spectrum calculation.

    Parameters
    ----------
    spectrum_config : :class:`SpectrumConfig`
        User configuration for spectrum orders, frequency bounds, precision, device, windowing, and
        related calculation options.
    data_config_list : list[:class:`DataConfig`]
        Data configurations containing the input data and sampling metadata.
    """

    # Validate and read the channels, number of data points, and the time step from the
    # SpectrumConfig and DataConfigs
    active_channels, n_data_points, dt, t_unit = _get_and_validate_selected_channels(
        spectrum_config, data_config_list
    )

    # Validate and resolve the frequency bounds
    f_max_allowed = 1 / (2 * dt)
    f_max = spectrum_config.f_max

    if f_max is None:
        f_max = f_max_allowed

        if f_max <= spectrum_config.f_min:
            raise ValueError("f_min is larger than the Nyquist frequency.")

    if f_max > f_max_allowed:
        raise ValueError("f_max is larger than the Nyquist frequency.")

    if spectrum_config.f_min < -f_max_allowed:
        raise ValueError("f_min outside of Nyquist frequency bounds.")

    # Compute how many points must be taken into account in one window to achieve the required
    # frequency spacing in the given frequency bounds
    window_T = (spectrum_config.frequency_points - 1) / (f_max - spectrum_config.f_min)
    window_points = int(np.round(window_T / dt))
    if window_points <= 0:
        raise ValueError("Calculated window_points must be greater than zero.")

    # Resolve spectra_channels, default to all auto-correlation spectra of order 1-4 if 
    # spectra_channels is not specified
    if spectrum_config.spectra_channels is None:
        spectra_channels = []
        for channel in active_channels:
            for order in range(1, 5):
                spectra_channels.append((channel,) * order)
    else:
        spectra_channels = spectrum_config.spectra_channels 

    # Check if enough data is available and try to lower the window count per cumulant/spectrum
    # estimate if needed
    required_points = window_points * spectrum_config.m
    if required_points > n_data_points:
        m = n_data_points // window_points
        print(
            "Values have been changed, because not enough data points were available."
            f"Old m: {spectrum_config.m}, new m: {m}"
        )
    else:
        m = spectrum_config.m

    orders = tuple(sorted({len(channels) for channels in spectra_channels}))
    if m < max(orders):
        raise ValueError("Not enough data points")

    # get the frequency axis
    use_full_fft = spectrum_config.f_min < 0
    if use_full_fft:
        freq_all = np.fft.fftfreq(window_points, dt)
        freq_all = np.fft.fftshift(freq_all)
    else:
        freq_all = np.fft.rfftfreq(window_points, dt)

    f_max_idx = int(np.sum(freq_all <= f_max))
    f_min_idx = int(np.sum(freq_all < spectrum_config.f_min))

    # determine the data types based on the given precision
    if spectrum_config.precision == "single":
        real_dtype = torch.float32
        complex_dtype = torch.complex64
    elif spectrum_config.precision == "double":
        real_dtype = torch.float64
        complex_dtype = torch.complex128
    else:
        if spectrum_config.device == "mps":
            real_dtype = torch.float32
            complex_dtype = torch.complex64
        else:
            real_dtype = torch.float64
            complex_dtype = torch.complex128

    # Determine the number of spectral estimates
    chunk_size = m * window_points
    unshifted_estimates = n_data_points // chunk_size

    if spectrum_config.spectral_estimates_max is None:
        spectral_estimates = unshifted_estimates
    else:
        spectral_estimates = min(spectrum_config.spectral_estimates_max, unshifted_estimates)

    # raise ValueError, if not a single shifted spectral estimate can be calculated when interlacing
    # is enabled.
    if spectrum_config.interlacing:
        shifted_estimates = (n_data_points - window_points // 2) // chunk_size
        if shifted_estimates < 1:
            raise ValueError(
                "Interlacing was requested, but the data is too short for a shifted spectral "
                "estimate. Disable interlacing or provide more data."
            )

    return RuntimeConfig(
        active_channels=active_channels,
        spectra_channels=tuple(spectra_channels),
        orders=orders,
        dt=dt,
        window_points=window_points,
        m=m,
        n_data_points=n_data_points,
        freq_band=freq_all[f_min_idx:f_max_idx],
        freq_unit=unit_conversion_time_to_freq(t_unit),
        f_min_idx=f_min_idx,
        f_max_idx=f_max_idx,
        use_full_fft=use_full_fft,
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,
        device=torch.device(spectrum_config.device),
        s3_calc=spectrum_config.s3_calc,
        spectral_estimates=spectral_estimates,
        interlacing=spectrum_config.interlacing,
        old_window=spectrum_config.old_window,
    )


def initialize_result_store(runtime: RuntimeConfig) -> SpectrumResultStore:
    """Create an initialized result store for a list of spectrum tasks.

    Each task is converted into a :class:`SpectrumResult` with matching channels.

    Parameters
    ----------
    runtime : :class:`RuntimeConfig`
        :class:`RuntimeConfig` that contains all necessary information to initialize result arrays.

    Returns
    -------
    SpectrumResultStore
        Store containing one initialized :class:`SpectrumResult` per task.
    """

    store = SpectrumResultStore()
    for channels in runtime.spectra_channels:
        store.add(SpectrumResult(channels))
    store.initialize_arrays(runtime)
    return store
