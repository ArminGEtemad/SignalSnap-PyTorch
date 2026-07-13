# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemed, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

import warnings
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import torch

from ..configurators import DataConfig, SpectrumConfig
from .utils import ChannelIndex, FrequencyUnits, TimeUnits, unit_conversion_time_to_freq


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
    freq_all : np.ndarray
        Full frequency axis.
    fft_freq_count : int
        Number of frequencies in ``freq_all``. Is equal to ``window_points``.
    freq_band : np.ndarray
        Selected frequency axis.
    band_start_idx, band_end_idx : int
        Slice indices selecting the configured frequency band.
    freq_unit : Literal["Hz", "kHz", "MHz", "GHz", "THz"]
        Unit of the frequency axis.
    real_dtype : torch.dtype
        Sets the dtype of floats.
    complex_dtype : torch.dtype
        Sets the dtype of complex numbers.
    device : torch.device
        Torch device used for calculation.
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
    freq_all: np.ndarray
    fft_freq_count: int
    freq_band: np.ndarray
    band_start_idx: int
    band_end_idx: int
    freq_unit: FrequencyUnits
    real_dtype: torch.dtype
    complex_dtype: torch.dtype
    device: torch.device
    spectral_estimates: int
    interlacing: bool
    old_window: bool


def _resolve_requested_spectra(
    requested_spectra: list[tuple[int, ...]] | None,
    channel_count: int,
) -> tuple[tuple[int, ...], ...]:
    """Validate and normalize the requested spectra.

    If requested_spectra is None, generate auto-correlation spectra of
    orders one through four for every available data channel.
    """
    if channel_count < 1:
        raise ValueError("At least one DataConfig is required.")

    if requested_spectra is None:
        return tuple(
            (channel,) * order for channel in range(channel_count) for order in range(1, 5)
        )

    if not requested_spectra:
        raise ValueError("requested_spectra must contain at least one spectrum.")

    resolved: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()

    for spectrum in requested_spectra:
        if not isinstance(spectrum, tuple):
            raise TypeError("Each spectrum request must be a tuple of channel indices.")

        order = len(spectrum)
        if not 1 <= order <= 4:
            raise ValueError(f"Spectrum order must be between 1 and 4; received order {order}.")

        resolved_channels: list[int] = []

        for channel in spectrum:
            # bool is technically an int in Python, but should not be accepted as a channel index.
            if isinstance(channel, (bool, np.bool_)):
                raise TypeError(f"Channel indices must be integers; received {channel!r}.")

            if not isinstance(channel, (int, np.integer)):
                raise TypeError(f"Channel indices must be integers; received {channel!r}.")

            if channel < 0:
                raise ValueError(f"Channel indices must be nonnegative; received {channel}.")

            if channel >= channel_count:
                raise ValueError(
                    f"Channel {channel} is out of bounds for "
                    f"{channel_count} available data channels."
                )

            resolved_channels.append(int(channel))

        resolved_spectrum = tuple(resolved_channels)

        if resolved_spectrum in seen:
            raise ValueError(
                f"requested_spectra cannot contain duplicates; "
                f"{resolved_spectrum} was requested more than once."
            )

        seen.add(resolved_spectrum)
        resolved.append(resolved_spectrum)

    return tuple(resolved)


def _get_and_validate_selected_channels(
    data_config_list: list[DataConfig],
    spectra_channels: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], int, float, TimeUnits]:
    """Resolve selected data-channel indices and validate the corresponding data."""

    active_channels: list[int] = []

    for channels in spectra_channels:
        for channel in channels:
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
    data_config_list: list[DataConfig],
    spectrum_config: SpectrumConfig,
    requested_spectra: list[tuple[int, ...]] | None,
) -> RuntimeConfig:
    """Resolve user configuration into immutable runtime calculation settings.

    Validates the selected data channels, derives the frequency axis and frequency-band indices,
    checks Nyquist-frequency bounds, resolves the effective window size, and
    selects torch dtypes and device settings used by the spectrum calculation.

    Parameters
    ----------
    data_config_list : list[:class:`DataConfig`]
        Data configurations containing the input data and sampling metadata.
    spectrum_config : :class:`SpectrumConfig`
        User configuration for frequency bounds, precision, device, windowing, and
        related calculation options.
    requested_spectra : list[tuple[int, ...]] | None
        Specifies which (multi-channel) spectra will be calculated. Each tuple represents one auto-
        or cross-correlation spectrum. Each tuple entry is a channel index which matches the index
        in ``data_config_list``. If ``None``, the auto-correlation spectra of orders 1 to 4 will be
        calculated for all available data channels.
    """

    # Validate and read the channels, number of data points, and the time step from the
    # SpectrumConfig and DataConfigs
    spectra_channels = _resolve_requested_spectra(
        requested_spectra,
        channel_count=len(data_config_list),
    )
    active_channels, n_data_points, dt, t_unit = _get_and_validate_selected_channels(
        data_config_list, spectra_channels
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

    # Check if enough data is available and try to lower the window count per cumulant/spectrum
    # estimate if needed
    required_points = window_points * spectrum_config.m
    if required_points > n_data_points:
        m = n_data_points // window_points
        warnings.warn(
            f"Not enough data points are available for m={spectrum_config.m}; using m={m} instead.",
            UserWarning,
            stacklevel=3,
        )
    else:
        m = spectrum_config.m

    orders = tuple(sorted({len(channels) for channels in spectra_channels}))
    if m < max(orders):
        raise ValueError("Not enough data points")

    # get the frequency axis
    freq_all = np.fft.fftfreq(window_points, dt)
    freq_all = np.fft.fftshift(freq_all)

    band_end_idx = int(np.sum(freq_all <= f_max))
    band_start_idx = int(np.sum(freq_all < spectrum_config.f_min))

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
        freq_all=freq_all,
        fft_freq_count=window_points,
        freq_band=freq_all[band_start_idx:band_end_idx],
        band_start_idx=band_start_idx,
        band_end_idx=band_end_idx,
        freq_unit=unit_conversion_time_to_freq(t_unit),
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,
        device=torch.device(spectrum_config.device),
        spectral_estimates=spectral_estimates,
        interlacing=spectrum_config.interlacing,
        old_window=spectrum_config.old_window,
    )


def iter_window_slices(runtime: RuntimeConfig) -> Iterator[tuple[int, int, bool]]:
    """Return the window slice indices.

    Each yielded ``(start, end, shifted)`` selects ``m * N`` samples from a one-dimensional data
    channel, where ``m = runtime.m`` and ``N = runtime.window_points``. With interlacing enabled,
    additional slices shifted by ``N // 2`` are yielded when they still fit inside the signal.
    """

    chunk_size = runtime.window_points * runtime.m

    for chunk_index in range(runtime.spectral_estimates):
        start = chunk_index * chunk_size
        end = start + chunk_size
        yield start, end, False

    if runtime.interlacing:
        shift = runtime.window_points // 2
        n_chunks_shifted = max(
            0, (runtime.n_data_points - runtime.window_points // 2) // chunk_size
        )
        shifted_estimates = min(runtime.spectral_estimates, n_chunks_shifted)
        for chunk_index in range(shifted_estimates):
            start = chunk_index * chunk_size + shift
            end = start + chunk_size
            yield start, end, True
