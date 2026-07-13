# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemed, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

import warnings

from ._core import accumulation as _accumulation
from ._core import fft as _fft
from ._core import planning as _planning
from ._core import spectra as _spectra
from .configurators import DataConfig, SpectrumConfig
from .results import SpectrumResultStore

__all__ = ["calculate_spectra"]


def calculate_spectra(
    data_config_list: list[DataConfig],
    spectrum_config: SpectrumConfig,
    *,
    requested_spectra: list[tuple[int, ...]] | None = None,
) -> SpectrumResultStore:
    """Calculate requested auto- and cross-polyspectra for one or more data channels.

    Builds the runtime configuration, expands the requested spectrum tasks, iterates over windowed
    signal chunks, computes Fourier coefficients, accumulates spectra, and finalizes mean spectra
    and standard-error estimates.

    Parameters
    ----------
    data_config_list : list[:class:`DataConfig`]
        Input signal channels and sampling metadata.
    spectrum_config : :class:`SpectrumConfig`
        Frequency, windowing, precision, and device options.
    requested_spectra : list[tuple[int, ...]] | None
        Specifies which (multi-channel) spectra will be calculated. Each tuple represents one auto-
        or cross-correlation spectrum. Each tuple entry is a channel index which matches the index
        in ``data_config_list``. If ``None``, the auto-correlation spectra of orders 1 to 4 will be
        calculated for all available data channels.

    Returns
    -------
    SpectrumResultStore
        Finalized spectra indexed by ``channels``.
    """

    # Resolve user inputs and initialize reusable calculation state.
    runtime = _planning.build_runtime_config(
        data_config_list=data_config_list,
        spectrum_config=spectrum_config,
        requested_spectra=requested_spectra,
    )
    window_buffer = _fft.prepare_window(runtime)
    third_order_cache = _spectra.build_third_order_cache(runtime) if 3 in runtime.orders else None
    accumulator_store = _accumulation.initialize_accumulator_store(runtime)

    failed_spectra: set[tuple[int, ...]] = set()

    # Each data slice contains runtime.m windows and produces one spectral estimate for every
    # requested spectrum.
    for start, end, shifted in _planning.iter_window_slices(runtime):
        coeffs_by_channel = {}

        # Compute Fourier coefficients for each active channel.
        for channel in runtime.active_channels:
            data = data_config_list[channel].data[start:end]
            chunk = _fft.reshape_window_chunk(data, runtime)
            chunk = _fft.to_device(chunk, runtime)
            coeffs_by_channel[channel] = _fft.compute_fft(
                chunk=chunk,
                window=window_buffer.window,
                runtime=runtime,
            )

        intermediate_buffer = _spectra.build_intermediate_slice_buffer(
            runtime=runtime,
            coeffs_by_channel=coeffs_by_channel,
            third_order_cache=third_order_cache,
        )

        # Compute and accumulate every requested spectrum for this data slice.
        for channels in runtime.spectra_channels:
            if channels in failed_spectra:
                continue

            accumulator = accumulator_store.get(channels)

            # Isolate calculation failures to the affected spectrum so the remaining spectrum
            # requests can continue.
            try:
                spectrum = _spectra.compute_single_spectrum(
                    channels=channels,
                    intermediate_buffer=intermediate_buffer,
                    window_buffer=window_buffer,
                    runtime=runtime,
                )
                _accumulation.accumulate_spectrum(
                    accumulator=accumulator,
                    single_spectrum=spectrum,
                    shifted=shifted,
                )
            except Exception as exc:
                failed_spectra.add(channels)
                warnings.warn(
                    f"Calculation failed for spectrum {channels}: {type(exc).__name__}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

    # Finalize accumulated spectra and their error estimates.
    result_store = SpectrumResultStore()
    for accumulator in accumulator_store:
        if accumulator.channels in failed_spectra:
            continue

        # Isolate finalization failures so other completed spectra can still be returned.
        try:
            result = _accumulation.finalize_result(accumulator)
        except Exception as exc:
            warnings.warn(
                f"Could not finalize spectrum for channels {accumulator.channels}: "
                f"{type(exc).__name__}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        result_store.add(result)

    return result_store
