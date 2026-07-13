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
    spectrum_config: SpectrumConfig, data_config_list: list[DataConfig]
) -> SpectrumResultStore:
    """Calculate requested auto- and cross-polyspectra for one or more data channels.

    Builds the runtime configuration, expands the requested spectrum tasks, iterates over windowed
    signal chunks, computes Fourier coefficients, accumulates spectra, and finalizes mean spectra
    and standard-error estimates.

    Parameters
    ----------
    spectrum_config : :class:`SpectrumConfig`
        Frequency, order, windowing, precision, and device options.
    data_config_list : list[:class:`DataConfig`]
        Input signal channels and sampling metadata.

    Returns
    -------
    SpectrumResultStore
        Finalized spectra indexed by ``channels``.
    """
    runtime = _planning.build_runtime_config(
        spectrum_config=spectrum_config, data_config_list=data_config_list
    )
    window_buffer = _fft.prepare_window(runtime)
    third_order_cache = _spectra.build_third_order_cache(runtime) if 3 in runtime.orders else None
    accumulator_store = _accumulation.initialize_accumulator_store(runtime)

    failed_spectra: set[tuple[int, ...]] = set()

    for start, end, shifted in _planning.iter_window_slices(runtime):
        coeffs_by_channel = {}

        for channel in runtime.active_channels:
            data = data_config_list[channel].data[start:end]
            chunk = _fft.reshape_window_chunk(data, runtime)
            chunk = _fft.to_device(chunk, runtime)
            coeffs_by_channel[channel] = _fft.compute_fft(chunk, window_buffer.window, runtime)

        intermediate_buffer = _spectra.build_intermediate_slice_buffer(
            runtime, coeffs_by_channel, third_order_cache
        )

        for channels in runtime.spectra_channels:
            if channels in failed_spectra:
                continue

            accumulator = accumulator_store.get(channels)

            try:
                spectrum = _spectra.compute_single_spectrum(
                    channels, intermediate_buffer, window_buffer, runtime
                )
                _accumulation.accumulate_spectrum(accumulator, spectrum, shifted)
            except Exception as exc:
                failed_spectra.add(channels)
                warnings.warn(
                    f"Calculation failed for spectrum {channels}: {type(exc).__name__}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

    result_store = SpectrumResultStore()
    for accumulator in accumulator_store:
        if accumulator.channels in failed_spectra:
            continue

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
