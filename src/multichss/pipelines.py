# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemed, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

from .aggregator import accumulate_spectrum, finalize_result
from .configurators import DataConfig, SpectrumConfig
from .fft import compute_fft, iter_window_slices, prepare_windows, reshape_window_chunk, to_device
from .planning import build_runtime_config, initialize_result_store
from .spectra import build_third_order_cache, compute_single_spectrum


def calculate_spectra(
    spectrum_config: SpectrumConfig,
    data_config_list: list[DataConfig],
    selected: list[int] | None = None,
):
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
    selected : list[int] | None, optional
        Channel indices to include. If ``None``, all channels are used.

    Returns
    -------
    SpectrumResultStore
        Finalized spectra indexed by ``channels``.
    """
    runtime = build_runtime_config(
        spectrum_config=spectrum_config, data_config_list=data_config_list, selected=selected
    )
    result_store = initialize_result_store(runtime)
    single_window, repeated_window = prepare_windows(runtime)

    third_order_cache = (
        build_third_order_cache(runtime)
        if any(len(channels) == 3 for channels in runtime.spectra)
        else None
    )

    for start, end, shifted in iter_window_slices(runtime):
        coeffs_by_channel = {}

        for channel in runtime.selected_channels:
            data = data_config_list[channel].data[start:end]
            chunk = reshape_window_chunk(data, runtime)
            chunk = to_device(chunk, runtime)
            coeffs_by_channel[channel] = compute_fft(chunk, repeated_window, runtime)

        for channels in runtime.spectra:
            spectrum = compute_single_spectrum(
                channels=channels,
                coeffs_by_channel=coeffs_by_channel,
                single_window=single_window,
                runtime=runtime,
                third_order_cache=third_order_cache,
            )
            result = result_store.get(channels)
            accumulate_spectrum(result, spectrum, shifted=shifted)

    for result in result_store.results.values():
        finalize_result(result)

    return result_store
