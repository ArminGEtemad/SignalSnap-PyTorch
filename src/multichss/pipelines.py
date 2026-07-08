# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemed, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from .aggregator import accumulate_spectrum, finalize_result
from .configurators import DataConfig, PlotConfig, SpectrumConfig
from .fft import compute_fft, iter_window_slices, prepare_window, reshape_window_chunk, to_device
from .planning import build_runtime_config, initialize_result_store
from .plotting import (
    plot_order_1,
    plot_order_2,
    plot_order_3_or_4,
    select_plot_results,
    split_results_by_order,
)
from .results import SpectrumResultStore
from .spectra import (
    build_intermediate_slice_buffer,
    build_third_order_cache,
    compute_single_spectrum,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def calculate_spectra(spectrum_config: SpectrumConfig, data_config_list: list[DataConfig]):
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
    runtime = build_runtime_config(
        spectrum_config=spectrum_config, data_config_list=data_config_list
    )
    result_store = initialize_result_store(runtime)
    window_buffer = prepare_window(runtime)

    third_order_cache = build_third_order_cache(runtime) if 3 in runtime.orders else None

    for start, end, shifted in iter_window_slices(runtime):
        coeffs_by_channel = {}

        for channel in runtime.active_channels:
            data = data_config_list[channel].data[start:end]
            chunk = reshape_window_chunk(data, runtime)
            chunk = to_device(chunk, runtime)
            coeffs_by_channel[channel] = compute_fft(chunk, window_buffer.window, runtime)

        intermediate_buffer = build_intermediate_slice_buffer(
            runtime, coeffs_by_channel, third_order_cache
        )

        for channels in runtime.spectra_channels:
            spectrum = compute_single_spectrum(
                channels=channels,
                intermediate_buffer=intermediate_buffer,
                window_buffer=window_buffer,
                runtime=runtime,
            )
            result = result_store.get(channels)
            accumulate_spectrum(result, spectrum, shifted=shifted)

    for result in result_store.results.values():
        finalize_result(result)

    return result_store


def plot_spectra(plot_config: PlotConfig, result_store: SpectrumResultStore)-> list[Figure]:
    """Plot calculated polyspectra from a SpectrumResultStore."""

    results = select_plot_results(plot_config, result_store)
    grouped_results = split_results_by_order(results)

    figures = []

    if grouped_results[1]:
        plot_order_1(grouped_results[1], plot_config)

    if grouped_results[2]:
        figures.extend(plot_order_2(grouped_results[2], plot_config))

    if grouped_results[3]:
        figures.extend(plot_order_3_or_4(grouped_results[3], plot_config, order=3))

    if grouped_results[4]:
        figures.extend(plot_order_3_or_4(grouped_results[4], plot_config, order=4))

    return figures
