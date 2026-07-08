# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemed, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

from .configurators import PlotConfig
from .results import SpectrumResult, SpectrumResultStore


def select_plot_results(
    plot_config: PlotConfig, result_store: SpectrumResultStore
) -> list[SpectrumResult]:
    if plot_config.spectra_channels is None:
        return list(result_store.results.values())

    results = []
    for channels in plot_config.spectra_channels:
        try:
            results.append(result_store.get(channels))
        except KeyError as exc:
            raise ValueError(
                f"Requested plot for channels {channels}, but no matching spectrum result exists."
            ) from exc

    return results


def split_results_by_order(results: list[SpectrumResult]) -> dict[int, list[SpectrumResult]]:
    grouped = {1: [], 2: [], 3: [], 4: []}
    for result in results:
        grouped[result.order].append(result)
    return grouped


def component_data(data: np.ndarray, component: str) -> np.ndarray:
    if component == "re":
        return np.real(data)
    if component == "im":
        return np.imag(data)
    raise ValueError(f"Unsupported plot component: {component}")


def component_label(component: str) -> str:
    return "Real" if component == "re" else "Imaginary"


def channels_label(channels: tuple[int, ...]) -> str:
    return "_".join(str(channel) for channel in channels)


def custom_colormap():
    colors = np.array([
        (23, 51, 107),
        (82, 137, 190),
        (165, 203, 230),
        (235, 235, 235),
        (235, 164, 120),
        (188, 84, 68),
        (107, 22, 38),
    ]) / 255.0

    return mcolors.LinearSegmentedColormap.from_list("multichss_spectrum", colors)


def custom_error_colormap(insignif_transparency: float):
    return LinearSegmentedColormap.from_list(
        "multichss_insignificant",
        [
            (0.0, 0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0, insignif_transparency),
        ],
    )


def save_or_show(fig: Figure, plot_config: PlotConfig, filename: str) -> None:
    if plot_config.output == "show":
        plt.show()
    elif plot_config.output == "save":
        fig.savefig(plot_config.output_folder / filename, bbox_inches="tight")
        plt.close(fig)
    else:
        raise ValueError(f"Invalid plot output: {plot_config.output}")
    

def format_order_1_rows(rows: list[dict[str, object]]) -> str:
    headers = ["Channels", "Real", "Imag", "Error real", "Error imag"]

    table = [
        [str(row.get(header, "")) for header in headers]
        for row in rows
    ]

    widths = [
        max(len(header), *(len(row[col]) for row in table))
        for col, header in enumerate(headers)
    ]

    header_line = "  ".join(
        header.ljust(widths[col])
        for col, header in enumerate(headers)
    )

    separator = "  ".join("-" * width for width in widths)

    body = [
        "  ".join(value.ljust(widths[col]) for col, value in enumerate(row))
        for row in table
    ]

    return "\n".join([header_line, separator, *body])


def plot_order_1(results: list[SpectrumResult], plot_config: PlotConfig):
    rows = []

    for result in results:
        if result.spectrum is None:
            continue

        value = result.spectrum[0]
        error = result.spectrum_error[0] if result.spectrum_error is not None else None

        rows.append({
            "Channels": result.channels,
            "Real": value.real,
            "Imag": value.imag,
            "Error real": None if error is None else error.real,
            "Error imag": None if error is None else error.imag,
        })

    if not rows:
        print("No order 1 results available.")
        return []

    if plot_config.output == "save":
        path = plot_config.output_folder / "s1_table.txt"
        path.write_text(format_order_1_rows(rows), encoding="utf-8")

    print(format_order_1_rows(rows))


def plot_order_2(results: list[SpectrumResult], plot_config: PlotConfig) -> list[Figure]:
    figures = []

    for result in results:
        if result.spectrum is None:
            continue

        if result.freq is None:
            raise RuntimeError(
                f"Spectrum result for channels {result.channels} has no frequency axis."
            )

        fig, axes = plt.subplots(
            len(plot_config.plot_format),
            1,
            figsize=(8, 3 * len(plot_config.plot_format)),
            squeeze=False,
            sharex=True,
        )

        for row, component in enumerate(plot_config.plot_format):
            ax = axes[row][0]
            y = component_data(result.spectrum, component)

            ax.plot(result.freq, y, label=f"S{result.order} {component_label(component)}")

            if result.spectrum_error is not None:
                err = np.abs(component_data(result.spectrum_error, component))
                for i in range(plot_config.significance):
                    width = (i + 1) * err
                    ax.fill_between(result.freq, y - width, y + width, alpha=0.15)

            ax.set_xlim(plot_config.f_min, plot_config.f_max)
            ax.set_ylabel(component_label(component))
            ax.set_title(f"S2 channels {result.channels}")
            ax.legend()

        axes[-1][0].set_xlabel(f"Frequency / {result.freq_unit}")
        fig.tight_layout()

        filename = f"s2_channels_{channels_label(result.channels)}.png"
        save_or_show(fig, plot_config, filename)
        figures.append(fig)

    return figures


def plot_order_3_or_4(
    results: list[SpectrumResult],
    plot_config: PlotConfig,
    order: int,
) -> list[Figure]:
    figures = []
    cmap = custom_colormap()
    error_cmap = custom_error_colormap(plot_config.insignif_transparency)

    for result in results:
        if result.spectrum is None:
            continue

        if result.freq is None:
            raise RuntimeError(
                f"Spectrum result for channels {result.channels} has no frequency axis."
            )

        fig, axes = plt.subplots(
            1,
            len(plot_config.plot_format),
            figsize=(6 * len(plot_config.plot_format), 5),
            squeeze=False,
        )

        x, y = np.meshgrid(result.freq, result.freq)

        for col, component in enumerate(plot_config.plot_format):
            ax = axes[0][col]
            z = component_data(result.spectrum, component)
            z = np.ma.masked_invalid(z)

            limit = np.nanmax(np.abs(z))
            mesh = ax.pcolormesh(
                x,
                y,
                z,
                cmap=cmap,
                vmin=-limit,
                vmax=limit,
                shading="auto",
            )

            if result.spectrum_error is not None:
                err = np.abs(component_data(result.spectrum_error, component))
                insignificant = np.abs(z) < plot_config.significance * err
                ax.pcolormesh(
                    x,
                    y,
                    insignificant.astype(float),
                    cmap=error_cmap,
                    vmin=0,
                    vmax=1,
                    shading="auto",
                )

            ax.set_xlim(plot_config.f_min, plot_config.f_max)
            ax.set_ylim(plot_config.f_min, plot_config.f_max)
            ax.set_xlabel(f"Frequency / {result.freq_unit}")
            ax.set_ylabel(f"Frequency / {result.freq_unit}")
            ax.set_title(f"S{order} {component_label(component)} channels {result.channels}")
            fig.colorbar(mesh, ax=ax)

        fig.tight_layout()

        filename = f"s{order}_channels_{channels_label(result.channels)}.png"
        save_or_show(fig, plot_config, filename)
        figures.append(fig)

    return figures