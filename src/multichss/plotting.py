# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemed, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from .configurators import PlotStyle
from .results import SpectrumResult, SpectrumResultStore


@dataclass(frozen=True)
class SpectrumFigure:
    figure: Figure
    order: int
    channels: tuple[int, ...]

    def filename(self, extension: str = "png") -> str:
        channel_label = "_".join(map(str, self.channels))
        return f"s{self.order}_channels_{channel_label}.{extension}"


def component_data(data: np.ndarray, component: str) -> np.ndarray:
    if component == "re":
        return np.real(data)
    if component == "im":
        return np.imag(data)
    raise ValueError(f"Unsupported plot component: {component}")


def component_label(component: str) -> str:
    return "Real" if component == "re" else "Imaginary"


def custom_colormap():
    colors = (
        np.array(
            [
                (23, 51, 107),
                (82, 137, 190),
                (165, 203, 230),
                (235, 235, 235),
                (235, 164, 120),
                (188, 84, 68),
                (107, 22, 38),
            ]
        )
        / 255.0
    )

    return mcolors.LinearSegmentedColormap.from_list("multichss_spectrum", colors)


def custom_error_colormap(insignif_transparency: float):
    return LinearSegmentedColormap.from_list(
        "multichss_insignificant",
        [
            (0.0, 0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0, insignif_transparency),
        ],
    )


def format_order_1_rows(rows: list[dict[str, object]]) -> str:
    headers = ["Channels", "Real", "Imag", "Error real", "Error imag"]

    table = [[str(row.get(header, "")) for header in headers] for row in rows]

    widths = [
        max(len(header), *(len(row[col]) for row in table)) for col, header in enumerate(headers)
    ]

    header_line = "  ".join(header.ljust(widths[col]) for col, header in enumerate(headers))

    separator = "  ".join("-" * width for width in widths)

    body = ["  ".join(value.ljust(widths[col]) for col, value in enumerate(row)) for row in table]

    return "\n".join([header_line, separator, *body])


def build_order_1_table(result_store: SpectrumResultStore) -> str:
    order_1_results = [result for result in result_store if result.order == 1]

    if not order_1_results:
        raise RuntimeError("No matching results at order 1.")

    rows = []
    for result in order_1_results:
        if result.spectrum is None:
            raise RuntimeError(f"Spectrum result for channel {result.channels} is not available.")

        value = result.spectrum[0]
        error = result.spectrum_error[0] if result.spectrum_error is not None else None

        rows.append(
            {
                "Channels": result.channels,
                "Real": value.real,
                "Imag": value.imag,
                "Error real": None if error is None else error.real,
                "Error imag": None if error is None else error.imag,
            }
        )
    return format_order_1_rows(rows)


def create_order_2_figure(result: SpectrumResult, plot_style: PlotStyle) -> Figure:
    if result.spectrum is None:
        raise RuntimeError(f"Spectrum result for channels {result.channels} is not available.")

    if result.freq is None:
        raise RuntimeError(f"Spectrum result for channels {result.channels} has no frequency axis.")

    fig, axes = plt.subplots(
        len(plot_style.plot_format),
        1,
        figsize=(8, 3 * len(plot_style.plot_format)),
        squeeze=False,
        sharex=True,
    )

    for row, component in enumerate(plot_style.plot_format):
        ax = axes[row][0]
        y = component_data(result.spectrum, component)

        ax.plot(result.freq, y, label=f"S{result.order} {component_label(component)}")

        if result.spectrum_error is not None:
            err = np.abs(component_data(result.spectrum_error, component))
            for i in range(plot_style.significance):
                width = (i + 1) * err
                ax.fill_between(result.freq, y - width, y + width, alpha=0.15)

        ax.set_xlim(plot_style.f_min, plot_style.f_max)
        ax.set_ylabel(component_label(component))
        ax.set_title(f"S2 of channels {result.channels}")
        ax.legend()

    axes[-1][0].set_xlabel(f"Frequency / {result.freq_unit}")
    fig.tight_layout()

    return fig


def create_order_3_or_4_figure(result: SpectrumResult, plot_style: PlotStyle) -> Figure:
    cmap = custom_colormap()
    error_cmap = custom_error_colormap(plot_style.insignif_transparency)

    if result.spectrum is None:
        raise RuntimeError(f"Spectrum result for channels {result.channels} is not available.")

    if result.freq is None:
        raise RuntimeError(f"Spectrum result for channels {result.channels} has no frequency axis.")

    fig, axes = plt.subplots(
        1,
        len(plot_style.plot_format),
        figsize=(6 * len(plot_style.plot_format), 5),
        squeeze=False,
    )

    x, y = np.meshgrid(result.freq, result.freq)

    for col, component in enumerate(plot_style.plot_format):
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
            insignificant = np.abs(z) < plot_style.significance * err
            ax.pcolormesh(
                x,
                y,
                insignificant.astype(float),
                cmap=error_cmap,
                vmin=0,
                vmax=1,
                shading="auto",
            )

        ax.set_xlim(plot_style.f_min, plot_style.f_max)
        ax.set_ylim(plot_style.f_min, plot_style.f_max)
        ax.set_xlabel(f"Frequency / {result.freq_unit}")
        ax.set_ylabel(f"Frequency / {result.freq_unit}")
        ax.set_title(f"S{result.order} {component_label(component)} channels {result.channels}")
        fig.colorbar(mesh, ax=ax)

    fig.tight_layout()

    return fig


def create_spectrum_figures(
    result_store: SpectrumResultStore, plot_style: PlotStyle
) -> list[SpectrumFigure]:
    """Create figures for calculated spectra of orders 2 through 4."""

    figures = []

    for result in result_store:
        if result.order == 1:
            continue
        elif result.order == 2:
            figure = create_order_2_figure(result, plot_style)
        elif result.order in (3, 4):
            figure = create_order_3_or_4_figure(result, plot_style)
        else:
            raise ValueError(f"Unsupported spectrum order: {result.order}")

        figures.append(
            SpectrumFigure(
                figure=figure,
                order=result.order,
                channels=result.channels,
            )
        )

    return figures


def save_figures(
    figures: Iterable[SpectrumFigure],
    output_folder: str | Path,
    *,
    extension: str = "png",
    dpi: int = 150,
    close: bool = True,
) -> list[Path]:
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    for spectrum_figure in figures:
        path = output_folder / spectrum_figure.filename(extension)

        spectrum_figure.figure.savefig(
            path,
            dpi=dpi,
            bbox_inches="tight",
        )
        saved_paths.append(path)

        if close:
            plt.close(spectrum_figure.figure)

    return saved_paths
