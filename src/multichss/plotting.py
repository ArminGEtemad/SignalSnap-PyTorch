# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemed, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from pydantic import BaseModel, Field, field_validator, model_validator

from ._core.data_access import get_sample_count, open_channels, read_channel
from ._core.planning import normalize_channel_index, resolve_frequencies
from ._core.utils import PlotComponent as _PlotComponent
from .configurators import _SHARED_CONFIG, DataConfig, SpectrumConfig
from .results import SpectrumResult, SpectrumResultStore


class PlotStyle(BaseModel):
    """Configuration for plotting calculated polyspectra.

    Attributes
    ----------
    f_min, f_max : float
        Frequency range displayed in plots.
    sigma : float
        Default standard-error level for second-order uncertainty intervals and the insignificance
        threshold for higher-order spectra.
    uncertainty_levels : list[float] | None
        Standard-error levels displayed as uncertainty bands for second-order spectra. If ``None``,
        a single band at ``sigma`` is displayed.
    arcsinh_ratio : float | None
        Relative width of the approximately linear region used for arcsinh display scaling. If
        ``None``, no scaling is applied.
    plot_format : list[Literal["re", "im"]]
        Spectrum components to plot.
    insignificance_alpha : float
        Opacity of the overlay marking insignificant values in third- and fourth-order spectra.
    """

    model_config = _SHARED_CONFIG

    f_min: float
    f_max: float

    sigma: Annotated[float, Field(gt=0)] = 1.0
    uncertainty_levels: (
        Annotated[list[Annotated[float, Field(gt=0)]], Field(min_length=1)] | None
    ) = None
    arcsinh_ratio: Annotated[float, Field(gt=0)] | None = None
    plot_format: Annotated[list[_PlotComponent], Field(min_length=1)] = ["re", "im"]
    insignificance_alpha: Annotated[float, Field(ge=0.0, le=1.0)] = 0.8

    @field_validator("plot_format")
    @classmethod
    def ensure_unique_formats(cls, v: list[_PlotComponent]) -> list[_PlotComponent]:
        """Ensure plot_format does not contain duplicate components."""
        if len(v) != len(set(v)):
            raise ValueError("plot_format cannot contain duplicate elements.")
        return v

    @model_validator(mode="after")
    def validate_limits(self) -> PlotStyle:
        if self.f_min >= self.f_max:
            raise ValueError(f"f_min ({self.f_min}) must be less than f_max ({self.f_max}).")
        return self
    

@dataclass(frozen=True)
class SpectrumFigure:
    """Matplotlib figure and metadata for one plotted spectrum.

    Attributes
    ----------
    figure : Figure
        Matplotlib figure containing the plotted spectrum.
    order : int
        Order of the plotted spectrum.
    channels : tuple[int, ...]
        Data-channel indices identifying the plotted auto- or cross-spectrum.
    """

    figure: Figure
    order: int
    channels: tuple[int, ...]

    def filename(self, extension: str = "png") -> str:
        """Build a filename from the spectrum order and channel indices.

        Parameters
        ----------
        extension : str = "png"
            Filename extension, without a leading period.

        Returns
        -------
        str
            Filename in the form ``s{order}_channels_{channels}.{extension}``.
        """

        channel_label = "_".join(map(str, self.channels))
        return f"s{self.order}_channels_{channel_label}.{extension}"


def _arcsinh_width(data: np.ndarray, ratio: float) -> float | None:
    """Calculate the width of the linear region for arcsinh display scaling.

    Parameters
    ----------
    data : np.ndarray
        Values whose finite maximum absolute magnitude sets the scale.
    ratio : float
        Fraction of the maximum magnitude used as the linear-region width.

    Returns
    -------
    float | None
        Scaling width, or ``None`` if the data has no finite nonzero values.
    """

    finite = np.asarray(data)[np.isfinite(data)]

    if finite.size == 0:
        return None

    maximum = float(np.max(np.abs(finite)))

    if maximum == 0.0:
        return None

    return ratio * maximum


def _component_data(data: np.ndarray, component: _PlotComponent) -> np.ndarray:
    """Extract the requested real or imaginary component of complex data.

    Parameters
    ----------
    data : np.ndarray
        Complex-valued spectrum data.
    component : Literal["re", "im"]
        Component to extract.

    Returns
    -------
    np.ndarray
        Real-valued view or array containing the selected component.

    Raises
    ------
    ValueError
        If ``component`` is not ``"re"`` or ``"im"``.
    """

    if component == "re":
        return np.real(data)
    if component == "im":
        return np.imag(data)
    raise ValueError(f"Unsupported plot component: {component}")


def _component_label(component: _PlotComponent) -> str:
    """Return the display label for a spectrum component.

    Parameters
    ----------
    component : Literal["re", "im"]
        Spectrum component to label.

    Returns
    -------
    str
        ``"Real"`` for ``"re"`` and ``"Imaginary"`` for ``"im"``.
    """

    return "Real" if component == "re" else "Imaginary"


def _custom_colormap() -> LinearSegmentedColormap:
    """Create the diverging colormap used for higher-order spectra.

    Returns
    -------
    LinearSegmentedColormap
        Blue-gray-red colormap centered on a neutral gray.
    """

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


def _custom_error_colormap(insignificance_alpha: float) -> LinearSegmentedColormap:
    """Create the transparency overlay used to mark insignificant values.

    Parameters
    ----------
    insignificance_alpha : float
        Opacity assigned to values marked as insignificant.

    Returns
    -------
    LinearSegmentedColormap
        Colormap ranging from transparent to white at the requested opacity.
    """

    return LinearSegmentedColormap.from_list(
        "multichss_insignificant",
        [
            (0.0, 0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0, insignificance_alpha),
        ],
    )


def create_first_window_figure(
    data_config: DataConfig,
    spectrum_config: SpectrumConfig,
    *,
    channels: Iterable[int] | None = None,
) -> Figure:
    """Create a figure of the first FFT window of one or more data channels.

    The window length is derived from ``spectrum_config`` and the sampling
    interval of the selected data channels, using the same frequency
    resolution logic as the spectrum calculation.

    Parameters
    ----------
    data_config : :class:`DataConfig`
        Input channels and their shared sampling metadata. Channels may be in-memory arrays or
        HDF5-backed channels.
    spectrum_config : :class:`SpectrumConfig`
        Configuration defining the requested frequency range and number of frequency points.
    channels : Iterable[int] | None, optional
        Indices of the channels to plot. Indices refer to entries in ``data_config.channels``. If
        ``None``, all available channels are plotted.

    Returns
    -------
    Figure
        Matplotlib figure containing one subplot per selected channel.

    Raises
    ------
    ValueError
        If a channel index is invalid or if a channel is shorter than one window.
    TypeError
        If a channel index is not an integer.
    """

    # Validate DataConfigs and requested channels.
    if channels is None:
        selected_channels = tuple(range(len(data_config.channels)))
    else:
        selected_channels = tuple(channels)

    if not selected_channels:
        raise ValueError("At least one channel must be selected.")

    normalized_channels: list[int] = []

    for channel in selected_channels:
        normalized = normalize_channel_index(channel, len(data_config.channels))

        if normalized in normalized_channels:
            raise ValueError(f"Channel {normalized} was selected more than once.")

        normalized_channels.append(normalized)

    # Plot first window
    window_points, _, _, _ = resolve_frequencies(spectrum_config=spectrum_config, dt=data_config.dt)

    with open_channels(data_config, normalized_channels) as opened_channels:
        for channel in normalized_channels:
            available_points = get_sample_count(opened_channels[channel])

            if available_points < window_points:
                raise ValueError(
                    f"Channel {channel} contains {available_points} samples, "
                    f"but one window requires {window_points} samples."
                )

        figure, axes = plt.subplots(
            nrows=len(normalized_channels),
            ncols=1,
            figsize=(14, 3 * len(normalized_channels)),
            squeeze=False,
            sharex=True,
        )

        time = np.arange(window_points) * data_config.dt

        for row, channel in enumerate(normalized_channels):
            axis = axes[row, 0]
            first_window = read_channel(opened_channels[channel], start=0, stop=window_points)

            axis.plot(time, first_window)
            axis.set_title(f"First window for channel {channel}")
            axis.set_ylabel("Amplitude")

            # Avoid identical x-axis limits for a one-sample window.
            if window_points > 1:
                axis.set_xlim(time[0], time[-1])

    axes[-1, 0].set_xlabel(f"t / {data_config.t_unit}")
    figure.tight_layout()

    return figure


def _format_order_1_rows(rows: list[dict[str, object]]) -> str:
    """Format order-1 spectrum values as a fixed-width text table.

    Parameters
    ----------
    rows : list[dict[str, object]]
        Table rows keyed by the order-1 column headings. Missing values are formatted as empty
        strings.

    Returns
    -------
    str
        Table containing a header and separator followed by the supplied rows. If ``rows`` is empty,
        only the header and separator are returned.
    """

    headers = ["Channels", "Real", "Imag", "Error real", "Error imag"]

    table = [[str(row.get(header, "")) for header in headers] for row in rows]

    widths = [
        max(len(header), max((len(row[col]) for row in table), default=0))
        for col, header in enumerate(headers)
    ]

    header_line = "  ".join(header.ljust(widths[col]) for col, header in enumerate(headers))

    separator = "  ".join("-" * width for width in widths)

    body = ["  ".join(value.ljust(widths[col]) for col, value in enumerate(row)) for row in table]

    return "\n".join([header_line, separator, *body])


def build_order_1_table(result_store: SpectrumResultStore) -> str:
    """Build a text table from the order-1 results in a result store.

    Parameters
    ----------
    result_store : :class:`SpectrumResultStore`
        Calculated spectra from which order-1 results are selected.

    Returns
    -------
    str
        Fixed-width table of channel indices, complex spectrum values, and error estimates.

    Warns
    -----
    RuntimeWarning
        If the store contains no order-1 results. A header-only table is still returned.
    """

    order_1_results = [result for result in result_store if result.order == 1]

    if not order_1_results:
        warnings.warn("No matching results at order 1.", RuntimeWarning, stacklevel=2)

    rows = []
    for result in order_1_results:
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
    return _format_order_1_rows(rows)


def _create_order_2_figure(result: SpectrumResult, plot_style: PlotStyle) -> Figure:
    """Create a line-plot figure for an order-2 spectrum result.

    Parameters
    ----------
    result : :class:`SpectrumResult`
        Order-2 spectrum and optional error estimate to plot.
    plot_style : :class:`PlotStyle`
        Frequency limits, components, scaling, and significance settings for the plot.

    Returns
    -------
    Figure
        Matplotlib figure containing one subplot per requested spectrum component.
    """

    fig, axes = plt.subplots(
        len(plot_style.plot_format),
        1,
        figsize=(8, 3 * len(plot_style.plot_format)),
        squeeze=False,
        sharex=True,
    )

    for row, component in enumerate(plot_style.plot_format):
        ax = axes[row][0]
        y = _component_data(result.spectrum, component)
        width = None

        if plot_style.arcsinh_ratio is not None:
            width = _arcsinh_width(y, plot_style.arcsinh_ratio)

        component_name = _component_label(component)
        if width is not None:
            ax.set_yscale("asinh", linear_width=width)
            component_name += " (arcsinh scaled)"

        ax.plot(result.freq, y, label=f"S{result.order} {component_name}")

        if result.spectrum_error is not None:
            err = np.abs(_component_data(result.spectrum_error, component))

            levels = (
                plot_style.uncertainty_levels
                if plot_style.uncertainty_levels is not None
                else [plot_style.sigma]
            )

            # Draw widest first so narrower bands remain visible.
            levels = sorted(set(levels), reverse=True)
            number_of_levels = len(levels)

            for index, level in enumerate(levels):
                interval = level * err
                alpha = 0.08 + 0.12 * (index + 1) / number_of_levels

                ax.fill_between(
                    result.freq,
                    y - interval,
                    y + interval,
                    alpha=alpha,
                    label=f"{level:g}σ uncertainty",
                )

        ax.set_xlim(plot_style.f_min, plot_style.f_max)
        ax.set_ylabel(component_name)
        ax.set_title(f"S2 of channels {result.channels}")
        ax.legend()

    axes[-1][0].set_xlabel(f"Frequency / {result.freq_unit}")

    fig.tight_layout()

    return fig


def _create_order_3_or_4_figure(result: SpectrumResult, plot_style: PlotStyle) -> Figure:
    """Create a two-dimensional color-map figure for an order-3 or order-4 result.

    Parameters
    ----------
    result : :class:`SpectrumResult`
        Order-3 or order-4 spectrum and optional error estimate to plot.
    plot_style : :class:`PlotStyle`
        Frequency limits, components, scaling, and significance settings for the plot.

    Returns
    -------
    Figure
        Matplotlib figure containing one color-map subplot per requested spectrum component.
    """

    cmap = _custom_colormap()
    error_cmap = _custom_error_colormap(plot_style.insignificance_alpha)

    fig, axes = plt.subplots(
        1,
        len(plot_style.plot_format),
        figsize=(6 * len(plot_style.plot_format), 5),
        squeeze=False,
    )

    x, y = np.meshgrid(result.freq, result.freq)

    for col, component in enumerate(plot_style.plot_format):
        ax = axes[0][col]
        raw_z = _component_data(result.spectrum, component)
        z = np.ma.masked_invalid(raw_z)

        limit = np.nanmax(np.abs(raw_z))
        width = None

        if plot_style.arcsinh_ratio is not None:
            width = _arcsinh_width(raw_z, plot_style.arcsinh_ratio)

        if width is None:
            norm = mcolors.Normalize(vmin=-limit, vmax=limit)
        else:
            norm = mcolors.AsinhNorm(linear_width=width, vmin=-limit, vmax=limit)

        mesh = ax.pcolormesh(x, y, z, cmap=cmap, norm=norm, shading="auto")

        if result.spectrum_error is not None:
            err = np.abs(_component_data(result.spectrum_error, component))
            insignificant = np.abs(raw_z) < plot_style.sigma * err
            ax.pcolormesh(
                x, y, insignificant.astype(float), cmap=error_cmap, vmin=0, vmax=1, shading="auto"
            )

        component_name = _component_label(component)

        if width is not None:
            component_name += " (arcsinh scaled)"

        ax.set_xlim(plot_style.f_min, plot_style.f_max)
        ax.set_ylim(plot_style.f_min, plot_style.f_max)
        ax.set_xlabel(f"Frequency / {result.freq_unit}")
        ax.set_ylabel(f"Frequency / {result.freq_unit}")
        ax.set_title(f"S{result.order} {component_name} channels {result.channels}")
        fig.colorbar(mesh, ax=ax)

    fig.tight_layout()

    return fig


def create_spectrum_figure(result: SpectrumResult, plot_style: PlotStyle) -> SpectrumFigure:
    """Create a figure for one calculated spectrum.

    Parameters
    ----------
    result : :class:`SpectrumResult`
        Order-2, order-3, or order-4 spectrum to plot.
    plot_style : :class:`PlotStyle`
        Frequency limits, components, scaling, and significance settings for the plot.

    Returns
    -------
    :class:`SpectrumFigure`
        Matplotlib figure together with its spectrum order and channel metadata.

    Raises
    ------
    ValueError
        If ``result`` is not an order-2, order-3, or order-4 spectrum.
    """

    if result.order == 2:
        figure = _create_order_2_figure(result, plot_style)
    elif result.order in (3, 4):
        figure = _create_order_3_or_4_figure(result, plot_style)
    else:
        raise ValueError(f"Unsupported spectrum order: {result.order}")

    return SpectrumFigure(figure=figure, order=result.order, channels=result.channels)


def create_spectrum_figures(
    result_store: SpectrumResultStore, plot_style: PlotStyle
) -> list[SpectrumFigure]:
    """Create figures for all order-2 through order-4 results in a result store.

    Order-1 results are skipped because they are represented as text by
    :func:`build_order_1_table`.

    Parameters
    ----------
    result_store : :class:`SpectrumResultStore`
        Calculated spectra to plot.
    plot_style : :class:`PlotStyle`
        Frequency limits, components, scaling, and significance settings shared by all figures.

    Returns
    -------
    list[:class:`SpectrumFigure`]
        Figures in result-store iteration order. An empty list is returned if the store contains no
        results of orders 2 through 4.
    """

    figures = []

    for result in result_store:
        if result.order == 1:
            continue

        figures.append(create_spectrum_figure(result, plot_style))

    return figures


def save_figures(
    figures: list[SpectrumFigure],
    output_folder: str | Path,
    *,
    extension: str = "png",
    dpi: int = 150,
    close: bool = True,
) -> list[Path]:
    """Save spectrum figures to an output folder.

    Filenames are generated from each figure's spectrum order and channel indices using
    :meth:`SpectrumFigure.filename`.

    Parameters
    ----------
    figures : list[:class:`SpectrumFigure`]
        Spectrum figures to save.
    output_folder : str | Path
        Destination directory. It and any missing parent directories are created automatically.
    extension : str = "png"
        Output filename extension, without a leading period.
    dpi : int = 150
        Resolution passed to Matplotlib when saving each figure.
    close : bool = True
        Whether to close each Matplotlib figure after saving it.

    Returns
    -------
    list[Path]
        Paths of the saved figures in input iteration order.
    """

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
