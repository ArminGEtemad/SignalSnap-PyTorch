# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemed, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, DirectoryPath, Field, field_validator, model_validator

from .utils import ChannelIndex, TimeUnits

os.environ["PYDANTIC_ERRORS_INCLUDE_URL"] = "0"
SHARED_CONFIG = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class DataConfig(BaseModel):
    """Configuration for data used in polyspectra calculations.

    These settings are later resolved together with :class:`SpectrumConfig` into a
    :class:`~multichss.planning.RuntimeConfig`.

    Together with ``df`` calculated based on parameters in :class:`SpectrumConfig`, ``dt`` will be
    used to determine the number of data points (``window_points``) used for each Fourier
    transform::

        window_points = 1 / (dt * df)

    and to determine the Nyquist frequency::

        f_nyquist = 1 / (2 * dt)

    Attributes
    ----------
    data : ArrayLike with a shape attribute
        The recorded (real) signal data.
    dt : float
        The time interval between two consecutive data points. Must be positive.
    t_unit : Literal["s", "ms", "us", "ns", "ps"]
        Unit of the time step. Defaults to ``"s"``.
    """

    model_config = SHARED_CONFIG

    data: Any
    dt: Annotated[float, Field(gt=0)]
    t_unit: TimeUnits = "s"

    @field_validator("data")
    @classmethod
    def validate_data(cls, v: Any) -> Any:
        if v is None:
            raise ValueError("data cannot be None.")
        if not hasattr(v, "shape"):
            raise ValueError("DataConfig.data must provide a shape attribute.")
        if len(v.shape) == 0 or v.shape[0] <= 0:
            raise ValueError("DataConfig.data must contain at least one sample.")
        if np.iscomplexobj(v):
            raise TypeError("Input data cannot be complex.")
        if len(v.shape) != 1:
            raise ValueError("DataConfig.data must be one-dimensional.")

        return v


class PlotConfig(BaseModel):
    """Configuration for plotting calculated polyspectra.

    Attributes
    ----------
    f_min, f_max : float
        Frequency range displayed in plots.
    spectra_channels : list[tuple[ChannelIndex, ...] | None = None
        Specifies which (multi-channel) spectra will be shown. Each tuple represents one auto-
        or cross-correlation spectrum. Each tuple entry is a channel index. If ``None``, all
        available spectra will be plotted.
    significance : int
        Number of error estimates used to mark insignificant regions.
    arcsinh_scale : tuple[bool, float]
        Whether to apply arcsinh scaling and the relative scale factor to use.
    plot_format : list[Literal["re", "im"]]
        Spectrum components to plot.
    insignif_transparency : float
        Overlay opacity for values below the configured significance threshold.
    output : Literal["show", "save"]
        Whether plots are shown interactively or saved.
    output_folder : DirectoryPath
        Destination folder used when ``output="save"``.
    """

    model_config = SHARED_CONFIG

    f_min: float
    f_max: float

    spectra_channels: (
        Annotated[
            list[
                tuple[ChannelIndex]
                | tuple[ChannelIndex, ChannelIndex]
                | tuple[ChannelIndex, ChannelIndex, ChannelIndex]
                | tuple[ChannelIndex, ChannelIndex, ChannelIndex, ChannelIndex]
            ],
            Field(min_length=1),
        ]
        | None
    ) = None
    significance: Annotated[int, Field(gt=0)] = 1
    arcsinh_scale: tuple[bool, Annotated[float, Field(ge=0)]] = (False, 0.02)
    plot_format: Annotated[list[Literal["re", "im"]], Field(min_length=1)] = ["re", "im"]
    insignif_transparency: Annotated[float, Field(ge=0.0, le=1.0)] = 0.8
    output: Literal["show", "save"] = "show"
    output_folder: DirectoryPath = Path(".").resolve()

    @field_validator("plot_format")
    @classmethod
    def ensure_unique_formats(cls, v: list[str]) -> list[str]:
        """Ensure plot_format does not contain duplicate components."""
        if len(v) != len(set(v)):
            raise ValueError("plot_format cannot contain duplicate elements.")
        return v

    @field_validator("output_folder")
    @classmethod
    def resolve_output_folder(cls, v: Path) -> Path:
        return v.resolve()

    @model_validator(mode="after")
    def validate_limits(self) -> PlotConfig:
        if self.f_min >= self.f_max:
            raise ValueError(f"f_min ({self.f_min}) must be less than f_max ({self.f_max}).")
        return self
    
    @field_validator("spectra_channels", mode="before")
    @classmethod
    def validate_spectrum_request(cls, spectra_channels):
        """
        Check if spectra_channels contains duplicates.
        """
        if spectra_channels is not None:
            if len(spectra_channels) != len(set(spectra_channels)):
                raise ValueError("spectra_channels cannot contain duplicates.")

        return spectra_channels


class SpectrumConfig(BaseModel):
    """Spectrum configuration for polyspectra calculations.

    :class:`SpectrumConfig` describes what the user asks the calculation to use: frequency bounds,
    number of frequency points, spectrum orders, window count per spectral estimate, backend torch
    device, and compatibility options. These settings are later resolved together with
    :class:`DataConfig` into a :class:`~multichss.planning.RuntimeConfig`.

    ``f_min``, ``f_max``, and ``frequency_points`` will be used to determine the frequency spacing::

        df = (f_max - f_min) / (frequency_points - 1)

    Together with ``dt`` from :class:`DataConfig`, ``df`` will be used to determine the number of
    data points (``window_points``) used for each Fourier transform::

        window_points = 1 / (dt * df)

    Attributes
    ----------
    f_min : float = 0.0
        Lower frequency bound. If omitted, zero is used.
    f_max : float | None = None
        Upper frequency bound. If omitted, the Nyquist frequency based on :class:`DataConfig`'s
        ``dt`` is used.
    frequency_points : int = 100
        Number of frequency points in the specified frequency range. Must be positive.
    spectra_channels : list[tuple[ChannelIndex, ...] | None = None
        Specifies which (multi-channel) spectra will be calculated. Each tuple represents one auto-
        or cross-correlation spectrum. Each tuple entry is a channel index. If ``None``, the auto-
        correlation spectra of orders 1 to 4 will be calculated of all available data channels.
    m : int = 10
        Number of windows used per spectral estimate. This may be reduced at runtime if the signal
        is too short. Must be positive.
    device : Literal["cpu", "mps", "cuda"]  = "cpu"
        Torch device requested for calculation.
    precision : Literal["auto", "single", "double"] = "auto"
        Floating point precision. ``single`` will result in ``float32`` and ``complex64``.
        ``double`` will result in ``float64`` and ``complex128``. ``auto`` will choose ``single`` if
        device is ``mps`` and ``double`` otherwise.
    spectral_estimates_max : int | None = int(1e6)
        Maximum number of unshifted spectral estimates. If ``None``, as many estimates as possible
        are calculated based on the data. The true number of spectral estimates may be lower if the
        data does not have enough samples. If ``interlacing=True``, up to the same number of
        additional shifted estimates are calculated. The number of shifted estimates may also be one
        less than the number of unshifted estimates if the final shifted windows do not fit. Must be
        positive.
    interlacing : bool = False
        Compute additional spectral estimates for windows shifted by half a window size, to
        compensate the low weight of data points produced by the window function near the original
        window edges. Error estimates are calculated separately for unshifted and shifted spectra;
        when both are available, the reported error is the component-wise maximum of both estimates.
    old_window : bool = False
        Compatibility option. If set to ``True``, the approximated confined Gaussian window from
        the old API is used as a window function.
    """

    model_config = SHARED_CONFIG

    f_min: float = 0.0
    f_max: float | None = None
    frequency_points: Annotated[int, Field(ge=2)] = 100
    spectra_channels: (
        Annotated[
            list[
                tuple[ChannelIndex]
                | tuple[ChannelIndex, ChannelIndex]
                | tuple[ChannelIndex, ChannelIndex, ChannelIndex]
                | tuple[ChannelIndex, ChannelIndex, ChannelIndex, ChannelIndex]
            ],
            Field(min_length=1),
        ]
        | None
    ) = None
    m: Annotated[int, Field(gt=0)] = 10
    device: Literal["cpu", "mps", "cuda"] = "cpu"
    precision: Literal["auto", "single", "double"] = "auto"
    spectral_estimates_max: Annotated[int, Field(gt=0)] | None = int(1e6)
    interlacing: bool = False
    old_window: bool = False

    @model_validator(mode="after")
    def validate_limits(self) -> SpectrumConfig:
        if self.f_max is not None:
            if self.f_min >= self.f_max:
                raise ValueError(f"f_min ({self.f_min}) must be less than f_max ({self.f_max}).")
        
        return self

    @field_validator("spectra_channels", mode="before")
    @classmethod
    def validate_spectrum_request(cls, spectra_channels):
        """
        Check if spectra_channels contains duplicates.
        """
        if spectra_channels is not None:
            if len(spectra_channels) != len(set(spectra_channels)):
                raise ValueError("spectra_channels cannot contain duplicates.")

        return spectra_channels
