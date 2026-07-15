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
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ._core.utils import TimeUnits as _TimeUnits

os.environ["PYDANTIC_ERRORS_INCLUDE_URL"] = "0"
_SHARED_CONFIG = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class DataConfig(BaseModel):
    """Configuration for data used in polyspectra calculations.

    These settings are later resolved together with :class:`SpectrumConfig` into the internal
    runtime configuration used by :func:`~multichss.calculate_spectra`.

    Together with ``df`` calculated based on parameters in :class:`SpectrumConfig`, ``dt`` will be
    used to determine the number of data points (``window_points``) used for each Fourier
    transform:

        window_points = 1 / (dt * df)

    and to determine the Nyquist frequency:

        f_nyquist = 1 / (2 * dt)

    Attributes
    ----------
    channels : tuple[Any, ...]
        List of data channels. Each channel is recorded (real) signal data and can either be an
        array with a shape and dtype attribute or a :class:`HDF5Channel`.
    dt : float
        The time interval between two consecutive data points. Must be positive.
    t_unit : Literal["s", "ms", "us", "ns", "ps"]
        Unit of the time step. Defaults to ``"s"``.
    """

    model_config = _SHARED_CONFIG

    channels: Annotated[tuple[Any, ...], Field(min_length=1)]
    dt: Annotated[float, Field(gt=0)]
    t_unit: _TimeUnits = "s"

    @field_validator("channels")
    @classmethod
    def validate_channels(cls, channels: tuple[Any, ...]) -> tuple[Any, ...]:
        for index, channel in enumerate(channels):
            if isinstance(channel, HDF5Channel):
                continue

            if channel is None:
                raise ValueError(f"Channel {index} cannot be None.")

            if not hasattr(channel, "shape"):
                raise TypeError(f"Array channel {index} must provide a shape attribute.")

            if len(channel.shape) != 1:
                raise ValueError(f"Array channel {index} must be one-dimensional.")

            if channel.shape[0] == 0:
                raise ValueError(f"Array channel {index} cannot be empty.")

            try:
                channel_dtype = np.dtype(channel.dtype)
            except TypeError:
                channel_dtype = np.asarray(channel).dtype

            if np.issubdtype(channel_dtype, np.complexfloating):
                raise TypeError(f"Array channel {index} cannot be complex.")

            is_numeric = np.issubdtype(channel_dtype, np.number)
            is_boolean = np.issubdtype(channel_dtype, np.bool_)

            if not is_numeric and not is_boolean:
                raise TypeError(
                    f"Array channel {index} must be numeric; received dtype {channel_dtype}."
                )

        return channels


class HDF5Channel(BaseModel):
    """Location of one signal channel inside an HDF5 dataset."""

    model_config = _SHARED_CONFIG

    file: Path
    dataset: str
    selection: tuple[Any, ...]

    @field_validator("dataset")
    @classmethod
    def validate_dataset(cls, value: str) -> str:
        if not value:
            raise ValueError("dataset cannot be empty.")
        return value

    @field_validator("selection")
    @classmethod
    def validate_selection(cls, value: tuple[Any, ...]) -> tuple[Any, ...]:
        if not value:
            raise ValueError("selection cannot be empty.")

        normalized = []

        for item in value:
            if isinstance(item, (bool, np.bool_)):
                raise TypeError("HDF5 selection entries must be integers or slices.")

            if isinstance(item, np.integer):
                item = int(item)

            if not isinstance(item, (int, slice)):
                raise TypeError("HDF5 selection entries must be integers or slices.")

            if isinstance(item, slice):
                if item.start is None:
                    start = None
                else:
                    if isinstance(item.start, (bool, np.bool_)):
                        raise TypeError("HDF5 slice start must be an integer or None.")
                    if not isinstance(item.start, (int, np.integer)):
                        raise TypeError("HDF5 slice start must be an integer or None.")
                    start = int(item.start)

                if item.stop is None:
                    stop = None
                else:
                    if isinstance(item.stop, (bool, np.bool_)):
                        raise TypeError("HDF5 slice stop must be an integer or None.")
                    if not isinstance(item.stop, (int, np.integer)):
                        raise TypeError("HDF5 slice stop must be an integer or None.")
                    stop = int(item.stop)

                if item.step is None:
                    step = None
                else:
                    if isinstance(item.step, (bool, np.bool_)):
                        raise TypeError("HDF5 slice step must be an integer or None.")
                    if not isinstance(item.step, (int, np.integer)):
                        raise TypeError("HDF5 slice step must be an integer or None.")
                    step = int(item.step)

                if step not in (None, 1):
                    raise ValueError("HDF5 slice steps other than 1 are not supported.")

                normalized.append(slice(start, stop, step))
            else:
                normalized.append(item)

        return tuple(normalized)


class SpectrumConfig(BaseModel):
    """Spectrum configuration for polyspectra calculations.

    :class:`SpectrumConfig` describes what the user asks the calculation to use: frequency bounds,
    number of frequency points, window count per spectral estimate, backend torch device, and
    compatibility options. These settings are later resolved together with :class:`DataConfig` into
    the internal runtime configuration used by :func:`~multichss.calculate_spectra`.

    ``f_min``, ``f_max``, and ``frequency_points`` will be used to determine the REQUESTED frequency
    spacing:

        df* = (f_max - f_min) / (frequency_points - 1)

    (!) The discrete Fourier transform cannot result in arbitrary frequency spacings with a given
    sample spacing. The library will use the closest available frequency spacing.

    (!) The discrete Fourier transform always includes the frequency ``f=0`` and all frequency
    points in the range between ``f_min`` and ``f_max`` are integer multiples of the actual df.

    Together with ``dt`` from :class:`DataConfig`, ``df`` will be used to determine the number of
    data points (``window_points``) used for each Fourier transform:

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

    model_config = _SHARED_CONFIG

    f_min: float = 0.0
    f_max: float | None = None
    frequency_points: Annotated[int, Field(ge=2)] = 100
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
