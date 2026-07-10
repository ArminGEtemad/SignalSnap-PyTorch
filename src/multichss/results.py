# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemed, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
from torch import Tensor

from .utils import FrequencyUnits


@dataclass(slots=True)
class SpectrumAccumulator:
    """Data container for the accumulation of spectral estimates.

    Stores the configuration metadata, accumulated hardware states, error buffers for a specific
    higher-order auto- or cross-spectrum calculation.

    Attributes
    ----------
    channels : tuple[int, ...]
        The indices identifying which channels are part of this calculation. For example,
        ``(0, 0, 0)`` indicates a third-order auto-spectrum on channel 0, while ``(0, 1)`` indicates
        a cross-spectrum between channels 0 and 1.
    freq : np.ndarray
        Frequency axis associated with the spectrum.
    freq_unit : Literal["Hz", "kHz", "MHz", "GHz", "THz"]
        Unit of the frequency axis.
    spectrum_sum_unshifted : torch.Tensor | None
        Running total sum of the unshifted calculated spectra on the active torch device.
    spectrum_sum_shifted : torch.Tensor | None
        Running total sum of the shifted calculated spectra on the active torch device. Only used
        when interlacing is enabled.
    squared_sum_unshifted : torch.Tensor | None
        Running total squared sum of the real and imaginary parts of the unshifted spectra on the
        active torch device. Real and imaginary parts are squared separately.
    squared_sum_shifted : torch.Tensor | None
        Running total squared sum of the real and imaginary parts of the shifted spectra on the
        active torch device. Real and imaginary parts are squared separately. Only used when
        interlacing is enabled.
    chunks_unshifted : int
        The total number of individual unshifted spectral estimates integrated into
        ``spectrum_sum_unshifted``.
    chunks_shifted : int
        The total number of individual shifted spectral estimates integrated into
        ``spectrum_sum_shifted``. When interlacing is enabled, this count never exceeds
        ``chunks_unshifted``.
    """

    channels: tuple[int, ...]
    freq: np.ndarray
    freq_unit: FrequencyUnits

    spectrum_sum_unshifted: Tensor | None = None
    spectrum_sum_shifted: Tensor | None = None
    squared_sum_unshifted: Tensor | None = None
    squared_sum_shifted: Tensor | None = None

    chunks_unshifted: int = 0
    chunks_shifted: int = 0

    @property
    def order(self) -> int:
        return len(self.channels)


@dataclass(slots=True)
class SpectrumAccumulatorStore:
    """Container for all :class:`SpectrumAccumulator` used by a calculation pipeline.

    Stores one :class:`SpectrumAccumulator` per channel tuple. Accumulators are indexed by
    ``channels``, where ``channels`` is a tuple of data-channel indices.

    This class owns collection-level bookkeeping only. Numerical accumulation, error estimation, and
    finalization are handled elsewhere.

    Attributes
    ----------
    accumulators : dict[tuple[int, ...], SpectrumAccumulator]
        Mapping from ``channels`` to the corresponding :class:`SpectrumAccumulator`. For example,
        ``(0, 0)`` identifies the second-order auto-spectrum of channel 0, while ``(0, 1)``
        identifies a second-order cross-spectrum between channels 0 and 1.
    """

    accumulators: dict[tuple[int, ...], SpectrumAccumulator] = field(default_factory=dict)

    def __iter__(self) -> Iterator[SpectrumAccumulator]:
        return iter(self.accumulators.values())

    def get(self, channels: tuple[int, ...]) -> SpectrumAccumulator:
        """Return the accumulator for a channel tuple."""
        return self.accumulators[channels]

    def add(self, accumulator: SpectrumAccumulator) -> None:
        """Add or replace a spectrum accumulator using its channels."""
        self.accumulators[accumulator.channels] = accumulator


@dataclass(frozen=True, slots=True)
class SpectrumResult:
    """Data container for the results of a single spectral calculation.

    Stores the configuration metadata, and final computed results for a specific higher-order auto-
    or cross-spectrum calculation.

    Attributes
    ----------
    channels : tuple[int, ...]
        The indices identifying which channels are part of this calculation. For example,
        ``(0, 0, 0)`` indicates a third-order auto-spectrum on channel 0, while ``(0, 1)`` indicates
        a cross-spectrum between channels 0 and 1.
    freq : np.ndarray
        Frequency axis associated with the spectrum.
    freq_unit : Literal["Hz", "kHz", "MHz", "GHz", "THz"]
        Unit of the frequency axis.
    spectrum : np.ndarray
        The final normalized spectral values transferred back to the CPU.
    spectrum_error : np.ndarray | None
        The final calculated standard error of the mean (SEM) values transferred back to the CPU. If
        interlacing was enabled, this is the maximum of the unshifted and shifted spectrum error. If
        only one shifted spectral estimate is available, ``spectrum_error`` is based on the
        unshifted estimates alone.
    """

    channels: tuple[int, ...]

    freq: np.ndarray
    freq_unit: FrequencyUnits
    spectrum: np.ndarray
    spectrum_error: np.ndarray | None = None

    @property
    def order(self) -> int:
        return len(self.channels)

    def __post_init__(self) -> None:
        if not 1 <= self.order <= 4:
            raise ValueError(f"Unsupported spectrum order {self.order}.")

        if self.freq.ndim != 1:
            raise ValueError("Frequency axis must be one-dimensional.")

        frequency_points = len(self.freq)

        expected_shape = {
            1: (1,),
            2: (frequency_points,),
            3: (frequency_points, frequency_points),
            4: (frequency_points, frequency_points),
        }[self.order]

        if self.spectrum.shape != expected_shape:
            raise ValueError(
                f"Order-{self.order} spectrum has shape "
                f"{self.spectrum.shape}; expected {expected_shape}."
            )

        if self.spectrum_error is not None and self.spectrum_error.shape != expected_shape:
            raise ValueError("Spectrum error must have the same shape as the spectrum.")


@dataclass(slots=True)
class SpectrumResultStore:
    """Container for all spectrum results produced by a calculation pipeline.

    Stores one :class:`SpectrumResult` per channel tuple. Results are indexed by ``channels``, where
    ``channels`` is a tuple of data-channel indices.

    This class owns collection-level bookkeeping only. Numerical accumulation, error estimation, and
    finalization are handled elsewhere.

    Attributes
    ----------
    results : dict[tuple[int, ...], SpectrumResult]
        Mapping from ``channels`` to the corresponding spectrum result. For example,
        ``(0, 0)`` identifies the second-order auto-spectrum of channel 0, while ``(0, 1)``
        identifies a second-order cross-spectrum between channels 0 and 1.
    """

    results: dict[tuple[int, ...], SpectrumResult] = field(default_factory=dict)

    def __iter__(self) -> Iterator[SpectrumResult]:
        return iter(self.results.values())

    def get(self, channels: tuple[int, ...]) -> SpectrumResult:
        """Return the result for a channel tuple."""
        return self.results[channels]

    def add(self, result: SpectrumResult) -> None:
        """Add or replace a spectrum result using its channels."""
        self.results[result.channels] = result

    def select(self, channels: list[tuple[int, ...]]) -> SpectrumResultStore:
        """Return a new store containing the selected results.

        The new store shares its SpectrumResult objects with this store.
        """
        selected = {}

        for channel_tuple in channels:
            try:
                selected[channel_tuple] = self.results[channel_tuple]
            except KeyError as exc:
                raise ValueError(
                    f"No spectrum result exists for channels {channel_tuple}."
                ) from exc

        return SpectrumResultStore(results=selected)
