# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemed, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from torch import Tensor

from .utils import FrequencyUnits

if TYPE_CHECKING:
    from .planning import RuntimeConfig


@dataclass(slots=True)
class SpectrumResult:
    """Data container for the state and results of a single spectral calculation.

    Stores the configuration metadata, accumulated hardware states, error buffers, and final
    computed results for a specific higher-order auto- or cross-spectrum calculation.

    Attributes
    ----------
    channels : tuple[int, ...]
        The indices identifying which channels are part of this calculation. For example,
        ``(0, 0, 0)`` indicates a third-order auto-spectrum on channel 0, while ``(0, 1)`` indicates
        a cross-spectrum between channels 0 and 1.
    freq : np.ndarray | None
        Frequency axis associated with the spectrum.
    freq_unit : Literal["Hz", "kHz", "MHz", "GHz", "THz"]
        Unit of the frequency axis.
    spectrum : np.ndarray | None
        The final normalized spectral values transferred back to the CPU.
    spectrum_error : np.ndarray | None
        The final calculated standard error of the mean (SEM) values transferred back to the CPU. If
        interlacing was enabled, this is the maximum of the unshifted and shifted spectrum error. If
        only one shifted spectral estimate is available, ``spectrum_error`` is based on the
        unshifted estimates alone.
    spectrum_accumulator_unshifted : torch.Tensor | None
        Running total accumulation buffer of the unshifted calculated spectra on the active torch
        device.
    spectrum_accumulator_shifted : torch.Tensor | None
        Running total accumulation buffer of the shifted calculated spectra on the active torch
        device. Only used when interlacing is enabled.
    error_accumulator_x_squared_unshifted : torch.Tensor | None
        Running total accumulation buffer of the real and imaginary parts of the unshifted spectra
        squared on the active torch device. Real and imaginary parts are squared separately.
    error_accumulator_x_squared_shifted : torch.Tensor | None
        Running total accumulation buffer of the real and imaginary parts of the shifted spectra
        squared on the active torch device. Real and imaginary parts are squared separately. Only
        used when interlacing is enabled.
    chunks_processed_unshifted : int
        The total number of individual unshifted spectral estimates integrated into
        ``spectrum_accumulator_unshifted``.
    chunks_processed_shifted : int
        The total number of individual shifted spectral estimates integrated into
        ``spectrum_accumulator_shifted``. When interlacing is enabled, this count never exceeds
        ``chunks_processed_unshifted``.
    """

    channels: tuple[int, ...]

    freq: np.ndarray | None = None
    freq_unit: FrequencyUnits | None = None
    spectrum: np.ndarray | None = None
    spectrum_error: np.ndarray | None = None

    spectrum_accumulator_unshifted: Tensor | None = None
    spectrum_accumulator_shifted: Tensor | None = None
    error_accumulator_x_squared_unshifted: Tensor | None = None
    error_accumulator_x_squared_shifted: Tensor | None = None

    chunks_processed_unshifted: int = 0
    chunks_processed_shifted: int = 0

    @property
    def order(self) -> int:
        return len(self.channels)

    def reset_state(self):
        """Clears accumulators to prepare for a fresh calculation."""
        self.freq = None
        self.freq_unit = None
        self.spectrum = None
        self.spectrum_error = None
        self.spectrum_accumulator_unshifted = None
        self.spectrum_accumulator_shifted = None
        self.error_accumulator_x_squared_unshifted = None
        self.error_accumulator_x_squared_shifted = None
        self.chunks_processed_unshifted = 0
        self.chunks_processed_shifted = 0

    def initialize_arrays(self, runtime: RuntimeConfig) -> None:
        """Initialize frequency axis and units from the resolved runtime configuration."""

        if self.order == 1:
            self.freq = np.asarray([0.0])
        else:
            self.freq = runtime.freq_band

        self.freq_unit = runtime.freq_unit


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

    def get(self, channels: tuple[int, ...]) -> SpectrumResult:
        """Return the result for a channel tuple."""
        return self.results[channels]

    def add(self, result: SpectrumResult) -> None:
        """Add or replace a spectrum result using its channels."""
        self.results[result.channels] = result

    def reset_all_states(self) -> None:
        """Reset the mutable calculation state of all stored results."""
        for result in self.results.values():
            result.reset_state()

    def initialize_arrays(self, runtime: RuntimeConfig) -> None:
        """Initialize frequency axes and units for every stored spectrum result."""
        for result in self.results.values():
            result.initialize_arrays(runtime)
