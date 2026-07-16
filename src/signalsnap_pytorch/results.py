# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemad, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

import numpy as np

from ._core.utils import FrequencyUnits as _FrequencyUnits


@dataclass(frozen=True, slots=True)
class SpectrumResult:
    """Data container for the results of a single spectral calculation.

    Stores the configuration metadata, and final computed results for a specific higher-order auto-
    or cross-spectrum calculation.

    Example: S^3(w1_i, w2_j) would be stored in ``SpectrumResult.spectrum[i, j]``.

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
    freq_unit: _FrequencyUnits
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

    def __post_init__(self) -> None:
        for channels, result in self.results.items():
            if not isinstance(result, SpectrumResult):
                raise TypeError(
                    "SpectrumResultStore values must be SpectrumResult objects; "
                    f"received {type(result).__name__} for key {channels}."
                )

            if channels != result.channels:
                raise ValueError(
                    f"Result key {channels} does not match "
                    f"SpectrumResult.channels {result.channels}."
                )

    def __contains__(self, channels: object) -> bool:
        """Return whether a result exists for a channel tuple."""
        return channels in self.results

    def __iter__(self) -> Iterator[SpectrumResult]:
        return iter(self.results.values())

    def __len__(self) -> int:
        """Return the number of stored results."""
        return len(self.results)

    def __getitem__(self, channels: tuple[int, ...]) -> SpectrumResult:
        """Return a result for a channel tuple."""
        return self.results[channels]

    def add(self, result: SpectrumResult) -> None:
        """Add or replace a spectrum result using its channels."""
        if not isinstance(result, SpectrumResult):
            raise TypeError(
                "SpectrumResultStore only accepts SpectrumResult objects; "
                f"received {type(result).__name__}."
            )

        self.results[result.channels] = result

    def select(self, channels: Iterable[tuple[int, ...]]) -> SpectrumResultStore:
        """Return a new store containing the selected results.

        The new store shares its SpectrumResult objects with this store.
        """
        selected: dict[tuple[int, ...], SpectrumResult] = {}

        for channel_tuple in channels:
            try:
                selected[channel_tuple] = self.results[channel_tuple]
            except KeyError as exc:
                raise ValueError(
                    f"No spectrum result exists for channels {channel_tuple}."
                ) from exc

        return SpectrumResultStore(results=selected)

    def select_by_order(self, order: int) -> SpectrumResultStore:
        """Return all results with the specified spectrum order.

        An empty store is returned when no matching results exist.
        """
        if isinstance(order, (bool, np.bool_)) or not isinstance(order, (int, np.integer)):
            raise TypeError("order must be an integer.")

        order = int(order)

        if not 1 <= order <= 4:
            raise ValueError("order must be between 1 and 4.")

        return self.select([result.channels for result in self if result.order == order])

    def select_by_channel(self, channel: int) -> SpectrumResultStore:
        """Return all results involving the specified data channel.

        A result matches when ``channel`` occurs anywhere in its channel tuple.
        An empty store is returned when no matching results exist.
        """
        if isinstance(channel, (bool, np.bool_)) or not isinstance(channel, (int, np.integer)):
            raise TypeError("channel must be an integer.")

        channel = int(channel)

        if channel < 0:
            raise ValueError("channel must be nonnegative.")

        return self.select([result.channels for result in self if channel in result.channels])
