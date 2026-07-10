# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemed, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
import warnings

import torch
from torch import Tensor
import numpy as np

from ..results import SpectrumResult
from .utils import FrequencyUnits


@dataclass(slots=True)
class SpectrumAccumulator:
    """Data container for the accumulation of spectral estimates.

    Stores the configuration metadata, accumulated hardware states, error buffers for a specific
    higher-order auto- or cross-spectrum calculation.

    Attributes
    ----------
    channels : tuple[int, ...]runtime configuration used by :func:`multichss.calculate_spectra`.
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


def accumulate_spectrum(
    accumulator: SpectrumAccumulator, single_spectrum: Tensor, shifted: bool = False
) -> None:
    """Accumulate one spectral estimate into the :class:`SpectrumAccumulator`.

    Adds the spectral estimate to the running sum and stores running sums of squared
    real and imaginary components used later to estimate the standard error of the mean. Spectral
    estimates and the squared component are accumulated separately for shifted and unshifted data.
    """

    if not shifted:
        if accumulator.spectrum_sum_unshifted is None:
            accumulator.spectrum_sum_unshifted = single_spectrum.clone()
        else:
            accumulator.spectrum_sum_unshifted += single_spectrum

        if accumulator.squared_sum_unshifted is None:
            accumulator.squared_sum_unshifted = torch.complex(
                torch.square(single_spectrum.real), torch.square(single_spectrum.imag)
            )
        else:
            accumulator.squared_sum_unshifted += torch.complex(
                torch.square(single_spectrum.real), torch.square(single_spectrum.imag)
            )

        accumulator.chunks_unshifted += 1

    else:
        if accumulator.spectrum_sum_shifted is None:
            accumulator.spectrum_sum_shifted = single_spectrum.clone()
        else:
            accumulator.spectrum_sum_shifted += single_spectrum

        if accumulator.squared_sum_shifted is None:
            accumulator.squared_sum_shifted = torch.complex(
                torch.square(single_spectrum.real), torch.square(single_spectrum.imag)
            )
        else:
            accumulator.squared_sum_shifted += torch.complex(
                torch.square(single_spectrum.real), torch.square(single_spectrum.imag)
            )

        accumulator.chunks_shifted += 1


def _check_accumulator_group(
    spectrum_sum: Tensor | None,
    squared_sum: Tensor | None,
    chunks: int,
) -> tuple[Tensor, Tensor, int] | None:
    if spectrum_sum is None:
        if squared_sum is not None or chunks != 0:
            raise RuntimeError("Spectrum accumulator state is inconsistent.")
        return None

    if squared_sum is None or chunks <= 0:
        raise RuntimeError("A spectrum accumulator state is inconsistent.")

    return spectrum_sum, squared_sum, chunks


def _finalize_accumulator_group(
    spectrum_sum: Tensor,
    squared_sum: Tensor,
    chunks_processed: int,
) -> tuple[Tensor, Tensor | None]:
    """
    Compute spectrum mean and error for each specified group, e.g. for shifted and unshifted data.
    """
    mean = spectrum_sum / chunks_processed

    if chunks_processed < 2:
        return mean, None

    mean_squared = squared_sum / chunks_processed
    variance = (chunks_processed / (chunks_processed - 1)) * (
        mean_squared
        - torch.complex(
            torch.square(mean.real),
            torch.square(mean.imag),
        )
    )

    var_re = torch.clamp_min(variance.real, 0.0)
    var_im = torch.clamp_min(variance.imag, 0.0)

    error = torch.complex(
        torch.sqrt(var_re / chunks_processed),
        torch.sqrt(var_im / chunks_processed),
    )

    return mean, error


def finalize_result(accumulator: SpectrumAccumulator) -> SpectrumResult:
    """Create a finalized result from an accumulator."""

    unshifted_group = _check_accumulator_group(
        accumulator.spectrum_sum_unshifted,
        accumulator.squared_sum_unshifted,
        accumulator.chunks_unshifted,
    )

    if unshifted_group is None:
        raise RuntimeError(
            f"Cannot finalize channels {accumulator.channels}: no spectra were accumulated."
        )

    groups = [unshifted_group]

    shifted_group = _check_accumulator_group(
        accumulator.spectrum_sum_shifted,
        accumulator.squared_sum_shifted,
        accumulator.chunks_shifted,
    )
    if shifted_group is not None:
        groups.append(shifted_group)

    total_chunks = 0
    total_spectrum = groups[0][0].clone().zero_()

    errors: list[Tensor] = []

    for spectrum_sum, squared_sum, chunks_processed in groups:
        total_spectrum += spectrum_sum
        total_chunks += chunks_processed

        _, error = _finalize_accumulator_group(
            spectrum_sum=spectrum_sum, squared_sum=squared_sum, chunks_processed=chunks_processed
        )
        if error is not None:
            errors.append(error)

    spectrum = (total_spectrum / total_chunks).cpu().resolve_conj().numpy()

    if not errors:
        spectrum_error = None
        warnings.warn(
            "Need at least two spectral estimates for an error estimation.",
            RuntimeWarning,
            stacklevel=3,
        )
    elif len(errors) == 1:
        spectrum_error = errors[0].cpu().resolve_conj().numpy()
    else:
        error_re = errors[0].real
        error_im = errors[0].imag

        for error in errors[1:]:
            error_re = torch.maximum(error_re, error.real)
            error_im = torch.maximum(error_im, error.imag)

        spectrum_error = torch.complex(error_re, error_im).cpu().resolve_conj().numpy()

    return SpectrumResult(
        channels=accumulator.channels,
        freq=accumulator.freq,
        freq_unit=accumulator.freq_unit,
        spectrum=spectrum,
        spectrum_error=spectrum_error,
    )
