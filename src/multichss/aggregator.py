# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemed, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

import torch
from torch import Tensor

from .results import SpectrumResult


def accumulate_spectrum(
    result: SpectrumResult, single_spectrum: Tensor, shifted: bool = False
) -> None:
    """Accumulate one spectral estimate into the result object.

    Adds the spectral estimate to the running mean accumulator and stores running sums of squared
    real and imaginary components used later to estimate the standard error of the mean. Spectral
    estimates and the squared component are accumulated separately for shifted and unshifted data.
    """

    if result.freq is None:
        raise ValueError("SpectrumResult must be initialized before accumulation.")

    if not shifted:
        if result.spectrum_accumulator_unshifted is None:
            result.spectrum_accumulator_unshifted = single_spectrum.clone()
        else:
            result.spectrum_accumulator_unshifted += single_spectrum

        if result.error_accumulator_x_squared_unshifted is None:
            result.error_accumulator_x_squared_unshifted = torch.complex(
                torch.square(single_spectrum.real), torch.square(single_spectrum.imag)
            )
        else:
            result.error_accumulator_x_squared_unshifted += torch.complex(
                torch.square(single_spectrum.real), torch.square(single_spectrum.imag)
            )

        result.chunks_processed_unshifted += 1

    else:
        if result.spectrum_accumulator_shifted is None:
            result.spectrum_accumulator_shifted = single_spectrum.clone()
        else:
            result.spectrum_accumulator_shifted += single_spectrum

        if result.error_accumulator_x_squared_shifted is None:
            result.error_accumulator_x_squared_shifted = torch.complex(
                torch.square(single_spectrum.real), torch.square(single_spectrum.imag)
            )
        else:
            result.error_accumulator_x_squared_shifted += torch.complex(
                torch.square(single_spectrum.real), torch.square(single_spectrum.imag)
            )

        result.chunks_processed_shifted += 1

def _check_result_group(
    spectrum_accumulator: Tensor | None,
    error_accumulator_x_squared: Tensor | None,
    chunks_processed: int,
) -> tuple[Tensor, Tensor, int] | None:
    if spectrum_accumulator is None:
        return None

    if error_accumulator_x_squared is None or chunks_processed == 0:
        raise RuntimeError("A spectrum result state is inconsistent.")

    return spectrum_accumulator, error_accumulator_x_squared, chunks_processed


def _finalize_result_group(
    spectrum_accumulator: Tensor,
    error_accumulator_x_squared: Tensor,
    chunks_processed: int,
) -> tuple[Tensor, Tensor | None]:
    """
    Compute spectrum mean and error for each specified group, e.g. for shifted and unshifted data.
    """
    mean = spectrum_accumulator / chunks_processed

    if chunks_processed < 2:
        return mean, None

    mean_squared = error_accumulator_x_squared / chunks_processed
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


def finalize_result(result: SpectrumResult) -> None:
    """Finalize accumulated spectra and error estimates on a result object."""

    unshifted_group = _check_result_group(
        result.spectrum_accumulator_unshifted,
        result.error_accumulator_x_squared_unshifted,
        result.chunks_processed_unshifted,
    )

    if unshifted_group is None:
        result.spectrum = None
        result.spectrum_error = None
        return

    groups = [unshifted_group]

    shifted_group = _check_result_group(
        result.spectrum_accumulator_shifted,
        result.error_accumulator_x_squared_shifted,
        result.chunks_processed_shifted,
    )
    if shifted_group is not None:
        groups.append(shifted_group)

    total_chunks = 0
    total_spectrum = groups[0][0].clone().zero_()

    errors: list[Tensor] = []

    for spectrum_sum, squared_sum, chunks_processed in groups:
        total_spectrum += spectrum_sum
        total_chunks += chunks_processed

        _, error = _finalize_result_group(
            spectrum_accumulator=spectrum_sum,
            error_accumulator_x_squared=squared_sum,
            chunks_processed=chunks_processed,
        )
        if error is not None:
            errors.append(error)

    result.spectrum = (total_spectrum / total_chunks).cpu().resolve_conj().numpy()

    if not errors:
        result.spectrum_error = None
        print("Need at least two spectral estimates for an error estimation.")
    elif len(errors) == 1:
        result.spectrum_error = errors[0].cpu().resolve_conj().numpy()
    else:
        error_re = errors[0].real
        error_im = errors[0].imag

        for error in errors[1:]:
            error_re = torch.maximum(error_re, error.real)
            error_im = torch.maximum(error_im, error.imag)

        result.spectrum_error = torch.complex(error_re, error_im).cpu().resolve_conj().numpy()
