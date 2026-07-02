# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemed, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from .results import SpectrumResult


def accumulate_spectrum(
    result: SpectrumResult, single_spectrum: Tensor, shifted: bool = False
) -> None:
    """Accumulate one spectral estimate into a result object.

    Adds the spectrum to the running mean accumulator and stores running sums of squared real and
    imaginary components used later to estimate the standard error of the mean.
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


def finalize_result(result: SpectrumResult) -> None:
    """Finalize accumulated spectra and error estimates on a result object."""

    if result.spectrum_accumulator_unshifted is None:
        result.spectrum = None
        result.spectrum_error = None
        return

    if result.chunks_processed_unshifted == 0:
        raise ValueError("Cannot finalize result without processed chunks.")

    if result.spectrum_accumulator_shifted is None:
        result.spectrum_accumulator_unshifted /= result.chunks_processed_unshifted
        result.spectrum = result.spectrum_accumulator_unshifted.cpu().resolve_conj().numpy()
    else:
        unified_spectrum = (
            result.spectrum_accumulator_unshifted + result.spectrum_accumulator_shifted
        ) / (result.chunks_processed_unshifted + result.chunks_processed_shifted)
        result.spectrum = unified_spectrum.cpu().resolve_conj().numpy()

        result.spectrum_accumulator_unshifted /= result.chunks_processed_unshifted
        result.spectrum_accumulator_shifted /= result.chunks_processed_shifted

    assert result.error_accumulator_x_squared_unshifted is not None

    if result.chunks_processed_unshifted == 1:
        result.spectrum_error = None
        print("Need at least two unshifted spectra estimates for an error estimation.")
    else:
        var_factor_unshifted = result.chunks_processed_unshifted / (
            result.chunks_processed_unshifted - 1
        )
        result.error_accumulator_x_squared_unshifted /= result.chunks_processed_unshifted
        spectrum_variance_unshifted = var_factor_unshifted * (
            result.error_accumulator_x_squared_unshifted
            - torch.complex(
                torch.square(result.spectrum_accumulator_unshifted.real),
                torch.square(result.spectrum_accumulator_unshifted.imag),
            )
        )
        var_re_unshifted = torch.clamp_min(spectrum_variance_unshifted.real, 0.0)
        var_im_unshifted = torch.clamp_min(spectrum_variance_unshifted.imag, 0.0)

        spectrum_error_unshifted = (
            torch.complex(
                torch.sqrt(var_re_unshifted / result.chunks_processed_unshifted),
                torch.sqrt(var_im_unshifted / result.chunks_processed_unshifted),
            )
            .cpu()
            .resolve_conj()
            .numpy()
        )
        if result.chunks_processed_shifted >= 2:
            assert result.error_accumulator_x_squared_shifted is not None
            assert result.spectrum_accumulator_shifted is not None

            var_factor_shifted = result.chunks_processed_shifted / (
                result.chunks_processed_shifted - 1
            )
            result.error_accumulator_x_squared_shifted /= result.chunks_processed_shifted
            spectrum_variance_shifted = var_factor_shifted * (
                result.error_accumulator_x_squared_shifted
                - torch.complex(
                    torch.square(result.spectrum_accumulator_shifted.real),
                    torch.square(result.spectrum_accumulator_shifted.imag),
                )
            )
            var_re_shifted = torch.clamp_min(spectrum_variance_shifted.real, 0.0)
            var_im_shifted = torch.clamp_min(spectrum_variance_shifted.imag, 0.0)

            spectrum_error_shifted = (
                torch.complex(
                    torch.sqrt(var_re_shifted / result.chunks_processed_shifted),
                    torch.sqrt(var_im_shifted / result.chunks_processed_shifted),
                )
                .cpu()
                .resolve_conj()
                .numpy()
            )

            error_re = np.maximum(
                np.real(spectrum_error_unshifted), np.real(spectrum_error_shifted)
            )
            error_im = np.maximum(
                np.imag(spectrum_error_unshifted), np.imag(spectrum_error_shifted)
            )
            result.spectrum_error = error_re + 1j * error_im
        else:
            if result.chunks_processed_shifted == 1:
                print(
                    "Only using spectrum error from the unshifted spectral estimates, since there"
                    "is only a single shifted estimate."
                )
            result.spectrum_error = spectrum_error_unshifted
