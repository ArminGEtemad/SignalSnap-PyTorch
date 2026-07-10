# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemed, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

import torch
from torch import Tensor


def build_s3_target_indices(axis_indices: Tensor, fft_freq_count: int) -> tuple[Tensor, Tensor]:
    """Map output-axis bins (w1, w2) to the shifted-FFT bin for w3 = -(w1 + w2).
    ``safe_indices`` is a 2D grid based on (w1, w2) which includes the corresponding indices of the
    shifted fft array for w3. Every w1 + w2 that is out of bounds will receive index 0.
    ``valid_mask`` is a 2D grid containing ``True`` when w1 + w2 is a valid frequency and ``False``
    otherwise, which can later be used to delete any values that were computed out of bounds.
    """

    zero_idx = fft_freq_count // 2
    axis_offsets = axis_indices - zero_idx

    target_offsets = -(axis_offsets[:, None] + axis_offsets[None, :])
    target_indices = target_offsets + zero_idx

    valid_mask = (target_indices >= 0) & (target_indices < fft_freq_count)
    safe_indices = torch.where(valid_mask, target_indices, torch.zeros_like(target_indices))
    return safe_indices, valid_mask


def gather_s3_third_factor(coeffs: Tensor, target_indices: Tensor, m: int) -> Tensor:
    """Gather a(w3) for w3 = -(w1 + w2), returning shape (m, F, F)."""
    return coeffs[:m, target_indices]


def _mean_outer(m: int, a: Tensor, b: Tensor) -> Tensor:
    """
    Computes the average outer product over the window axis (frist axis).

    ``a`` and ``b`` are Tensors of shape ``(m, F)``.
    Returns a ``(F, F)`` Tensor: result[f, g] = (1/m)* sum_i a[i, f] * b[i, g], so for every window
    with index ``i`` the ``(F, F)`` grid is computed and at the end the windows are averaged.
    """
    return torch.einsum("mf,mg->fg", a, b) / m


def c2_factorized(m: int, centered_x: Tensor, centered_y: Tensor) -> Tensor:
    """Second-order cumulant.

        C2(x, y) = m/(m-1) * ((x-x.mean)*(y-y.mean)).mean

    ``centered_x`` and ``centered_y`` are the Fourier coefficients of the specified user band with
    the mean (calculated over the m windows) subtracted and have shape ``(m, F)``, with
    ``F=runtime.band_max_idx - runtime.band_min_idx``.

    Returns a ``(F,)`` shaped spectrum.
    """

    s2 = m / (m - 1) * torch.mean(centered_x * centered_y, dim=0)
    return s2


def c3_factorized(m: int, centered_x: Tensor, centered_y: Tensor, centered_z: Tensor) -> Tensor:
    """Third-order cumulant.

        C3(x, y) = (m^2)/((m-1)(m-2)) * ((x-x.mean)*(y-y.mean)*(z-z.mean)).mean

    ``centered_x`` and ``centered_y`` are the Fourier coefficients of the specified user band with
    the mean (calculated over the m windows) subtracted and have shape ``(m, F)``, with
    ``F=runtime.band_max_idx - runtime.band_min_idx``.
    ``centered_z`` is the precomputed, centered ``(F, F)`` grid of Fourier coefficients for the
    third component of the cumulant.

    Returns a ``(F, F)`` shaped spectrum.
    """

    s3 = (
        m**2
        / ((m - 1) * (m - 2))
        * torch.mean(centered_x[:, None, :] * centered_y[:, :, None] * centered_z, dim=0)
    )
    return s3


def c4_factorized(
    m: int, centered_x: Tensor, centered_y: Tensor, centered_z: Tensor, centered_w: Tensor
) -> Tensor:
    """Fourth-order cumulant.

        C4(x, y) = (m^2)/((m-1)(m-2)(m-3))
                    * ((m+1) * ((x-x.mean)*(y-y.mean)*(z-z.mean)*(w-w.mean)).mean
                        -(m-1) *(
                                (((x-x.mean)*(y-y.mean)).mean * ((z-z.mean)*(w-w.mean)).mean)
                                +(((x-x.mean)*(z-z.mean)).mean * ((y-y.mean)*(w-w.mean)).mean)
                                +(((x-x.mean)*(w-w.mean)).mean * ((y-y.mean)*(z-z.mean)).mean)
                        )

                    )

    ``centered_x``, ``centered_y``, ``centered_z``, and ``centered_w`` are the Fourier coefficients
    of the specified user band with the mean (calculated over the m windows) subtracted and have
    shape ``(m, F)``, with ``F=runtime.band_max_idx - runtime.band_min_idx``.

    Returns a ``(F, F)`` shaped spectrum. ``centered_x`` and ``centered_y`` are varied in the first
    component of the result and ``centered_z`` and ``centered_z`` are varied in the second component
    of the result.
    """

    centered_xy = centered_x * centered_y
    centered_zw = centered_z * centered_w
    s4 = (
        m**2
        / ((m - 1) * (m - 2) * (m - 3))
        * (
            (m + 1) * _mean_outer(m, centered_xy, centered_zw)
            - (m - 1)
            * (
                torch.outer(centered_xy.mean(dim=0), centered_zw.mean(dim=0))
                + _mean_outer(m, centered_x, centered_z) * _mean_outer(m, centered_y, centered_w)
                + _mean_outer(m, centered_x, centered_w) * _mean_outer(m, centered_y, centered_z)
            )
        )
    )
    return s4
