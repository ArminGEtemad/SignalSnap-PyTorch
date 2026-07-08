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
    """Gather a(w3) for w3 = -(w1 + w2), returning shape (F, F, m)."""
    coeffs_by_freq = coeffs.permute(1, 2, 0)  # (N, 1, m)
    return coeffs_by_freq[target_indices, 0, :m]


def c1(a_w: Tensor) -> Tensor:
    """First-order cumulant.


    ``a_w`` has shape ``(m, F, 1)`` with ``F = runtime.band_end_idx - runtime.band_start_idx``. The
    returned tensor has shape ``(1,)`` and contains the DC component: index 0 for real FFT input, or
    the center frequency for full FFT input.
    """

    s1 = torch.mean(a_w, dim=0)
    dc_index = s1.shape[0] // 2
    result = s1[dc_index]

    return result


def c2_factorized(m: int, centered_x: Tensor, centered_y: Tensor) -> Tensor:
    s2 = m / (m - 1) * torch.mean(centered_x * centered_y, dim=0)
    return s2.squeeze(-1)


def c3_factorized(m: int, centered_x: Tensor, centered_y: Tensor, centered_z: Tensor) -> Tensor:
    s3 = m**2 / ((m - 1) * (m - 2)) * torch.mean(centered_x * centered_y * centered_z, dim=0)
    return s3.squeeze(-1)


def c4_factorized(
    m: int, centered_x: Tensor, centered_y: Tensor, centered_z: Tensor, centered_w: Tensor
) -> Tensor:
    s4 = (
        m**2
        / ((m - 1) * (m - 2) * (m - 3))
        * (
            (m + 1)
            * torch.matmul(
                centered_x * centered_y, (centered_z * centered_w).transpose(-1, -2)
            ).mean(dim=0)
            - (m - 1)
            * (
                torch.matmul(
                    (centered_x * centered_y).mean(dim=0),
                    (centered_z * centered_w).mean(dim=0).transpose(-1, -2),
                )
                + torch.matmul(centered_x, centered_z.transpose(-1, -2)).mean(dim=0)
                * torch.matmul(centered_y, centered_w.transpose(-1, -2)).mean(dim=0)
                + torch.matmul(centered_x, centered_w.transpose(-1, -2)).mean(dim=0)
                * torch.matmul(centered_y, centered_z.transpose(-1, -2)).mean(dim=0)
            )
        )
    )
    return s4
