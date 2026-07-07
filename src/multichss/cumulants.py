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
    """Map output-axis bins (w1, w2) to the shifted-FFT bin for w3 = -(w1 + w2)."""
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


def c2(m: int, a_w1: Tensor, a_w2: Tensor) -> Tensor:
    """Second-order cumulant is the covariance.
    
    ``a_w1`` and ``a_w2`` must both have the same shape ``(m, X, 1)``. The returned tensor has shape
    ``(X,)``.
    """

    a_w2_star = torch.conj(a_w2)

    factor = m / (m - 1)
    term_1 = torch.mean(a_w1 * a_w2_star, dim=0)
    term_2 = torch.mean(a_w1, dim=0) * torch.mean(a_w2_star, dim=0)
    s2 = factor * (term_1 - term_2)
    return s2.squeeze(-1)


def c3(m: int, a_w1: Tensor, a_w2: Tensor, a_w3: Tensor) -> Tensor:
    """
    Third-order cumulant::

        C_3 = m^2 / [(m - 1) * (m - 2)] * {
                < a_w1 * a_w2 * a_w3 >
                - < a_w1 >< a_w2 * a_w3 > - < a_w1 * a_w2 >< a_w3 > - < a_w1 * a_w3 >< a_w2 >
                + 2 < a_w1 >< a_w2 >< a_w3 >
            }
    
    with w3 = - w1 - w2 and as before <...> denotes the mean and the factor m^2 / (m - 1)(m - 2) is
    the unbiased estimator for the third order cumulant. ``m`` is the number of windows per spectral 
    estimate. The estimator requires ``m > 2``.
    (see arXiv:1904.12154)
    """

    a_w1_modified = a_w1.transpose(-1, -2)
    a_w1_modified_stacked = a_w1_modified.expand(
        a_w1_modified.size(0), a_w2.size(1), a_w1_modified.size(2)
    )

    a_w2_modified_stacked = a_w2.expand((a_w2.size(0), a_w2.size(1), a_w1.size(1)))

    a_w3_modified = a_w3.permute(2, 0, 1)

    d_12 = a_w1_modified_stacked * a_w2_modified_stacked
    d_13 = a_w1_modified_stacked * a_w3_modified
    d_23 = a_w2_modified_stacked * a_w3_modified
    d_123 = d_12 * a_w3_modified

    d_means = [
        torch.mean(d, dim=0)
        for d in [
            a_w1_modified_stacked,
            a_w2_modified_stacked,
            a_w3_modified,
            d_12,
            d_13,
            d_23,
            d_123,
        ]
    ]

    d_1_mean, d_2_mean, d_3_mean, d_12_mean, d_13_mean, d_23_mean, d_123_mean = d_means
    s3 = (
        m**2
        / ((m - 1) * (m - 2))
        * (
            d_123_mean
            - d_12_mean * d_3_mean
            - d_13_mean * d_2_mean
            - d_23_mean * d_1_mean
            + 2 * d_1_mean * d_2_mean * d_3_mean
        )
    )

    return s3


def c4(m: int, a_w1: Tensor, a_w2: Tensor, a_w3: Tensor, a_w4: Tensor) -> Tensor:
    """
    Fourth-order cumulant::

        C_4 = m^2 / [(m - 1) * (m - 2) * (m - 3)] * {
                (m + 1) * <(a_w1 - <a_w1>) * (a_w2 - <a_w2>) * (a_w3 - <a_w3>) * (a_w4 - <a_w4>)>
                -(m - 1) * [
                           <(a_w1 - <a_w1>) * (a_w2 - <a_w2>)> * <(a_w3 - <a_w3>) * (a_w4 - <a_w4>)>
                           + 2 o.p.
                ]
            }
    
    <...> denotes the mean. ``m`` is the number of windows per spectral 
    estimate. The estimator requires ``m > 3``.
    (see arXiv:1904.12154)

    All input tensors must have shape ``(m, F, 1)``, where 
    ``F = runtime.band_end_idx - runtime.band_start_idx`` is the selected frequency-band length.
    The returned tensor has shape ``(F, F)``.

    The second and fourth inputs are conjugated internally, matching the convention used for 
    fourth-order auto- and cross-spectra.
    """

    # --- for a better readability ---
    x = a_w1
    y = torch.conj(a_w2)
    z = a_w3
    w = torch.conj(a_w4)
    # --------------------------------

    x_mean = x - x.mean(dim=0, keepdim=True)
    y_mean = y - y.mean(dim=0, keepdim=True)
    z_mean = z - z.mean(dim=0, keepdim=True)
    w_mean = w - w.mean(dim=0, keepdim=True)

    # Compute product and various partial means
    xyzw = torch.matmul((x_mean * y_mean), (z_mean * w_mean).transpose(-1, -2))
    xyzw_mean = xyzw.mean(dim=0)

    xy_mean = (x_mean * y_mean).mean(dim=0)
    zw_mean = (z_mean * w_mean).mean(dim=0)
    xy_zw_mean = torch.matmul(xy_mean, zw_mean.transpose(-1, -2))

    xz_mean = torch.matmul(x_mean, z_mean.transpose(-1, -2)).mean(dim=0)
    yw_mean = torch.matmul(y_mean, w_mean.transpose(-1, -2)).mean(dim=0)
    xz_yw_mean = xz_mean * yw_mean

    xw_mean = torch.matmul(x_mean, w_mean.transpose(-1, -2)).mean(dim=0)
    yz_mean = torch.matmul(y_mean, z_mean.transpose(-1, -2)).mean(dim=0)
    xw_yz_mean = xw_mean * yz_mean

    # Final combination
    s4 = (m**2 / ((m - 1) * (m - 2) * (m - 3))) * (
        (m + 1) * xyzw_mean - (m - 1) * (xy_zw_mean + xz_yw_mean + xw_yz_mean)
    )

    return s4
