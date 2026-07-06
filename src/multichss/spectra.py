# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemed, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .cumulants import a_w3_gen, c1, c2, c3, c4, calc_a_w3, index_generation_to_aw_3
from .fft import WindowBuffer
from .planning import RuntimeConfig


@dataclass(slots=True)
class ThirdOrderCache:
    """Reusable tensors needed to assemble third-order Fourier coefficient products."""
    a_w3_init: Tensor
    indices: Tensor


def build_third_order_cache(runtime: RuntimeConfig) -> ThirdOrderCache | None:
    """Precompute third-order index and work tensors for the active runtime configuration."""
    
    return ThirdOrderCache(
        a_w3_init=a_w3_gen(
            runtime.s3_calc,
            runtime.f_max_idx,
            runtime.m,
            device=runtime.device,
            dtype=runtime.complex_dtype,
        ),
        indices=index_generation_to_aw_3(runtime.s3_calc, runtime.f_max_idx, device=runtime.device),
    )


def compute_single_spectrum(
    channels: tuple[int, ...],
    coeffs_by_channel: dict[int, Tensor],
    window_buffer: WindowBuffer,
    runtime: RuntimeConfig,
    third_order_cache: ThirdOrderCache | None = None,
) -> Tensor:
    """Compute one normalized spectrum from channel Fourier coefficients.

    Dispatches to the cumulant implementation for orders 1 through 4 and applies the matching window
    normalization.
    
    ``coeffs_by_channel`` maps each channel index to FFT coefficients with shape ``(m, K, 1)``,
    where ``K = N`` for full FFTs and ``K = N // 2 + 1`` for real FFTs. The selected frequency band
    has length ``F = runtime.f_max_idx - runtime.f_min_idx``.

    Returns a conjugated, window-normalized spectrum. Output shape depends on order: order 1 returns
    ``(1,)``, order 2 returns ``(F,)``, order 3 returns ``(H, H)`` for ``s3_calc="1/4"`` or
    ``(H, 2 * H - 1)`` for ``s3_calc="1/2"`` with ``H = runtime.f_max_idx // 2``, and order 4
    returns ``(F, F)``.

    Third-order shapes assume the calculation starts at ``runtime.f_min_idx == 0``.
    """
    order = len(channels)
    f_min_idx = runtime.f_min_idx
    f_max_idx = runtime.f_max_idx

    if order == 1:
        coeffs = coeffs_by_channel[channels[0]]
        single_spectrum = c1(runtime.use_full_fft, coeffs)

    elif order == 2:
        a_w1 = coeffs_by_channel[channels[0]][:, f_min_idx:f_max_idx, :]
        a_w2 = coeffs_by_channel[channels[1]][:, f_min_idx:f_max_idx, :]
        single_spectrum = c2(runtime.m, a_w1, a_w2)
        
    elif order == 3:
        if third_order_cache is None:
            raise ValueError("Third-order spectra require third_order_cache.")

        a_w1 = coeffs_by_channel[channels[0]][:, f_min_idx : f_max_idx // 2, :]
        if channels[0] == channels[1]:
            a_w2 = a_w1
        else:
            a_w2 = coeffs_by_channel[channels[1]][:, f_min_idx : f_max_idx // 2, :]

        coeffs_gpu_p = coeffs_by_channel[channels[2]].permute((1, 2, 0))

        if runtime.s3_calc == "1/2":
            a_w1 = torch.cat((a_w1[:, 1:, :].flip([1]).conj(), a_w1), dim=1)
            coeffs_gpu_p = torch.cat(
                (coeffs_gpu_p, torch.conj(coeffs_gpu_p[1:, :, :].flip([0]))),
                dim=0,
            )

        a_w3 = calc_a_w3(
            coeffs_gpu_p,
            f_max_idx,
            runtime.m,
            third_order_cache.a_w3_init,
            third_order_cache.indices,
        )

        single_spectrum = c3(runtime.m, a_w1, a_w2, a_w3)

    elif order == 4:
        a_w1 = coeffs_by_channel[channels[0]][:, f_min_idx:f_max_idx, :]
        a_w2 = coeffs_by_channel[channels[1]][:, f_min_idx:f_max_idx, :]
        a_w3 = coeffs_by_channel[channels[2]][:, f_min_idx:f_max_idx, :]
        a_w4 = coeffs_by_channel[channels[3]][:, f_min_idx:f_max_idx, :]

        single_spectrum = c4(runtime.m, a_w1, a_w2, a_w3, a_w4)

    else:
        raise ValueError(f"Unsupported spectrum order: {order}.")

    return torch.conj(single_spectrum / window_buffer.norm(order))
