# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemed, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from .cumulants import a_w3_gen, c1, c2, c3, c4, calc_a_w3, index_generation_to_aw_3
from .fft import WindowBuffer
from .planning import RuntimeConfig
from .utils import S3Calcs


@dataclass(slots=True)
class ThirdOrderCache:
    """Reusable tensors needed to assemble third-order Fourier coefficient products."""

    a_w3_init: Tensor
    indices: Tensor


@dataclass(slots=True)
class IntermediateSliceBuffer:
    """Stores precomputed intermediate results used in compute_single_spectrum()."""

    f_min_idx: int
    f_max_idx: int
    m: int
    s3_calc: S3Calcs
    third_order_cache: ThirdOrderCache | None = None
    coeffs_by_channel: dict[int, Tensor] = field(default_factory=dict)

    _coeffs_by_channel_band: dict[int, Tensor] = field(default_factory=dict)
    _coeffs_by_channel_half_band: dict[int, Tensor] = field(default_factory=dict)
    _third_order_factor_by_channel: dict[int, Tensor] = field(default_factory=dict)

    def coeffs_by_channel_band(self, channel: int) -> Tensor:
        if channel not in self._coeffs_by_channel_band:
            self._coeffs_by_channel_band[channel] = self.coeffs_by_channel[channel][
                :, self.f_min_idx : self.f_max_idx, :
            ]
        return self._coeffs_by_channel_band[channel]

    def coeffs_by_channel_half_band(self, channel: int) -> Tensor:
        if channel not in self._coeffs_by_channel_half_band:
            self._coeffs_by_channel_half_band[channel] = self.coeffs_by_channel[channel][
                :, self.f_min_idx : self.f_max_idx // 2, :
            ]
        return self._coeffs_by_channel_half_band[channel]

    def third_order_factor(self, channel: int) -> Tensor:
        if channel not in self._third_order_factor_by_channel:
            if self.third_order_cache is None:
                raise ValueError("Third-order spectra require third_order_cache.")

            coeffs_perm = self.coeffs_by_channel[channel].permute((1, 2, 0))

            if self.s3_calc == "1/2":
                coeffs_perm = torch.cat(
                    (coeffs_perm, torch.conj(coeffs_perm[1:, :, :].flip([0]))), dim=0
                )

            self._third_order_factor_by_channel[channel] = calc_a_w3(
                coeffs_perm,
                self.f_max_idx,
                self.m,
                self.third_order_cache.a_w3_init.clone(),
                self.third_order_cache.indices,
            )

        return self._third_order_factor_by_channel[channel]


def build_third_order_cache(runtime: RuntimeConfig) -> ThirdOrderCache:
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


def build_intermediate_slice_buffer(
    runtime: RuntimeConfig,
    coeffs_by_channel: dict[int, Tensor],
    third_order_cache: ThirdOrderCache | None,
) -> IntermediateSliceBuffer:
    return IntermediateSliceBuffer(
        f_min_idx=runtime.f_min_idx,
        f_max_idx=runtime.f_max_idx,
        coeffs_by_channel=coeffs_by_channel,
        m=runtime.m,
        s3_calc=runtime.s3_calc,
        third_order_cache=third_order_cache,
    )


def compute_single_spectrum(
    channels: tuple[int, ...],
    intermediate_buffer: IntermediateSliceBuffer,
    window_buffer: WindowBuffer,
    runtime: RuntimeConfig,
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

    if order == 1:
        single_spectrum = c1(
            runtime.use_full_fft,
            intermediate_buffer.coeffs_by_channel[channels[0]],
        )

    elif order == 2:
        single_spectrum = c2(
            runtime.m,
            intermediate_buffer.coeffs_by_channel_band(channels[0]),
            intermediate_buffer.coeffs_by_channel_band(channels[1]),
        )

    elif order == 3:
        a_w1 = intermediate_buffer.coeffs_by_channel_half_band(channels[0])
        if runtime.s3_calc == "1/2":
            a_w1 = torch.cat((a_w1[:, 1:, :].flip([1]).conj(), a_w1), dim=1)

        a_w2 = intermediate_buffer.coeffs_by_channel_half_band(channels[1])
        a_w3 = intermediate_buffer.third_order_factor(channels[2])

        single_spectrum = c3(runtime.m, a_w1, a_w2, a_w3)

    elif order == 4:
        single_spectrum = c4(
            runtime.m,
            intermediate_buffer.coeffs_by_channel_band(channels[0]),
            intermediate_buffer.coeffs_by_channel_band(channels[1]),
            intermediate_buffer.coeffs_by_channel_band(channels[2]),
            intermediate_buffer.coeffs_by_channel_band(channels[3]),
        )

    else:
        raise ValueError(f"Unsupported spectrum order: {order}.")

    return torch.conj(single_spectrum / window_buffer.norm(order))
