# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemed, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import torch
from torch import Tensor

from .cumulants import build_s3_target_indices, c1, c2, c3, c4, gather_s3_third_factor
from .fft import WindowBuffer
from .planning import RuntimeConfig


@dataclass(slots=True)
class ThirdOrderIndexCache:
    target_indices: Tensor
    valid_mask: Tensor


@dataclass(slots=True)
class ThirdOrderPrepared:
    a_w1: Tensor
    a_w2: Tensor
    a_w3: Tensor
    valid_mask: Tensor


@dataclass(slots=True)
class IntermediateSliceBuffer:
    """Stores precomputed intermediate results used in compute_single_spectrum()."""

    band_start_idx: int
    band_end_idx: int
    m: int
    fft_freq_count: int
    coeffs_by_channel: dict[int, Tensor] = field(default_factory=dict)
    third_order_cache: ThirdOrderIndexCache | None = None

    _coeffs_by_channel_band: dict[int, Tensor] = field(default_factory=dict)
    _third_order_prepared_by_channels: dict[tuple[int, int, int], ThirdOrderPrepared] = field(
        default_factory=dict
    )

    def coeffs_by_channel_band(self, channel: int) -> Tensor:
        if channel not in self._coeffs_by_channel_band:
            self._coeffs_by_channel_band[channel] = self.coeffs_by_channel[channel][
                :, self.band_start_idx : self.band_end_idx, :
            ]
        return self._coeffs_by_channel_band[channel]

    def third_order_prepared(self, channels: tuple[int, int, int]) -> ThirdOrderPrepared:
        if channels not in self._third_order_prepared_by_channels:
            if self.third_order_cache is None:
                raise ValueError("Third-order spectra require third_order_cache.")

            a_w1 = self.coeffs_by_channel_band(channels[0])
            a_w2 = self.coeffs_by_channel_band(channels[1])
            a_w3 = gather_s3_third_factor(
                self.coeffs_by_channel[channels[2]],
                self.third_order_cache.target_indices,
                self.m,
            )

            self._third_order_prepared_by_channels[channels] = ThirdOrderPrepared(
                a_w1=a_w1,
                a_w2=a_w2,
                a_w3=a_w3,
                valid_mask=self.third_order_cache.valid_mask,
            )

        return self._third_order_prepared_by_channels[channels]


def build_third_order_cache(runtime: RuntimeConfig) -> ThirdOrderIndexCache:
    axis_indices = torch.arange(
        runtime.band_start_idx,
        runtime.band_end_idx,
        device=runtime.device,
    )
    target_indices, valid_mask = build_s3_target_indices(axis_indices, runtime.fft_freq_count)
    return ThirdOrderIndexCache(target_indices=target_indices, valid_mask=valid_mask)


def build_intermediate_slice_buffer(
    runtime: RuntimeConfig,
    coeffs_by_channel: dict[int, Tensor],
    third_order_cache: ThirdOrderIndexCache | None,
) -> IntermediateSliceBuffer:
    return IntermediateSliceBuffer(
        band_start_idx=runtime.band_start_idx,
        band_end_idx=runtime.band_end_idx,
        m=runtime.m,
        fft_freq_count=runtime.fft_freq_count,
        coeffs_by_channel=coeffs_by_channel,
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

    ``coeffs_by_channel`` maps each channel index to shifted full-FFT coefficients with shape
    ``(m, N, 1)``. The selected frequency band has length
    ``F = runtime.band_end_idx - runtime.band_start_idx``.

    Returns a conjugated, window-normalized spectrum. Output shape depends on order: order 1 returns
    ``(1,)``, order 2 returns ``(F,)``, and orders 3 and 4 return ``(F, F)``. Invalid third-order
    points, where ``w3 = -(w1 + w2)`` is outside the shifted FFT support, are filled with ``NaN``.
    """
    order = len(channels)

    if order == 1:
        single_spectrum = c1(intermediate_buffer.coeffs_by_channel[channels[0]])

    elif order == 2:
        single_spectrum = c2(
            runtime.m,
            intermediate_buffer.coeffs_by_channel_band(channels[0]),
            intermediate_buffer.coeffs_by_channel_band(channels[1]),
        )

    elif order == 3:
        order3_channels = cast(tuple[int, int, int], channels)
        prepared = intermediate_buffer.third_order_prepared(order3_channels)
        single_spectrum = c3(runtime.m, prepared.a_w1, prepared.a_w2, prepared.a_w3)

        nan_value = torch.full_like(single_spectrum, complex(float("nan"), 0.0))
        single_spectrum = torch.where(prepared.valid_mask, single_spectrum, nan_value)

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
