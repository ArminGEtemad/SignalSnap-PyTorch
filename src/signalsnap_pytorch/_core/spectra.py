# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemad, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from .cumulants import (
    build_s3_target_indices,
    c2_factorized,
    c3_factorized,
    c4_factorized,
    gather_s3_third_factor,
)
from .fft import WindowBuffer
from .planning import RuntimeConfig


@dataclass(slots=True)
class ThirdOrderIndexCache:
    """Describes where the third frequency lies for the corresponding frequency axis used in the
    calculation.


    """

    target_indices: Tensor
    valid_mask: Tensor


@dataclass(slots=True)
class ThirdOrderFactor:
    """Stores the Fourier coefficients for the third frequency based on the indices in
    :class:`ThirdOrderIndexCache`.
    """

    centered_a_w3: Tensor
    valid_mask: Tensor


@dataclass(slots=True)
class IntermediateSliceBuffer:
    """Stores precomputed intermediate results used in :func:`compute_single_spectrum`.

    Attributes
    ----------
    band_start_idx, band_end_idx : int
        Start and end index (corresponding to `f_min` and `f_max`) of the selected frequency band in
        the full Fourier coefficients.
    m : int
        Number of windows per spectral estimate.
    fft_freq_count : int
        Length of the full Fourier coefficients.
    coeffs_by_channel : dict[int, Tensor]
        Full Fourier coefficients by channel.
    third_order_cache : :class:`ThirdOrderIndexCache` | None
        Indices of the third frequency for the corresponding frequency axis.
    """

    band_start_idx: int
    band_end_idx: int
    m: int
    fft_freq_count: int
    coeffs_by_channel: dict[int, Tensor] = field(default_factory=dict)
    third_order_cache: ThirdOrderIndexCache | None = None

    _centered_coeffs_by_channel_band: dict[int, Tensor] = field(default_factory=dict)
    _centered_c3_third_factor_by_channel: dict[int, ThirdOrderFactor] = field(default_factory=dict)

    def centered_coeffs_by_channel_band(self, channel: int, conjugated: bool = False) -> Tensor:
        """Returns the centered Fourier coefficients in the specified frequency band"""
        if channel not in self._centered_coeffs_by_channel_band:
            coeffs = self.coeffs_by_channel[channel][:, self.band_start_idx : self.band_end_idx]
            self._centered_coeffs_by_channel_band[channel] = coeffs - torch.mean(coeffs, dim=0)

        if conjugated:
            return torch.conj(self._centered_coeffs_by_channel_band[channel])
        else:
            return self._centered_coeffs_by_channel_band[channel]

    def centered_c3_third_factor_by_channel(self, channel: int) -> ThirdOrderFactor:
        """Returns the centered Fourier coefficients for the c3 third factor in the specified
        frequency band.
        """
        if channel not in self._centered_c3_third_factor_by_channel:
            if self.third_order_cache is None:
                raise ValueError("Third-order spectra require third_order_cache.")
            centered_a_w3 = gather_s3_third_factor(
                self.coeffs_by_channel[channel]
                - torch.mean(self.coeffs_by_channel[channel], dim=0),
                self.third_order_cache.target_indices,
                self.m,
            )
            self._centered_c3_third_factor_by_channel[channel] = ThirdOrderFactor(
                centered_a_w3=centered_a_w3,
                valid_mask=self.third_order_cache.valid_mask,
            )

        return self._centered_c3_third_factor_by_channel[channel]


def build_third_order_cache(runtime: RuntimeConfig) -> ThirdOrderIndexCache:
    axis_indices = torch.arange(
        runtime.band_start_idx,
        runtime.band_end_idx,
        device=runtime.device,
    )
    target_indices, valid_mask = build_s3_target_indices(axis_indices, runtime.window_points)
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
        fft_freq_count=runtime.window_points,
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

    Parameters
    ----------
    channels : tuple[int, ...]
        Specifies the corresponding channels of the spectra to be computed, e.g. (0, 0, 0) for a
        third-order auto-spectrum.
    intermediate_buffer : :class:`IntermediateSliceBuffer`
        Stores the precomputed computed Fourier coefficients and bands for the current slice.
    window_buffer : :class:`WindowBuffer`
        Stores all information related to the window function.
    runtime : :class:`RuntimeConfig`
        Resolved calculation settings derived from user configuration.

    Returns
    -------
    Tensor
        Single spectral estimate for the specified spectrum. Output shape depends on order: order 1
        returns `(1,)`, order 2 returns `(F,)`, and orders 3 and 4 return `(F, F)`, with F being the
        length of the selected frequency band. Invalid third-order points, where `w3 = -(w1 + w2)`
        is outside the shifted FFT support, are filled with `NaN`.
    """
    order = len(channels)

    if order == 1:
        a_w = intermediate_buffer.coeffs_by_channel[channels[0]]
        dc_index = a_w.shape[1] // 2
        single_cumulant = torch.mean(a_w[:, dc_index], dim=0).reshape(1)

    elif order == 2:
        single_cumulant = c2_factorized(
            runtime.m,
            intermediate_buffer.centered_coeffs_by_channel_band(channels[0]),
            intermediate_buffer.centered_coeffs_by_channel_band(channels[1], conjugated=True),
        )

    elif order == 3:
        prepared = intermediate_buffer.centered_c3_third_factor_by_channel(channels[2])

        single_cumulant = c3_factorized(
            runtime.m,
            intermediate_buffer.centered_coeffs_by_channel_band(channels[0]),
            intermediate_buffer.centered_coeffs_by_channel_band(channels[1]),
            prepared.centered_a_w3,
        )

        nan_value = torch.full_like(single_cumulant, complex(float("nan"), 0.0))
        single_cumulant = torch.where(prepared.valid_mask, single_cumulant, nan_value)

    elif order == 4:
        single_cumulant = c4_factorized(
            runtime.m,
            intermediate_buffer.centered_coeffs_by_channel_band(channels[0]),
            intermediate_buffer.centered_coeffs_by_channel_band(channels[1], conjugated=True),
            intermediate_buffer.centered_coeffs_by_channel_band(channels[2]),
            intermediate_buffer.centered_coeffs_by_channel_band(channels[3], conjugated=True),
        )
    else:
        raise ValueError(f"Unsupported spectrum order: {order}.")

    return single_cumulant / window_buffer.norm(order)
