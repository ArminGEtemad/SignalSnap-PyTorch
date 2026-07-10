# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemed, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import Field

TimeUnits: TypeAlias = Literal["s", "ms", "us", "ns", "ps"]
FrequencyUnits: TypeAlias = Literal["Hz", "kHz", "MHz", "GHz", "THz"]
ChannelIndex: TypeAlias = Annotated[int, Field(ge=0)]
PlotComponent: TypeAlias = Literal["re", "im"]


def unit_conversion_time_to_freq(t_unit: TimeUnits) -> FrequencyUnits:
    """Return the frequency unit corresponding to a time-step unit."""
    mapping: dict[TimeUnits, FrequencyUnits] = {
        "s": "Hz",
        "ms": "kHz",
        "us": "MHz",
        "ns": "GHz",
        "ps": "THz",
    }

    try:
        return mapping[t_unit]
    except KeyError:
        raise ValueError(f"Unknown time unit: {t_unit}")
