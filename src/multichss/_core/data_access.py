# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemed, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from ..configurators import DataConfig, HDF5Channel

NormalizedSelector = int | slice


@dataclass(frozen=True, slots=True)
class HDF5ChannelState:
    """
    Opened runtime version of a user specified :class:`HDF5Channel`.
    """

    dataset: h5py.Dataset
    selection: tuple[NormalizedSelector, ...]
    selected_shape: tuple[int, ...]

    @property
    def sample_count(self) -> int:
        return math.prod(self.selected_shape)

    def read(self, start: int, stop: int) -> np.ndarray:
        if start == stop:
            return np.empty(0, dtype=self.dataset.dtype)

        if len(self.selected_shape) == 1:
            return self._read_1d(start, stop)

        if len(self.selected_shape) == 2:
            return self._read_2d(start, stop)

        raise RuntimeError(
            f"Unsupported selected shape {self.selected_shape}. "
            "At most two non-fixed dimensions are supported."
        )

    def _build_dataset_index(
        self, logical_selectors: tuple[int | slice, ...]
    ) -> tuple[int | slice, ...]:
        if len(logical_selectors) != len(self.selected_shape):
            raise ValueError(
                f"Expected {len(self.selected_shape)} logical selectors, "
                f"received {len(logical_selectors)}."
            )

        dataset_index: list[int | slice] = []
        logical_axis = 0

        for base_selector in self.selection:
            if isinstance(base_selector, int):
                dataset_index.append(base_selector)
                continue

            logical_selector = logical_selectors[logical_axis]
            logical_axis += 1

            # Slices were normalized earlier, so start is always an integer.
            base_start = base_selector.start
            assert base_start is not None

            if isinstance(logical_selector, int):
                dataset_index.append(base_start + logical_selector)
            else:
                logical_start = logical_selector.start
                logical_stop = logical_selector.stop

                if logical_start is None or logical_stop is None:
                    raise ValueError("Internal logical slices must have explicit bounds.")

                dataset_index.append(slice(base_start + logical_start, base_start + logical_stop))

        return tuple(dataset_index)

    def _allocate_output(self, size: int) -> np.ndarray:
        """Let HDF5 convert byte order while reading, avoiding a later full copy."""
        dtype = self.dataset.dtype.newbyteorder("=")
        return np.empty(size, dtype=dtype)

    def _read_direct(
        self, destination: np.ndarray, logical_selectors: tuple[int | slice, ...]
    ) -> None:
        dataset_index = self._build_dataset_index(logical_selectors)
        self.dataset.read_direct(destination, source_sel=dataset_index)

    def _read_1d(self, start: int, stop: int) -> np.ndarray:
        result = self._allocate_output(stop - start)
        self._read_direct(result, (slice(start, stop),))
        return result

    def _read_2d(self, start: int, stop: int) -> np.ndarray:
        _, column_count = self.selected_shape
        result = self._allocate_output(stop - start)

        cursor = start
        output_cursor = 0

        # Partial first row.
        start_row, start_column = divmod(cursor, column_count)

        if start_column:
            count = min(stop - cursor, column_count - start_column)

            self._read_direct(
                result[output_cursor : output_cursor + count],
                (start_row, slice(start_column, start_column + count)),
            )

            cursor += count
            output_cursor += count

        # Complete middle rows.
        complete_rows = (stop - cursor) // column_count

        if complete_rows:
            first_row = cursor // column_count
            count = complete_rows * column_count

            destination = result[output_cursor : output_cursor + count].reshape(
                complete_rows, column_count
            )

            self._read_direct(
                destination,
                (
                    slice(first_row, first_row + complete_rows),
                    slice(0, column_count),
                ),
            )

            cursor += count
            output_cursor += count

        # Partial final row.
        if cursor < stop:
            final_row = cursor // column_count
            count = stop - cursor

            self._read_direct(
                result[output_cursor : output_cursor + count],
                (final_row, slice(0, count)),
            )

            cursor += count
            output_cursor += count

        if cursor != stop or output_cursor != result.size:
            raise RuntimeError(
                f"HDF5 reading stopped at sample {cursor}; expected to stop at {stop}."
            )

        return result


def normalize_hdf5_selection(
    dataset: h5py.Dataset, channel: HDF5Channel
) -> tuple[tuple[NormalizedSelector, ...], tuple[int, ...]]:

    if len(channel.selection) != dataset.ndim:
        raise ValueError(
            f"Selection for dataset {channel.dataset!r} contains {len(channel.selection)} entries, "
            f"but the dataset has {dataset.ndim} dimensions."
        )

    normalized = []
    selected_shape = []

    for axis, (dimension_size, selector) in enumerate(zip(dataset.shape, channel.selection)):
        if isinstance(selector, int) and not isinstance(selector, bool):
            index = selector

            if index < 0:
                index += dimension_size

            if index < 0 or index >= dimension_size:
                raise IndexError(
                    f"Index {selector} is out of bounds for axis {axis} with size {dimension_size}."
                )

            normalized.append(index)
            continue

        elif isinstance(selector, slice):
            start, stop, step = selector.indices(dimension_size)
            if step != 1:
                raise ValueError("Only HDF5 slices with step 1 are supported.")

            normalized.append(slice(start, stop))
            selected_shape.append(max(0, stop - start))
        else:
            raise TypeError(
                f"Selection entry for axis {axis} must be an integer or slice; "
                f"received {type(selector).__name__}."
            )

    if not selected_shape:
        raise ValueError("The HDF5 selection selects a scalar, not a signal channel.")

    if any(size == 0 for size in selected_shape):
        raise ValueError("The HDF5 channel selection is empty.")

    if len(selected_shape) > 2:
        raise ValueError(
            "An HDF5 channel may have at most two non-fixed dimensions. "
            "Use integer indices to fix additional dimensions."
        )

    return tuple(normalized), tuple(selected_shape)


def validate_hdf5_dataset(file: h5py.File, channel: HDF5Channel) -> h5py.Dataset:
    if channel.dataset not in file:
        raise KeyError(
            f"Dataset {channel.dataset!r} does not exist in HDF5 file {str(channel.file)!r}."
        )

    obj = file[channel.dataset]

    if not isinstance(obj, h5py.Dataset):
        raise TypeError(f"HDF5 path {channel.dataset!r} is not a dataset.")

    if np.issubdtype(obj.dtype, np.complexfloating):
        raise TypeError("Complex HDF5 datasets are not supported.")

    if not np.issubdtype(obj.dtype, np.number) and not np.issubdtype(obj.dtype, np.bool_):
        raise TypeError(f"HDF5 dataset dtype {obj.dtype} is not numeric.")

    return obj


RuntimeChannel = Any | HDF5ChannelState


@contextmanager
def open_channels(
    data_config: DataConfig,
    channel_indices: Iterable[int],
) -> Iterator[dict[int, RuntimeChannel]]:
    with ExitStack() as stack:
        files: dict[Path, h5py.File] = {}
        opened_channels: dict[int, RuntimeChannel] = {}

        for channel_index in channel_indices:
            channel = data_config.channels[channel_index]

            if not isinstance(channel, HDF5Channel):
                opened_channels[channel_index] = channel
                continue

            path = channel.file.expanduser().resolve()

            if path not in files:
                files[path] = stack.enter_context(h5py.File(path, mode="r"))

            dataset = validate_hdf5_dataset(files[path], channel)
            selection, selected_shape = normalize_hdf5_selection(dataset, channel)

            opened_channels[channel_index] = HDF5ChannelState(
                dataset=dataset, selection=selection, selected_shape=selected_shape
            )

        yield opened_channels


def get_sample_count(channel: RuntimeChannel) -> int:
    if isinstance(channel, HDF5ChannelState):
        return channel.sample_count

    return int(channel.shape[0])


def validate_read_range(channel: RuntimeChannel, start: int, stop: int) -> tuple[int, int]:
    if isinstance(start, bool) or not isinstance(start, (int, np.integer)):
        raise TypeError("start must be an integer.")

    if isinstance(stop, bool) or not isinstance(stop, (int, np.integer)):
        raise TypeError("stop must be an integer.")

    start = int(start)
    stop = int(stop)

    if start < 0:
        raise ValueError("start cannot be negative.")

    if stop < start:
        raise ValueError("stop cannot be smaller than start.")

    sample_count = get_sample_count(channel)

    if stop > sample_count:
        raise ValueError(
            f"Cannot read until sample {stop}; the channel contains {sample_count} samples."
        )

    return start, stop


def read_channel(channel: RuntimeChannel, start: int, stop: int) -> np.ndarray:
    start, stop = validate_read_range(channel, start, stop)

    if isinstance(channel, HDF5ChannelState):
        result = channel.read(start, stop)
    else:
        result = channel[start:stop]

    result = np.asarray(result)

    if result.ndim != 1:
        raise RuntimeError(
            f"Reading returned shape {result.shape}; expected a one-dimensional array."
        )

    expected_size = stop - start

    if result.shape[0] != expected_size:
        raise RuntimeError(f"Reading returned {result.shape[0]} samples; expected {expected_size}.")

    if not result.dtype.isnative:
        native_dtype = result.dtype.newbyteorder("=")
        result = result.astype(native_dtype, copy=False)

    return np.ascontiguousarray(result)
