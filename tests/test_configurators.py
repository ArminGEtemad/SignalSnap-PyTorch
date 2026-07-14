from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from multichss import DataConfig, HDF5Channel, SpectrumConfig


def test_spectrum_config_accepts_negative_frequency_band():
    SpectrumConfig(f_min=-1, f_max=1)


def test_spectrum_config_defaults_to_no_interlacing():
    assert SpectrumConfig().interlacing is False


def test_data_config_accepts_array_channels():
    config = DataConfig(
        channels=[np.arange(10), np.arange(10) * 2],
        dt=0.1,
    )

    assert len(config.channels) == 2
    assert isinstance(config.channels, tuple)
    assert config.dt == 0.1
    assert config.t_unit == "s"


def test_data_config_accepts_array_and_hdf5_channels():
    hdf5_channel = HDF5Channel(
        file=Path("data.h5"),
        dataset="/signals",
        selection=(slice(None), slice(None), 0),
    )

    config = DataConfig(
        channels=[np.arange(10), hdf5_channel],
        dt=0.1,
    )

    assert len(config.channels) == 2
    assert config.channels[1] == hdf5_channel


@pytest.mark.parametrize(
    ("channels", "message"),
    [
        ([], "at least 1 item"),
        ([None], "cannot be None"),
        ([object()], "shape attribute"),
        ([np.ones((2, 3))], "one-dimensional"),
        ([np.array([])], "cannot be empty"),
        ([np.array([1 + 2j])], "cannot be complex"),
        ([np.array(["a", "b"])], "must be numeric"),
        ([np.array([object()], dtype=object)], "must be numeric"),
    ],
)
def test_data_config_rejects_invalid_array_channels(channels, message):
    with pytest.raises((ValidationError, TypeError), match=message):
        DataConfig(channels=channels, dt=1.0)


@pytest.mark.parametrize(
    ("dataset", "selection", "message"),
    [
        ("", (slice(None),), "dataset cannot be empty"),
        ("/signals", (), "selection cannot be empty"),
        ("/signals", (True,), "integers or slices"),
        ("/signals", ("channel-0",), "integers or slices"),
        ("/signals", (slice(None, None, 2),), "steps other than 1"),
        ("/signals", (slice(None, None, -1),), "steps other than 1"),
    ],
)
def test_hdf5_channel_rejects_invalid_configuration(dataset, selection, message):
    with pytest.raises((ValidationError, TypeError), match=message):
        HDF5Channel(
            file=Path("data.h5"),
            dataset=dataset,
            selection=selection,
        )
