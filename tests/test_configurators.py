from typing import Any

import pytest
from pydantic import ValidationError

from multichss import SpectrumConfig


@pytest.mark.parametrize(
    "config_kwargs",
    [
        pytest.param(
            {"f_min": -1, "f_max": 1, "spectra_channels": [(0, 0, 0)]},
            id="negative-band-order-3-auto",
        ),
        pytest.param(
            {"f_min": -1, "f_max": 1, "spectra_channels": [(0, 1, 1)]},
            id="negative-band-order-3-cross",
        ),
        pytest.param(
            {"spectra_channels": [(1, 1, 0)]},
            id="repeated-channels",
        ),
    ],
)
def test_spectrum_config_accepts_valid_spectrum_requests(config_kwargs: dict[str, Any]):
    SpectrumConfig(**config_kwargs)


def test_spectrum_config_rejects_duplicate_spectra():
    with pytest.raises(ValidationError, match="cannot contain duplicates"):
        SpectrumConfig(spectra_channels=[(1, 1, 0), (1, 1, 0)])


def test_spectrum_config_rejects_empty_spectrum_request():
    with pytest.raises(ValidationError, match="at least 1 item"):
        SpectrumConfig(spectra_channels=[])


def test_spectrum_config_rejects_negative_spectra_channel_indices():
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        SpectrumConfig(spectra_channels=[(-1,)])


def test_spectrum_config_defaults_to_no_interlacing():
    assert SpectrumConfig().interlacing is False
