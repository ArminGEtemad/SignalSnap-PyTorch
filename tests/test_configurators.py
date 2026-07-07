import pytest
from pydantic import ValidationError

from multichss.configurators import SpectrumConfig


def test_spectrum_config_allows_negative_f_min_for_order_3_auto_spectrum():
    SpectrumConfig(f_min=-1, f_max=1, spectra_channels=[(0, 0, 0)])


def test_spectrum_config_allows_negative_f_min_for_order_3_cross_spectrum():
    SpectrumConfig(f_min=-1, f_max=1, spectra_channels=[(0, 1, 1)])


def test_spectrum_config_allows_repeated_channels_in_spectra():
    SpectrumConfig(spectra_channels=[(1, 1, 0)])


def test_spectrum_config_rejects_duplicate_spectra():
    with pytest.raises(ValidationError, match="cannot contain duplicates"):
        SpectrumConfig(spectra_channels=[(1, 1, 0), (1, 1, 0)])


def test_spectrum_config_rejects_empty_spectrum_request():
    with pytest.raises(ValidationError, match="at least 1 item"):
        SpectrumConfig(spectra_channels=[])


def test_spectrum_config_rejects_negative_spectra_channel_indices():
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        SpectrumConfig(spectra_channels=[(-1,)])
