import pytest
from pydantic import ValidationError

from multichss.configurators import SpectrumConfig


def test_spectrum_config_rejects_nonzero_f_min_for_order_3_auto_spectrum():
    with pytest.raises(ValidationError, match="Third-order spectra cannot be requested"):
        SpectrumConfig(f_min=-1, f_max=1, auto_spectra_orders=[3])


def test_spectrum_config_rejects_nonzero_f_min_for_order_3_cross_spectrum():
    with pytest.raises(ValidationError, match="Third-order spectra cannot be requested"):
        SpectrumConfig(
            f_min=-1,
            f_max=1,
            auto_spectra_orders=[],
            cross_spectra=[(0, 1, 1)],
        )


def test_spectrum_config_allows_repeated_channels_in_cross_spectra():
    SpectrumConfig(auto_spectra_orders=[], cross_spectra=[(1, 1, 0)])


def test_spectrum_config_rejects_auto_spectra_in_cross_spectra():
    with pytest.raises(ValidationError, match="cannot include auto-spectra"):
        SpectrumConfig(auto_spectra_orders=[], cross_spectra=[(1, 1, 1)])


def test_spectrum_config_rejects_duplicate_cross_spectra():
    with pytest.raises(ValidationError, match="cannot contain duplicates"):
        SpectrumConfig(auto_spectra_orders=[], cross_spectra=[(1, 1, 0), (1, 1, 0)])


def test_spectrum_config_rejects_duplicate_auto_spectra_orders():
    with pytest.raises(ValidationError, match="Auto-spectrum orders cannot contain duplicates"):
        SpectrumConfig(auto_spectra_orders=[1, 2, 2])


def test_spectrum_config_rejects_empty_spectrum_request():
    with pytest.raises(ValidationError, match="At least one auto-order or cross-spectrum"):
        SpectrumConfig(auto_spectra_orders=[], cross_spectra=None)


def test_spectrum_config_allows_cross_only_request():
    config = SpectrumConfig(auto_spectra_orders=[], cross_spectra=[(0, 1)])

    assert config.auto_spectra_orders == []
    assert config.cross_spectra == [(0, 1)]