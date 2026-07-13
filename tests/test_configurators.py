from multichss import SpectrumConfig


def test_spectrum_config_accepts_negative_frequency_band():
    SpectrumConfig(f_min=-1, f_max=1)


def test_spectrum_config_defaults_to_no_interlacing():
    assert SpectrumConfig().interlacing is False
