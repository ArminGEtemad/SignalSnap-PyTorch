import numpy as np

from multichss import DataConfig, SpectrumConfig, calculate_spectra


def test_c1_returns_correct_mean():
    """The first-order cumulant equals the signal mean."""
    dt = 0.01
    samples_per_window = 1_000
    centered_samples = np.arange(samples_per_window) - (samples_per_window - 1) / 2
    window_signal = np.sin(2 * np.pi * 10 * centered_samples / samples_per_window) + 2
    signal = np.tile(window_signal, 20)

    data_config = DataConfig(data=signal, dt=dt, t_unit="s")
    spectrum_config = SpectrumConfig(
        f_min=0,
        f_max=2,
        device="cpu",
        frequency_points=21,
    )

    result = calculate_spectra(
        [data_config], spectrum_config, requested_spectra=[(0,)]
    ).get((0,))

    np.testing.assert_allclose(result.spectrum, np.asarray([2.0 + 0.0j]), atol=1e-12)


def test_c1_returns_mean_when_selected_band_excludes_dc():
    signal = np.full(10_000, 2.0)
    data_config = DataConfig(data=signal, dt=0.01, t_unit="s")
    spectrum_config = SpectrumConfig(
        f_min=1,
        f_max=2,
        device="cpu",
        frequency_points=3,
    )

    result = calculate_spectra(
        [data_config], spectrum_config, requested_spectra=[(0,)]
    ).get((0,))

    np.testing.assert_allclose(result.spectrum, np.asarray([2.0 + 0.0j]), atol=1e-12)
    np.testing.assert_array_equal(result.freq, np.asarray([0.0]))
