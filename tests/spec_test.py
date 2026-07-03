# This test doesn't showcase what higher order spectra are actually meant for
# since I am using them on trigonometrical functions with no noise. 
# However i know the results so they are good for testing.

import numpy as np
from multichss.pipelines import calculate_spectra
from multichss.configurators import SpectrumConfig, DataConfig

def test_c1_returns_correct_mean():
    """
    Test that the first-order cumulant correctly returns the mean of the signal.
    Since the signal is sin(2πt) + 2, the mean should be 2.0.
    """

    # Generate test signal
    t = np.linspace(0, 10000, 1000000)
    y = np.sin(2 * np.pi * t) + 2  # known mean = 2.0

    # Wrap into config objects
    config1 = DataConfig(data=y, dt=0.01, t_unit="s")
    selected_data = [0]

    sconfig = SpectrumConfig(
        f_min=0, f_max=2, s3_calc='1/4', device='cpu', auto_spectra_orders=[1],
        frequency_points=100,
    )

    # Run the spectrum calculator
    result_store = calculate_spectra(sconfig, [config1], selected_data)
    result = result_store.get((0,))

    assert result.spectrum is not None

    # Grab the component from the first-order spectrum
    real_part = result.spectrum[0].real
    imag_part = result.spectrum[0].imag

    print(result.spectrum)

    # Assert
    assert abs(real_part - 2.0) < 1e-6, f"Expected real=2.0, got {real_part}"
    assert abs(imag_part - 0.0) < 1e-6, f"Expected imag=0.0, got {imag_part}"


def test_c1_returns_mean_when_selected_band_excludes_dc():
    y = np.ones(10000) * 2.0
    config1 = DataConfig(data=y, dt=0.01, t_unit="s")

    sconfig = SpectrumConfig(
        f_min=1,
        f_max=2,
        device="cpu",
        auto_spectra_orders=[1],
        frequency_points=100,
    )

    result_store = calculate_spectra(sconfig, [config1], selected=[0])
    result = result_store.get((0,))

    assert result.spectrum is not None
    assert result.freq is not None
    np.testing.assert_allclose(result.spectrum, np.asarray([2.0 + 0.0j]), atol=1e-12)
    np.testing.assert_allclose(result.freq, np.asarray([0.0]), atol=0.0)
