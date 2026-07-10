import numpy as np
import pytest
from contextlib import nullcontext

from multichss.configurators import DataConfig, SpectrumConfig
from multichss.fft import iter_window_slices
from multichss.pipelines import calculate_spectra
from multichss.planning import build_runtime_config

auto_spectra = [(0,), (0, 0)]


@pytest.mark.parametrize(
    (
        "n_data_points",
        "spectral_estimates_max",
        "auto_spectra_channels",
        "frequency_points",
        "f_max",
        "m",
        "expected_unshifted_estimates",
        "expected_m",
    ),
    [
        pytest.param(80, None, auto_spectra, 9, 0.5, 4, 1, 4, id="uncapped"),
        pytest.param(80, 1, auto_spectra, 9, 0.5, 4, 1, 4, id="capped-below-available"),
        pytest.param(128, 2, auto_spectra, 9, 0.5, 4, 2, 4, id="cap-below-boundary"),
        pytest.param(136, 10, auto_spectra, 9, 0.5, 4, 2, 4, id="cap-above-available"),
        pytest.param(136, 4, auto_spectra, 9, 0.5, 4, 2, 4, id="cap-equals-available"),
        pytest.param(127, None, auto_spectra, 9, 0.5, 4, 1, 4, id="one-before-next-base"),
        pytest.param(63, None, auto_spectra, 9, 0.5, 4, 1, 3, id="m-reduced-at-short-boundary"),
        pytest.param(
            256,
            3,
            auto_spectra + [(0, 0, 0), (0, 0, 0, 0)],
            9,
            0.5,
            4,
            3,
            4,
            id="higher-orders-capped",
        ),
        pytest.param(96, None, auto_spectra, 6, 1 / 3, 3, 2, 3, id="odd-window-before-half-shift"),
        pytest.param(97, None, auto_spectra, 6, 1 / 3, 3, 2, 3, id="odd-window-at-half-shift"),
    ],
)
def test_spectral_estimates_in_runtime_config(
    n_data_points,
    spectral_estimates_max,
    auto_spectra_channels,
    frequency_points,
    f_max,
    m,
    expected_unshifted_estimates,
    expected_m,
):
    spectrum_config = SpectrumConfig(
        f_min=0.0,
        f_max=f_max,
        frequency_points=frequency_points,
        spectra_channels=auto_spectra_channels,
        m=m,
        spectral_estimates_max=spectral_estimates_max,
    )
    data_config = DataConfig(data=np.ones(n_data_points), dt=1.0)

    warning_context = (
        pytest.warns(UserWarning, match=f"using m={expected_m} instead")
        if expected_m != m
        else nullcontext()
    )

    with warning_context:
        runtime = build_runtime_config(spectrum_config, [data_config])

    assert runtime.m == expected_m
    assert runtime.spectral_estimates == expected_unshifted_estimates


@pytest.mark.parametrize(
    ("interlacing", "expected_slices", "expected_spectral_estimates"),
    [
        pytest.param(
            True,
            [(0, 64, False), (64, 128, False), (8, 72, True), (72, 136, True)],
            2,
            id="interlacing-enabled",
        ),
        pytest.param(
            False,
            [(0, 64, False), (64, 128, False)],
            2,
            id="interlacing-disabled",
        ),
    ],
)
def test_window_slices_respect_interlacing(
    interlacing,
    expected_slices,
    expected_spectral_estimates,
):
    spectrum_config = SpectrumConfig(
        f_min=0.0,
        f_max=0.5,
        frequency_points=9,
        spectra_channels=auto_spectra,
        m=4,
        spectral_estimates_max=None,
        interlacing=interlacing,
    )
    data_config = DataConfig(data=np.ones(136), dt=1.0)

    runtime = build_runtime_config(spectrum_config, [data_config])

    assert runtime.interlacing is interlacing
    assert runtime.spectral_estimates == expected_spectral_estimates
    assert list(iter_window_slices(runtime)) == expected_slices


def test_pipeline_returns_full_axis_third_order_spectrum_with_invalid_points_masked():
    spectrum_config = SpectrumConfig(
        f_min=-0.25,
        f_max=0.25,
        frequency_points=5,
        spectra_channels=[(0, 0, 0)],
        m=4,
        spectral_estimates_max=1,
    )
    data_config = DataConfig(data=np.ones(64), dt=1.0)

    with pytest.warns(RuntimeWarning, match="at least two spectral estimates"):
        result_store = calculate_spectra(spectrum_config, [data_config])
    result = result_store.get((0, 0, 0))

    assert result.spectrum.shape == (result.freq.size, result.freq.size)
    assert np.isnan(result.spectrum).any()


def test_runtime_config_keeps_m_for_exact_unshifted_fit_without_interlacing():
    spectrum_config = SpectrumConfig(
        f_min=0.0,
        f_max=0.5,
        frequency_points=9,
        spectra_channels=auto_spectra,
        m=4,
        interlacing=False,
        spectral_estimates_max=None,
    )
    data_config = DataConfig(data=np.ones(64), dt=1.0)

    runtime = build_runtime_config(spectrum_config, [data_config])

    assert runtime.m == 4
    assert runtime.spectral_estimates == 1


def test_runtime_config_raises_when_interlacing_has_no_shifted_estimate():
    spectrum_config = SpectrumConfig(
        f_min=0.0,
        f_max=0.5,
        frequency_points=9,
        spectra_channels=auto_spectra,
        m=4,
        interlacing=True,
        spectral_estimates_max=None,
    )
    data_config = DataConfig(data=np.ones(64), dt=1.0)

    with pytest.raises(ValueError, match="Interlacing was requested"):
        build_runtime_config(spectrum_config, [data_config])


@pytest.mark.parametrize(
    ("auto_spectra_channels", "m"),
    [
        pytest.param([(0, 0)], 1, id="order-2-needs-at-least-two-windows"),
        pytest.param([(0, 0, 0)], 2, id="order-3-needs-at-least-three-windows"),
        pytest.param([(0, 0, 0, 0)], 3, id="order-4-needs-at-least-four-windows"),
    ],
)
def test_runtime_config_rejects_m_below_requested_order(auto_spectra_channels, m):
    spectrum_config = SpectrumConfig(
        f_min=0.0,
        f_max=20.0,
        frequency_points=16,
        spectra_channels=auto_spectra_channels,
        m=m,
    )
    data_config = DataConfig(data=np.ones(50000), dt=0.001)

    with pytest.raises(ValueError, match="Not enough data points"):
        build_runtime_config(spectrum_config, [data_config])


def test_runtime_config_defaults_to_all_auto_spectra_for_all_channels():
    spectrum_config = SpectrumConfig(
        f_min=0.0,
        f_max=0.5,
        frequency_points=9,
        m=4,
    )
    data_config_list = [
        DataConfig(data=np.ones(136), dt=1.0),
        DataConfig(data=np.ones(136), dt=1.0),
    ]

    runtime = build_runtime_config(spectrum_config, data_config_list)

    assert runtime.active_channels == (0, 1)
    assert runtime.spectra_channels == (
        (0,),
        (0, 0),
        (0, 0, 0),
        (0, 0, 0, 0),
        (1,),
        (1, 1),
        (1, 1, 1),
        (1, 1, 1, 1),
    )


def test_runtime_config_rejects_out_of_bounds_spectra_channel_indices():
    spectrum_config = SpectrumConfig(
        f_min=0.0,
        f_max=0.5,
        frequency_points=9,
        spectra_channels=[(1,)],
        m=4,
    )
    data_config = DataConfig(data=np.ones(136), dt=1.0)

    with pytest.raises(ValueError, match="Channel indices must be in the range"):
        build_runtime_config(spectrum_config, [data_config])


def test_spectrum_config_defaults_to_no_interlacing():
    assert SpectrumConfig().interlacing is False
