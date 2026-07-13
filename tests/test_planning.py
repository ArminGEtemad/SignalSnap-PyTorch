from contextlib import nullcontext

import numpy as np
import pytest

from multichss import DataConfig, SpectrumConfig, calculate_spectra
from multichss._core.planning import (
    _resolve_requested_spectra,
    build_runtime_config,
    iter_window_slices,
)

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
        pytest.param(136, 1, auto_spectra, 9, 0.5, 4, 1, 4, id="cap-below-available"),
        pytest.param(128, 2, auto_spectra, 9, 0.5, 4, 2, 4, id="cap-equals-available"),
        pytest.param(136, 10, auto_spectra, 9, 0.5, 4, 2, 4, id="cap-above-available"),
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
        runtime = build_runtime_config(
            [data_config], spectrum_config, auto_spectra_channels
        )

    assert runtime.m == expected_m
    assert runtime.spectral_estimates == expected_unshifted_estimates


@pytest.mark.parametrize(
    (
        "n_data_points",
        "frequency_points",
        "f_max",
        "m",
        "interlacing",
        "expected_slices",
        "expected_spectral_estimates",
    ),
    [
        pytest.param(
            136,
            9,
            0.5,
            4,
            True,
            [(0, 64, False), (64, 128, False), (8, 72, True), (72, 136, True)],
            2,
            id="even-window-interlacing-enabled",
        ),
        pytest.param(
            136,
            9,
            0.5,
            4,
            False,
            [(0, 64, False), (64, 128, False)],
            2,
            id="even-window-interlacing-disabled",
        ),
        pytest.param(
            96,
            6,
            1 / 3,
            3,
            True,
            [(0, 45, False), (45, 90, False), (7, 52, True)],
            2,
            id="odd-window-before-second-shifted-estimate",
        ),
        pytest.param(
            97,
            6,
            1 / 3,
            3,
            True,
            [(0, 45, False), (45, 90, False), (7, 52, True), (52, 97, True)],
            2,
            id="odd-window-at-second-shifted-estimate",
        ),
    ],
)
def test_window_slices_respect_interlacing(
    n_data_points,
    frequency_points,
    f_max,
    m,
    interlacing,
    expected_slices,
    expected_spectral_estimates,
):
    spectrum_config = SpectrumConfig(
        f_min=0.0,
        f_max=f_max,
        frequency_points=frequency_points,
        m=m,
        spectral_estimates_max=None,
        interlacing=interlacing,
    )
    data_config = DataConfig(data=np.ones(n_data_points), dt=1.0)

    runtime = build_runtime_config([data_config], spectrum_config, auto_spectra)

    assert runtime.interlacing is interlacing
    assert runtime.spectral_estimates == expected_spectral_estimates
    assert list(iter_window_slices(runtime)) == expected_slices


def test_pipeline_returns_full_axis_third_order_spectrum_with_invalid_points_masked():
    spectrum_config = SpectrumConfig(
        f_min=-0.25,
        f_max=0.25,
        frequency_points=5,
        m=4,
        spectral_estimates_max=1,
    )
    data_config = DataConfig(data=np.ones(64), dt=1.0)

    with pytest.warns(RuntimeWarning, match="at least two spectral estimates"):
        result_store = calculate_spectra(
            [data_config], spectrum_config, requested_spectra=[(0, 0, 0)]
        )
    result = result_store.get((0, 0, 0))

    assert result.spectrum.shape == (result.freq.size, result.freq.size)

    assert spectrum_config.f_max is not None
    window_duration = (spectrum_config.frequency_points - 1) / (
        spectrum_config.f_max - spectrum_config.f_min
    )
    window_points = int(np.round(window_duration / data_config.dt))
    full_fft_freq = np.fft.fftshift(np.fft.fftfreq(window_points, data_config.dt))
    third_factor_freq = -(result.freq[:, None] + result.freq[None, :])
    expected_valid_mask = np.isclose(
        third_factor_freq[..., None],
        full_fft_freq,
        rtol=0.0,
        atol=1e-12,
    ).any(axis=-1)

    np.testing.assert_array_equal(np.isnan(result.spectrum), ~expected_valid_mask)
    assert np.isfinite(result.spectrum[expected_valid_mask]).all()


def test_runtime_config_keeps_m_for_exact_unshifted_fit_without_interlacing():
    spectrum_config = SpectrumConfig(
        f_min=0.0,
        f_max=0.5,
        frequency_points=9,
        m=4,
        interlacing=False,
        spectral_estimates_max=None,
    )
    data_config = DataConfig(data=np.ones(64), dt=1.0)

    runtime = build_runtime_config([data_config], spectrum_config, auto_spectra)

    assert runtime.m == 4
    assert runtime.spectral_estimates == 1


def test_runtime_config_raises_when_interlacing_has_no_shifted_estimate():
    spectrum_config = SpectrumConfig(
        f_min=0.0,
        f_max=0.5,
        frequency_points=9,
        m=4,
        interlacing=True,
        spectral_estimates_max=None,
    )
    data_config = DataConfig(data=np.ones(64), dt=1.0)

    with pytest.raises(ValueError, match="Interlacing was requested"):
        build_runtime_config([data_config], spectrum_config, auto_spectra)


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
        m=m,
    )
    data_config = DataConfig(data=np.ones(50000), dt=0.001)

    with pytest.raises(ValueError, match="Not enough data points"):
        build_runtime_config(
            [data_config], spectrum_config, auto_spectra_channels
        )


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

    runtime = build_runtime_config(data_config_list, spectrum_config, None)

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
        m=4,
    )
    data_config = DataConfig(data=np.ones(136), dt=1.0)

    with pytest.raises(ValueError, match="out of bounds"):
        build_runtime_config([data_config], spectrum_config, [(1,)])


@pytest.mark.parametrize(
    ("requested_spectra", "exception_type", "message"),
    [
        pytest.param([], ValueError, "at least one spectrum", id="empty-request"),
        pytest.param([[]], TypeError, "must be a tuple", id="non-tuple-spectrum"),
        pytest.param([()], ValueError, "between 1 and 4", id="order-zero"),
        pytest.param(
            [(0, 0, 0, 0, 0)], ValueError, "between 1 and 4", id="order-five"
        ),
        pytest.param([(True,)], TypeError, "must be integers", id="boolean-index"),
        pytest.param([(0.0,)], TypeError, "must be integers", id="float-index"),
        pytest.param([(-1,)], ValueError, "nonnegative", id="negative-index"),
        pytest.param([(2,)], ValueError, "out of bounds", id="out-of-bounds-index"),
        pytest.param(
            [(0,), (0,)], ValueError, "cannot contain duplicates", id="duplicate-spectrum"
        ),
    ],
)
def test_resolve_requested_spectra_rejects_invalid_requests(
    requested_spectra, exception_type, message
):
    with pytest.raises(exception_type, match=message):
        _resolve_requested_spectra(requested_spectra, channel_count=2)


def test_resolve_requested_spectra_normalizes_numpy_integer_indices():
    requested_spectra = [(np.int64(0),), (np.int32(0), np.int64(1))]

    resolved = _resolve_requested_spectra(requested_spectra, channel_count=2)

    assert resolved == ((0,), (0, 1))
    assert all(type(channel) is int for spectrum in resolved for channel in spectrum)


def test_resolve_requested_spectra_rejects_missing_data_channels():
    with pytest.raises(ValueError, match="At least one DataConfig is required"):
        _resolve_requested_spectra(None, channel_count=0)
