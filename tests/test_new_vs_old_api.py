# This code tries to compute the same spectrum in the new API as the old version in
# https://github.com/ArminGEtemad/SignalSnap-PyTorch/blob/main/Examples/Straightforward_Example.ipynb

import h5py
import numpy as np
import pytest

from multichss.configurators import DataConfig, SpectrumConfig
from multichss.pipelines import calculate_spectra

# compare_error should be set to False for real github tests, since the old api calculates a short
# term error which is different from what is currently implemented in the new api
compare_error = False

auto_keys = [(0,), (0, 0), (0, 0, 0), (0, 0, 0, 0), (1,), (1, 1), (1, 1, 1), (1, 1, 1, 1)]
cross_keys_ch24 = [
    (0, 1),
    (1, 0),
    (1, 0, 0, 1),
    (1, 1, 0, 0),
]
cross_keys_ch3 = [
    (0, 1, 1),
    (1, 0, 0),
    (0, 0, 1),
]


def _common_region(actual, expected):
    common_shape = tuple(min(a, e) for a, e in zip(actual.shape, expected.shape))
    common_slices = tuple(slice(0, size) for size in common_shape)
    return actual[common_slices], expected[common_slices]


def _indices_for_freqs(actual_freq, expected_freq):
    indices = []
    for freq in expected_freq:
        matches = np.flatnonzero(np.isclose(actual_freq, freq, rtol=0.0, atol=1e-12))
        if matches.size != 1:
            raise AssertionError(f"Frequency {freq} is not represented exactly once.")
        indices.append(matches[0])
    return np.asarray(indices)


def _legacy_order3_region(actual_spectrum, actual_freq, expected_spectrum, expected_freq):
    if expected_spectrum.shape == (expected_freq.size, expected_freq.size):
        row_freq = expected_freq
        col_freq = expected_freq
    elif expected_spectrum.ndim == 2 and expected_spectrum.shape[1] == expected_freq.size:
        row_freq = expected_freq[expected_freq.size // 2 :]
        col_freq = expected_freq
    else:
        raise AssertionError(
            f"Unsupported legacy third-order shape {expected_spectrum.shape} "
            f"for frequency axis length {expected_freq.size}."
        )

    row_indices = _indices_for_freqs(actual_freq, row_freq)
    col_indices = _indices_for_freqs(actual_freq, col_freq)
    return actual_spectrum[np.ix_(row_indices, col_indices)], expected_spectrum


@pytest.fixture(scope="module")
def prepared_data():
    with h5py.File("./tests/test_data/datasets/5Qubit_short_data.h5", "r") as f:
        x_test_dataset = f["/X_test"]
        assert isinstance(x_test_dataset, h5py.Dataset)
        X_test = x_test_dataset[...]

    return [
        DataConfig(data=X_test[:1000, :, 0].reshape(-1), dt=2.0, t_unit="ns"),
        DataConfig(data=X_test[:1000, :, 1].reshape(-1), dt=2.0, t_unit="ns"),
    ]


@pytest.mark.parametrize(
    ("name", "reference_file", "keys"),
    [
        pytest.param(
            "auto",
            "./tests/test_data/references/5Qubit_short_data_auto_corr.npz",
            auto_keys,
            id="auto",
        ),
        pytest.param(
            "cross_ch24",
            "./tests/test_data/references/5Qubit_short_data_cross_corr_ch124.npz",
            cross_keys_ch24,
            id="cross_ch24",
        ),
        pytest.param(
            "cross_ch3",
            "./tests/test_data/references/5Qubit_short_data_cross_corr_ch3.npz",
            cross_keys_ch3,
            id="cross_ch3",
        ),
    ],
)
def test_new_vs_old_api(name, reference_file, keys, prepared_data):
    """
    Tests if the refactor implements the same calculations. To check if the
    implementation is correct, the old window function needs to be used in
    SpectrumConfig.
    """
    dataconfig_list = prepared_data
    benchmark_spectra = np.load(reference_file, allow_pickle=True)
    old_spectra = benchmark_spectra["spectra"].item()

    if compare_error:
        old_error = benchmark_spectra["error"].item()
    else:
        old_error = None

    old_freqs = benchmark_spectra["freqs"].item()

    if name == "auto":
        legacy_s3_freq = np.asarray(old_freqs[0][3])
        sconfig = SpectrumConfig(
            f_min=float(legacy_s3_freq[0]),
            f_max=float(legacy_s3_freq[-1]),
            device="cpu",
            spectra_channels=keys,
            frequency_points=legacy_s3_freq.size,
            interlacing=True,
            old_window=True,
        )
    elif name == "cross_ch24":
        sconfig = SpectrumConfig(
            f_min=-0.25,
            f_max=0.25,
            device="cpu",
            spectra_channels=keys,
            frequency_points=100,
            interlacing=True,
            old_window=True,
        )
    elif name == "cross_ch3":
        legacy_s3_freq = np.asarray(old_freqs[keys[0]][3])
        sconfig = SpectrumConfig(
            f_min=float(legacy_s3_freq[0]),
            f_max=float(legacy_s3_freq[-1]),
            device="cpu",
            spectra_channels=keys,
            frequency_points=legacy_s3_freq.size,
            interlacing=True,
            old_window=True,
        )
    else:
        raise AssertionError(f"Update test parameters to include a test for {name}")

    result_store = calculate_spectra(sconfig, dataconfig_list)

    for channel in keys:
        order = len(channel)
        result = result_store.get(channel)
        assert old_spectra is not None
        assert result.spectrum is not None

        if name == "auto":
            normalized_channel = channel[0]
        else:
            normalized_channel = channel

        if order == 3:
            assert result.freq is not None
            assert result.spectrum.shape == (result.freq.size, result.freq.size)
            actual_spectrum = np.asarray(result.spectrum)
            expected_spectrum = np.asarray(old_spectra[normalized_channel][order])
            actual_spectrum, expected_spectrum = _legacy_order3_region(
                actual_spectrum,
                np.asarray(result.freq),
                expected_spectrum,
                np.asarray(old_freqs[normalized_channel][order]),
            )
            np.testing.assert_allclose(
                actual_spectrum,
                expected_spectrum,
                rtol=1e-6,
                atol=1e-8,
                err_msg=(
                    f"Legacy third-order region for channel {channel} doesn't match "
                    "the corresponding new full-axis result."
                ),
            )
            continue

        actual_spectrum = np.asarray(result.spectrum)
        expected_spectrum = np.asarray(old_spectra[normalized_channel][order])
        actual_spectrum, expected_spectrum = _common_region(actual_spectrum, expected_spectrum)
        np.testing.assert_allclose(
            actual_spectrum,
            expected_spectrum,
            rtol=1e-6,
            atol=1e-8,
            err_msg=f"Spectrum at order {order} for channel {channel} doesn't match.",
        )
        assert old_freqs is not None
        assert result.freq is not None
        expected_freq = (
            np.asarray([0.0]) if order == 1 else np.asarray(old_freqs[normalized_channel][order])
        )
        actual_freq = np.asarray(result.freq)
        actual_freq, expected_freq = _common_region(actual_freq, expected_freq)
        np.testing.assert_allclose(
            actual_freq,
            expected_freq,
            rtol=0,
            atol=1e-12,
            err_msg=f"Frequency axis at order {order} for channel {channel} doesn't match.",
        )

        if compare_error:
            assert old_error is not None
            assert result.spectrum_error is not None

            np.testing.assert_allclose(
                np.asarray(result.spectrum_error),
                np.asarray(old_error[normalized_channel][order]),
                rtol=1e-6,
                atol=1e-8,
                err_msg=f"Spectrum error at order {order} for channel {channel} doesn't match.",
            )
