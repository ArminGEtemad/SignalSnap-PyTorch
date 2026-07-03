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

auto_orders = [1, 2, 3, 4]
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


@pytest.fixture(scope="module")
def prepared_data():
    with h5py.File("./tests/test_data/datasets/5Qubit_short_data.h5", "r") as f:
        x_test_dataset = f["/X_test"]
        assert isinstance(x_test_dataset, h5py.Dataset)
        X_test = x_test_dataset[...]

    return [
        DataConfig(data=X_test[:1000, :, 0].reshape(-1), dt=2.0, t_unit="ns"),
        DataConfig(data=X_test[:1000, :, 1].reshape(-1), dt=2.0, t_unit="ns"),
    ], [0, 1]


@pytest.mark.parametrize(
    ("name", "reference_file", "keys_or_orders"),
    [
        pytest.param(
            "auto",
            "./tests/test_data/references/5Qubit_short_data_auto_corr.npz",
            auto_orders,
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
def test_new_vs_old_api(name, reference_file, keys_or_orders, prepared_data):
    """
    Tests if the refactor implements the same calculations. To check if the
    implementation is correct, the old window function needs to be used in
    SpectrumConfig.
    """
    dataconfig_list, selected_data = prepared_data
    benchmark_spectra = np.load(reference_file, allow_pickle=True)
    old_spectra = benchmark_spectra["spectra"].item()

    if compare_error:
        old_error = benchmark_spectra["error"].item()
    else:
        old_error = None

    old_freqs = benchmark_spectra["freqs"].item()
    
    auto_spectra_orders = []
    cross_spectra = []
    if isinstance(keys_or_orders[0],  int):
        auto_spectra_orders = keys_or_orders
    else:
        cross_spectra = keys_or_orders

    if name == "auto":
        sconfig = SpectrumConfig(
            f_min=0,
            f_max=0.25,
            s3_calc="1/4",
            device="cpu",
            auto_spectra_orders=auto_spectra_orders,   
            frequency_points=100,
            interlacing=True,
            old_window=True,
        )
    elif name == "cross_ch24":
        sconfig = SpectrumConfig(
            f_min=-0.25,
            f_max=0.25,
            s3_calc="1/4",
            device="cpu",
            auto_spectra_orders=[],
            cross_spectra=cross_spectra,
            frequency_points=100,
            interlacing=True,
            old_window=True,
        )
    elif name == "cross_ch3":
        sconfig = SpectrumConfig(
            f_min=0,
            f_max=0.25,
            s3_calc="1/2",
            device="cpu",
            auto_spectra_orders=[],
            cross_spectra=cross_spectra,
            frequency_points=100,
            interlacing=True,
            old_window=True,
        )
    else:
        raise AssertionError(f"Update test parameters to include a test for {name}")

    result_store = calculate_spectra(sconfig, dataconfig_list, selected=selected_data)

    
    if auto_spectra_orders:
        keys_or_orders = []
        for channel in selected_data:
            for order in auto_spectra_orders:
                channels = (channel,) * order
                keys_or_orders.append(channels)

    
    for channel in keys_or_orders:
        order = len(channel)
        result = result_store.get(channel)
        assert old_spectra is not None
        assert result.spectrum is not None

        if name == "auto":
            normalized_channel = channel[0]
        else:
            normalized_channel = channel

        np.testing.assert_allclose(
            np.asarray(result.spectrum),
            np.asarray(old_spectra[normalized_channel][order]),
            rtol=1e-6,
            atol=1e-8,
            err_msg=f"Spectrum at order {order} for channel {channel} doesn't match.",
        )
        assert old_freqs is not None
        assert result.freq is not None
        expected_freq = (
            np.asarray([0.0]) if order == 1 else np.asarray(old_freqs[normalized_channel][order])
        )
        np.testing.assert_allclose(
            np.asarray(result.freq),
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
