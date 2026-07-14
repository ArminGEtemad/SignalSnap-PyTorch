import h5py
import numpy as np

from multichss import DataConfig, HDF5Channel, SpectrumConfig, calculate_spectra


REQUESTED_SPECTRA = [
    (0,),
    (0, 0),
    (0, 1),
    (0, 1, 0),
    (0, 1, 0, 1),
]


def _assert_result_stores_equal(actual_store, expected_store):
    for channels in REQUESTED_SPECTRA:
        actual = actual_store.get(channels)
        expected = expected_store.get(channels)

        np.testing.assert_array_equal(actual.freq, expected.freq)
        assert actual.freq_unit == expected.freq_unit
        assert actual.channels == expected.channels
        np.testing.assert_allclose(
            actual.spectrum,
            expected.spectrum,
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )

        if expected.spectrum_error is None:
            assert actual.spectrum_error is None
        else:
            assert actual.spectrum_error is not None
            np.testing.assert_allclose(
                actual.spectrum_error,
                expected.spectrum_error,
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            )


def test_hdf5_and_mixed_channels_match_eager_array_pipeline(tmp_path):
    rng = np.random.default_rng(12345)
    stored = rng.normal(size=(10, 64, 3))
    path = tmp_path / "signals.h5"

    with h5py.File(path, "w") as file:
        file.create_dataset("/signals", data=stored, chunks=(2, 64, 1))

    eager_channels = [
        stored[1:9, :, 0].reshape(-1),
        stored[1:9, :, 1].reshape(-1),
    ]
    eager_config = DataConfig(channels=eager_channels, dt=0.1, t_unit="s")
    hdf5_config = DataConfig(
        channels=[
            HDF5Channel(
                file=path,
                dataset="/signals",
                selection=(slice(1, 9), slice(None), 0),
            ),
            HDF5Channel(
                file=path,
                dataset="/signals",
                selection=(slice(1, 9), slice(None), 1),
            ),
        ],
        dt=0.1,
        t_unit="s",
    )
    mixed_config = DataConfig(
        channels=[hdf5_config.channels[0], eager_channels[1]],
        dt=0.1,
        t_unit="s",
    )
    spectrum_config = SpectrumConfig(
        f_min=-2.5,
        f_max=2.5,
        frequency_points=9,
        m=4,
        spectral_estimates_max=3,
        interlacing=True,
    )

    expected = calculate_spectra(
        eager_config,
        spectrum_config,
        requested_spectra=REQUESTED_SPECTRA,
    )

    for actual_config in (hdf5_config, mixed_config):
        actual = calculate_spectra(
            actual_config,
            spectrum_config,
            requested_spectra=REQUESTED_SPECTRA,
        )
        _assert_result_stores_equal(actual, expected)


def test_pipeline_does_not_open_unrequested_hdf5_channel(tmp_path):
    data_config = DataConfig(
        channels=[
            np.arange(128, dtype=float),
            HDF5Channel(
                file=tmp_path / "missing.h5",
                dataset="/signals",
                selection=(slice(None),),
            ),
        ],
        dt=1.0,
    )
    spectrum_config = SpectrumConfig(
        f_min=0.0,
        f_max=0.5,
        frequency_points=5,
        m=4,
    )

    result_store = calculate_spectra(
        data_config,
        spectrum_config,
        requested_spectra=[(0, 0)],
    )

    assert result_store.get((0, 0)).channels == (0, 0)
