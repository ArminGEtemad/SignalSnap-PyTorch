import numpy as np
import pytest

from signalsnap_pytorch import SpectrumResult, SpectrumResultStore


@pytest.mark.parametrize(
    ("freq", "spectrum", "spectrum_error", "message"),
    [
        pytest.param(
            np.zeros((2, 2)),
            np.zeros(2, dtype=complex),
            None,
            "Frequency axis must be one-dimensional",
            id="frequency-dimensions",
        ),
        pytest.param(
            np.zeros(2),
            np.zeros(3, dtype=complex),
            None,
            "spectrum has shape",
            id="spectrum-shape",
        ),
        pytest.param(
            np.zeros(2),
            np.zeros(2, dtype=complex),
            np.zeros(3, dtype=complex),
            "Spectrum error must have the same shape",
            id="error-shape",
        ),
    ],
)
def test_spectrum_result_rejects_inconsistent_array_shapes(
    freq,
    spectrum,
    spectrum_error,
    message,
):
    with pytest.raises(ValueError, match=message):
        SpectrumResult(
            channels=(0, 0),
            freq=freq,
            freq_unit="Hz",
            spectrum=spectrum,
            spectrum_error=spectrum_error,
        )


def make_result(channels: tuple[int, ...]) -> SpectrumResult:
    frequency_points = 3
    order = len(channels)

    shape = {
        1: (1,),
        2: (frequency_points,),
        3: (frequency_points, frequency_points),
        4: (frequency_points, frequency_points),
    }[order]

    freq = np.array([0.0]) if order == 1 else np.arange(frequency_points)

    return SpectrumResult(
        channels=channels,
        freq=freq,
        freq_unit="Hz",
        spectrum=np.zeros(shape, dtype=complex),
    )


def test_result_store_supports_dictionary_style_access():
    result = make_result((0, 1))
    store = SpectrumResultStore()
    store.add(result)

    assert len(store) == 1
    assert (0, 1) in store
    assert store[(0, 1)] is result
    assert store.get((0, 1)) is result
    assert store.get((1, 0)) is None
    assert list(store.keys()) == [(0, 1)]
    assert list(store.values()) == [result]
    assert list(store.items()) == [((0, 1), result)]


def test_result_store_iteration_still_yields_results():
    result = make_result((0, 1))
    store = SpectrumResultStore()
    store.add(result)

    assert list(store) == [result]


def test_result_store_indexing_raises_for_missing_result():
    store = SpectrumResultStore()

    with pytest.raises(KeyError):
        _ = store[(0, 1)]
