import numpy as np
import pytest
import torch

from multichss.aggregator import accumulate_spectrum, finalize_result
from multichss.results import SpectrumAccumulator, SpectrumResult


def make_accumulator(
    channels: tuple[int, ...] = (0, 0),
    frequency_points: int = 2,
) -> SpectrumAccumulator:
    freq = np.asarray([0.0]) if len(channels) == 1 else np.arange(frequency_points, dtype=float)
    return SpectrumAccumulator(channels=channels, freq=freq, freq_unit="Hz")


def test_accumulate_spectrum_keeps_shifted_and_unshifted_state_separate():
    accumulator = make_accumulator()
    unshifted = torch.tensor([1 + 2j, 3 + 4j], dtype=torch.complex128)
    shifted = torch.tensor([5 + 6j, 7 + 8j], dtype=torch.complex128)

    accumulate_spectrum(accumulator, unshifted)
    accumulate_spectrum(accumulator, shifted, shifted=True)

    torch.testing.assert_close(accumulator.spectrum_sum_unshifted, unshifted)
    torch.testing.assert_close(accumulator.spectrum_sum_shifted, shifted)
    torch.testing.assert_close(
        accumulator.squared_sum_unshifted,
        torch.tensor([1 + 4j, 9 + 16j], dtype=torch.complex128),
    )
    torch.testing.assert_close(
        accumulator.squared_sum_shifted,
        torch.tensor([25 + 36j, 49 + 64j], dtype=torch.complex128),
    )
    assert accumulator.chunks_unshifted == 1
    assert accumulator.chunks_shifted == 1


def test_finalize_result_calculates_mean_and_componentwise_sem():
    accumulator = make_accumulator()
    accumulate_spectrum(
        accumulator,
        torch.tensor([1 + 2j, 3 + 4j], dtype=torch.complex128),
    )
    accumulate_spectrum(
        accumulator,
        torch.tensor([3 + 4j, 5 + 8j], dtype=torch.complex128),
    )

    result = finalize_result(accumulator)

    np.testing.assert_allclose(result.spectrum, np.asarray([2 + 3j, 4 + 6j]))
    np.testing.assert_allclose(result.spectrum_error, np.asarray([1 + 1j, 1 + 2j]))
    assert result.channels == accumulator.channels
    assert result.freq is accumulator.freq
    assert result.freq_unit == accumulator.freq_unit


def test_finalize_result_combines_groups_and_uses_larger_componentwise_sem():
    accumulator = make_accumulator(channels=(0,), frequency_points=1)

    for value in (1 + 1j, 3 + 3j):
        accumulate_spectrum(accumulator, torch.tensor([value], dtype=torch.complex128))

    for value in (10 + 2j, 14 + 6j):
        accumulate_spectrum(
            accumulator,
            torch.tensor([value], dtype=torch.complex128),
            shifted=True,
        )

    result = finalize_result(accumulator)

    np.testing.assert_allclose(result.spectrum, np.asarray([7 + 3j]))
    np.testing.assert_allclose(result.spectrum_error, np.asarray([2 + 2j]))


def test_finalize_result_with_one_estimate_warns_and_returns_no_error():
    accumulator = make_accumulator()
    estimate = torch.tensor([1 + 2j, 3 + 4j], dtype=torch.complex128)
    accumulate_spectrum(accumulator, estimate)
    sum_before = accumulator.spectrum_sum_unshifted.clone()

    with pytest.warns(RuntimeWarning, match="at least two spectral estimates"):
        result = finalize_result(accumulator)

    np.testing.assert_allclose(result.spectrum, estimate.numpy())
    assert result.spectrum_error is None
    torch.testing.assert_close(accumulator.spectrum_sum_unshifted, sum_before)
    assert accumulator.chunks_unshifted == 1


def test_finalize_result_rejects_empty_accumulator():
    accumulator = make_accumulator()

    with pytest.raises(RuntimeError, match="no spectra were accumulated"):
        finalize_result(accumulator)


@pytest.mark.parametrize(
    ("spectrum_sum", "squared_sum", "chunks"),
    [
        pytest.param(None, torch.ones(2, dtype=torch.complex128), 0, id="squares-without-sum"),
        pytest.param(None, None, 1, id="count-without-sum"),
        pytest.param(torch.ones(2, dtype=torch.complex128), None, 1, id="sum-without-squares"),
        pytest.param(
            torch.ones(2, dtype=torch.complex128),
            torch.ones(2, dtype=torch.complex128),
            0,
            id="sum-without-count",
        ),
    ],
)
def test_finalize_result_rejects_inconsistent_accumulator_state(
    spectrum_sum,
    squared_sum,
    chunks,
):
    accumulator = make_accumulator()
    accumulator.spectrum_sum_unshifted = spectrum_sum
    accumulator.squared_sum_unshifted = squared_sum
    accumulator.chunks_unshifted = chunks

    with pytest.raises(RuntimeError, match="accumulator state is inconsistent"):
        finalize_result(accumulator)


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
