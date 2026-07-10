import numpy as np
import pytest

from multichss import SpectrumResult


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
