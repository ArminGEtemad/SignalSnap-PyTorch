# HDF5 input

[Documentation index](README.md) · [Plotting your results](plotting.md)

SignalSnap can read only the required slices of HDF5 datasets instead of loading an entire
measurement into memory. This allows datasets much larger than system memory to be processed
incrementally.

Install the optional dependency:

```bash
python -m pip install ".[hdf5]"
```

## Defining channels

An `HDF5Channel` identifies a file, dataset, and selection. The selected values per spectral
estimate are flattened into one signal channel:

```python
from pathlib import Path

from signalsnap_pytorch import DataConfig, HDF5Channel

data_config = DataConfig(
    channels=(
        HDF5Channel(
            file=Path("measurement.h5"),
            dataset="/signals",
            selection=(slice(None), slice(None), 0),
        ),
        HDF5Channel(
            file=Path("measurement.h5"),
            dataset="/signals",
            selection=(slice(None), slice(None), 1),
        ),
    ),
    dt=2.0,
    t_unit="ns",
)
```

This example turns the last-axis entries `0` and `1` into separate SignalSnap channels.

## Selection rules

- Selection entries must be integers or `slice(...)`.
- Slice steps other than `1` are not supported.
- A selection may leave at most two dataset dimensions unfixed.
- Remaining dimensions are flattened in C/row-major order.
- The resulting channel must be nonempty and contain real-valued numeric or Boolean data.
- All active channels must contain the same number of selected samples.

## Mixing storage types

In-memory arrays and `HDF5Channel` objects may be combined in one `DataConfig`:

```python
data_config = DataConfig(
    channels=(in_memory_reference, hdf5_measurement),
    dt=2.0,
    t_unit="ns",
)
```

Only channels used by `requested_spectra` are opened and read. During calculation, SignalSnap reads
one required chunk at a time, so a selected dataset can exceed system memory. FFT coefficients and
accumulators still require memory on the selected compute device; the exact amount depends on the
frequency grid, requested orders, and channel tuples.

Next: [Scientific background](scientific-background.md).