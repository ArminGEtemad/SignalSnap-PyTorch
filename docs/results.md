# Working with results

[Documentation index](README.md) · [Calculation configuration](configuration.md)

`calculate_spectra` returns a `SpectrumResultStore`. Retrieve a result using the same channel tuple
used in the request:

```python
result = results[(0, 1)]

result.channels         # (0, 1)
result.order            # 2
result.freq             # one-dimensional NumPy frequency axis
result.freq_unit        # for example, "Hz"
result.spectrum         # complex NumPy array
result.spectrum_error   # complex NumPy array, or None
```

## Shapes and frequency coordinates

`result.freq` is the authoritative frequency axis. `f_min`, `f_max`, and `df` from `SpectrumConfig`
should not be used to reconstruct the frequency axis.

For `N = len(result.freq)`, the returned shapes are:

| Order | `spectrum` shape | Interpretation |
| --- | --- | --- |
| 1 | `(1,)` | Zero-frequency component corresponding to the signal mean |
| 2 | `(N,)` | Spectrum evaluated along one frequency axis |
| 3 | `(N, N)` | Bispectrum evaluated on a two-dimensional grid |
| 4 | `(N, N)` | Diagonal two-dimensional trispectrum slice |

For order one, `result.freq` is just `[0]`.

A third-order value `S^(3)(frequency[i], frequency[j])` is stored in
`result.spectrum[i, j]`. It is `NaN` when the required third frequency,
`-(frequency[i] + frequency[j])`, lies outside the FFT support.

An order-four value at `[i, j]` belongs to the diagonal slice
`(frequency[i], -frequency[i], frequency[j], -frequency[j])`. SignalSnap does not currently return
the full three-dimensional trispectrum.

## Standard errors

`spectrum_error` has the same shape as `spectrum`.
Its real and imaginary components independently store the standard errors of the corresponding
spectrum components; the complex values themselves have no statistical interpretation.

```python
real_sem = result.spectrum_error.real
imaginary_sem = result.spectrum_error.imag
```

It is `None` unless at least two unshifted estimates are available.

See [Calculation configuration](configuration.md) to learn how the `interlacing` option influences
the result.

## Physical units

SignalSnap does not attach amplitude units to channels. If channel $k$ has amplitude unit $X_k$,
an order-$n$ spectrum has units

```math
\left(\prod_{k=1}^{n} X_k \right)\mathrm{t\_unit}^{n-1}
```

or equivalently


```math
\left(\prod_{k=1}^{n} X_k \right)\mathrm{freq\_unit}^{1-n}.
```

`freq_unit` is the inverse-time unit corresponding to the `t_unit` supplied in `DataConfig`.

## Using the result store

The store preserves insertion order. Iteration yields `SpectrumResult` objects rather than channel
tuples:

```python
for result in results:
    print(result.channels, result.spectrum.shape)

print(f"Received {len(results)} results")

if (0, 1) in results:
    cross_spectrum = results[(0, 1)]
```

Accessing an unavailable tuple raises `KeyError`, distinguishing a missing result from one
containing zeros or `NaN` values.

Select an explicit set of results:

```python
selected = results.select([(0, 0), (0, 1)])
```

Filter by order or by the presence of a channel anywhere in the tuple:

```python
second_order = results.select_by_order(2)
involving_channel_zero = results.select_by_channel(0)
```

Filtered selections return an empty `SpectrumResultStore` when nothing matches. Every selection
returns a new store that shares the same `SpectrumResult` objects and underlying arrays with the
original store rather than copying them.

## Partial failures and warnings

Failures while computing, accumulating, or finalizing an individual requested spectrum are isolated
to the affected channel tuple. SignalSnap emits a `RuntimeWarning`, omits that result, and
continues with the other requests. Treat these warnings as significant and verify that the returned
store contains every requested tuple.

Next: [Plotting your results](plotting.md).
