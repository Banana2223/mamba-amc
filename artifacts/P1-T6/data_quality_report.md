# P1-T6 RML2018.01A data exploration and quality report

- Generated: `2026-10-14T13:26:50+08:00`
- Scan mode: `full`
- Dataset: `data/raw/RML2018.01a/GOLD_XYZ_OSC.0001_1024.hdf5`
- Validation passed: `True`
- P1-T6 completion ready: `True`

## Structure

| Dataset | Shape | Dtype |
|---|---:|---|
| X | `(2555904, 1024, 2)` | `float32` |
| Y | `(2555904, 24)` | `int64` |
| Z | `(2555904, 1)` | `int64` |

## Distribution

- Total samples: `2,555,904`
- Classes: `24`
- SNR levels: `26`
- Class-SNR groups: `624`
- Samples per group: `4096` to `4096`
- Invalid one-hot rows: `0`

## IQ quality

- Scope: `full_dataset`
- Samples checked: `2,555,904`
- Non-finite values: `0`
- Samples containing non-finite values: `0`
- All-zero samples: `0`
- I range: `-68.3234` to `51.5626`
- Q range: `-40.2193` to `46.9531`

## Figures

- class_snr_distribution: `artifacts/P1-T6/class_snr_distribution.png`
- sample_waveforms: `artifacts/P1-T6/sample_waveforms.png`
- sample_constellations: `artifacts/P1-T6/sample_constellations.png`
- iq_value_distribution: `artifacts/P1-T6/iq_value_distribution.png`
- signal_power_by_snr: `artifacts/P1-T6/signal_power_by_snr.png`

## Conclusion

Full X scan completed; P1-T6 can be closed.

The dataset is synthetic historical research data. Passing this quality check does not remove the need to report dataset errata and the lack of real over-the-air validation as limitations.
