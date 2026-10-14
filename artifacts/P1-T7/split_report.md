# P1-T7 RML2018.01A split and loading report

- Generated: `2026-10-14T16:58:52+08:00`
- Random seed: `2026`
- Validation passed: `True`
- Loader smoke test passed: `True`

## Fixed split

| Split | Samples per class-SNR group | Total samples | Actual ratio |
|---|---:|---:|---:|
| train | 2868 | 1,789,632 | 70.0195% |
| val | 614 | 383,136 | 14.9902% |
| test | 614 | 383,136 | 14.9902% |

Each of the 624 class-SNR strata is shuffled independently before assignment. All models must reuse this split file and seed.

## Leakage and coverage checks

- Duplicate indices: `{'train': 0, 'val': 0, 'test': 0}`
- Cross-split overlaps: `0`
- Unassigned rows: `0`
- Full coverage: `True`

These checks establish index-level separation. RML2018.01A does not provide transmitter/session identity, so session-level leakage cannot be tested and must not be claimed.

## DataLoader smoke test

- train: `[64, 2, 1024]` `torch.float32`, first batch `0.1024s`, passed `True`
- val: `[64, 2, 1024]` `torch.float32`, first batch `0.0294s`, passed `True`
- test: `[64, 2, 1024]` `torch.float32`, first batch `0.0252s`, passed `True`

The loader opens HDF5 lazily per process and returns IQ as `(2, 1024)`. Normalization and augmentation are intentionally deferred to P1-T8.
