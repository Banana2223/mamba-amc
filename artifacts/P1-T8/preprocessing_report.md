# P1-T8 IQ preprocessing and augmentation report

- Generated: `2026-10-14T18:29:02+08:00`
- Validation passed: `True`
- Example: QPSK at `20` dB, HDF5 row `507904`

## Fixed preprocessing protocol

- Input representation: real tensor `(2, length)`, channel order I then Q.
- Default sequence length: 1024; shorter experiments use random train crop and center validation/test crop.
- Normalization: per-sample complex RMS, with one shared scale for I and Q.
- No per-channel centering: carrier/DC information that may distinguish AM subclasses is retained.
- Validation and test transforms are deterministic and never augmented.
- Training augmentation is disabled for the first baseline and enabled only in explicitly marked experiments.

## Optional training augmentations

- AWGN: probability 0.25, added-noise SNR sampled from 15 to 30 dB.
- Constant phase rotation: probability 0.50, angle up to +/-pi.
- Carrier frequency offset: probability 0.25, up to +/-0.005 cycles/sample.
- Multipath echoes: probability 0.20, at most 2 echoes and 16-sample delay.

The original dataset SNR metadata is retained as provenance. After synthetic channel augmentation it must not be interpreted as the exact effective SNR.

## Verification

- eval_is_deterministic: `True`
- seeded_augmentation_is_reproducible: `True`
- seeded_dataloader_augmentation_is_reproducible: `True`
- different_seed_changes_augmentation: `True`
- all_outputs_are_finite: `True`
- full_length_shape_is_2x1024: `True`
- short_crop_shape_is_2x512: `True`
- normalized_power_is_one: `True`
- augmented_power_is_one: `True`
- loader_batch_shape_is_32x2x1024: `True`
- loader_metadata_shapes_are_valid: `True`
- loader_samples_have_unit_power: `True`

- Raw complex power: `0.997186`
- Normalized complex power: `1.000000`
- Augmented/normalized complex power: `1.000000`
- Figure: `artifacts/P1-T8/preprocessing_examples.png`
