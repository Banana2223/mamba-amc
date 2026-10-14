# P1-T9 minimal end-to-end chain report

- Generated: `2026-10-14T19:12:28+08:00`
- Validation passed: `True`
- Device: `cuda`
- Probe model parameters: `10,712`
- Train steps: `3`
- Total elapsed: `1.3098` seconds

## Verified chain

`fixed split -> HDF5 on-demand read -> RMS preprocessing -> DataLoader -> TinyAMCProbe -> cross-entropy -> backward -> optimizer update -> validation`

## Checks

- input_iq_shape: `True`
- input_iq_dtype: `True`
- input_iq_finite: `True`
- input_unit_complex_power: `True`
- input_label_shape: `True`
- input_label_dtype: `True`
- input_label_range: `True`
- input_snr_shape: `True`
- input_snr_values: `True`
- input_index_shape: `True`
- input_unique_indices: `True`
- same_seed_first_batch_reproducible: `True`
- logits_shape_is_batch_x_24: `True`
- train_logits_finite: `True`
- all_train_losses_finite: `True`
- gradient_tensors_exist: `True`
- gradients_finite: `True`
- gradients_nonzero: `True`
- optimizer_changed_parameters: `True`
- validation_logits_shape_is_batch_x_24: `True`
- validation_logits_finite: `True`
- validation_loss_finite: `True`

## Tensor contract

- Input batch: `[64, 2, 1024]` `torch.float32`
- Labels: `[64]` `torch.int64`
- Logits: `[64, 24]`

## Training smoke values

- Losses: `[3.211574077606201, 3.2109100818634033, 3.218555212020874]`
- Last gradient L2 norm: `0.547790`
- Validation loss on one batch: `3.223816`

These loss values are only software sanity checks. They are not model performance results and must not be used in the thesis comparison.

- CUDA peak memory: `37.75` MiB
