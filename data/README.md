# Dataset storage

This directory stores local datasets used by the project. Dataset binaries are
not committed to Git.

## RADIOML 2018.01A

- Official page: <https://www.deepsig.ai/datasets/>
- Publisher: DeepSig Inc.
- License stated by the publisher: CC BY-NC-SA 4.0
- Expected archive: `raw/RML2018.01a/2018.01.OSC.0001_1024x2M.h5.tar.gz`
- Accepted ZIP archive: `raw/RML2018.01a/RadioML201801A.zip`
- Expected data file: `raw/RML2018.01a/GOLD_XYZ_OSC.0001_1024.hdf5`
- Class order file: `raw/RML2018.01a/classes.txt`
- Preferred corrected class order: `raw/RML2018.01a/classes-fixed.json`
- License file: `raw/RML2018.01a/LICENSE.TXT`

DeepSig currently requires a short contact form before download. Complete that
step in a browser and save the archive under `raw/RML2018.01a/`. Do not use an
unattributed mirror when preparing the final experimental record.

Initialize the local directory:

```bash
/home/yzl/anaconda3/envs/qam-mamba/bin/python P1-T5-prepare_dataset.py init
```

If the browser provides a direct authorized URL, the script can download with
resume support:

```bash
/home/yzl/anaconda3/envs/qam-mamba/bin/python P1-T5-prepare_dataset.py \
  download --url 'AUTHORIZED_DIRECT_URL'
```

Extract and run the complete validation:

```bash
/home/yzl/anaconda3/envs/qam-mamba/bin/python P1-T5-prepare_dataset.py run
```

The validation reads only HDF5 metadata and tiny boundary slices. It does not
load the full `X` array into memory. The final record is written to
`artifacts/P1-T5/dataset_manifest.json`.

The official page describes this as a historical research dataset with known
errata. Its synthetic nature and lack of real over-the-air validation must be
reported as limitations in the final thesis.
