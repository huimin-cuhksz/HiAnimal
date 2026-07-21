# HiAnimal training and reconstruction

This directory contains the minimal code and bundled assets needed to train the
HiAnimal Occupancy-UV network and run a single-image reconstruction example.

## Contents

- `apps/` and `lib/`: training, inference, model, sampling, and mesh utilities.
- `training_examples/`: 10 complete training examples for a reproducible smoke test.
- `test_cases/`: cat and sheep images with their estimated normal maps.
- `checkpoints/netG_latest`: checkpoint used by the reconstruction example.
- `test_results/cat/`: example OBJ reconstruction and predicted UV coordinates.

Use the Conda environment described in the repository's root README.

## Training

Run from any directory:

```bash
./hianimal_train/train.sh
```

The launcher uses GPU 2, batch size 8, and two data-loading workers. It trains
on the 10 bundled examples and writes new weights to
`hianimal_train/checkpoints/example/`.

Each training example contains:

```text
training_examples/
├── 36views/<subject>/
│   ├── sn_normal/<view>.png
│   └── calib/<view>.txt
└── 36views_sample_uv_whole/<subject>/
    └── geo/scale_uv.npz
```

- `sn_normal/<view>.png` is the 3-channel normal-map input.
- `calib/<view>.txt` is the matching camera calibration.
- `scale_uv.npz` contains 3D points with occupancy and UV supervision shared
  across the views of one subject.

The full experiment used 3,065 subjects and 19 views per subject from 100 to
280 degrees in 10-degree increments. The bundled 10-example subset is intended
for code verification and demonstration, not for reproducing the full training.

## Single-image reconstruction

Run the bundled cat example:

```bash
./hianimal_train/predict.sh
```

The script loads `checkpoints/netG_latest`, reads
`test_cases/1_cat/cat_normal.png`, and writes:

```text
test_results/cat/cat_prediction.obj
test_results/cat/cat_prediction.npz
```

The OBJ stores the reconstructed mesh. The NPZ stores one UV coordinate per
mesh vertex. To test another normal map, change the input and output paths in
`predict.sh` or invoke `python -m apps.predict_single` directly.

## Notes

- `CUDA_VISIBLE_DEVICES` is set in `train.sh` and `predict.sh`; change it for
  your machine if GPU 2 is unavailable.
- Run-time caches, new experiment results, and newly trained checkpoints under
  `checkpoints/example/` are excluded by `.gitignore`.
- The hourglass implementation retains the upstream PIFu MIT license notice in
  `lib/model/HGFilters_hd.py`.
