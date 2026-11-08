# Changelog

## 0.2.0 (unreleased)

### Added
- **PyTorch inference**: `predict_torch.py` runs all eleven models with PyTorch (`fetch.pytorch`). It takes
  `predict.py`'s arguments and writes the same `results_<model>.csv`; `--model` can name several models
  (`--model abcdefghijk`), which share one read of the candidates, and `--data_dir` can be repeated.
  - The models are built from the original Keras weight files, using each file's own `model_config`, so the
    architectures can't drift from the trained weights. Batch normalisation is folded into the convolutions for
    inference.
  - On the same candidates the probabilities match TensorFlow FETCH's to within a few times 1e-4 for all eleven
    models, with no label changes ([docs/validation.md](docs/validation.md)). It runs in full float32 (TF32 off),
    so repeated runs give identical results.
  - Per model it is about 2x (Pascal) to 4x (Ampere) faster than `predict.py` on a GPU.
  - A candidate containing NaN is scored with the NaNs set to 0, where `predict.py` stops at it.
- `FETCH_MODEL_DIR` sets where the weight files are read from (for example inside a container); by default
  they are shared with `predict.py` in `~/.keras/models`.
- Tests that need neither TensorFlow nor the weights (graph builder, candidate reader, weight download), tests
  of all eleven models against TensorFlow's probabilities, and a GitHub Actions workflow running them on CPU with
  model a.

### Changed
- TensorFlow and PyTorch are optional extras: `pip install ".[torch]"` or `pip install ".[tensorflow]"`.
  The tensorflow extra also installs scikit-learn, which `train.py` needs.

### Fixed
- `train.py` held out `1 - --val_split` of the data for validation instead of `--val_split`, so by default it
  trained on 20% of the candidates and validated on 80%.
