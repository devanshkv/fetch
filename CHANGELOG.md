# Changelog

## 0.2.0 (unreleased)

### Added
- **PyTorch inference**: `predict_torch.py` runs all eleven models with PyTorch (`fetch.pytorch`). It takes
  `predict.py`'s arguments and writes the same `results_<model>.csv`; `--model` can name several models
  (`--model abcdefghijk`), which share one read of the candidates, and `--data_dir` can be repeated.
  - The models are built from the original Keras weight files, using each file's own `model_config`, so the
    architectures can't drift from the trained weights. Batch normalisation is folded into the convolutions for
    inference.
  - On 41,993 candidates the probabilities match TensorFlow FETCH's (TF32 off) to within 3.7e-4 for ten of the
    eleven models and 1.4e-3 for model j, where TensorFlow's GPU result is the one that's off, with no label
    changes ([docs/validation.md](docs/validation.md)). It runs in full float32 (TF32 off), so repeated runs give
    identical results.
  - Per model it is about 2x faster than `predict.py` on a GPU, on Pascal and Ampere (1.9x to 3.7x on an
    8 GB Ampere GPU).
  - A candidate containing NaN is scored with the NaNs set to 0, where `predict.py` stops at it.
- `FETCH_MODEL_DIR` sets where the weight files are read from (for example inside a container); by default
  they are shared with `predict.py` in `~/.keras/models`.
- Tests that need neither TensorFlow nor the weights (graph builder, candidate reader, weight download), tests
  of all eleven models against TensorFlow's probabilities, and a GitHub Actions workflow running them on CPU with
  model a.

### Changed
- Python 3.11 or newer is required, as for current PyTorch (2.15); CI tests 3.11 and 3.14.
- TensorFlow and PyTorch are optional extras: `pip install ".[torch]"` or `pip install ".[tensorflow]"`.
  The tensorflow extra also installs scikit-learn, which `train.py` needs.

### Fixed
- `train.py` held out `1 - --val_split` of the data for validation instead of `--val_split`, so by default it
  trained on 20% of the candidates and validated on 80%.
