Validation of the PyTorch models
===

`predict_torch.py` was compared with TensorFlow FETCH on real survey candidates: single-pulse candidates and
folded periodicity candidates. Each table gives, per model, the
largest and the median absolute difference in probability, the number of positives (p >= 0.5) and the number of
candidates whose label changes. Every comparison found no label changes.

Against FETCH 0.1.5
---
FETCH 0.1.5 (Python 3.6, TensorFlow 1.10, Keras 2.2.4, each model built from its original .yaml), on 6,793
single-pulse candidates.

| Model | Largest difference | Median difference | Positives | Label changes |
|---|---|---|---|---|
| a | 1.9e-5 | 3.8e-11 | 52 | 0 |
| b | 6.9e-6 | 7.0e-10 | 242 | 0 |
| c | 1.7e-5 | 1.5e-11 | 194 | 0 |
| d | 8.7e-6 | 1.4e-9 | 287 | 0 |
| e | 7.7e-6 | 3.6e-11 | 89 | 0 |
| f | 1.1e-5 | 4.2e-13 | 75 | 0 |
| g | 9.3e-6 | 4.0e-13 | 71 | 0 |
| h | 4.9e-5 | 1.4e-12 | 162 | 0 |
| i | 4.2e-6 | 9.4e-9 | 272 | 0 |
| j | 6.7e-5 | 1.2e-10 | 123 | 0 |
| k | 2.0e-4 | 3.7e-11 | 495 | 0 |

Against this version's `predict.py`
---
`predict.py` with TensorFlow 2.15 and TF32 off (`tf.config.experimental.enable_tensor_float_32_execution(False)`),
each model loaded from its weight file's own architecture, on 3,033 single-pulse and folded candidates. The
InceptionResNetV2 `Lambda` layers of models h and j were given as named functions, because the original lambdas
can't be unmarshalled by newer Pythons.

| Model | Largest difference | Median difference | Positives | Label changes |
|---|---|---|---|---|
| a | 1.9e-5 | 4.5e-10 | 155 | 0 |
| b | 1.7e-5 | 4.2e-10 | 150 | 0 |
| c | 8.3e-6 | 1.3e-10 | 178 | 0 |
| d | 5.3e-6 | 7.2e-10 | 113 | 0 |
| e | 6.1e-6 | 5.3e-11 | 84 | 0 |
| f | 2.9e-6 | 9.7e-14 | 16 | 0 |
| g | 2.7e-6 | 4.7e-13 | 6 | 0 |
| h | 1.2e-4 | 4.1e-12 | 60 | 0 |
| i | 1.0e-5 | 3.8e-9 | 87 | 0 |
| j | 8.4e-5 | 6.6e-10 | 41 | 0 |
| k | 3.4e-4 | 2.0e-9 | 189 | 0 |

Model a on a larger set
---
`predict.py` with TensorFlow's default settings (TF32 on where the GPU supports it), model a, on 32,998
single-pulse and folded candidates: largest difference 1.2e-2, 99th percentile 2.6e-4, median 3.4e-8; 155
positives with either, and no label changes. The larger differences come from TF32, not from the port (see
below).

Repeatability
---
- `predict_torch.py` runs in full float32. Two runs on the same 4,854 candidates, all eleven models, differed by
  at most 1.0e-6 (median 0).
- `predict.py` with TF32 off gives the same probabilities on every run (h and j within 3e-6).
- With TF32 on, `predict.py`'s probabilities differ from its own TF32-off results by up to ~1e-2 and between
  repeated runs by up to 6.3e-4, mostly for probabilities between 0.05 and 0.95. So with TF32 on, TensorFlow's
  labels for candidates close to 0.5 can change from run to run.

Tests
---
`tests/test_pytorch_models.py` keeps TensorFlow's probabilities for all eleven models on two candidates,
`tests/test.h5` (a bright pulse every model is sure of) and `tests/synthetic.h5` (a faint pulse the models
disagree about, from `tests/make_synthetic.py`), and checks that the PyTorch models stay within 1e-4 of them.
