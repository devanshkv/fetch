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

On 21,998 single-pulse candidates
---
The same `predict.py` setup (TensorFlow 2.15, TF32 off, batch 64) on an 8 GB Ampere GPU, on 21,998 single-pulse
candidates. 18 more candidates of the set, whose frequency-time data is constant, were left out because `predict.py`
stops at them. Positives are the same with either.

| Model | Largest difference | 99th percentile | Median difference | Positives | Label changes |
|---|---|---|---|---|---|
| a | 7.6e-6 | 1.9e-6 | 3.8e-11 | 7,300 | 0 |
| b | 5.1e-5 | 3.8e-6 | 2.7e-12 | 10,484 | 0 |
| c | 9.2e-6 | 1.7e-6 | 1.6e-11 | 9,118 | 0 |
| d | 6.0e-6 | 1.8e-6 | 5.1e-11 | 8,676 | 0 |
| e | 1.1e-5 | 1.8e-6 | 5.0e-12 | 7,143 | 0 |
| f | 2.8e-5 | 6.0e-6 | 1.4e-15 | 9,318 | 0 |
| g | 7.7e-5 | 5.7e-6 | 1.3e-12 | 8,470 | 0 |
| h | 6.9e-5 | 3.9e-6 | 5.1e-14 | 6,910 | 0 |
| i | 1.8e-5 | 3.1e-6 | 3.3e-10 | 11,063 | 0 |
| j | 1.4e-3 | 5.1e-5 | 3.5e-10 | 8,047 | 0 |
| k | 1.5e-4 | 2.8e-6 | 2.4e-12 | 10,260 | 0 |

Model j's larger differences are TensorFlow's. On j's 100 most different candidates, against a float64 reference
(the PyTorch model in float64 on the CPU): TensorFlow on the GPU is off by up to 1.4e-3 (median 1.2e-4),
TensorFlow on the CPU by up to 1.0e-5, and `predict_torch.py` by up to 2.3e-5, with or without batch
normalisation folded. Even with cuDNN's autotuning off (`TF_CUDNN_USE_AUTOTUNE=0`), TensorFlow on the GPU stays
4.3e-4 off for these candidates.

On another 19,995 single-pulse candidates
---
The same setup on the same GPU, on 19,995 single-pulse candidates drawn at random from a different set (5 more with
constant frequency-time data left out). This set has few positives, so it tests the probabilities more than
the labels near 0.5. Positives are the same with either.

| Model | Largest difference | 99th percentile | Median difference | Positives | Label changes |
|---|---|---|---|---|---|
| a | 4.3e-6 | 3.3e-7 | 6.4e-11 | 0 | 0 |
| b | 1.1e-5 | 1.4e-6 | 9.2e-11 | 445 | 0 |
| c | 1.1e-5 | 9.6e-7 | 5.2e-10 | 226 | 0 |
| d | 6.3e-6 | 8.6e-7 | 5.9e-10 | 272 | 0 |
| e | 5.4e-6 | 2.2e-7 | 1.5e-11 | 38 | 0 |
| f | 9.5e-6 | 4.7e-7 | 6.0e-13 | 120 | 0 |
| g | 8.8e-6 | 6.2e-8 | 3.1e-13 | 40 | 0 |
| h | 1.1e-5 | 7.5e-7 | 2.8e-13 | 116 | 0 |
| i | 2.6e-5 | 1.4e-6 | 6.5e-9 | 380 | 0 |
| j | 7.1e-5 | 4.1e-6 | 7.6e-12 | 99 | 0 |
| k | 3.7e-4 | 4.5e-5 | 2.0e-10 | 427 | 0 |

TensorFlow's model g stopped on this 8 GB GPU with "CUDNN failed to allocate the scratch space"
(also with `TF_CUDNN_USE_AUTOTUNE=0`); it ran with `TF_GPU_ALLOCATOR=cuda_malloc_async`.

Speed
---
On the 21,998 candidates, on the same GPU, each model run on its own (seconds):

| | a | b | c | d | e | f | g | h | i | j | k | Total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `predict.py` (TF32 off) | 434 | 443 | 438 | 433 | 409 | 430 | 444 | 456 | 443 | 443 | 438 | 4,813 |
| `predict_torch.py` | 180 | 159 | 194 | 219 | 211 | 177 | 203 | 235 | 202 | 213 | 118 | 2,111 |

`predict_torch.py --model abcdefghijk`, all eleven on one read of the candidates, took 2,002 s. On the 19,995
other candidates, `predict.py` took 353 to 383 s per model (4,061 s for all eleven) and
`predict_torch.py --model abcdefghijk` 1,842 s.

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
- `predict.py` with TF32 off can still change between runs, because cuDNN's autotuning can choose different
  convolution algorithms each time: two runs with the same settings on the same 200 candidates differed by up to
  1.8e-3 for model j and 1.3e-4 for model h. With `TF_CUDNN_USE_AUTOTUNE=0` the runs were identical.
- With TF32 on, `predict.py`'s probabilities differ from its own TF32-off results by up to ~1e-2 and between
  repeated runs by up to 6.3e-4, mostly for probabilities between 0.05 and 0.95. So with TF32 on, TensorFlow's
  labels for candidates close to 0.5 can change from run to run.

Tests
---
`tests/test_pytorch_models.py` keeps TensorFlow's probabilities for all eleven models on two candidates,
`tests/test.h5` (a bright pulse every model is sure of) and `tests/synthetic.h5` (a faint pulse the models
disagree about, from `tests/make_synthetic.py`), and checks that the PyTorch models stay within 1e-4 of them.
