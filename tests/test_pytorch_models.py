"""
The eleven PyTorch models against TensorFlow FETCH, and predict_torch.py end to end.

They need the Keras weight files in $FETCH_MODEL_DIR or ~/.keras/models; a model whose file
isn't there is skipped, unless FETCH_TEST_DOWNLOAD=1, which downloads it (100-400 MB each).
FETCH_TEST_MODELS (e.g. "a") limits the tests to those models; the rest are skipped.
"""
import os
import string
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from fetch.pytorch.data import read_candidate
from fetch.pytorch.models import CACHE_DIR, MODEL_LIST, get_model

HERE = os.path.dirname(os.path.abspath(__file__))
TEST_FILES = {
    # upstream's test candidate (from the onnx branch): a bright pulse every model is sure of
    "test.h5": os.path.join(HERE, "test.h5"),
    # tests/make_synthetic.py 1.0: a faint pulse the models disagree about
    "synthetic.h5": os.path.join(HERE, "synthetic.h5"),
}

# TensorFlow 2.15 FETCH loading each weight file's own architecture (tf.keras.models.load_model);
# h and j from FETCH 0.1.5 on greenburst (Python 3.6, TF 1.10, Keras 2.2.4, the original .yaml),
# since their Lambda layers don't load under newer Pythons. Upstream's ONNX h and j are wrong
# (h is model a; j gives ~0.9989 for everything), so they aren't used.
EXPECTED = {
    "test.h5": dict(a=1.0, b=1.0, c=1.0, d=0.99703765, e=1.0, f=1.0, g=0.9999999, h=1.0,
                    i=0.9999931, j=0.99987185, k=1.0),
    "synthetic.h5": dict(a=0.017186861, b=1.45475005e-05, c=0.00046073744, d=6.6513436e-05,
                         e=4.0048e-06, f=1.6065685e-05, g=0.47861606, h=0.604673, i=0.00028426308,
                         j=0.19959074, k=0.11035772),
}
MODELS = string.ascii_lowercase[:11]


def have_weights(m):
    if m not in os.environ.get("FETCH_TEST_MODELS", MODELS):
        return False
    name = pd.read_csv(MODEL_LIST)["model"][MODELS.index(m)]
    return os.path.exists(os.path.join(CACHE_DIR, name)) or os.environ.get("FETCH_TEST_DOWNLOAD") == "1"


def needs_weights(*models):
    missing = [m for m in models if not have_weights(m)]
    return pytest.mark.skipif(bool(missing), reason=f"no weights for {missing} (FETCH_TEST_DOWNLOAD=1 fetches them)")


def batch(names):
    x = [read_candidate(TEST_FILES[n]) for n in names]
    ft = torch.stack([torch.from_numpy(a) for a, _ in x]).contiguous(memory_format=torch.channels_last)
    dt = torch.stack([torch.from_numpy(b) for _, b in x]).contiguous(memory_format=torch.channels_last)
    return ft, dt


def close(p, expected):
    """Within 1e-4 in probability and 1e-3 in log10 probability."""
    lp, le = np.log10(np.clip(p, 1e-12, 1)), np.log10(np.clip(expected, 1e-12, 1))
    return abs(p - expected) < 1e-4 and abs(lp - le) < 1e-3


@pytest.fixture(scope="module")
def models():
    cache = {}

    def load(m, **kwargs):
        key = (m, tuple(sorted(kwargs.items())))
        if key not in cache:
            cache[key] = get_model(m, "cpu", **kwargs)
        return cache[key]

    return load


@pytest.mark.parametrize("m", MODELS)
def test_matches_tensorflow(models, m):
    if not have_weights(m):
        pytest.skip(f"no weights for model {m}")
    names = list(TEST_FILES)
    with torch.inference_mode():
        p = models(m)(*batch(names))[:, 1].numpy()
    for name, prob in zip(names, p):
        assert close(prob, EXPECTED[name][m]), f"model {m}, {name}: {prob} vs {EXPECTED[name][m]}"


@pytest.mark.parametrize("m", ["a", "h"])
def test_batching_doesnt_change_results(models, m):
    if not have_weights(m):
        pytest.skip(f"no weights for model {m}")
    names = list(TEST_FILES)
    with torch.inference_mode():
        together = models(m)(*batch(names))[:, 1].numpy()
        alone = [models(m)(*batch([n]))[0, 1].item() for n in names]
    np.testing.assert_allclose(together, alone, rtol=1e-5, atol=1e-7)


@needs_weights("a")
def test_folding_batchnorm_doesnt_change_results(models):
    ft, dt = batch(list(TEST_FILES))
    with torch.inference_mode():
        folded = models("a")(ft, dt).numpy()
        unfolded = models("a", fold_batchnorm=False)(ft, dt).numpy()
    np.testing.assert_allclose(folded, unfolded, rtol=1e-4, atol=1e-6)


@needs_weights("a")
def test_predict_torch_script(tmp_path):
    # two models when j is available too, to cover one read of the h5s feeding several models
    run = "aj" if have_weights("j") else "a"
    for name, path in TEST_FILES.items():
        os.symlink(path, tmp_path / name)
    script = os.path.join(os.path.dirname(HERE), "bin", "predict_torch.py")
    env = dict(os.environ, PYTHONPATH=os.path.dirname(HERE))
    subprocess.run([sys.executable, script, "-c", str(tmp_path), "-m", run, "-g", "-1", "-n", "1", "-b", "2"],
                   check=True, env=env)
    for m in run:
        results = pd.read_csv(tmp_path / f"results_{m}.csv", index_col=0)
        assert list(results.columns) == ["candidate", "probability", "label"]
        for _, row in results.iterrows():
            expected = EXPECTED[os.path.basename(row.candidate)][m]
            assert close(row.probability, expected)
            assert row.label == float(expected >= 0.5)
