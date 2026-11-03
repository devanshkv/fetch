"""
The PyTorch side's candidate reader against the preprocessing in fetch/data_sequence.py, and the weight
download's md5 check. Needs neither TensorFlow nor the weight files.
"""
import hashlib
import os

import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.signal as s

from fetch.pytorch import models
from fetch.pytorch.data import CandidateDataset, read_candidate

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = [os.path.join(HERE, "test.h5"), os.path.join(HERE, "synthetic.h5")]


def datagenerator_inputs(path):
    """The steps DataGenerator.__data_generation takes for one candidate, without Keras."""
    X = np.empty((1, 256, 256, 1))
    Y = np.empty((1, 256, 256, 1))
    with h5py.File(path, "r") as f:
        data_ft = s.detrend(np.array(f["data_freq_time"], dtype=np.float32).T)
        data_ft /= np.std(data_ft)
        data_ft -= np.median(data_ft)
        data_dt = np.array(f["data_dm_time"], dtype=np.float32)
        data_dt /= np.std(data_dt)
        data_dt -= np.median(data_dt)
        X[0,] = np.reshape(data_ft, (256, 256, 1))
        Y[0,] = np.reshape(data_dt, (256, 256, 1))
    X[X != X] = 0.0
    Y[Y != Y] = 0.0
    return X[0, ..., 0].astype(np.float32), Y[0, ..., 0].astype(np.float32)


@pytest.mark.parametrize("path", FILES)
def test_reader_matches_datagenerator(path):
    ft, dt = read_candidate(path)
    expected_ft, expected_dt = datagenerator_inputs(path)
    assert ft.shape == dt.shape == (1, 256, 256)
    assert ft.dtype == dt.dtype == np.float32
    np.testing.assert_array_equal(ft[0], expected_ft)
    np.testing.assert_array_equal(dt[0], expected_dt)


def copy_with(tmp_path, name, edit):
    path = tmp_path / name
    with h5py.File(FILES[0], "r") as src, h5py.File(path, "w") as dst:
        for key in ("data_freq_time", "data_dm_time"):
            data = src[key][...]
            dst.create_dataset(key, data=edit(key, data))
    return str(path)


def test_nan_candidate_is_scored_as_if_nan_were_zero(tmp_path):
    # DataGenerator raises here (scipy's detrend refuses NaN), stopping predict.py for the whole folder
    def put_nan(key, data):
        data = data.copy()
        data[10, 20] = np.nan
        return data

    def put_zero(key, data):
        data = data.copy()
        data[10, 20] = 0
        return data

    with_nan = copy_with(tmp_path, "nan.h5", put_nan)
    with pytest.raises(ValueError):
        datagenerator_inputs(with_nan)
    ft, dt = read_candidate(with_nan)
    zero_ft, zero_dt = read_candidate(copy_with(tmp_path, "zero.h5", put_zero))
    assert np.isfinite(ft).all() and np.isfinite(dt).all()
    np.testing.assert_array_equal(ft, zero_ft)
    np.testing.assert_array_equal(dt, zero_dt)


def test_candidate_without_a_dataset_reads_as_zeros(tmp_path):
    path = tmp_path / "no_dm_time.h5"
    with h5py.File(FILES[0], "r") as src, h5py.File(path, "w") as dst:
        dst.create_dataset("data_freq_time", data=src["data_freq_time"][...])
    ft, dt = read_candidate(str(path))
    assert not ft.any() and not dt.any()


def test_dataset_items_are_tensors():
    ft, dt = CandidateDataset(FILES)[1]
    assert tuple(ft.shape) == tuple(dt.shape) == (1, 256, 256)
    np.testing.assert_array_equal(ft.numpy(), read_candidate(FILES[1])[0])


@pytest.fixture
def fake_weights(tmp_path, monkeypatch):
    """A model list whose model a is a small file served by a fake download."""
    content = b"not really a keras file"
    served = {"bytes": content}
    listing = tmp_path / "model_list.csv"
    pd.DataFrame({"model": ["a_weights.h5"], "hash": [hashlib.md5(content).hexdigest()]}).to_csv(listing)
    monkeypatch.setattr(models, "MODEL_LIST", str(listing))
    downloads = []

    def urlretrieve(url, out):
        downloads.append(url)
        with open(out, "wb") as f:
            f.write(served["bytes"])

    monkeypatch.setattr(models.urllib.request, "urlretrieve", urlretrieve)
    return tmp_path / "cache", served, downloads


def test_weights_are_downloaded_once(fake_weights):
    cache, served, downloads = fake_weights
    path = models.weights_path("a", cache_dir=str(cache))
    assert open(path, "rb").read() == served["bytes"]
    assert downloads == [models.PATH_TO_WEIGHTS + "a_weights.h5"]
    assert models.weights_path("a", cache_dir=str(cache)) == path
    assert len(downloads) == 1


def test_corrupt_download_is_refused(fake_weights):
    cache, served, downloads = fake_weights
    served["bytes"] = b"truncated"
    with pytest.raises(IOError, match="md5"):
        models.weights_path("a", cache_dir=str(cache))
    assert not os.listdir(cache)  # neither the bad file nor its .part is left behind
