"""FETCH's eleven models in PyTorch, built from the same Keras weight files the TensorFlow version uses."""
import hashlib
import logging
import os
import string
import urllib.request

import pandas as pd
import torch

from fetch.pytorch.keras_graph import from_keras_h5

logger = logging.getLogger(__name__)

PATH_TO_WEIGHTS = "https://zenodo.org/records/5029590/files/"
# FETCH_MODEL_DIR (e.g. weights baked into a container), else where tf.keras.utils.get_file
# keeps them, so the two versions share one download
CACHE_DIR = os.environ.get("FETCH_MODEL_DIR") or os.path.join(os.path.expanduser("~"), ".keras", "models")
MODEL_LIST = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "model_list.csv")


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def weights_path(model_idx, cache_dir=CACHE_DIR):
    """The model's Keras h5 file, downloaded and checked against its md5 if it isn't cached."""
    model_list = pd.read_csv(MODEL_LIST)
    row = model_list.iloc[string.ascii_lowercase.index(model_idx)]
    path = os.path.join(cache_dir, row["model"])
    if os.path.exists(path) and _md5(path) == row["hash"]:
        return path
    os.makedirs(cache_dir, exist_ok=True)
    logger.info(f"Downloading {row['model']}")
    tmp = path + ".part"
    urllib.request.urlretrieve(PATH_TO_WEIGHTS + row["model"], tmp)
    if _md5(tmp) != row["hash"]:
        os.remove(tmp)
        raise IOError(f"{row['model']} failed its md5 check")
    os.replace(tmp, path)
    return path


def get_model(model_idx, device="cpu", fold_batchnorm=True):
    """
    :param model_idx: model letter, a--k
    :param device: torch device to put it on
    :param fold_batchnorm: fold batch normalisation into the convolutions (faster; inference only)
    :return: the model in eval mode, in channels-last memory format; it takes the (N, 1, 256, 256)
        frequency-time and DM-time batches and returns (N, 2) class probabilities, as FETCH's
        Keras models do
    """
    if model_idx not in string.ascii_lowercase[:11]:
        raise ValueError("Models range from a to k")
    logger.info(f"Getting model {model_idx}")
    model = from_keras_h5(weights_path(model_idx))
    if fold_batchnorm:
        model.fold_batchnorm()
    return model.to(device, memory_format=torch.channels_last)
