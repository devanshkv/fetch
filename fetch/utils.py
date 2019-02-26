#!/usr/bin/env python3

import glob
import logging
import os
import string

import numpy as np
import pandas as pd
from keras.models import Model
from keras.models import model_from_yaml
from keras.optimizers import Adam
from keras.utils import get_file

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
PATH_TO_WEIGHTS = 'http://psrpop.phys.wvu.edu/download.php?val='

logger = logging.getLogger(__name__)


def open_n_layers_for_training(model, nlayers):
    """
    Makes nlayers of the model trainable.
    nlayers start from the top of the model.
    Top (or head) refers to the classification layer. The opening of layers for training starts from top.

    :param model: Model to open layers of
    :type model: Model
    :param nlayers: Number of (trainable) layers to open.
    :type nlayers: int
    :return: model
    """
    mask = np.zeros(len(model.layers), dtype=np.bool)
    mask[-nlayers:] = True
    for layer, mask_val in zip(model.layers, mask):
        layer.trainable = mask_val
    return model


def ready_for_train(model, nf, ndt, nft):
    """
    This makes the model ready for training, it opens the layers for training and complies it.

    :param model: model to train
    :type: Model
    :param nf: Number of layers to train post FT and DT models
    :type nf: int
    :param ndt: Number of layers in DT model to train
    :type ndt: int
    :param nft: Number of layers in FT model to train
    :type nft : int
    :return: compiled model ready for training
    """

    # Make all layers non trainable first
    model.trainable = False
    model = open_n_layers_for_training(model, nf)

    # Get the FT and DT models to open them up for training
    model.layers[4] = open_n_layers_for_training(model.layers[4], nft)
    model.layers[5] = open_n_layers_for_training(model.layers[5], ndt)

    model_trainable = Model(model.inputs, model.outputs)

    # Adam optimizer with imagenet defaults
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    # Compile
    model_trainable.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model_trainable


def get_model(model_idx):
    """

    :param model_idx: model string between a--j
    :type model_idx: str
    :return: Model
    """
    # Get the model from the folder
    logging.info(f'Getting model {model_idx}')
    model_yaml = glob.glob(f'models/{model_idx}_FT*/*yaml')[0]

    # Read the model from the yaml
    with open(model_yaml, 'r') as y:
        model = model_from_yaml(y.read())

    # get the model weights, if not present download them.
    model_list = pd.read_csv('models/model_list.csv')
    model_index = string.ascii_lowercase.index(model_idx)

    weights = get_file(model_list['model'][model_index], PATH_TO_WEIGHTS + model_list['model'][model_index],
                       file_hash=model_list['hash'][model_index], cache_subdir='models', hash_algorithm='md5')

    # dump weights
    model.load_weights(weights)

    return model
