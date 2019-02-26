#!/usr/bin/env python3

import argparse
import logging
import os
import string

import pandas as pd
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from keras.models import Model
from sklearn.model_selection import train_test_split

from fetch.data_sequence import DataGenerator
from fetch.utils import get_model
from fetch.utils import ready_for_train

logger = logging.getLogger(__name__)

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def train(model, epochs, patience, output_path, nproc, train_obj, val_obj):
    """

    :param model: model to train (must be compiled)
    :type model: Model
    :param epochs: max number of epochs to train.
    :type epochs: int
    :param patience: Stop after these many layers if val. loss doesn't decrease
    :type patience: int
    :param output_path: paths to save weights and logs
    :type output_path: str
    :param nproc: number of processors for training
    :type nproc: int
    :param train_obj: DataGenerator training object for training
    :type train_obj: DataGenerator
    :param val_obj: DataGenerator training object for validation
    :type val_obj: DataGenerator
    :return: model, history object
    """
    if nproc == 1:
        use_multiprocessing = False
    else:
        use_multiprocessing = True

    # Callbacks for training and validation
    ES = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=patience, verbose=1, mode='min',
                       restore_best_weights=True)
    CK = ModelCheckpoint(output_path + 'weights.h5', monitor='val_loss', verbose=1, save_best_only=True,
                         save_weights_only=False,
                         mode='min')
    csv_name = output_path + 'training_log.csv'
    LO = CSVLogger(csv_name, append=False)

    callbacks = [ES, CK, LO]

    train_history = model.fit_generator(generator=train_obj, validation_data=val_obj, epochs=epochs,
                                        use_multiprocessing=use_multiprocessing, max_queue_size=10, workers=nproc,
                                        shuffle=True, callbacks=callbacks, verbose=1)
    return model, train_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast Extragalactic Transient Candiate Hunter (FETCH)")
    parser.add_argument('-v', '--verbose', help='Be verbose', action='store_true')
    parser.add_argument('-g', '--gpu_id', help='GPU ID', type=int, required=False, default=0)
    parser.add_argument('-n', '--nproc', help='Number of processors for training', default=4, type=int)
    parser.add_argument('-c', '--data_csv', help='CSV with candidate h5 paths and labels',
                        required=True, type=str)
    parser.add_argument('-b', '--batch_size', help='Batch size for training data', default=8, type=int)
    parser.add_argument('-e', '--epochs', help='Number of epochs for training', default=15, type=int)
    parser.add_argument('-p', '--patience', help='Layer patience, stop training if validation loss does not decreate',
                        default=3, type=int)
    parser.add_argument('-nft', '--n_ft_layers', help='Number of layers in FT model to train', default=0, type=int)
    parser.add_argument('-ndt', '--n_dt_layers', help='Number of layers in DT model to train', default=0, type=int)
    parser.add_argument('-nf', '--n_fusion_layers', help='Number of layers to train post FT and DT models', default=1,
                        type=int)
    parser.add_argument('-o', '--output_path', help='Place to save the weights and training logs', type=str,
                        required=True)
    parser.add_argument('-vs', '--val_split', help='Percent of data to use for validation', type=float, default=0.2)
    parser.add_argument('-m', '--model', help='Index of the model to train', required=True, type=str)
    args = parser.parse_args()

    logging_format = '%(asctime)s - %(funcName)s -%(name)s - %(levelname)s - %(message)s'

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format=logging_format)
    else:
        logging.basicConfig(level=logging.INFO, format=logging_format)

    if args.model not in list(string.ascii_lowercase)[:11]:
        raise ValueError(f'Model only range from a -- j.')

    if args.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = f'{args.gpu_id}'

    if args.n_fusion_layers >= 9:
        raise ValueError(
            f'Cannot open {args.n_fusion_layers} for training. Models only have 6 layers after FT and DT models.')

    data_df = pd.read_csv(args.data_csv)

    train_df, val_df = train_test_split(data_df, test_size=(1 - args.val_split), random_state=1993)
    train_data_generator = DataGenerator(list_IDs=list(train_df['h5']), labels=list(train_df['label']), noise=True,
                                         shuffle=True)
    validate_data_generator = DataGenerator(list_IDs=list(val_df['h5']), labels=list(val_df['label']), noise=False,
                                            shuffle=False)

    model_to_train = get_model(args.model)

    model_to_train = ready_for_train(model_to_train, ndt=args.n_dt_layers, nft=args.n_ft_layers,
                                     nf=args.n_fusion_layers)

    trained_model, history = train(model_to_train, epochs=args.epochs, patience=args.patience,
                                   output_path=args.output_path,
                                   nproc=args.nproc, train_obj=train_data_generator, val_obj=validate_data_generator)
