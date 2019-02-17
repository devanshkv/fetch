"""
https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
"""
import logging
import os

import h5py
import keras
import numpy as np
import scipy.signal as s

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
logger = logging.getLogger(__name__)


class DataGenerator(keras.utils.Sequence):

    def __init__(self, list_IDs, labels, batch_size=32, ft_dim=(256, 256), dt_dim=(256, 256), n_channels=1,
                 n_classes=2, shuffle=True, noise=False, noise_mean=0.0, noise_std=1.0):
        """

        :param list_IDs: list of h5 files
        :type list_IDs: list
        :param labels: list of labels (use fake labels when using predict)
        :type labels: list
        :param batch_size: Batch size (def = 32)
        :type batch_size: int
        :param ft_dim: 2D shape (def (256, 256)
        :type dt_dim tuple
        :param dt_dim: 2D shape (def (256, 256)
        :type ft_dim tuple
        :param n_channels: number of channels in data (def = 1)
        :type n_channels: int
        :param n_classes: number of classes to classify data into (def = 2)
        :type n_classes: ints
        :param shuffle: to shuffle or not to shuffle?
        :type shuffle: bool
        :param noise: to add noise or not to?
        :type noise: bool
        :param noise_mean: mean of gaussian noise
        :type noise_mean: float
        :param noise_std: standard deviation of gaussian noise
        :type noise_std: float
        """
        self.ft_dim = ft_dim
        self.dt_dim = dt_dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.noise = noise
        self.labels = labels
        self.noise_mean = noise_mean
        self.noise_std = noise_std

    def __len__(self):
        """

        :return: Number of batches per epoch
        """
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """

        :param index: index
        :return: Data dictionary and categorical labels
        """
        if index < self.__len__():
            indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        else:
            indexes = self.indexes[index * self.batch_size:]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp, indexes)

        return X, y

    def on_epoch_end(self):
        """

        :return: Updates the indices at the end of the epoch
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _read_h5(self, ID, key):
        """

        :param ID: h5 file to read
        :type ID: str
        :param key: data_freq_time or data_dm_time
        :type key: str
        :return: 3D numpy array
        """
        try:
            with h5py.File(ID, 'r') as f:
                if key == 'data_freq_time':
                    data = np.array(f[key], dtype=np.float32).T
                    X = np.reshape(data, (*self.ft_dim, self.n_channels))
                else:
                    data = np.array(f[key], dtype=np.float32)
                    X = np.reshape(data, (*self.dt_dim, self.n_channels))
                X[X != X] = 0.0
            return X
        except KeyError:
            print(ID, key)

    def __data_generation(self, list_IDs_temp, indexes):
        """

        :param list_IDs_temp: list of h5 files to read
        :param indexes: indexes
        :return: Batch of data and labels
        """
        X = np.empty((len(list_IDs_temp), *self.ft_dim, self.n_channels))
        Y = np.empty((len(list_IDs_temp), *self.dt_dim, self.n_channels))
        y = np.empty((len(list_IDs_temp)), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            try:
                with h5py.File(ID, 'r') as f:
                    data_ft = s.detrend(np.array(f['data_freq_time'], dtype=np.float32).T)
                    data_ft /= np.std(data_ft)
                    data_ft -= np.median(data_ft)
                    data_dt = np.array(f['data_dm_time'], dtype=np.float32)
                    X[i,] = np.reshape(data_ft, (*self.ft_dim, self.n_channels))
                    Y[i,] = np.reshape(data_dt, (*self.dt_dim, self.n_channels))
            except KeyError:
                print(ID)
            y[i] = self.labels[indexes[i]]
        X[X != X] = 0.0
        Y[Y != Y] = 0.0

        if self.noise:
            X += np.random.normal(loc=self.noise_mean, scale=self.noise_std, size=X.shape)
        return {'data_freq_time': X, 'data_dm_time': Y}, keras.utils.to_categorical(y, num_classes=self.n_classes)
