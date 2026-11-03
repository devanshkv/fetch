"""
Reads candidate h5 files into model inputs, preprocessed exactly as fetch.data_sequence.DataGenerator does.

The one difference is a candidate containing NaN: scipy's detrend refuses it, so DataGenerator raises and
predict.py stops for the whole directory. Here NaNs are set to 0 before detrending and the candidate is scored.
"""
import os

import h5py
import numpy as np
import scipy.signal as s
import torch

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def read_candidate(path, ft_dim=(256, 256), dt_dim=(256, 256)):
    """(frequency-time, DM-time) arrays of one h5 file, each (1, 256, 256) float32; zeros if unreadable."""
    ft = np.zeros(ft_dim, dtype=np.float64)
    dt = np.zeros(dt_dim, dtype=np.float64)
    try:
        with h5py.File(path, "r") as f:
            data_ft = s.detrend(np.nan_to_num(np.array(f["data_freq_time"], dtype=np.float32).T))
            data_ft /= np.std(data_ft)
            data_ft -= np.median(data_ft)
            data_dt = np.nan_to_num(np.array(f["data_dm_time"], dtype=np.float32))
            data_dt /= np.std(data_dt)
            data_dt -= np.median(data_dt)
            ft = np.reshape(data_ft, ft_dim)
            dt = np.reshape(data_dt, dt_dim)
    except KeyError:
        print(path)
    # DataGenerator fills float64 arrays and zeroes NaNs before handing them to Keras as float32
    ft = np.where(ft != ft, 0.0, ft)
    dt = np.where(dt != dt, 0.0, dt)
    return ft.astype(np.float32)[None], dt.astype(np.float32)[None]


class CandidateDataset(torch.utils.data.Dataset):
    def __init__(self, files):
        self.files = list(files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        ft, dt = read_candidate(self.files[index])
        return torch.from_numpy(ft), torch.from_numpy(dt)
