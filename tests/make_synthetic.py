#!/usr/bin/env python3
"""
Writes tests/synthetic.h5: Gaussian noise with a faint dispersed pulse, a candidate the models
disagree about, so its probabilities aren't all saturated at 0 or 1.
usage: python tests/make_synthetic.py [amplitude]
"""
import os
import sys

import h5py
import numpy as np

amplitude = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
rng = np.random.default_rng(20261029)
freq_time = rng.standard_normal((256, 256)).astype(np.float32)  # (time, frequency)
dm_time = rng.standard_normal((256, 256)).astype(np.float32)  # (DM, time)
t = np.arange(256)
for chan in range(256):
    freq_time[:, chan] += amplitude * np.exp(-0.5 * ((t - 128 - chan // 32) / 2.0) ** 2)
for dm in range(256):
    width = 2.0 + abs(dm - 128) / 8
    dm_time[dm] += amplitude * 2 / width * np.exp(-0.5 * ((t - 128) / width) ** 2)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic.h5")
with h5py.File(out, "w") as f:
    f.create_dataset("data_freq_time", data=freq_time)
    f.create_dataset("data_dm_time", data=dm_time)
print(out)
