#!/usr/bin/env python3
"""predict.py with the PyTorch models: same arguments, same results_<model>.csv."""
import argparse
import glob
import logging
import string

import numpy as np
import pandas as pd
import torch

from fetch.pytorch.data import CandidateDataset
from fetch.pytorch.models import get_model

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fast Extragalactic Transient Candiate Hunter (FETCH), PyTorch",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", help="Be verbose", action="store_true")
    parser.add_argument(
        "-g", "--gpu_id", help="GPU ID (use -1 for CPU)", type=int, required=False, default=0
    )
    parser.add_argument(
        "-n", "--nproc", help="Number of processes reading the h5 files", default=4, type=int
    )
    parser.add_argument(
        "-c", "--data_dir", help="Directory with candidate h5s.", required=True, type=str, action="append"
    )
    parser.add_argument("-b", "--batch_size", help="Batch size", default=8, type=int)
    parser.add_argument(
        "-m",
        "--model",
        help="Index of the model(s) to use; several (e.g. abc) share one read of the h5s",
        required=True,
    )
    parser.add_argument("-p", "--probability", help="Detection threshold", default=0.5, type=float)
    args = parser.parse_args()

    logging_format = "%(asctime)s - %(funcName)s -%(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format=logging_format)

    for m in args.model:
        if m not in string.ascii_lowercase[:11]:
            raise ValueError("Models range from a to k.")

    if args.gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cpu")
    logger.info(f"Using {device}")
    # TF32 would move the probabilities by up to ~0.004 on Ampere and newer GPUs
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    models = {m: get_model(m, device) for m in args.model}

    for data_dir in args.data_dir:
        cands_to_eval = glob.glob(f"{data_dir}/*h5")
        if len(cands_to_eval) == 0:
            logger.warning(f"No candidates to evaluate in directory: {data_dir}")
            continue
        logger.debug(f"Read {len(cands_to_eval)} candidates")

        loader = torch.utils.data.DataLoader(
            CandidateDataset(cands_to_eval),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.nproc if args.nproc > 1 else 0,
        )
        probs = {m: [] for m in models}
        with torch.inference_mode():
            for ft, dt in loader:
                ft = ft.to(device, memory_format=torch.channels_last)
                dt = dt.to(device, memory_format=torch.channels_last)
                for m, model in models.items():
                    probs[m].append(model(ft, dt)[:, 1].float().cpu().numpy())

        for m in models:
            p = np.concatenate(probs[m])
            results = pd.DataFrame(
                {"candidate": cands_to_eval, "probability": p, "label": np.round(p >= args.probability)}
            )
            results.to_csv(data_dir + f"/results_{m}.csv")
