#!/usr/bin/env python3

import argparse
import glob
import logging
import os
import string

import numpy as np
import pandas as pd
from tensorflow.keras.utils import OrderedEnqueuer, Progbar

from fetch.data_sequence import DataGenerator
from fetch.utils import get_model

logger = logging.getLogger(__name__)

MODEL_INDICES = list(string.ascii_lowercase)[:11]


def batches(sequence, workers, use_multiprocessing, max_queue_size=10):
    """
    Yield a Sequence's batches in order, reading ahead on `workers` processes.
    This is the reading half of predict_generator, split out so that one pass
    over the candidates can feed several models.
    """
    if workers <= 1:
        for index in range(len(sequence)):
            yield sequence[index]
        return

    enqueuer = OrderedEnqueuer(
        sequence, use_multiprocessing=use_multiprocessing, shuffle=False
    )
    enqueuer.start(workers=workers, max_queue_size=max_queue_size)
    try:
        output = enqueuer.get()
        for _ in range(len(sequence)):
            yield next(output)
    finally:
        enqueuer.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fast Extragalactic Transient Candiate Hunter (FETCH)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", help="Be verbose", action="store_true")
    parser.add_argument(
        "-g",
        "--gpu_id",
        help="GPU ID (use -1 for CPU)",
        type=int,
        required=False,
        default=0,
    )
    parser.add_argument(
        "-n", "--nproc", help="Number of processors for training", default=4, type=int
    )
    parser.add_argument(
        "-c",
        "--data_dir",
        help="Directory with candidate h5s.",
        required=True,
        type=str,
        action='append'
    )
    parser.add_argument(
        "-b", "--batch_size", help="Batch size for training data", default=8, type=int
    )
    parser.add_argument(
        "-m",
        "--model",
        help="Index of the model to use. Several may be given; they share one "
        "pass over the candidates and each writes its own results_<model>.csv.",
        required=True,
        nargs="+",
    )
    parser.add_argument(
        "-p", "--probability", help="Detection threshold", default=0.5, type=float
    )
    args = parser.parse_args()

    logging_format = (
        "%(asctime)s - %(funcName)s -%(name)s - %(levelname)s - %(message)s"
    )

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format=logging_format)
    else:
        logging.basicConfig(level=logging.INFO, format=logging_format)

    for model_idx in args.model:
        if model_idx not in MODEL_INDICES:
            raise ValueError(
                f"Model {model_idx} unknown: models only range from "
                f"{MODEL_INDICES[0]} -- {MODEL_INDICES[-1]}."
            )
    if len(set(args.model)) != len(args.model):
        raise ValueError(f"Model given more than once: {args.model}")

    if args.gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu_id}"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if args.nproc > 1:
        use_multiprocessing = True
        logging.info(f"Using multiprocessing with {args.nproc} workers")
    else:
        use_multiprocessing = False

    models = [(model_idx, get_model(model_idx)) for model_idx in args.model]

    for data_dir in args.data_dir:

        cands_to_eval = glob.glob(f"{data_dir}/*h5")

        if len(cands_to_eval) == 0:
            logger.warning(f"No candidates to evaluate in directory: {data_dir}")
            continue

        logging.debug(f"Read {len(cands_to_eval)} candidates")

        # Get the data generator, make sure noise and shuffle are off.
        cand_datagen = DataGenerator(
            list_IDs=cands_to_eval,
            labels=[0] * len(cands_to_eval),
            shuffle=False,
            noise=False,
            batch_size=args.batch_size,
        )

        # get's get predicting. Read once, show each batch to every model. No
        # shuffle, so the predictions concatenate in the order of cands_to_eval.
        batch_probs = {model_idx: [] for model_idx, _ in models}
        progbar = Progbar(target=len(cand_datagen))
        for step, (data, _) in enumerate(
            batches(cand_datagen, args.nproc, use_multiprocessing), start=1
        ):
            for model_idx, model in models:
                batch_probs[model_idx].append(model.predict_on_batch(data))
            progbar.update(step)

        # Save results
        for model_idx, _ in models:
            probs = np.concatenate(batch_probs[model_idx])
            results_dict = {}
            results_dict["candidate"] = cands_to_eval
            results_dict["probability"] = probs[:, 1]
            results_dict["label"] = np.round(probs[:, 1] >= args.probability)
            results_file = data_dir + f"/results_{model_idx}.csv"
            pd.DataFrame(results_dict).to_csv(results_file)
