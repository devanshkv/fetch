#!/usr/bin/env python3

import argparse
import glob
import hashlib
import logging
import os
import string
from pathlib import Path

import h5py
import numpy as np
import onnxruntime as ort
import pandas as pd
import requests
import scipy.signal as s

# Configure logging
logger = logging.getLogger(__name__)


def get_default_onnx_dir():
    """
    Get the default ONNX models directory.
    Uses ONNX_HOME environment variable if set, otherwise defaults to $HOME/onnx_models.
    """
    onnx_home = os.environ.get("ONNX_HOME")
    if onnx_home:
        return Path(onnx_home)
    else:
        home_dir = os.environ.get("HOME", os.getcwd())
        default_dir = Path(home_dir) / "onnx_models"
        logger.warning(f"ONNX_HOME not set, using default directory: {default_dir}")
        return default_dir


# Model registry mapping model indices to Zenodo ONNX files
MODEL_REGISTRY = {
    "a": {
        "url": "https://zenodo.org/api/records/15699208/files/model_a.onnx/content",
        "md5": "7a8a627129817418963c7b77b962e0bd",
        "size_mb": 114.80,
    },
    "b": {
        "url": "https://zenodo.org/api/records/15699208/files/model_b.onnx/content",
        "md5": "aecb4c022e673f80f6cf73ced9e4c373",
        "size_mb": 87.28,
    },
    "c": {
        "url": "https://zenodo.org/api/records/15699208/files/model_c.onnx/content",
        "md5": "027ab7bf064944b9782ab51ef1dd8416",
        "size_mb": 139.52,  # estimated based on pattern
    },
    "d": {
        "url": "https://zenodo.org/api/records/15699208/files/model_d.onnx/content",
        "md5": "9d33f3c9e3db15b5903ada9043e93126",
        "size_mb": 157.33,
    },
    "e": {
        "url": "https://zenodo.org/api/records/15699208/files/model_e.onnx/content",
        "md5": "d40fff195b94c75d0842660b913a3bc5",
        "size_mb": 164.86,
    },
    "f": {
        "url": "https://zenodo.org/api/records/15699208/files/model_f.onnx/content",
        "md5": "e5391f7b501f2b50666438015b2221f1",
        "size_mb": 114.00,
    },
    "g": {
        "url": "https://zenodo.org/api/records/15699208/files/model_g.onnx/content",
        "md5": "1282ebaff6999e93d091dc0a689131af",
        "size_mb": 139.52,
    },
    "h": {
        "url": "https://zenodo.org/api/records/15699208/files/model_h.onnx/content",
        "md5": "60c5d33688573a7498190238840c8752",
        "size_mb": 114.80,
    },
    "i": {
        "url": "https://zenodo.org/api/records/15699208/files/model_i.onnx/content",
        "md5": "5c21e392e2517bdede04034f188876d3",
        "size_mb": 132.58,
    },
    "j": {
        "url": "https://zenodo.org/api/records/15699208/files/model_j.onnx/content",
        "md5": "7beac5476ff03f6fa96ec304bf44acb1",
        "size_mb": 301.24,
    },
    "k": {
        "url": "https://zenodo.org/api/records/15699208/files/model_k.onnx/content",
        "md5": "b8e0c9a24275a1813bd007088e1b19f7",
        "size_mb": 116.16,
    },
}


def calculate_md5(file_path):
    """Calculate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def download_model(model_idx, onnx_models_dir):
    """
    Download ONNX model from Zenodo if not already cached.

    :param model_idx: Model index (a-k)
    :param onnx_models_dir: Directory to store ONNX models
    :return: Path to downloaded model file
    """
    if model_idx not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_idx} not found in registry")

    model_info = MODEL_REGISTRY[model_idx]
    model_path = onnx_models_dir / f"model_{model_idx}.onnx"

    # Check if model already exists and has correct hash
    if model_path.exists():
        logger.info(f"Model {model_idx} found locally, verifying hash...")
        if calculate_md5(model_path) == model_info["md5"]:
            logger.info(f"Model {model_idx} hash verified, using cached version")
            return model_path
        else:
            logger.warning(f"Model {model_idx} hash mismatch, re-downloading...")
            model_path.unlink()

    # Download model
    logger.info(
        f"Downloading model {model_idx} ({model_info['size_mb']:.1f} MB) from Zenodo..."
    )

    try:
        response = requests.get(model_info["url"], stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(
                            f"\rDownload progress: {progress:.1f}%", end="", flush=True
                        )

        print()  # New line after progress

        # Verify downloaded file hash
        logger.info(f"Verifying hash for model {model_idx}...")
        if calculate_md5(model_path) != model_info["md5"]:
            model_path.unlink()
            raise ValueError(f"Downloaded model {model_idx} hash verification failed")

        logger.info(f"Model {model_idx} downloaded and verified successfully")
        return model_path

    except Exception as e:
        if model_path.exists():
            model_path.unlink()
        raise RuntimeError(f"Failed to download model {model_idx}: {str(e)}")


def preprocess_ft_data(data):
    """
    Apply FT (Frequency-Time) preprocessing pipeline.
    Extracted from DataGenerator.__data_generation.
    """
    # Replace NaNs and convert to float32
    data = np.nan_to_num(data.astype(np.float32))

    # Apply detrending (remove linear trend)
    data = s.detrend(data)

    # Normalize: subtract median, divide by standard deviation
    data = data - np.median(data)
    data = data / np.std(data)

    # Handle any remaining NaNs (in case std was 0)
    data = np.nan_to_num(data)

    return data


def preprocess_dt_data(data):
    """
    Apply DT (DM-Time) preprocessing pipeline.
    Extracted from DataGenerator.__data_generation.
    """
    # Replace NaNs and convert to float32
    data = np.nan_to_num(data.astype(np.float32))

    # Normalize: subtract median, divide by standard deviation
    data = data - np.median(data)
    data = data / np.std(data)

    # Handle any remaining NaNs (in case std was 0)
    data = np.nan_to_num(data)

    return data


def load_and_preprocess_h5_data(h5_file_path, ft_dim=(256, 256), dt_dim=(256, 256)):
    """
    Load and preprocess data from H5 file.

    :param h5_file_path: Path to H5 file
    :param ft_dim: Expected FT data dimensions
    :param dt_dim: Expected DT data dimensions
    :return: Tuple of (ft_data, dt_data) with shape (H, W, 1)
    """
    try:
        with h5py.File(h5_file_path, "r") as f:
            # Load raw data
            data_ft_raw = np.array(f["data_freq_time"], dtype=np.float32).T
            data_dt_raw = np.array(f["data_dm_time"], dtype=np.float32)

            # Apply preprocessing
            data_ft = preprocess_ft_data(data_ft_raw)
            data_dt = preprocess_dt_data(data_dt_raw)

            # Reshape to expected dimensions with channel dimension
            ft_data = np.reshape(data_ft, (*ft_dim, 1))
            dt_data = np.reshape(data_dt, (*dt_dim, 1))

            return ft_data, dt_data

    except Exception as e:
        logger.error(f"Failed to load/preprocess {h5_file_path}: {str(e)}")
        raise


def process_batch(h5_files, batch_size=8, ft_dim=(256, 256), dt_dim=(256, 256)):
    """
    Process a batch of H5 files and return preprocessed data.

    :param h5_files: List of H5 file paths
    :param batch_size: Batch size for processing
    :param ft_dim: FT data dimensions
    :param dt_dim: DT data dimensions
    :return: Generator yielding (ft_batch, dt_batch, file_paths_batch)
    """
    for i in range(0, len(h5_files), batch_size):
        batch_files = h5_files[i : i + batch_size]

        ft_batch = []
        dt_batch = []
        valid_files = []

        for h5_file in batch_files:
            try:
                ft_data, dt_data = load_and_preprocess_h5_data(h5_file, ft_dim, dt_dim)
                ft_batch.append(ft_data)
                dt_batch.append(dt_data)
                valid_files.append(h5_file)
            except Exception as e:
                logger.warning(f"Skipping {h5_file}: {str(e)}")
                continue

        if valid_files:
            # Convert to numpy arrays with batch dimension
            ft_batch = np.array(ft_batch)
            dt_batch = np.array(dt_batch)

            yield ft_batch, dt_batch, valid_files


def run_onnx_inference(onnx_session, ft_data, dt_data):
    """
    Run inference using ONNX model.

    :param onnx_session: ONNX Runtime inference session
    :param ft_data: FT data batch
    :param dt_data: DT data batch
    :return: Model predictions
    """
    # Get input names from ONNX model
    input_names = [inp.name for inp in onnx_session.get_inputs()]

    # Create input dictionary
    # ONNX models typically expect input names without ':0' suffix
    onnx_inputs = {}
    for idx, name in enumerate(input_names):
        clean_name = name.split(":")[0]  # Remove ':0' if present
        if idx == 0:
            onnx_inputs[clean_name] = ft_data
        elif idx == 1:
            onnx_inputs[clean_name] = dt_data
        else:
            logger.warning(f"Unexpected input {idx}: {name}")

    # Run inference
    outputs = onnx_session.run(None, onnx_inputs)
    return outputs[0]  # Return first (and typically only) output


def main():
    parser = argparse.ArgumentParser(
        description="Fast Extragalactic Transient Candidate Hunter (FETCH) - ONNX Version",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", help="Be verbose", action="store_true")
    parser.add_argument(
        "-c",
        "--data_dir",
        help="Directory with candidate h5s.",
        required=True,
        type=str,
        action="append",
    )
    parser.add_argument(
        "-b", "--batch_size", help="Batch size for inference", default=8, type=int
    )
    parser.add_argument(
        "-m", "--model", help="Index of the model to use (a-k)", required=True
    )
    parser.add_argument(
        "-p", "--probability", help="Detection threshold", default=0.5, type=float
    )
    parser.add_argument(
        "--onnx_dir",
        help="Directory to store ONNX models (default: $ONNX_HOME or $HOME/onnx_models)",
        default=None,
        type=str,
    )
    parser.add_argument(
        "-g",
        "--gpu_id",
        help="GPU ID (use -1 for CPU only)",
        type=int,
        required=False,
        default=0,
    )
    args = parser.parse_args()

    # Setup logging
    logging_format = (
        "%(asctime)s - %(funcName)s - %(name)s - %(levelname)s - %(message)s"
    )
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format=logging_format)
    else:
        logging.basicConfig(level=logging.INFO, format=logging_format)

    # Validate model index
    if args.model not in list(string.ascii_lowercase)[:11]:
        raise ValueError(f"Model must be between a-k, got: {args.model}")

    # Setup ONNX models directory
    if args.onnx_dir:
        onnx_models_dir = Path(args.onnx_dir)
    else:
        onnx_models_dir = get_default_onnx_dir()

    onnx_models_dir.mkdir(exist_ok=True)
    logger.info(f"Using ONNX models directory: {onnx_models_dir}")

    # Setup execution providers (GPU/CPU)
    providers = []
    if args.gpu_id >= 0:
        # Try to use GPU
        available_providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in available_providers:
            providers.append(("CUDAExecutionProvider", {"device_id": args.gpu_id}))
            logger.info(f"Using GPU {args.gpu_id} with CUDA")
        elif "ROCMExecutionProvider" in available_providers:
            providers.append(("ROCMExecutionProvider", {"device_id": args.gpu_id}))
            logger.info(f"Using GPU {args.gpu_id} with ROCm")
        else:
            logger.warning(
                "GPU requested but no GPU providers available, falling back to CPU"
            )

    # Always add CPU as fallback
    providers.append("CPUExecutionProvider")
    if not providers or providers == ["CPUExecutionProvider"]:
        logger.info("Using CPU execution")

    # Download and load ONNX model
    try:
        model_path = download_model(args.model, onnx_models_dir)
        logger.info(f"Loading ONNX model from {model_path}")

        # Create ONNX Runtime session
        onnx_session = ort.InferenceSession(str(model_path), providers=providers)

        # Log model info
        input_names = [inp.name for inp in onnx_session.get_inputs()]
        output_names = [out.name for out in onnx_session.get_outputs()]
        logger.info(f"Model inputs: {input_names}")
        logger.info(f"Model outputs: {output_names}")

    except Exception as e:
        logger.error(f"Failed to load ONNX model {args.model}: {str(e)}")
        return 1

    # Process each data directory
    for data_dir in args.data_dir:
        logger.info(f"Processing directory: {data_dir}")

        # Find H5 files
        cands_to_eval = glob.glob(f"{data_dir}/*h5")

        if len(cands_to_eval) == 0:
            logger.warning(f"No candidates to evaluate in directory: {data_dir}")
            continue

        logger.info(f"Found {len(cands_to_eval)} candidate files")

        # Collect results
        all_candidates = []
        all_probabilities = []

        # Process in batches
        for ft_batch, dt_batch, batch_files in process_batch(
            cands_to_eval, args.batch_size
        ):
            try:
                # Run ONNX inference
                batch_probs = run_onnx_inference(onnx_session, ft_batch, dt_batch)

                # Store results
                all_candidates.extend(batch_files)
                all_probabilities.extend(
                    batch_probs[:, 1]
                )  # Probability of positive class

                logger.debug(f"Processed batch of {len(batch_files)} files")

            except Exception as e:
                logger.error(f"Inference failed for batch: {str(e)}")
                continue

        # Save results
        if all_candidates:
            results_dict = {
                "candidate": all_candidates,
                "probability": all_probabilities,
                "label": np.round(
                    np.array(all_probabilities) >= args.probability
                ).astype(int),
            }

            results_file = f"{data_dir}/results_{args.model}_onnx.csv"
            pd.DataFrame(results_dict).to_csv(results_file, index=False)
            logger.info(f"Results saved to: {results_file}")

            # Log summary
            num_detections = sum(results_dict["label"])
            logger.info(
                f"Processed {len(all_candidates)} candidates, {num_detections} detections above threshold {args.probability}"
            )
        else:
            logger.warning(f"No valid candidates processed in {data_dir}")

    return 0


if __name__ == "__main__":
    exit(main())
