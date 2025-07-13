#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

from fetch.predict_onnx import (
    download_model,
    get_default_onnx_dir,
    load_and_preprocess_h5_data,
    run_onnx_inference,
)

# Expected probabilities
EXPECTED_ONNX_PROBABILITIES = {
    "a": 1.0,
    "b": 1.0,
    "c": 1.0,
    "d": 0.99703777,
    "e": 1.0,
    "f": 1.0,
    "g": 0.9999999,
    "h": 1.0,
    "i": 0.9999931,
    "j": 0.99885,
    "k": 1.0,
}

# Test data file
TEST_DATA_FILE = Path(__file__).parent / "test.h5"

# Tolerance for probability comparison
PROBABILITY_TOLERANCE = 1e-6


@pytest.mark.parametrize("model_idx", list(EXPECTED_ONNX_PROBABILITIES.keys()))
def test_onnx_inference_model(model_idx):
    """Test ONNX inference for each model"""
    if not TEST_DATA_FILE.exists():
        pytest.skip(f"Test data file not found: {TEST_DATA_FILE}")

    expected_prob = EXPECTED_ONNX_PROBABILITIES[model_idx]

    # Load and preprocess test data
    ft_data, dt_data = load_and_preprocess_h5_data(TEST_DATA_FILE)
    ft_batch = np.expand_dims(ft_data, axis=0)
    dt_batch = np.expand_dims(dt_data, axis=0)

    # Use persistent ONNX models directory
    onnx_dir = get_default_onnx_dir()
    onnx_dir.mkdir(exist_ok=True)

    # Download the model
    model_path = download_model(model_idx, onnx_dir)

    # Create ONNX Runtime session (CPU only)
    onnx_session = ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )

    # Run inference
    predictions = run_onnx_inference(onnx_session, ft_batch, dt_batch)

    # Extract probability of positive class (index 1)
    actual_prob = float(predictions[0, 1])

    # Compare with expected probability
    diff = abs(actual_prob - expected_prob)

    assert diff <= PROBABILITY_TOLERANCE, (
        f"Model {model_idx} probability mismatch: "
        f"Expected={expected_prob}, Actual={actual_prob}, Diff={diff}"
    )

    print(f"Model {model_idx}: Expected={expected_prob}, Actual={actual_prob:.8f} ✓")


def test_gpu_detection():
    """Test GPU detection without requiring GPU to be present"""
    import onnxruntime as ort
    available_providers = ort.get_available_providers()
    
    # Should always have CPU
    assert "CPUExecutionProvider" in available_providers
    
    # Log what's available
    print(f"Available providers: {available_providers}")
    
    # Check for common GPU providers
    gpu_providers = [
        "CUDAExecutionProvider",
        "ROCMExecutionProvider", 
        "DmlExecutionProvider",
        "OpenVINOExecutionProvider",
        "TensorrtExecutionProvider",
    ]
    
    found_gpu = any(provider in available_providers for provider in gpu_providers)
    if found_gpu:
        print("✅ GPU providers detected")
    else:
        print("ℹ️ No GPU providers found (CPU-only mode)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
