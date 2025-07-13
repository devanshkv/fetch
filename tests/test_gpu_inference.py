#!/usr/bin/env python3

import pytest
import numpy as np
import onnxruntime as ort
from pathlib import Path

from fetch.predict_onnx import (
    download_model,
    get_default_onnx_dir,
    load_and_preprocess_h5_data,
    run_onnx_inference,
)

# Test data file
TEST_DATA_FILE = Path(__file__).parent / "test.h5"

# Expected probabilities for GPU inference (should match CPU)
EXPECTED_GPU_PROBABILITIES = {
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

# Tolerance for probability comparison
PROBABILITY_TOLERANCE = 1e-6


def has_gpu_support():
    """Check if GPU execution providers are available"""
    available_providers = ort.get_available_providers()
    gpu_providers = [
        "CUDAExecutionProvider",
        "ROCMExecutionProvider", 
        "DmlExecutionProvider",  # DirectML for Windows
        "OpenVINOExecutionProvider",
        "TensorrtExecutionProvider",
    ]
    return any(provider in available_providers for provider in gpu_providers)


def get_gpu_providers():
    """Get available GPU execution providers"""
    available_providers = ort.get_available_providers()
    gpu_providers = []
    
    # CUDA (NVIDIA)
    if "CUDAExecutionProvider" in available_providers:
        gpu_providers.append(("CUDAExecutionProvider", {"device_id": 0}))
    
    # ROCm (AMD)
    if "ROCMExecutionProvider" in available_providers:
        gpu_providers.append(("ROCMExecutionProvider", {"device_id": 0}))
    
    # DirectML (Windows)
    if "DmlExecutionProvider" in available_providers:
        gpu_providers.append("DmlExecutionProvider")
    
    # OpenVINO (Intel)
    if "OpenVINOExecutionProvider" in available_providers:
        gpu_providers.append("OpenVINOExecutionProvider")
    
    # TensorRT (NVIDIA)
    if "TensorrtExecutionProvider" in available_providers:
        gpu_providers.append("TensorrtExecutionProvider")
    
    # Always add CPU as fallback
    gpu_providers.append("CPUExecutionProvider")
    
    return gpu_providers


@pytest.mark.skipif(not has_gpu_support(), reason="No GPU execution providers available")
class TestGPUInference:
    """Test GPU inference functionality"""
    
    def test_gpu_providers_available(self):
        """Test that GPU providers are detected correctly"""
        providers = get_gpu_providers()
        assert len(providers) > 1  # Should have at least GPU + CPU
        assert "CPUExecutionProvider" in [p if isinstance(p, str) else p[0] for p in providers]
        print(f"Available GPU providers: {providers}")
    
    @pytest.mark.skipif(not TEST_DATA_FILE.exists(), reason="Test data file not found")
    def test_gpu_session_creation(self):
        """Test that ONNX sessions can be created with GPU providers"""
        # Use persistent ONNX models directory
        onnx_dir = get_default_onnx_dir()
        onnx_dir.mkdir(exist_ok=True)
        
        # Download model 'a' (good for testing)
        model_path = download_model("a", onnx_dir)
        
        # Create session with GPU providers
        providers = get_gpu_providers()
        session = ort.InferenceSession(str(model_path), providers=providers)
        
        # Verify session was created successfully
        assert session is not None
        
        # Check which provider was actually used
        used_providers = session.get_providers()
        print(f"Session using providers: {used_providers}")
        
        # Should prefer GPU over CPU if available
        if len(providers) > 1:  # More than just CPU
            assert used_providers[0] != "CPUExecutionProvider"
    
    @pytest.mark.skipif(not TEST_DATA_FILE.exists(), reason="Test data file not found")
    @pytest.mark.parametrize("model_idx", ["a", "b"])  # Test subset for speed
    def test_gpu_inference_accuracy(self, model_idx):
        """Test that GPU inference produces same results as expected"""
        expected_prob = EXPECTED_GPU_PROBABILITIES[model_idx]
        
        # Load and preprocess test data
        ft_data, dt_data = load_and_preprocess_h5_data(TEST_DATA_FILE)
        ft_batch = np.expand_dims(ft_data, axis=0)
        dt_batch = np.expand_dims(dt_data, axis=0)
        
        # Use persistent ONNX models directory
        onnx_dir = get_default_onnx_dir()
        onnx_dir.mkdir(exist_ok=True)
        
        # Download the model
        model_path = download_model(model_idx, onnx_dir)
        
        # Create GPU session
        providers = get_gpu_providers()
        gpu_session = ort.InferenceSession(str(model_path), providers=providers)
        
        # Run GPU inference
        gpu_predictions = run_onnx_inference(gpu_session, ft_batch, dt_batch)
        gpu_prob = float(gpu_predictions[0, 1])
        
        # Compare with expected probability
        diff = abs(gpu_prob - expected_prob)
        
        assert diff <= PROBABILITY_TOLERANCE, (
            f"GPU Model {model_idx} probability mismatch: "
            f"Expected={expected_prob}, GPU={gpu_prob}, Diff={diff}"
        )
        
        print(f"GPU Model {model_idx}: Expected={expected_prob}, GPU={gpu_prob:.8f} ✓")
    
    @pytest.mark.skipif(not TEST_DATA_FILE.exists(), reason="Test data file not found")
    def test_gpu_vs_cpu_consistency(self):
        """Test that GPU and CPU inference produce consistent results"""
        model_idx = "a"  # Use model 'a' for consistency test
        
        # Load and preprocess test data
        ft_data, dt_data = load_and_preprocess_h5_data(TEST_DATA_FILE)
        ft_batch = np.expand_dims(ft_data, axis=0)
        dt_batch = np.expand_dims(dt_data, axis=0)
        
        # Use persistent ONNX models directory
        onnx_dir = get_default_onnx_dir()
        onnx_dir.mkdir(exist_ok=True)
        
        # Download the model
        model_path = download_model(model_idx, onnx_dir)
        
        # Create CPU session
        cpu_session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        
        # Create GPU session
        gpu_providers = get_gpu_providers()
        gpu_session = ort.InferenceSession(str(model_path), providers=gpu_providers)
        
        # Run both inferences
        cpu_predictions = run_onnx_inference(cpu_session, ft_batch, dt_batch)
        gpu_predictions = run_onnx_inference(gpu_session, ft_batch, dt_batch)
        
        # Extract probabilities
        cpu_prob = float(cpu_predictions[0, 1])
        gpu_prob = float(gpu_predictions[0, 1])
        
        # Compare CPU vs GPU (should be very close)
        diff = abs(cpu_prob - gpu_prob)
        
        # Allow slightly larger tolerance for GPU vs CPU comparison
        gpu_tolerance = 1e-5
        
        assert diff <= gpu_tolerance, (
            f"CPU vs GPU mismatch for model {model_idx}: "
            f"CPU={cpu_prob}, GPU={gpu_prob}, Diff={diff}"
        )
        
        print(f"CPU vs GPU consistency: CPU={cpu_prob:.8f}, GPU={gpu_prob:.8f}, Diff={diff:.2e} ✓")
    
    @pytest.mark.skipif(not TEST_DATA_FILE.exists(), reason="Test data file not found")
    def test_gpu_batch_processing(self):
        """Test GPU inference with different batch sizes"""
        model_idx = "a"
        
        # Load and preprocess test data
        ft_data, dt_data = load_and_preprocess_h5_data(TEST_DATA_FILE)
        
        # Use persistent ONNX models directory
        onnx_dir = get_default_onnx_dir()
        onnx_dir.mkdir(exist_ok=True)
        
        # Download the model
        model_path = download_model(model_idx, onnx_dir)
        
        # Create GPU session
        providers = get_gpu_providers()
        session = ort.InferenceSession(str(model_path), providers=providers)
        
        # Test different batch sizes
        for batch_size in [1, 2, 4, 8]:
            # Create batch by repeating the same data
            ft_batch = np.repeat(ft_data[np.newaxis, :, :, :], batch_size, axis=0)
            dt_batch = np.repeat(dt_data[np.newaxis, :, :, :], batch_size, axis=0)
            
            # Run inference
            predictions = run_onnx_inference(session, ft_batch, dt_batch)
            
            # Verify output shape
            assert predictions.shape == (batch_size, 2), (
                f"Unexpected output shape for batch_size={batch_size}: {predictions.shape}"
            )
            
            # Verify all predictions are identical (same input)
            first_pred = predictions[0]
            for i in range(1, batch_size):
                diff = np.abs(predictions[i] - first_pred).max()
                assert diff < 1e-6, f"Batch predictions not consistent at index {i}"
            
            print(f"GPU batch size {batch_size}: ✓")


def test_gpu_availability_info():
    """Always runs - provides info about GPU availability"""
    available_providers = ort.get_available_providers()
    print(f"\nONNX Runtime available providers: {available_providers}")
    
    if has_gpu_support():
        gpu_providers = get_gpu_providers()
        print(f"GPU providers configured: {gpu_providers}")
        print("✅ GPU tests will run")
    else:
        print("❌ No GPU providers available - GPU tests will be skipped")
        print("To enable GPU support, install: pip install onnxruntime-gpu")


if __name__ == "__main__":
    # Run with verbose output to see GPU detection
    pytest.main([__file__, "-v", "-s", "--tb=short"])