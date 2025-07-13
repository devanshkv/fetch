#!/usr/bin/env python3
"""
GPU Test Runner for FETCH ONNX Edition

This script runs GPU-specific tests and provides clear feedback about GPU availability.
"""

import sys
import subprocess
from pathlib import Path
import onnxruntime as ort


def check_gpu_availability():
    """Check what GPU providers are available"""
    available_providers = ort.get_available_providers()
    
    gpu_providers = {
        "CUDAExecutionProvider": "NVIDIA CUDA",
        "ROCMExecutionProvider": "AMD ROCm", 
        "DmlExecutionProvider": "DirectML (Windows)",
        "OpenVINOExecutionProvider": "Intel OpenVINO",
        "TensorrtExecutionProvider": "NVIDIA TensorRT",
    }
    
    found_gpu_providers = []
    for provider, description in gpu_providers.items():
        if provider in available_providers:
            found_gpu_providers.append(f"  ✅ {description} ({provider})")
    
    print("🔍 GPU Availability Check")
    print("=" * 50)
    print(f"ONNX Runtime providers: {len(available_providers)}")
    print(f"  - CPUExecutionProvider: {'✅' if 'CPUExecutionProvider' in available_providers else '❌'}")
    
    if found_gpu_providers:
        print("  - GPU Providers:")
        for provider in found_gpu_providers:
            print(provider)
        return True
    else:
        print("  - GPU Providers: ❌ None found")
        print("\n💡 To enable GPU support:")
        print("   pip install onnxruntime-gpu  # For CUDA/TensorRT")
        print("   # or install onnxruntime-rocm for AMD")
        return False


def run_tests():
    """Run the test suite with appropriate GPU handling"""
    has_gpu = check_gpu_availability()
    
    print(f"\n🧪 Running Tests")
    print("=" * 50)
    
    # Base test command
    cmd = [sys.executable, "-m", "pytest", "tests/", "-v", "-s"]
    
    if has_gpu:
        print("🚀 Running ALL tests (including GPU tests)")
        # Run all tests including GPU
        result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    else:
        print("🖥️  Running CPU-only tests (GPU tests will be skipped)")
        # Skip GPU-specific tests but run everything else
        result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    
    print(f"\n📊 Test Results")
    print("=" * 50)
    if result.returncode == 0:
        print("✅ All tests passed!")
    else:
        print(f"❌ Some tests failed (exit code: {result.returncode})")
    
    return result.returncode


def main():
    """Main entry point"""
    print("🔬 FETCH ONNX GPU Test Runner")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("tests").exists():
        print("❌ Error: tests/ directory not found")
        print("Please run this script from the project root directory")
        return 1
    
    return run_tests()


if __name__ == "__main__":
    sys.exit(main())