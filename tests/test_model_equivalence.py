import numpy as np
import tensorflow as tf
import torch
from fetch.models.a_FT_DenseNet121_2_DMT_Xception_13_256.a4 import CombinedModel, load_custom_keras_model_weights
from fetch.utils import get_model

def test_model_equivalence(h5_file_path):
    # Generate two separate inputs
    np.random.seed(42)
    input_shape = (1, 256, 256, 1)  # TF shape: (batch, height, width, channels)
    dummy_input_ft = np.random.randn(*input_shape).astype(np.float32)
    dummy_input_dt = np.random.randn(*input_shape).astype(np.float32)
    
    # TensorFlow model setup
    tf_model = get_model('a')
    
    # PyTorch model setup
    torch_model = CombinedModel(num_classes=2)
    load_custom_keras_model_weights(torch_model, h5_file_path)
    torch_model.eval()
    
    # TF prediction - pass both inputs separately
    tf_output = tf_model.predict([dummy_input_ft, dummy_input_dt])
    
    # Convert to channels-first for PyTorch: (N, C, H, W)
    torch_input_ft = torch.from_numpy(dummy_input_ft).permute(0, 3, 1, 2)
    torch_input_dt = torch.from_numpy(dummy_input_dt).permute(0, 3, 1, 2)
    
    # PyTorch prediction - pass both inputs with correct shape
    with torch.no_grad():
        torch_output = torch_model(torch_input_ft, torch_input_dt).numpy()
    
    # Compare outputs
    print("\n--- Results ---")
    print(f"TensorFlow output: {tf_output[0]}")
    print(f"PyTorch output: {torch_output[0]}")
    print(f"Output difference: {np.abs(tf_output - torch_output).max()}")
    
    # Check if outputs are close (using relaxed tolerance due to numerical differences)
    if np.allclose(tf_output, torch_output, atol=1e-4):
        print("✅ Outputs match within tolerance!")
    else:
        print("❌ Outputs differ!")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python test_model_equivalence.py <path_to_model_weights.h5>")
        sys.exit(1)
    test_model_equivalence(sys.argv[1])
