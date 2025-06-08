import numpy as np
import h5py
import tensorflow as tf
import torch
from fetch.models.a_FT_DenseNet121_2_DMT_Xception_13_256.a4 import CombinedModel, load_custom_keras_model_weights
from fetch.utils import get_model, ready_for_train

def test_model_equivalence(h5_file_path):
    # Generate dummy input
    np.random.seed(42)
    input_shape = (1, 1, 256, 256)  # TF input shape (batch, height, width, channels)
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    
    # TensorFlow model setup
    tf_model = get_model('a')
    #tf_model = ready_for_train(tf_model, ndt=0, nft=0, nf=1)
    
    # PyTorch model setup
    torch_model = CombinedModel(num_classes=2)
    load_custom_keras_model_weights(torch_model, h5_file_path)
    torch_model.eval()
    
    # TF prediction
    tf_input = tf.convert_to_tensor(dummy_input)
    tf_output = tf_model.predict(tf_input)
    
    # Torch prediction (transpose to channels-first)
    torch_input = torch.from_numpy(dummy_input)  # (batch, channels, height,  width)
    with torch.no_grad():
        torch_output = torch_model(torch_input, torch_input).numpy()
    
    # Compare outputs
    print("\n--- Results ---")
    print(f"TensorFlow output: {tf_output[0]}")
    print(f"PyTorch output: {torch_output[0]}")
    print(f"Output difference: {np.abs(tf_output - torch_output).max()}")
    
    # Check if outputs are close
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
