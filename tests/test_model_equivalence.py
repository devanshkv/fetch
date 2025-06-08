import numpy as np
import h5py  # Added to read H5 files
import scipy.signal as s
import tensorflow as tf
import torch
from fetch.models.a_FT_DenseNet121_2_DMT_Xception_13_256.a4 import CombinedModel, load_custom_keras_model_weights
from fetch.utils import get_model

def preprocess_ft_data(data):
    """Apply FT preprocessing pipeline"""
    data = np.nan_to_num(data)  # Replace NaNs
    data = s.detrend(data)       # Remove linear trend
    data = data - np.median(data)
    data = data / np.std(data)
    return data

def preprocess_dt_data(data):
    """Apply DT preprocessing pipeline"""
    data = np.nan_to_num(data)  # Replace NaNs
    data = data - np.median(data)
    data = data / np.std(data)
    return data

def test_model_equivalence(h5_file_path, test_data_path):
    # Load and preprocess test data from H5 file
    with h5py.File(test_data_path, 'r') as f:
        # Read FT data and transpose to match generator processing
        ft_data = f['data_freq_time'][:].T.astype(np.float32)
        dt_data = f['data_dm_time'][:].astype(np.float32)
    
    # Apply separate preprocessing for FT/DT data
    ft_processed = preprocess_ft_data(ft_data)
    dt_processed = preprocess_dt_data(dt_data)
    
    # Prepare inputs (add batch and channel dimensions)
    tf_input_ft = ft_processed[None, :, :, None]  # Batch, H, W, Channels
    tf_input_dt = dt_processed[None, :, :, None]
    
    torch_input_ft = torch.from_numpy(ft_processed)[None, None, :, :]  # Batch, C, H, W
    torch_input_dt = torch.from_numpy(dt_processed)[None, None, :, :]

    # TensorFlow model setup
    tf_model = get_model('a')
    
    # PyTorch model setup
    torch_model = CombinedModel(num_classes=2)
    load_custom_keras_model_weights(torch_model, h5_file_path)
    torch_model.eval()
    
    # TF prediction
    tf_output = tf_model.predict([tf_input_ft, tf_input_dt])
    
    # PyTorch prediction
    with torch.no_grad():
        torch_output = torch_model(torch_input_ft, torch_input_dt).numpy()
    
    # Compare outputs
    print("\n--- Results with test.h5 ---")
    print(f"TensorFlow output: {tf_output[0]}")
    print(f"PyTorch output: {torch_output[0]}")
    print(f"Output difference: {np.abs(tf_output - torch_output).max()}")
    
    if np.allclose(tf_output, torch_output, atol=1e-4):
        print("✅ Outputs match within tolerance!")
    else:
        print("❌ Outputs differ!")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python test_model_equivalence.py <path_to_model_weights.h5>")
        sys.exit(1)
    
    test_file = "/workspaces/fetch/test.h5"  # Set test data path
    test_model_equivalence(sys.argv[1], test_file)
