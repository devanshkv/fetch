import numpy as np
import h5py
import pandas as pd
import torch
import scipy.signal as s
from tqdm import tqdm
from tensorflow import keras
from fetch.models.a_FT_DenseNet121_2_DMT_Xception_13_256.a4 import CombinedModel, load_custom_keras_model_weights
from fetch.utils import get_model


def preprocess_ft_data(data):
    """Apply FT preprocessing pipeline"""
    data = np.nan_to_num(data)  # Replace NaNs
    data = s.detrend(data)       # Remove linear trend
    data = data - np.median(data)
    data = data / np.std(data)
    return data

def preprocess_dm_data(data):
    "Apply DM preprocessing: subtract median, divide by std"
    data = data.copy()
    data -= np.median(data)
    data /= np.std(data)
    return data

def test_bulk_data(keras_weights_path, bulk_data_path):
    "Compare TensorFlow and PyTorch model outputs on bulk dataset"
    # Load TensorFlow model
    tf_model = get_model('a')
    
    # Load PyTorch model
    pt_model = CombinedModel(num_classes=2)
    load_custom_keras_model_weights(pt_model, keras_weights_path)
    pt_model.eval()
    
    # Open bulk dataset
    results = []
    batch_size = 64  # Process in batches for memory efficiency
    
    with h5py.File(bulk_data_path, 'r') as hf:
        dm_dset = hf['data_dm_time']
        ft_dset = hf['data_freq_time']
        labels_dset = hf['data_labels']
        total_samples = len(labels_dset)
        
        for start_idx in tqdm(range(0, total_samples, batch_size)):
            end_idx = min(start_idx + batch_size, total_samples)
            
            # Load batch data
            dm_batch = dm_dset[start_idx:end_idx]
            ft_batch = ft_dset[start_idx:end_idx]
            labels_batch = labels_dset[start_idx:end_idx]
            
            for i, (dm_sample, ft_sample, label) in enumerate(zip(dm_batch, ft_batch, labels_batch)):
                idx = start_idx + i  # Global sample index
                
                # Preprocess FT data
                ft_processed = preprocess_ft_data(ft_sample[..., 0].T)  # Transpose for FT
                # Preprocess DM data
                dm_processed = preprocess_dm_data(dm_sample[..., 0])
                
                # TF prediction
                tf_in_ft = ft_processed[np.newaxis, ..., np.newaxis]
                tf_in_dm = dm_processed[np.newaxis, ..., np.newaxis]
                tf_prob = tf_model.predict([tf_in_ft, tf_in_dm], verbose=0)[0, 1]
                
                # PyTorch prediction
                pt_in_ft = torch.tensor(ft_processed[np.newaxis, np.newaxis, ...])
                pt_in_dm = torch.tensor(dm_processed[np.newaxis, np.newaxis, ...])
                with torch.no_grad():
                    pt_out = pt_model(pt_in_ft, pt_in_dm).numpy()[0]
                pt_prob = pt_out[1]
                
                # Collect results
                results.append({
                    'sample_id': idx,
                    'tf_output': tf_prob,
                    'pt_output': pt_prob,
                    'label': int(label)
                })
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv('bulk_model_comparison.csv', index=False)
    print(f"Saved results for {len(df)} samples to bulk_model_comparison.csv")


if __name__ == "__main__":
    keras_weights = "/workspaces/fetch/weights/a_ft_DenseNet121_2_dt_Xception_13_256.h5"
    bulk_data = "/workspaces/fetch/test_data.hdf5"
    test_bulk_data(keras_weights, bulk_data)
