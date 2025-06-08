import numpy as np
import h5py
import pandas as pd
import torch
from tqdm import tqdm
from tensorflow import keras
from fetch.models.a_FT_DenseNet121_2_DMT_Xception_13_256.a4 import CombinedModel, load_custom_keras_model_weights
from fetch.utils import get_model


def preprocess_ft_data(data):
    "Apply FT preprocessing: transpose, subtract median, divide by std"
    data = data.T
    data = data.copy()
    data -= np.median(data)
    data /= np.std(data)
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
    with h5py.File(bulk_data_path, 'r') as hf:
        # Load all data into memory
        dm_data = hf['data_dm_time'][:]
        ft_data = hf['data_freq_time'][:]
        labels = hf['data_labels'][:]
        total_samples = len(labels)
        
        results = []
        batch_size = 100  # Process in batches for efficiency
        
        for start_idx in tqdm(range(0, total_samples, batch_size)):
            end_idx = min(start_idx + batch_size, total_samples)
            batch_results = []
            
            for i in range(start_idx, end_idx):
                # Preprocess data
                ft_processed = preprocess_ft_data(ft_data[i, ..., 0])
                dm_processed = preprocess_dm_data(dm_data[i, ..., 0])
                
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
                batch_results.append({
                    'sample_id': i,
                    'tf_output': tf_prob,
                    'pt_output': pt_prob,
                    'label': int(labels[i])
                })
            
            results.extend(batch_results)
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv('bulk_model_comparison.csv', index=False)
    print(f"Saved results for {len(df)} samples to bulk_model_comparison.csv")


if __name__ == "__main__":
    keras_weights = "/workspaces/fetch/weights/a_ft_DenseNet121_2_dt_Xception_13_256.h5"
    bulk_data = "/workspaces/fetch/test_data.hdf5"
    test_bulk_data(keras_weights, bulk_data)
