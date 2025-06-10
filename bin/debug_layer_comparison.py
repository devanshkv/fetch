import numpy as np
import h5py
import torch
from tensorflow import keras
from fetch.models.a_FT_DenseNet121_2_DMT_Xception_13_256.a4 import CombinedModel, load_custom_keras_model_weights
from fetch.utils import get_model
from rich.console import Console
from rich.table import Table

c = Console()

def preprocess_ft_data(data):
    data = np.nan_to_num(data)
    data = data - np.median(data)
    data = data / np.std(data)
    return data

def preprocess_dm_data(data):
    data = data.copy()
    data -= np.median(data)
    data /= np.std(data)
    return data

def debug_sample(keras_weights_path, bulk_data_path, sample_id):
    # Load TF model
    tf_model = get_model("a")

    # Load PyTorch model
    pt_model = CombinedModel(num_classes=2)
    load_custom_keras_model_weights(pt_model, keras_weights_path)
    pt_model.eval()

    # Open bulk dataset
    with h5py.File(bulk_data_path, "r") as hf:
        dm_sample = hf["data_dm_time"][sample_id]
        ft_sample = hf["data_freq_time"][sample_id]
        label = hf["data_labels"][sample_id]

    # Preprocess
    ft_processed = preprocess_ft_data(ft_sample[..., 0].T)
    dm_processed = preprocess_dm_data(dm_sample[..., 0])

    # TF input
    tf_in_ft = ft_processed[np.newaxis, ..., np.newaxis]
    tf_in_dm = dm_processed[np.newaxis, ..., np.newaxis]

    # PyTorch input
    pt_in_ft = torch.tensor(ft_processed[np.newaxis, np.newaxis, ...], dtype=torch.float32)
    pt_in_dm = torch.tensor(dm_processed[np.newaxis, np.newaxis, ...], dtype=torch.float32)

    # ====== TF INTERMEDIATE LOGIC ======
    # Create new model capturing intermediate values
    layer_names = ['conv2d_1__0', 'conv2d_2__1',
                  'densenet121__0', 'xception__1',
                  'batch_normalization_5', 'batch_normalization_6',
                  'dropout__0', 'dropout__1',
                  'dense_1', 'dense_2',
                  'batch_normalization_7']

    outputs = [tf_model.get_layer(name).output for name in layer_names]
    debug_model = keras.Model(inputs=tf_model.inputs, outputs=outputs)
    tf_intermediate = debug_model([tf_in_ft, tf_in_dm])

    # ====== PT INTERMEDIATE LOGIC ======
    with torch.no_grad():
        pt_output, pt_intermediate = pt_model(pt_in_ft, pt_in_dm)

    # Prepare comparison table
    table = Table(title=f"Layer-wise Comparison - Sample ID {sample_id}")
    table.add_column("Layer", justify="left")
    table.add_column("TensorFlow", justify="center")
    table.add_column("PyTorch", justify="center")
    table.add_column("Max Diff", justify="center")

    # Compare key layers
    layers_to_compare = {
        'input_freq': (tf_in_ft, pt_in_ft),
        'input_dm': (tf_in_dm, pt_in_dm),
        'conv_freq': (tf_intermediate[0], pt_intermediate['after_conv_freq']),
        'conv_dm': (tf_intermediate[1], pt_intermediate['after_conv_dm']),
        'ft_features': (tf_intermediate[2], pt_intermediate['ft_features']),
        'dt_features': (tf_intermediate[3], pt_intermediate['dt_features']),
        'ft_bn': (tf_intermediate[4], pt_intermediate['ft_bn_out']),
        'dt_bn': (tf_intermediate[5], pt_intermediate['dt_bn_out']),
        'after_dense1': (tf_intermediate[8], pt_intermediate['after_dense1']),
        'after_dense2': (tf_intermediate[9], pt_intermediate['after_dense2']),
        'after_multiply': (None, pt_intermediate['after_multiply'])
    }

    for name, (tf_val, pt_val) in layers_to_compare.items():
        if tf_val is None:
            # Multiplication layer doesn't exist in TF
            continue

        # Convert to numpy and flatten for comparison
        pt_np = pt_val.numpy()
        diff = np.abs(tf_val - pt_np).max()

        # Format values for display
        def format_arr(arr):
            arr = arr.squeeze()
            if arr.size > 3:
                return f"min: {arr.min():.4f}, max: {arr.max():.4f}, mean: {arr.mean():.4f}"
            return str(arr.round(4))

        table.add_row(
            name,
            format_arr(tf_val),
            format_arr(pt_np),
            f"{diff:.6f}"
        )

    c.print(table)
    c.print(f"[bold]Sample Label:[/bold] {label}")

if __name__ == "__main__":
    keras_weights = "/workspaces/fetch/weights/a_ft_DenseNet121_2_dt_Xception_13_256.h5"
    bulk_data = "/workspaces/fetch/test_data.hdf5"

    # Run for sample 2931 (shown discrepancy)
    debug_sample(keras_weights, bulk_data, 2931)

    # Run for sample 3709 (shown discrepancy)
    debug_sample(keras_weights, bulk_data, 3709)
