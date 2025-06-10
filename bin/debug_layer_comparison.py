import numpy as np
import h5py
import torch
import tensorflow as tf
from tensorflow.keras.models import Model
from fetch.models.a_FT_DenseNet121_2_DMT_Xception_13_256.a4 import CombinedModel, load_custom_keras_model_weights
from fetch.utils import get_model
from rich.console import Console
from rich.table import Table

c = Console()

def create_tf_intermediate_function(tf_model):
    """Create a function to get intermediate outputs from TensorFlow model"""
    from tensorflow.keras import backend as K

    # Simplify layer selection - focus on critical comparison points
    layer_names = [
        'conv2d_1__0',     # FT branch output
        'conv2d_2__1',     # DM branch output
        'ft_features',      # Post DenseNet features
        'dt_features',      # Post Xception features
        'after_dense1',
        'after_dense2',
        'dense_3'           # Final output layer
    ]

    # Create output tensors using more robust layer retrieval
    outputs = []
    valid_names = []
    for name in layer_names:
        try:
            # Special handling for head layers
            if name == 'ft_features':
                outputs.append(tf_model.get_layer('dense_1').input)
            elif name == 'dt_features':
                outputs.append(tf_model.get_layer('dense_2').input)
            else:
                outputs.append(tf_model.get_layer(name).output)
            valid_names.append(name)
        except:
            logger.warning(f"Layer {name} not found in model")

    return K.function(tf_model.inputs, outputs), valid_names

def debug_sample(keras_weights_path, bulk_data_path, sample_id):
    # Load TensorFlow model and create intermediate models
    tf_model = get_model("a")
    tf_intermediate_function, tf_layer_names = create_tf_intermediate_function(tf_model)

    # Load PyTorch model
    pt_model = CombinedModel(num_classes=2)
    load_custom_keras_model_weights(pt_model, keras_weights_path)
    pt_model.eval()

    with h5py.File(bulk_data_path, "r") as hf:
        dm_sample = hf["data_dm_time"][sample_id]
        ft_sample = hf["data_freq_time"][sample_id]
        label = hf["data_labels"][sample_id]

    # Preprocess data
    ft_processed = (ft_sample[..., 0].T - np.median(ft_sample[..., 0])) / np.std(ft_sample[..., 0])
    dm_processed = (dm_sample[..., 0] - np.median(dm_sample[..., 0])) / np.std(dm_sample[..., 0])

    # TensorFlow input
    tf_in_ft = ft_processed[np.newaxis, ..., np.newaxis]
    tf_in_dm = dm_processed[np.newaxis, ..., np.newaxis]

    # PyTorch input
    pt_in_ft = torch.tensor(ft_processed[np.newaxis, np.newaxis, ...], dtype=torch.float32)
    pt_in_dm = torch.tensor(dm_processed[np.newaxis, np.newaxis, ...], dtype=torch.float32)

    # Get TensorFlow intermediate outputs
    tf_outputs = tf_intermediate_function([tf_in_ft, tf_in_dm])

    # Get PyTorch intermediate outputs
    with torch.no_grad():
        _, pt_debug = pt_model(pt_in_ft, pt_in_dm)

    # Prepare comparison table
    table = Table(title=f"Layer-wise Comparison - Sample ID {sample_id}")
    table.add_column("Layer", justify="left")
    table.add_column("TensorFlow Shape", justify="center")
    table.add_column("PyTorch Shape", justify="center")
    table.add_column("Max Abs Diff", justify="center")
    table.add_column("Max Rel Diff (%)", justify="center")

    # Updated layer mapping in debug_sample
    tf_to_pt_map = {
        'conv2d_1__0': 'after_conv_freq',
        'conv2d_2__1': 'after_conv_dm',
        'ft_features': 'ft_features',
        'dt_features': 'dt_features',
        'after_dense1': 'after_dense1',
        'after_dense2': 'after_dense2',
        'dense_3': 'final'  # Final output
    }

    for tf_layer_name, tf_out in zip(tf_layer_names, tf_outputs):
        # Handle TF outputs (convert to numpy if needed)
        tf_data = tf_out.squeeze()

        # Get corresponding PyTorch output
        if tf_layer_name in tf_to_pt_map:
            pt_key = tf_to_pt_map[tf_layer_name]
            pt_data = pt_debug[pt_key].squeeze().numpy()
            valid_comparison = True
        else:
            c.print(f"[yellow]Warning: No mapping for {tf_layer_name}[/yellow]")
            pt_data = np.zeros_like(tf_data)
            valid_comparison = False

        # Calculate differences
        abs_diff = np.abs(tf_data - pt_data).max()
        rel_diff = 100 * abs_diff / (np.abs(tf_data).max() + 1e-9)

        # Add table row - handle final output differently
        tf_shape = str(tf_out.squeeze().shape)
        if 'final' in pt_key:
            pt_shape = str(pt_data.shape)
        else:
            pt_shape = str(pt_data.squeeze().shape)

        table.add_row(
            tf_layer_name,
            tf_shape,
            pt_shape,
            f"{abs_diff:.4e}",
            f"{rel_diff:.2f}%" if valid_comparison else "N/A"
        )

    c.print(table)
    c.print(f"[bold]Sample Label:[/bold] {label}")
    c.print(f"[bold green]TF Final Output:[/bold green] {tf_model.predict([tf_in_ft, tf_in_dm])}")
    with torch.no_grad():
        c.print(f"[bold blue]PT Final Output:[/bold blue] {pt_model(pt_in_ft, pt_in_dm)[0].numpy()}")

if __name__ == "__main__":
    keras_weights = "/workspaces/fetch/weights/a_ft_DenseNet121_2_dt_Xception_13_256.h5"
    bulk_data = "/workspaces/fetch/test_data.hdf5"

    # Test with sample 2931
    debug_sample(keras_weights, bulk_data, 2931)

    # Add more samples as needed
    # debug_sample(keras_weights, bulk_data, 3709)
