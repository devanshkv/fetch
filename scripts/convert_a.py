import argparse
import numpy as np
import tensorflow as tf
import torch
import tf2onnx
from onnx2torch import convert

from fetch import utils
import fetch.global_max_pool  # register GlobalMaxPool converter


def build_sample_inputs(ft_shape, dt_shape):
    """Generate random FT and DM time inputs."""
    ft_sample = tf.random.uniform((1,) + ft_shape)
    dt_sample = tf.random.uniform((1,) + dt_shape)
    return [ft_sample, dt_sample]


def convert_keras_to_pytorch(keras_model, sample_inputs):
    """Convert a Keras model to PyTorch using sample inputs."""
    input_spec = [
        tf.TensorSpec(inp.shape, dtype=inp.dtype, name=f"input{i}")
        for i, inp in enumerate(sample_inputs)
    ]
    onnx_model, _ = tf2onnx.convert.from_keras(
        keras_model, input_signature=input_spec
    )
    try:
        pt_model = convert(onnx_model)
    except NotImplementedError as e:
        print(f"Conversion failed: {e}")
        return None
    pt_model.eval()
    return pt_model


def compare_outputs(keras_model, pt_model, sample_inputs, verbose=False):
    """Run both models on the given inputs and return their outputs."""
    keras_out = keras_model(sample_inputs, training=False).numpy()
    pt_model.eval()
    with torch.no_grad():
        torch_inputs = [torch.from_numpy(inp.numpy()) for inp in sample_inputs]
        pt_out = pt_model(*torch_inputs)
    pt_out_np = pt_out.detach().cpu().numpy()
    if verbose:
        print("Keras output:", keras_out)
        print("Torch output:", pt_out_np)
    return keras_out, pt_out_np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="a",
        help="Model index (a-k) or path to Keras .h5 model; default downloads model 'a'",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=3,
        help="Number of random examples to test",
    )
    args = parser.parse_args()

    if len(args.model) == 1 and args.model.isalpha():
        print(f"Fetching model '{args.model}' from Zenodo...")
        keras_model = utils.get_model(args.model)
    else:
        print(f"Loading Keras model from {args.model}...")
        keras_model = tf.keras.models.load_model(args.model)

    input_shapes = keras_model.input_shape
    if isinstance(input_shapes, list):
        ft_shape = input_shapes[0][1:]
        dt_shape = input_shapes[1][1:]
    else:
        ft_shape = dt_shape = input_shapes[1:]

    for i in range(args.examples):
        sample_inputs = build_sample_inputs(ft_shape, dt_shape)
        print(
            f"\nRunning example {i+1} with input shapes {sample_inputs[0].shape} and {sample_inputs[1].shape}..."
        )
        pt_model = convert_keras_to_pytorch(keras_model, sample_inputs)
        if pt_model is None:
            print("Skipping remaining examples due to conversion failure")
            break
        k_out, p_out = compare_outputs(
            keras_model, pt_model, sample_inputs, verbose=True
        )
        diff = np.abs(k_out - p_out).mean()
        print(f"Difference for example {i+1}: {diff}")


if __name__ == "__main__":
    main()
