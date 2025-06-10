"""Convert Keras model *a* to PyTorch using ONNX.

This script demonstrates how model *a* can be exported to ONNX with
``tf2onnx`` and then converted to a PyTorch ``nn.Module`` via
``onnx2torch``.  All three packages (``tf2onnx``, ``onnx``, and
``onnx2torch``) must be installed for the conversion to work.  If they
are missing the script will print a helpful message instead of failing
with an import error.
"""

import os
import requests
import numpy as np
import tensorflow as tf
import torch
from tensorflow.keras.models import model_from_json

try:  # optional deps
    import tf2onnx  # type: ignore
    import onnx  # type: ignore
    from onnx2torch import convert  # type: ignore
    from onnx2torch.node_converters.registry import add_converter  # type: ignore
    from onnx2torch.onnx_graph import OnnxGraph  # type: ignore
    from onnx2torch.onnx_node import OnnxNode  # type: ignore
    from onnx2torch.utils.common import (
        OperationConverterResult,
        get_shape_from_value_info,
        onnx_mapping_from_node,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - only triggered when deps missing
    missing = str(exc.name)
    raise SystemExit(
        f"Required package '{missing}' is not installed.\n"
        "Install tf2onnx, onnx and onnx2torch to run the conversion."
    ) from exc


class OnnxGlobalMaxPool(torch.nn.Module):
    def __init__(self, dims=None):
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dims = self.dims if self.dims is not None else list(range(2, len(x.shape)))
        return torch.amax(x, dim=dims, keepdim=True)


@add_converter(operation_type='GlobalMaxPool', version=1)
def _global_max_pool(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    input_value_info = graph.value_info[node.input_values[0]]
    input_shape = get_shape_from_value_info(input_value_info)
    dims = list(range(2, len(input_shape))) if input_shape is not None else None
    torch_module = OnnxGlobalMaxPool(dims)
    return OperationConverterResult(
        torch_module=torch_module,
        onnx_mapping=onnx_mapping_from_node(node=node),
    )


def convert_keras_to_pytorch(model_json_path='fetch/models/a_FT_DenseNet121_2_DMT_Xception_13_256/ft_DenseNet121_2_dt_Xception_13_256.json',
                             weights_path='a_ft.h5',
                             onnx_path='model_a.onnx'):
    """Load Keras model ``model_json_path`` and return PyTorch equivalent.

    If ``weights_path`` is missing it will be downloaded from Zenodo.
    """

    weights_url = (
        'https://zenodo.org/api/records/5029590/'
        'files/a_ft_DenseNet121_2_dt_Xception_13_256.h5/content'
    )

    if not os.path.exists(weights_path):
        print(f'Downloading weights to {weights_path}...')
        resp = requests.get(weights_url, stream=True, timeout=60)
        resp.raise_for_status()
        with open(weights_path, 'wb') as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)

    with open(model_json_path, 'r') as f:
        keras_model = model_from_json(f.read())
    keras_model.load_weights(weights_path)

    # build tf input signature for tf2onnx
    spec = [tf.TensorSpec(tensor.shape, tf.float32, name=tensor.name.split(':')[0])
            for tensor in keras_model.inputs]

    onnx_model, _ = tf2onnx.convert.from_keras(keras_model, input_signature=spec, opset=11)
    onnx.save(onnx_model, onnx_path)

    pt_model = convert(onnx_model)
    return keras_model, pt_model


def compare_outputs(keras_model, pt_model, seed=0):
    np.random.seed(seed)
    dummy_inputs = [
        np.random.rand(1, *tensor.shape[1:]).astype(np.float32)
        for tensor in keras_model.inputs
    ]

    keras_out = keras_model.predict(dummy_inputs)

    # convert inputs to torch in NCHW
    torch_inputs = [torch.from_numpy(arr.transpose(0,3,1,2)) for arr in dummy_inputs]
    with torch.no_grad():
        pt_out = pt_model(*torch_inputs)
    return keras_out, pt_out.numpy()


if __name__ == '__main__':
    km, tm = convert_keras_to_pytorch()
    k_out, t_out = compare_outputs(km, tm)
    print('Keras output:', k_out)
    print('Torch output:', t_out)
    print('Difference:', np.abs(k_out - t_out))
