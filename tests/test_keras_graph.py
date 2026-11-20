"""
Unit tests for fetch.pytorch.keras_graph against plain numpy versions of TensorFlow's semantics.
They need no model weights: each builds a small Keras-format h5 file.
"""
import base64
import json

import h5py
import numpy as np
import pytest
import torch

from fetch.pytorch import keras_graph as kg

rng = np.random.default_rng(7)


def tf_same_pad_np(size, kernel, stride):
    out = -(-size // stride)
    total = max((out - 1) * stride + kernel - size, 0)
    return total // 2, total - total // 2


def conv2d_np(x, kernel, bias, stride, same):
    """TF Conv2D, channels last: x (h, w, cin), kernel (kh, kw, cin, cout)."""
    kh, kw = kernel.shape[:2]
    if same:
        x = np.pad(x, (tf_same_pad_np(x.shape[0], kh, stride), tf_same_pad_np(x.shape[1], kw, stride), (0, 0)))
    oh = (x.shape[0] - kh) // stride + 1
    ow = (x.shape[1] - kw) // stride + 1
    out = np.zeros((oh, ow, kernel.shape[3]))
    for i in range(oh):
        for j in range(ow):
            patch = x[i * stride:i * stride + kh, j * stride:j * stride + kw]
            out[i, j] = np.tensordot(patch, kernel, axes=3)
    return out + bias


def pool_np(x, size, stride, same, kind):
    """TF pooling, channels last; average pooling counts only real pixels."""
    if same:
        pads = (tf_same_pad_np(x.shape[0], size, stride), tf_same_pad_np(x.shape[1], size, stride), (0, 0))
        mask = np.pad(np.ones(x.shape[:2] + (1,)), pads)
        x = np.pad(x, pads, constant_values=-np.inf if kind == "max" else 0)
    else:
        mask = np.ones(x.shape[:2] + (1,))
    oh = (x.shape[0] - size) // stride + 1
    ow = (x.shape[1] - size) // stride + 1
    out = np.zeros((oh, ow, x.shape[2]))
    for i in range(oh):
        for j in range(ow):
            win = x[i * stride:i * stride + size, j * stride:j * stride + size]
            if kind == "max":
                out[i, j] = win.max(axis=(0, 1))
            else:
                m = mask[i * stride:i * stride + size, j * stride:j * stride + size]
                out[i, j] = win.sum(axis=(0, 1)) / m.sum()
    return out


def to_torch(x):
    """(h, w, c) numpy -> (1, c, h, w) tensor."""
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32))


def to_np(t):
    return t[0].permute(1, 2, 0).numpy()


@pytest.mark.parametrize(
    "size,kernel,stride,expected",
    [(256, 3, 1, (1, 1)), (256, 3, 2, (0, 1)), (255, 3, 2, (1, 1)), (8, 2, 1, (0, 1)), (7, 1, 2, (0, 0))],
)
def test_tf_same_pad(size, kernel, stride, expected):
    assert kg._tf_same_pad(size, kernel, stride) == expected


@pytest.mark.parametrize("stride,same", [(1, True), (2, True), (2, False)])
def test_conv2d(stride, same):
    x = rng.standard_normal((8, 7, 3))
    kernel = rng.standard_normal((3, 3, 3, 4))
    bias = rng.standard_normal(4)
    layer = kg.Conv2D(dict(name="c", kernel_size=[3, 3], strides=[stride, stride],
                           padding="same" if same else "valid", activation="linear", use_bias=True))
    layer.load({"kernel": kernel, "bias": bias})
    with torch.inference_mode():
        out = to_np(layer(to_torch(x)))
    np.testing.assert_allclose(out, conv2d_np(x, kernel, bias, stride, same), rtol=1e-5, atol=1e-5)


def test_separable_conv2d_with_depth_multiplier():
    x = rng.standard_normal((6, 6, 3))
    depthwise = rng.standard_normal((3, 3, 3, 2))
    pointwise = rng.standard_normal((1, 1, 6, 5))
    layer = kg.SeparableConv2D(dict(name="s", kernel_size=[3, 3], strides=[1, 1], padding="same",
                                    activation="linear", use_bias=False))
    layer.load({"depthwise_kernel": depthwise, "pointwise_kernel": pointwise})
    # TF's depthwise output channel c * multiplier + m is input channel c through filter m
    mid = np.concatenate(
        [conv2d_np(x[..., c:c + 1], depthwise[:, :, c:c + 1, m:m + 1], 0, 1, True) for c in range(3) for m in range(2)],
        axis=2,
    )
    expected = conv2d_np(mid, pointwise, 0, 1, False)
    with torch.inference_mode():
        out = to_np(layer(to_torch(x)))
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("kind,size,stride", [("max", 3, 2), ("max", 3, 1), ("avg", 3, 1), ("avg", 2, 2), ("avg", 3, 2)])
def test_same_pooling(kind, size, stride):
    x = rng.standard_normal((8, 7, 2))
    cls = kg.MaxPooling2D if kind == "max" else kg.AveragePooling2D
    layer = cls(dict(name="p", pool_size=[size, size], strides=[stride, stride], padding="same"))
    with torch.inference_mode():
        out = to_np(layer(to_torch(x)))
    np.testing.assert_allclose(out, pool_np(x, size, stride, True, kind), rtol=1e-5, atol=1e-6)


def test_batchnorm_without_scale():
    x = rng.standard_normal((4, 4, 3))
    w = {"beta": rng.standard_normal(3), "moving_mean": rng.standard_normal(3),
         "moving_variance": rng.uniform(0.5, 2, 3)}
    layer = kg.BatchNormalization(dict(name="bn", axis=-1, epsilon=1e-3, scale=False, center=True))
    layer.load(w)
    with torch.inference_mode():
        out = to_np(layer(to_torch(x)))
    expected = (x - w["moving_mean"]) / np.sqrt(w["moving_variance"] + 1e-3) + w["beta"]
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)


def lambda_config(source_file):
    """A Lambda config whose 'bytecode' names source_file, as Keras' func_dump output does."""
    code = base64.b64encode(b"\xe3\x02\x00inputs\x05scale" + source_file.encode() + b"\xda\x08<lambda>").decode()
    return dict(name="l", function=[code, None, None], function_type="lambda", arguments={"scale": 0.17})


def test_scaled_residual_lambda():
    layer = kg.ScaledResidual(lambda_config("/x/keras_applications/inception_resnet_v2.py"))
    a, b = torch.randn(2, 3, 4, 4), torch.randn(2, 3, 4, 4)
    torch.testing.assert_close(layer([a, b]), a + b * 0.17)


def test_unknown_lambda_is_refused():
    with pytest.raises(NotImplementedError):
        kg.ScaledResidual(lambda_config("/x/my_layers.py"))


def node(*names):
    return [[[n, 0, 0, {}] for n in names]]


def write_keras_h5(path, config, weights):
    """A whole-model Keras h5: weights is {top-level layer: {weight name: array}}."""
    with h5py.File(path, "w") as f:
        f.attrs["model_config"] = json.dumps(config).encode()
        g = f.create_group("model_weights")
        g.attrs["layer_names"] = [n.encode() for n in weights]
        for layer, ws in weights.items():
            lg = g.create_group(layer)
            lg.attrs["weight_names"] = [w.encode() for w in ws]
            for w, a in ws.items():
                lg.create_dataset(w, data=a)


def small_model(tmp_path):
    """
    Input -> nested model (conv "block1_conv1", saved as block1_conv1_1 the way TensorFlow
    uniquified it, then batch norm) -> global max pool -> dense softmax. Returns the h5 path and a
    numpy function computing the same thing.
    """
    conv = dict(name="block1_conv1", kernel_size=[3, 3], strides=[1, 1], padding="same",
                activation="linear", use_bias=True)
    bn = dict(name="block1_bn", axis=3, epsilon=1e-3, scale=True, center=True)
    inner = dict(
        name="base",
        layers=[
            dict(class_name="InputLayer", config=dict(name="inner_in"), inbound_nodes=[]),
            dict(class_name="Conv2D", config=conv, inbound_nodes=node("inner_in")),
            dict(class_name="BatchNormalization", config=bn, inbound_nodes=node("block1_conv1")),
            dict(class_name="Activation", config=dict(name="relu", activation="relu"), inbound_nodes=node("block1_bn")),
            dict(class_name="GlobalMaxPooling2D", config=dict(name="pool"), inbound_nodes=node("relu")),
        ],
        input_layers=[["inner_in", 0, 0]],
        output_layers=[["pool", 0, 0]],
    )
    config = dict(
        class_name="Model",
        config=dict(
            name="top",
            layers=[
                dict(class_name="InputLayer", config=dict(name="x"), inbound_nodes=[]),
                dict(class_name="Model", config=inner, inbound_nodes=[[["x", 0, 0, {}]]]),
                dict(class_name="Dense", config=dict(name="dense", activation="softmax", use_bias=True),
                     inbound_nodes=[[["base", 1, 0, {}]]]),
            ],
            input_layers=[["x", 0, 0]],
            output_layers=[["dense", 0, 0]],
        ),
    )
    w = dict(kernel=rng.standard_normal((3, 3, 2, 4)), bias=rng.standard_normal(4),
             gamma=rng.uniform(0.5, 2, 4), beta=rng.standard_normal(4),
             mean=rng.standard_normal(4), var=rng.uniform(0.5, 2, 4),
             dense_k=rng.standard_normal((4, 2)), dense_b=rng.standard_normal(2))
    path = tmp_path / "small.h5"
    write_keras_h5(path, config, {
        "x": {},
        "base": {
            "block1_conv1_1/kernel:0": w["kernel"], "block1_conv1_1/bias:0": w["bias"],
            "block1_bn/gamma:0": w["gamma"], "block1_bn/beta:0": w["beta"],
            "block1_bn/moving_mean:0": w["mean"], "block1_bn/moving_variance:0": w["var"],
        },
        "dense": {"dense/kernel:0": w["dense_k"], "dense/bias:0": w["dense_b"]},
    })

    def reference(x):
        y = conv2d_np(x, w["kernel"], w["bias"], 1, True)
        y = (y - w["mean"]) / np.sqrt(w["var"] + 1e-3) * w["gamma"] + w["beta"]
        y = np.maximum(y, 0).max(axis=(0, 1)) @ w["dense_k"] + w["dense_b"]
        e = np.exp(y - y.max())
        return e / e.sum()

    return path, reference


def test_graph_from_h5(tmp_path):
    path, reference = small_model(tmp_path)
    model = kg.from_keras_h5(path)
    x = rng.standard_normal((9, 8, 2))
    with torch.inference_mode():
        out = model(to_torch(x))[0].numpy()
    np.testing.assert_allclose(out, reference(x), rtol=1e-5, atol=1e-6)


def test_fold_batchnorm(tmp_path):
    path, reference = small_model(tmp_path)
    model = kg.from_keras_h5(path).fold_batchnorm()
    assert model.layers["base"].layers["block1_bn"].folded
    x = rng.standard_normal((9, 8, 2))
    with torch.inference_mode():
        out = model(to_torch(x))[0].numpy()
    np.testing.assert_allclose(out, reference(x), rtol=1e-5, atol=1e-6)


def test_missing_weights_are_an_error(tmp_path):
    path, _ = small_model(tmp_path)
    with h5py.File(path, "a") as f:
        g = f["model_weights/base"]
        names = [w.decode() if isinstance(w, bytes) else w for w in g.attrs["weight_names"]]
        g.attrs["weight_names"] = [w.encode() for w in names if "bias" not in w]
    with pytest.raises(KeyError):
        kg.from_keras_h5(path)
