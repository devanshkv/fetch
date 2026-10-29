"""
Rebuilds a Keras 2 functional model in PyTorch from its saved config and loads its weights.

FETCH's weight files are whole Keras 2.2.2 models: the architecture is the `model_config`
attribute and the weights are under `model_weights`, so both come from the same file. Only the
layers FETCH's eleven models use are implemented. Tensors stay channels first (N, C, H, W);
TensorFlow's "same" padding, which puts the extra row or column at the end, is reproduced exactly.
"""
import base64
import json
import math
import re

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def _tf_same_pad(size, kernel, stride, dilation=1):
    """(before, after) padding TensorFlow uses for "same" along one axis."""
    out = math.ceil(size / stride)
    total = max((out - 1) * stride + (kernel - 1) * dilation + 1 - size, 0)
    return total // 2, total - total // 2


def _pad_same(x, kernel, stride, dilation=(1, 1), value=0.0):
    """
    (x, padding): when TensorFlow's padding is the same on both sides it's left to the op's own
    padding argument, which saves a copy; otherwise x comes back padded and padding is 0.
    """
    top, bottom = _tf_same_pad(x.shape[2], kernel[0], stride[0], dilation[0])
    left, right = _tf_same_pad(x.shape[3], kernel[1], stride[1], dilation[1])
    if top == bottom and left == right:
        return x, (top, left)
    return F.pad(x, (left, right, top, bottom), value=value), 0


def _activation(name):
    if name in (None, "linear"):
        return lambda x: x
    if name == "relu":
        return F.relu
    if name == "sigmoid":
        return torch.sigmoid
    if name == "tanh":
        return torch.tanh
    if name == "softmax":
        return lambda x: F.softmax(x, dim=1 if x.dim() == 2 else -1)
    raise NotImplementedError(f"Activation {name}")


def _channel_axis(axis, ndim):
    """Keras channels-last axis -> axis of the channels-first tensor."""
    axis = axis % ndim
    if ndim == 4:
        return {0: 0, 1: 2, 2: 3, 3: 1}[axis]
    return axis


class KerasLayer(nn.Module):
    """A layer with Keras weights; `params` lists them in the order Keras saves them."""

    params = ()

    def __init__(self, config):
        super().__init__()
        self.config = config

    def load(self, weights):
        """Loads {param name: array} as given by the h5 file."""
        if set(self.params) != set(weights):
            raise KeyError(f"{self.config['name']}: weights {sorted(weights)}, expected {sorted(self.params)}")
        for name in self.params:
            getattr(self, name).data.copy_(torch.from_numpy(self.convert(name, weights[name])))

    def convert(self, name, array):
        return np.asarray(array, dtype=np.float32)


class Conv2D(KerasLayer):
    def __init__(self, config):
        super().__init__(config)
        self.kernel_size = tuple(config["kernel_size"])
        self.strides = tuple(config["strides"])
        self.dilation = tuple(config.get("dilation_rate", (1, 1)))
        self.same = config["padding"] == "same"
        self.act = _activation(config.get("activation"))
        self.params = ("kernel", "bias") if config.get("use_bias", True) else ("kernel",)
        self.kernel = nn.Parameter(torch.empty(0), requires_grad=False)
        self.bias = nn.Parameter(torch.empty(0), requires_grad=False) if "bias" in self.params else None

    def load(self, weights):
        self.kernel.data = torch.empty(weights["kernel"].shape[::-1][:2] + self.kernel_size)
        if self.bias is not None:
            self.bias.data = torch.empty(weights["bias"].shape)
        super().load(weights)

    def convert(self, name, array):
        array = np.asarray(array, dtype=np.float32)
        return np.ascontiguousarray(array.transpose(3, 2, 0, 1)) if name == "kernel" else array

    def fold(self, scale, shift):
        """Applies a following x * scale + shift (per output channel) to the weights."""
        bias = self.bias.data if self.bias is not None else torch.zeros_like(shift)
        self.kernel.data = self.kernel.data * scale.view(-1, 1, 1, 1)
        self.bias = nn.Parameter(bias * scale + shift, requires_grad=False)

    def forward(self, x):
        padding = 0
        if self.same:
            x, padding = _pad_same(x, self.kernel_size, self.strides, self.dilation)
        return self.act(F.conv2d(x, self.kernel, self.bias, self.strides, padding, self.dilation))


class SeparableConv2D(KerasLayer):
    def __init__(self, config):
        super().__init__(config)
        self.kernel_size = tuple(config["kernel_size"])
        self.strides = tuple(config["strides"])
        self.dilation = tuple(config.get("dilation_rate", (1, 1)))
        self.same = config["padding"] == "same"
        self.act = _activation(config.get("activation"))
        names = ("depthwise_kernel", "pointwise_kernel")
        self.params = names + ("bias",) if config.get("use_bias", True) else names
        for name in self.params:
            setattr(self, name, nn.Parameter(torch.empty(0), requires_grad=False))

    def load(self, weights):
        kh, kw, cin, mult = weights["depthwise_kernel"].shape
        self.groups = cin
        self.depthwise_kernel.data = torch.empty(cin * mult, 1, kh, kw)
        self.pointwise_kernel.data = torch.empty(weights["pointwise_kernel"].shape[3], cin * mult, 1, 1)
        if "bias" in self.params:
            self.bias.data = torch.empty(weights["bias"].shape)
        super().load(weights)

    def convert(self, name, array):
        array = np.asarray(array, dtype=np.float32)
        if name == "depthwise_kernel":
            # (kh, kw, in, mult) -> (in * mult, 1, kh, kw); output channel c * mult + m, as in TF
            kh, kw, cin, mult = array.shape
            return np.ascontiguousarray(array.reshape(kh, kw, cin * mult).transpose(2, 0, 1)[:, None])
        if name == "pointwise_kernel":
            return np.ascontiguousarray(array.transpose(3, 2, 0, 1))
        return array

    def fold(self, scale, shift):
        """Applies a following x * scale + shift (per output channel) to the pointwise weights."""
        bias = self.bias.data if "bias" in self.params else torch.zeros_like(shift)
        self.pointwise_kernel.data = self.pointwise_kernel.data * scale.view(-1, 1, 1, 1)
        self.bias = nn.Parameter(bias * scale + shift, requires_grad=False)
        self.params = tuple(self.params) + (() if "bias" in self.params else ("bias",))

    def forward(self, x):
        padding = 0
        if self.same:
            x, padding = _pad_same(x, self.kernel_size, self.strides, self.dilation)
        x = F.conv2d(x, self.depthwise_kernel, None, self.strides, padding, self.dilation, self.groups)
        bias = self.bias if "bias" in self.params else None
        return self.act(F.conv2d(x, self.pointwise_kernel, bias))


class BatchNormalization(KerasLayer):
    def __init__(self, config):
        super().__init__(config)
        self.eps = config["epsilon"]
        axis = config["axis"]
        self.axis = axis[0] if isinstance(axis, (list, tuple)) else axis
        names = []
        if config.get("scale", True):
            names.append("gamma")
        if config.get("center", True):
            names.append("beta")
        self.params = tuple(names) + ("moving_mean", "moving_variance")
        for name in self.params:
            setattr(self, name, nn.Parameter(torch.empty(0), requires_grad=False))
        self.folded = False  # set once fold_batchnorm has moved it into the convolution before it

    def load(self, weights):
        for name in self.params:
            getattr(self, name).data = torch.empty(weights[name].shape)
        super().load(weights)

    def scale_shift(self):
        """(scale, shift) per channel: the layer is x * scale + shift."""
        scale = torch.rsqrt(self.moving_variance + self.eps)
        if "gamma" in self.params:
            scale = scale * self.gamma
        shift = -self.moving_mean * scale
        if "beta" in self.params:
            shift = shift + self.beta
        return scale, shift

    def forward(self, x):
        if self.folded:
            return x
        shape = [1] * x.dim()
        shape[_channel_axis(self.axis, x.dim())] = -1
        scale, shift = self.scale_shift()
        return x * scale.view(shape) + shift.view(shape)


class Dense(KerasLayer):
    def __init__(self, config):
        super().__init__(config)
        self.act = _activation(config.get("activation"))
        self.params = ("kernel", "bias") if config.get("use_bias", True) else ("kernel",)
        self.kernel = nn.Parameter(torch.empty(0), requires_grad=False)
        self.bias = nn.Parameter(torch.empty(0), requires_grad=False) if "bias" in self.params else None

    def load(self, weights):
        self.kernel.data = torch.empty(weights["kernel"].shape[::-1])
        if self.bias is not None:
            self.bias.data = torch.empty(weights["bias"].shape)
        super().load(weights)

    def convert(self, name, array):
        array = np.asarray(array, dtype=np.float32)
        return np.ascontiguousarray(array.T) if name == "kernel" else array

    def forward(self, x):
        if x.dim() != 2:
            raise NotImplementedError("Dense on a tensor that isn't 2D")
        return self.act(F.linear(x, self.kernel, self.bias))


class Activation(KerasLayer):
    def __init__(self, config):
        super().__init__(config)
        self.act = _activation(config["activation"])

    def forward(self, x):
        return self.act(x)


class Dropout(KerasLayer):
    def forward(self, x):
        return x


class ZeroPadding2D(KerasLayer):
    def __init__(self, config):
        super().__init__(config)
        (self.top, self.bottom), (self.left, self.right) = config["padding"]

    def forward(self, x):
        return F.pad(x, (self.left, self.right, self.top, self.bottom))


class MaxPooling2D(KerasLayer):
    def __init__(self, config):
        super().__init__(config)
        self.pool = tuple(config["pool_size"])
        self.strides = tuple(config["strides"] or config["pool_size"])
        self.same = config["padding"] == "same"

    def forward(self, x):
        padding = 0
        if self.same:  # max_pool2d's own padding is -inf too
            x, padding = _pad_same(x, self.pool, self.strides, value=-math.inf)
        return F.max_pool2d(x, self.pool, self.strides, padding)


class AveragePooling2D(MaxPooling2D):
    def forward(self, x):
        if not self.same:
            return F.avg_pool2d(x, self.pool, self.strides)
        # TF averages over the window's real pixels only
        padded, padding = _pad_same(x, self.pool, self.strides)
        if padding:
            return F.avg_pool2d(padded, self.pool, self.strides, padding, count_include_pad=False)
        ones, _ = _pad_same(torch.ones_like(x[:1, :1]), self.pool, self.strides)
        return F.avg_pool2d(padded, self.pool, self.strides) / F.avg_pool2d(ones, self.pool, self.strides)


class GlobalMaxPooling2D(KerasLayer):
    def forward(self, x):
        return torch.amax(x, dim=(2, 3))


class GlobalAveragePooling2D(KerasLayer):
    def forward(self, x):
        return torch.mean(x, dim=(2, 3))


class Concatenate(KerasLayer):
    def forward(self, *xs):
        return torch.cat(xs, dim=_channel_axis(self.config.get("axis", -1), xs[0].dim()))


class Add(KerasLayer):
    def forward(self, *xs):
        out = xs[0]
        for x in xs[1:]:
            out = out + x
        return out


class Multiply(KerasLayer):
    def forward(self, *xs):
        out = xs[0]
        for x in xs[1:]:
            out = out * x
        return out


class ScaledResidual(KerasLayer):
    """
    keras_applications' InceptionResNetV2 Lambda, `inputs[0] + inputs[1] * scale`.

    Keras saves a Lambda as marshalled Python bytecode, which only loads under the Python that
    wrote it, so it's recognised here by the source file named in that bytecode (decoded as text,
    never run) and rebuilt from its `scale` argument.
    """

    def __init__(self, config):
        super().__init__(config)
        code = config["function"][0] if isinstance(config["function"], (list, tuple)) else config["function"]
        if b"inception_resnet_v2.py" not in base64.b64decode(code):
            raise NotImplementedError(f"Lambda {config['name']} isn't InceptionResNetV2's scaled residual")
        self.scale = float(config["arguments"]["scale"])

    def forward(self, xs):
        return xs[0] + xs[1] * self.scale


LAYERS = {
    "Conv2D": Conv2D,
    "SeparableConv2D": SeparableConv2D,
    "BatchNormalization": BatchNormalization,
    "Dense": Dense,
    "Activation": Activation,
    "Dropout": Dropout,
    "ZeroPadding2D": ZeroPadding2D,
    "MaxPooling2D": MaxPooling2D,
    "AveragePooling2D": AveragePooling2D,
    "GlobalMaxPooling2D": GlobalMaxPooling2D,
    "GlobalAveragePooling2D": GlobalAveragePooling2D,
    "Concatenate": Concatenate,
    "Add": Add,
    "Multiply": Multiply,
    "Lambda": ScaledResidual,
}


class KerasGraph(nn.Module):
    """A Keras functional model (class_name Model): its layers run in the saved, topological order."""

    def __init__(self, config):
        super().__init__()
        self.name = config["name"]
        self.inputs = [name for name, _, _ in config["input_layers"]]
        self.outputs = [name for name, _, _ in config["output_layers"]]
        self.layers = nn.ModuleDict()
        self.order = []  # (layer name, [input layer names], Lambda takes a list)
        for layer in config["layers"]:
            kind, name = layer["class_name"], layer["config"]["name"]
            if kind == "InputLayer":
                continue
            if len(layer["inbound_nodes"]) != 1:
                raise NotImplementedError(f"{name} is called {len(layer['inbound_nodes'])} times")
            # every layer is called once (a nested model's call is its node 1), so the name is enough
            for _, _, tensor_index, *_ in layer["inbound_nodes"][0]:
                if tensor_index:
                    raise NotImplementedError(f"{name} takes a multi-output layer's output")
            if kind in ("Model", "Functional"):
                self.layers[name] = KerasGraph(layer["config"])
            elif kind in LAYERS:
                self.layers[name] = LAYERS[kind](layer["config"])
            else:
                raise NotImplementedError(f"Keras layer {kind} ({name})")
            self.order.append((name, [n[0] for n in layer["inbound_nodes"][0]], kind == "Lambda"))
        # tensors to drop after each step, once their last consumer has run
        last_use = {}
        for step, (_, sources, _) in enumerate(self.order):
            for s in sources:
                last_use[s] = step
        self.release = [[] for _ in self.order]
        for name, step in last_use.items():
            if name not in self.outputs:
                self.release[step].append(name)

    def forward(self, *inputs):
        tensors = dict(zip(self.inputs, inputs))
        for (name, sources, as_list), release in zip(self.order, self.release):
            args = [tensors[s] for s in sources]
            tensors[name] = self.layers[name](args) if as_list else self.layers[name](*args)
            del args
            for s in release:
                del tensors[s]
        out = [tensors[name] for name in self.outputs]
        return out[0] if len(out) == 1 else tuple(out)

    def fold_batchnorm(self):
        """
        Folds each BatchNormalization into the convolution before it, when that convolution has
        no activation and feeds nothing else. Inference only; the outputs change by rounding.
        """
        consumers = {}
        for _, sources, _ in self.order:
            for s in sources:
                consumers[s] = consumers.get(s, 0) + 1
        for name, sources, _ in self.order:
            layer = self.layers[name]
            if isinstance(layer, KerasGraph):
                layer.fold_batchnorm()
                continue
            if not isinstance(layer, BatchNormalization) or len(sources) != 1:
                continue
            src = self.layers[sources[0]] if sources[0] in self.layers else None
            if (
                isinstance(src, (Conv2D, SeparableConv2D))
                and src.config.get("activation") in (None, "linear")
                and consumers[sources[0]] == 1
                and sources[0] not in self.outputs
                and layer.axis % 4 == 3
            ):
                src.fold(*layer.scale_shift())
                layer.folded = True
        return self

    def load_h5(self, group):
        """
        Loads weights from a Keras `model_weights` group (or a nested model's group).

        At the top level each layer has its own group; a nested model's weights all sit in one
        group, named "<inner layer>/<param>:0".
        """
        for name, layer in self.layers.items():
            if isinstance(layer, KerasGraph):
                layer._load_nested(group[name])
            elif name in group and len(group[name].attrs["weight_names"]):
                g = group[name]
                weights = {}
                for w in g.attrs["weight_names"]:
                    w = w.decode() if isinstance(w, bytes) else w
                    weights[w.rsplit("/", 1)[-1].split(":")[0]] = g[w][()]
                layer.load(weights)

    def _load_nested(self, group):
        by_layer = {}
        for w in group.attrs["weight_names"]:
            w = w.decode() if isinstance(w, bytes) else w
            layer_name, param = w.rsplit("/", 1)
            by_layer.setdefault(layer_name, {})[param.split(":")[0]] = group[w][()]
        # TensorFlow uniquified variables whose layer name another nested model also used
        # (VGG's and Xception's block1_conv1): block1_conv1_1/kernel:0 belongs to block1_conv1
        for prefix in [p for p in by_layer if p not in self.layers]:
            base = re.sub(r"_\d+$", "", prefix)
            if base in self.layers and base not in by_layer:
                by_layer[base] = by_layer.pop(prefix)
        for name, layer in self.layers.items():
            if isinstance(layer, KerasGraph):
                raise NotImplementedError("Model nested two deep")
            if name in by_layer:
                layer.load(by_layer.pop(name))
            elif layer.params:
                raise KeyError(f"{self.name}: no weights for {name}")
        if by_layer:
            raise KeyError(f"{self.name}: weights for unknown layers {sorted(by_layer)[:5]}")


def from_keras_h5(path):
    """The PyTorch model and its weights from a whole-model Keras h5 file, in eval mode."""
    import h5py

    with h5py.File(path, "r") as f:
        config = f.attrs["model_config"]
        config = json.loads(config.decode() if isinstance(config, bytes) else config)
        if config["class_name"] not in ("Model", "Functional"):
            raise NotImplementedError(f"Top-level {config['class_name']}")
        model = KerasGraph(config["config"])
        model.load_h5(f["model_weights"])
    return model.eval()
