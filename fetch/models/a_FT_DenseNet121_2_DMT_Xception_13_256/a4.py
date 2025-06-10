import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import h5py  # For loading Keras weights
import numpy as np

class KerasDenseNet121FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.densenet = timm.create_model(
            "densenet121", pretrained=False, num_classes=0, global_pool=""
        )
        self.num_features = self.densenet.num_features

    def forward(self, x):
        features = self.densenet.forward_features(x)
        pooled_features = F.adaptive_max_pool2d(features, (1, 1)).flatten(start_dim=1)
        return pooled_features


class KerasXceptionFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.xception = timm.create_model(
            "legacy_xception", pretrained=False, num_classes=0, global_pool=""
        )
        self.num_features = self.xception.num_features

    def forward(self, x):
        features = self.xception.forward_features(x)
        pooled_features = F.adaptive_max_pool2d(features, (1, 1)).flatten(start_dim=1)
        return pooled_features


class CombinedModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv_freq = nn.Conv2d(1, 3, kernel_size=2, stride=1, padding=0)
        self.relu_freq = nn.ReLU()
        self.conv_dm = nn.Conv2d(1, 3, kernel_size=2, stride=1, padding=0)
        self.relu_dm = nn.ReLU()

        self.densenet_features_extractor = KerasDenseNet121FeatureExtractor()
        self.xception_features_extractor = KerasXceptionFeatureExtractor()

        densenet_output_features = self.densenet_features_extractor.num_features
        xception_output_features = self.xception_features_extractor.num_features

        self.bn_densenet_out = nn.BatchNorm1d(
            densenet_output_features, eps=0.001, momentum=0.99
        )
        self.bn_xception_out = nn.BatchNorm1d(
            xception_output_features, eps=0.001, momentum=0.99
        )

        self.dropout_densenet = nn.Dropout(0.3)
        self.dropout_xception = nn.Dropout(0.3)

        self.dense1 = nn.Linear(densenet_output_features, 256)
        self.dense2 = nn.Linear(xception_output_features, 256)

        self.bn_multiply_out = nn.BatchNorm1d(256, eps=0.001, momentum=0.99)
        self.relu_multiply_out = nn.ReLU()

        self.final_dense = nn.Linear(256, num_classes)

    def forward(self, data_freq_time, data_dm_time):
        x_freq = self.relu_freq(self.conv_freq(data_freq_time))
        x_dm = self.relu_dm(self.conv_dm(data_dm_time))

        d_features = self.densenet_features_extractor(x_freq)
        x_features = self.xception_features_extractor(x_dm)

        d_bn = self.bn_densenet_out(d_features)
        x_bn = self.bn_xception_out(x_features)

        d_dropped = self.dropout_densenet(d_bn)
        x_dropped = self.dropout_xception(x_bn)

        d_dense = self.dense1(d_dropped)
        x_dense = self.dense2(x_dropped)

        multiplied = d_dense * x_dense

        bn_multiplied = self.bn_multiply_out(multiplied)
        activated_multiplied = self.relu_multiply_out(bn_multiplied)

        output = self.final_dense(activated_multiplied)
        output = F.softmax(output, -1)
        return output


# --- Keras Weight Loading Utilities (ensure these are the latest correct versions from previous steps) ---
def _get_original_keras_layer_name(keras_layer_unique_name):
    if "__" in keras_layer_unique_name:
        return keras_layer_unique_name.split("__")[0]
    return keras_layer_unique_name


def _try_get_keras_dataset(hf, base_path_in_h5, original_layer_name, dataset_suffixes):
    for suffix in dataset_suffixes:
        try:
            return hf[f"{base_path_in_h5}/{suffix}"][:]
        except KeyError:
            continue
    nested_path_base = f"{base_path_in_h5}/{original_layer_name}"
    for suffix in dataset_suffixes:
        try:
            return hf[f"{nested_path_base}/{suffix}"][:]
        except KeyError:
            continue
    return None


def _list_group_keys(hf, group_path):
    paths_to_check = [group_path]
    original_name = _get_original_keras_layer_name(group_path.split("/")[-1])
    if original_name != group_path.split("/")[-1]:
        paths_to_check.append(f"{group_path}/{original_name}")
    for p in paths_to_check:
        if p in hf:
            keys = list(hf[p].keys())
            print(f"    Keys in HDF5 group '{p}': {keys}")
            for k_item in keys:
                item_path = f"{p}/{k_item}"
                if item_path in hf and isinstance(hf[item_path], h5py.Group):
                    print(
                        f"      Subgroup '{k_item}' keys: {list(hf[item_path].keys())}"
                    )
        else:
            print(f"    HDF5 group '{p}' not found for listing keys.")


def _load_keras_weights_conv2d(pytorch_conv_layer, hf, keras_layer_full_path_base):
    original_name = _get_original_keras_layer_name(
        keras_layer_full_path_base.split("/")[-1]
    )
    kernel_suffixes = ["kernel:0", "kernel"]
    bias_suffixes = ["bias:0", "bias"]
    weights_np = _try_get_keras_dataset(
        hf, keras_layer_full_path_base, original_name, kernel_suffixes
    )
    if weights_np is None:
        print(
            f"KeyError loading kernel for Conv2D {keras_layer_full_path_base}. Original name: {original_name}. Tried suffixes: {kernel_suffixes}."
        )
        _list_group_keys(hf, keras_layer_full_path_base)
        return
    bias_np = None
    if pytorch_conv_layer.bias is not None:
        bias_np = _try_get_keras_dataset(
            hf, keras_layer_full_path_base, original_name, bias_suffixes
        )
        if bias_np is None:
            print(
                f"    Note: Bias not found for Conv2D {keras_layer_full_path_base}. Tried suffixes: {bias_suffixes}"
            )
    try:
        weights_torch = torch.from_numpy(weights_np).permute(3, 2, 0, 1).contiguous()
        pytorch_conv_layer.weight.data = weights_torch
        if pytorch_conv_layer.bias is not None and bias_np is not None:
            pytorch_conv_layer.bias.data = torch.from_numpy(bias_np)
    except Exception as e:
        print(
            f"Error processing/assigning Conv2D weights for {keras_layer_full_path_base}: {e}"
        )


def _load_keras_weights_dense(pytorch_linear_layer, hf, keras_layer_full_path_base):
    original_name = _get_original_keras_layer_name(
        keras_layer_full_path_base.split("/")[-1]
    )
    kernel_suffixes = ["kernel:0", "kernel"]
    bias_suffixes = ["bias:0", "bias"]
    weights_np = _try_get_keras_dataset(
        hf, keras_layer_full_path_base, original_name, kernel_suffixes
    )
    if weights_np is None:
        print(
            f"KeyError loading kernel for Dense {keras_layer_full_path_base}. Original name: {original_name}. Tried suffixes: {kernel_suffixes}."
        )
        _list_group_keys(hf, keras_layer_full_path_base)
        return
    bias_np = None
    if pytorch_linear_layer.bias is not None:
        bias_np = _try_get_keras_dataset(
            hf, keras_layer_full_path_base, original_name, bias_suffixes
        )
        if bias_np is None:
            print(
                f"    Note: Bias not found for Dense {keras_layer_full_path_base}. Tried suffixes: {bias_suffixes}"
            )
    try:
        weights_torch = torch.from_numpy(weights_np).permute(1, 0).contiguous()
        pytorch_linear_layer.weight.data = weights_torch
        if pytorch_linear_layer.bias is not None and bias_np is not None:
            pytorch_linear_layer.bias.data = torch.from_numpy(bias_np)
    except Exception as e:
        print(
            f"Error processing/assigning Dense weights for {keras_layer_full_path_base}: {e}"
        )


def _load_keras_weights_batchnorm(pytorch_bn_layer, hf, keras_layer_full_path_base):
    original_name = _get_original_keras_layer_name(
        keras_layer_full_path_base.split("/")[-1]
    )
    beta_suffixes = ["beta:0", "beta"]
    gamma_suffixes = ["gamma:0", "gamma"]
    mean_suffixes = ["moving_mean:0", "moving_mean"]
    var_suffixes = ["moving_variance:0", "moving_variance"]
    beta = _try_get_keras_dataset(
        hf, keras_layer_full_path_base, original_name, beta_suffixes
    )
    gamma = _try_get_keras_dataset(
        hf, keras_layer_full_path_base, original_name, gamma_suffixes
    )
    mean = _try_get_keras_dataset(
        hf, keras_layer_full_path_base, original_name, mean_suffixes
    )
    variance = _try_get_keras_dataset(
        hf, keras_layer_full_path_base, original_name, var_suffixes
    )
    if beta is None or gamma is None or mean is None or variance is None:
        print(
            f"KeyError loading BatchNorm weights for {keras_layer_full_path_base}. Original name: {original_name}. Check H5 and dataset names."
        )
        _list_group_keys(hf, keras_layer_full_path_base)
        return
    try:
        pytorch_bn_layer.bias.data = torch.from_numpy(beta)
        pytorch_bn_layer.weight.data = torch.from_numpy(gamma)
        pytorch_bn_layer.running_mean.data = torch.from_numpy(mean)
        pytorch_bn_layer.running_var.data = torch.from_numpy(variance)
    except Exception as e:
        print(
            f"Error assigning BatchNorm weights for {keras_layer_full_path_base}: {e}"
        )


def _load_keras_weights_separable_conv2d(
    timm_sep_conv_layer, hf, keras_layer_full_path_base
):
    original_name = _get_original_keras_layer_name(
        keras_layer_full_path_base.split("/")[-1]
    )
    dw_kernel_suffixes = ["depthwise_kernel:0", "depthwise_kernel"]
    pw_kernel_suffixes = ["pointwise_kernel:0", "pointwise_kernel"]
    dw_kernel_np = _try_get_keras_dataset(
        hf, keras_layer_full_path_base, original_name, dw_kernel_suffixes
    )
    pw_kernel_np = _try_get_keras_dataset(
        hf, keras_layer_full_path_base, original_name, pw_kernel_suffixes
    )
    if dw_kernel_np is None or pw_kernel_np is None:
        print(
            f"KeyError loading SeparableConv2D kernels for {keras_layer_full_path_base}. Original name: {original_name}. Check H5."
        )
        _list_group_keys(hf, keras_layer_full_path_base)
        return
    try:
        dw_kernel_torch = (
            torch.from_numpy(dw_kernel_np).permute(2, 3, 0, 1).contiguous()
        )
        timm_sep_conv_layer.conv1.weight.data = dw_kernel_torch
        pw_kernel_torch = (
            torch.from_numpy(pw_kernel_np).permute(3, 2, 0, 1).contiguous()
        )
        timm_sep_conv_layer.pointwise.weight.data = pw_kernel_torch
    except Exception as e:
        print(
            f"Error processing/assigning SeparableConv2D weights for {keras_layer_full_path_base}: {e}"
        )


# --- load_timm_densenet_weights (should be correct from previous version) ---
def load_timm_densenet_weights(timm_densenet_model, hf, keras_model_base_path_in_h5):
    print(f"Loading DenseNet weights from H5 base path: {keras_model_base_path_in_h5}")
    _load_keras_weights_conv2d(
        timm_densenet_model.features.conv0,
        hf,
        f"{keras_model_base_path_in_h5}/conv1/conv",
    )
    _load_keras_weights_batchnorm(
        timm_densenet_model.features.norm0,
        hf,
        f"{keras_model_base_path_in_h5}/conv1/bn",
    )
    densenet121_block_config = (6, 12, 24, 16)
    keras_block_prefixes = ["conv2_block", "conv3_block", "conv4_block", "conv5_block"]
    timm_block_names = ["denseblock1", "denseblock2", "denseblock3", "denseblock4"]
    timm_transition_names = ["transition1", "transition2", "transition3"]
    keras_pool_transition_prefixes = ["pool2", "pool3", "pool4"]
    for i, num_layers_in_block in enumerate(densenet121_block_config):
        timm_block = getattr(timm_densenet_model.features, timm_block_names[i])
        keras_b_prefix = keras_block_prefixes[i]
        for layer_num in range(1, num_layers_in_block + 1):
            timm_dense_layer = getattr(timm_block, f"denselayer{layer_num}")
            _load_keras_weights_batchnorm(
                timm_dense_layer.norm1,
                hf,
                f"{keras_model_base_path_in_h5}/{keras_b_prefix}{layer_num}_0_bn",
            )
            _load_keras_weights_conv2d(
                timm_dense_layer.conv1,
                hf,
                f"{keras_model_base_path_in_h5}/{keras_b_prefix}{layer_num}_1_conv",
            )
            _load_keras_weights_batchnorm(
                timm_dense_layer.norm2,
                hf,
                f"{keras_model_base_path_in_h5}/{keras_b_prefix}{layer_num}_1_bn",
            )
            _load_keras_weights_conv2d(
                timm_dense_layer.conv2,
                hf,
                f"{keras_model_base_path_in_h5}/{keras_b_prefix}{layer_num}_2_conv",
            )
        if i < len(timm_transition_names):
            timm_transition = getattr(
                timm_densenet_model.features, timm_transition_names[i]
            )
            keras_pool_prefix = keras_pool_transition_prefixes[i]
            _load_keras_weights_batchnorm(
                timm_transition.norm,
                hf,
                f"{keras_model_base_path_in_h5}/{keras_pool_prefix}_bn",
            )
            _load_keras_weights_conv2d(
                timm_transition.conv,
                hf,
                f"{keras_model_base_path_in_h5}/{keras_pool_prefix}_conv",
            )
    _load_keras_weights_batchnorm(
        timm_densenet_model.features.norm5, hf, f"{keras_model_base_path_in_h5}/bn"
    )
    print("DenseNet weight loading attempt finished.")


def load_timm_xception_weights(timm_xception_model, hf, keras_model_base_path_in_h5):
    print(f"Loading Xception weights from H5 base path: {keras_model_base_path_in_h5}")

    _load_keras_weights_conv2d(
        timm_xception_model.conv1, hf, f"{keras_model_base_path_in_h5}/block1_conv1"
    )
    _load_keras_weights_batchnorm(
        timm_xception_model.bn1, hf, f"{keras_model_base_path_in_h5}/block1_conv1_bn"
    )
    _load_keras_weights_conv2d(
        timm_xception_model.conv2, hf, f"{keras_model_base_path_in_h5}/block1_conv2"
    )
    _load_keras_weights_batchnorm(
        timm_xception_model.bn2, hf, f"{keras_model_base_path_in_h5}/block1_conv2_bn"
    )

    # Block 1 (Keras block2, timm block1, start_with_relu=False)
    # rep = [SeparableConv2d, BatchNorm2d, ReLU, SeparableConv2d, BatchNorm2d]
    # Keras JSON: block2_sepconv1, block2_sepconv1_bn, block2_sepconv2, block2_sepconv2_bn
    # Keras JSON: conv2d_3 (skip conv), batch_normalization_1 (skip bn)
    t_block = timm_xception_model.block1
    _load_keras_weights_separable_conv2d(
        t_block.rep[0], hf, f"{keras_model_base_path_in_h5}/block2_sepconv1"
    )
    _load_keras_weights_batchnorm(
        t_block.rep[1], hf, f"{keras_model_base_path_in_h5}/block2_sepconv1_bn"
    )
    _load_keras_weights_separable_conv2d(
        t_block.rep[3], hf, f"{keras_model_base_path_in_h5}/block2_sepconv2"
    )
    _load_keras_weights_batchnorm(
        t_block.rep[4], hf, f"{keras_model_base_path_in_h5}/block2_sepconv2_bn"
    )
    if t_block.skip is not None:
        _load_keras_weights_conv2d(
            t_block.skip, hf, f"{keras_model_base_path_in_h5}/conv2d_3"
        )
        _load_keras_weights_batchnorm(
            t_block.skipbn, hf, f"{keras_model_base_path_in_h5}/batch_normalization_1"
        )

    # Block 2 (Keras block3, timm block2, start_with_relu=True)
    # rep = [ReLU, SeparableConv2d, BatchNorm2d, ReLU, SeparableConv2d, BatchNorm2d]
    # Keras JSON: block3_sepconv1, block3_sepconv1_bn, block3_sepconv2, block3_sepconv2_bn
    # Keras JSON: conv2d_4 (skip conv), batch_normalization_2 (skip bn)
    t_block = timm_xception_model.block2
    _load_keras_weights_separable_conv2d(
        t_block.rep[1], hf, f"{keras_model_base_path_in_h5}/block3_sepconv1"
    )
    _load_keras_weights_batchnorm(
        t_block.rep[2], hf, f"{keras_model_base_path_in_h5}/block3_sepconv1_bn"
    )
    _load_keras_weights_separable_conv2d(
        t_block.rep[4], hf, f"{keras_model_base_path_in_h5}/block3_sepconv2"
    )
    _load_keras_weights_batchnorm(
        t_block.rep[5], hf, f"{keras_model_base_path_in_h5}/block3_sepconv2_bn"
    )
    if t_block.skip is not None:
        _load_keras_weights_conv2d(
            t_block.skip, hf, f"{keras_model_base_path_in_h5}/conv2d_4"
        )
        _load_keras_weights_batchnorm(
            t_block.skipbn, hf, f"{keras_model_base_path_in_h5}/batch_normalization_2"
        )

    # Block 3 (Keras block4, timm block3, start_with_relu=True)
    # rep = [ReLU, SeparableConv2d, BatchNorm2d, ReLU, SeparableConv2d, BatchNorm2d]
    # Keras JSON: block4_sepconv1, block4_sepconv1_bn, block4_sepconv2, block4_sepconv2_bn
    # Keras JSON: conv2d_5 (skip conv), batch_normalization_3 (skip bn)
    t_block = timm_xception_model.block3
    _load_keras_weights_separable_conv2d(
        t_block.rep[1], hf, f"{keras_model_base_path_in_h5}/block4_sepconv1"
    )
    _load_keras_weights_batchnorm(
        t_block.rep[2], hf, f"{keras_model_base_path_in_h5}/block4_sepconv1_bn"
    )
    _load_keras_weights_separable_conv2d(
        t_block.rep[4], hf, f"{keras_model_base_path_in_h5}/block4_sepconv2"
    )
    _load_keras_weights_batchnorm(
        t_block.rep[5], hf, f"{keras_model_base_path_in_h5}/block4_sepconv2_bn"
    )
    if t_block.skip is not None:
        _load_keras_weights_conv2d(
            t_block.skip, hf, f"{keras_model_base_path_in_h5}/conv2d_5"
        )
        _load_keras_weights_batchnorm(
            t_block.skipbn, hf, f"{keras_model_base_path_in_h5}/batch_normalization_3"
        )

    # Middle flow (Keras blocks 5-12 -> timm blocks 4-11, start_with_relu=True)
    # Each block has 3 reps of (ReLU, SepConv, BN)
    # Keras JSON: blockX_sepconv1, _bn, blockX_sepconv2, _bn, blockX_sepconv3, _bn
    for i in range(4, 12):
        t_block = getattr(timm_xception_model, f"block{i}")  # timm block i
        keras_block_num = i + 1  # Keras block number
        # rep[0]=ReLU, rep[1]=SepConv, rep[2]=BN
        # rep[3]=ReLU, rep[4]=SepConv, rep[5]=BN
        # rep[6]=ReLU, rep[7]=SepConv, rep[8]=BN
        _load_keras_weights_separable_conv2d(
            t_block.rep[1],
            hf,
            f"{keras_model_base_path_in_h5}/block{keras_block_num}_sepconv1",
        )
        _load_keras_weights_batchnorm(
            t_block.rep[2],
            hf,
            f"{keras_model_base_path_in_h5}/block{keras_block_num}_sepconv1_bn",
        )
        _load_keras_weights_separable_conv2d(
            t_block.rep[4],
            hf,
            f"{keras_model_base_path_in_h5}/block{keras_block_num}_sepconv2",
        )
        _load_keras_weights_batchnorm(
            t_block.rep[5],
            hf,
            f"{keras_model_base_path_in_h5}/block{keras_block_num}_sepconv2_bn",
        )
        _load_keras_weights_separable_conv2d(
            t_block.rep[7],
            hf,
            f"{keras_model_base_path_in_h5}/block{keras_block_num}_sepconv3",
        )
        _load_keras_weights_batchnorm(
            t_block.rep[8],
            hf,
            f"{keras_model_base_path_in_h5}/block{keras_block_num}_sepconv3_bn",
        )

    # Block 12 (Keras block13, timm block12, start_with_relu=True, grow_first=False, 2 reps)
    # rep = [ReLU, SeparableConv2d, BatchNorm2d, ReLU, SeparableConv2d, BatchNorm2d]
    # Keras JSON: block13_sepconv1, _bn, block13_sepconv2, _bn
    # Keras JSON: conv2d_6 (skip conv), batch_normalization_4 (skip bn)
    t_block = timm_xception_model.block12
    _load_keras_weights_separable_conv2d(
        t_block.rep[1], hf, f"{keras_model_base_path_in_h5}/block13_sepconv1"
    )
    _load_keras_weights_batchnorm(
        t_block.rep[2], hf, f"{keras_model_base_path_in_h5}/block13_sepconv1_bn"
    )
    _load_keras_weights_separable_conv2d(
        t_block.rep[4], hf, f"{keras_model_base_path_in_h5}/block13_sepconv2"
    )
    _load_keras_weights_batchnorm(
        t_block.rep[5], hf, f"{keras_model_base_path_in_h5}/block13_sepconv2_bn"
    )
    if t_block.skip is not None:
        _load_keras_weights_conv2d(
            t_block.skip, hf, f"{keras_model_base_path_in_h5}/conv2d_6"
        )
        _load_keras_weights_batchnorm(
            t_block.skipbn, hf, f"{keras_model_base_path_in_h5}/batch_normalization_4"
        )

    # Final SeparableConvs (Keras block14) -> timm conv3, bn3, conv4, bn4
    _load_keras_weights_separable_conv2d(
        timm_xception_model.conv3, hf, f"{keras_model_base_path_in_h5}/block14_sepconv1"
    )
    _load_keras_weights_batchnorm(
        timm_xception_model.bn3,
        hf,
        f"{keras_model_base_path_in_h5}/block14_sepconv1_bn",
    )
    _load_keras_weights_separable_conv2d(
        timm_xception_model.conv4, hf, f"{keras_model_base_path_in_h5}/block14_sepconv2"
    )
    _load_keras_weights_batchnorm(
        timm_xception_model.bn4,
        hf,
        f"{keras_model_base_path_in_h5}/block14_sepconv2_bn",
    )
    print("Xception weight loading attempt finished.")


# --- load_custom_keras_model_weights (main loader) ---
def load_custom_keras_model_weights(pytorch_model, keras_h5_path):
    h5_file_prefix = "model_weights"
    try:
        hf = h5py.File(keras_h5_path, "r")
    except Exception as e:
        print(f"Error opening HDF5 file {keras_h5_path}: {e}")
        return

    with torch.no_grad():
        pytorch_model.eval()
        print("Loading weights for initial Conv2D layers...")
        _load_keras_weights_conv2d(
            pytorch_model.conv_freq, hf, f"{h5_file_prefix}/conv2d_1__0"
        )
        _load_keras_weights_conv2d(
            pytorch_model.conv_dm, hf, f"{h5_file_prefix}/conv2d_2__1"
        )

        print("\nLoading weights for DenseNet features...")
        load_timm_densenet_weights(
            pytorch_model.densenet_features_extractor.densenet,
            hf,
            f"{h5_file_prefix}/densenet121__0",
        )

        print("\nLoading weights for Xception features...")
        load_timm_xception_weights(
            pytorch_model.xception_features_extractor.xception,
            hf,
            f"{h5_file_prefix}/xception__1",
        )

        print("\nLoading weights for final classification head...")
        _load_keras_weights_batchnorm(
            pytorch_model.bn_densenet_out, hf, f"{h5_file_prefix}/batch_normalization_5"
        )
        _load_keras_weights_batchnorm(
            pytorch_model.bn_xception_out, hf, f"{h5_file_prefix}/batch_normalization_6"
        )
        _load_keras_weights_dense(pytorch_model.dense1, hf, f"{h5_file_prefix}/dense_1")
        _load_keras_weights_dense(pytorch_model.dense2, hf, f"{h5_file_prefix}/dense_2")
        _load_keras_weights_batchnorm(
            pytorch_model.bn_multiply_out, hf, f"{h5_file_prefix}/batch_normalization_7"
        )
        _load_keras_weights_dense(
            pytorch_model.final_dense, hf, f"{h5_file_prefix}/dense_3"
        )

    hf.close()
    print(
        f"\nFinished attempt to load Keras weights from {keras_h5_path} into PyTorch model."
    )
    print(
        "Review any 'KeyError' or other error messages above to debug specific layer loading issues."
    )


if __name__ == "__main__":
    pytorch_model = CombinedModel(num_classes=2)
    pytorch_model.eval()

    dummy_freq_time = torch.randn(1, 1, 256, 256)
    dummy_dm_time = torch.randn(1, 1, 256, 256)

    with h5py.File("/workspaces/fetch/test.h5", "r") as hf:
        ft = np.array(hf["data_freq_time"])
        dt = np.array(hf["data_dm_time"])

    dummy_freq_time = torch.from_numpy(ft)[None, None, :, :]
    dummy_dm_time = torch.from_numpy(dt)[None, None, :, :]

    print(dummy_freq_time.shape)

    try:
        output_random_weights = pytorch_model(dummy_freq_time, dummy_dm_time)
        print(
            "PyTorch model instantiated. Dummy forward pass with random weights successful."
        )
        print("Output shape:", output_random_weights.shape)
    except Exception as e:
        print(f"Error during dummy forward pass: {e}")
        import traceback

        traceback.print_exc()

    print("\n--- Weight Loading ---")
    keras_weights_h5_file = "/workspaces/fetch/weights/a_ft_DenseNet121_2_dt_Xception_13_256.h5"  # USE YOUR ACTUAL PATH

    import os

    if not os.path.exists(keras_weights_h5_file):
        print(f"ERROR: Keras weights file not found at '{keras_weights_h5_file}'.")
    else:
        print(f"Attempting to load weights from: {keras_weights_h5_file}")
        try:
            load_custom_keras_model_weights(pytorch_model, keras_weights_h5_file)

            print("\n--- Forward pass after weight loading attempt ---")
            output_keras_weights = pytorch_model(dummy_freq_time, dummy_dm_time)
            print(
                "PyTorch model forward pass after Keras weight loading attempt successful."
            )
            print(
                "Output with loaded weights (or partially loaded if errors occurred):",
                output_keras_weights,
            )

        except Exception as e:
            print(f"An error occurred during or after weight loading: {e}")
            import traceback

            traceback.print_exc()
