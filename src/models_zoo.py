"""
models_zoo.py

Model builders shared by the benchmark scripts. Contains the lightweight CNN
backbone used by PAF plus the comparison architectures: CNN-LSTM, ResNet1D, a
memory-fitted Transformer, InceptionTime and xresnet1d, the last two being the
architectures that top the published PTB-XL benchmark of Strodthoff et al.

All builders share the signature builder(input_shape, num_classes) and return an
uncompiled Keras model, so the benchmark harness can treat them uniformly.
"""

from keras.models import Sequential, Model
from keras.layers import (Conv1D, Dense, Dropout, MaxPooling1D, Flatten, LSTM,
                          Bidirectional, Input, Add, Activation, Concatenate,
                          BatchNormalization, GlobalAveragePooling1D,
                          LayerNormalization, MultiHeadAttention)


# ---------------------------------------------------------------------------
# PAF backbone
# ---------------------------------------------------------------------------
def build_cnn_ours(input_shape, num_classes):
    """The compact CNN used as the PAF backbone (three conv blocks)."""
    return Sequential([
        Conv1D(128, 11, activation="relu", input_shape=input_shape),
        MaxPooling1D(), Dropout(0.2),
        Conv1D(128, 13, activation="relu"), MaxPooling1D(), Dropout(0.2),
        Conv1D(128, 7, activation="relu"), Flatten(),
        Dense(128, activation="relu"), Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ], name="cnn_ours")


# ---------------------------------------------------------------------------
# Recurrent and residual baselines
# ---------------------------------------------------------------------------
def build_cnn_lstm(input_shape, num_classes):
    return Sequential([
        Conv1D(128, 11, activation="relu", input_shape=input_shape),
        MaxPooling1D(2), Dropout(0.2),
        Conv1D(128, 9, activation="relu"),
        MaxPooling1D(2), Dropout(0.2),
        Bidirectional(LSTM(64)), Dropout(0.3),
        Dense(128, activation="relu"), Dropout(0.2),
        Dense(num_classes, activation="softmax"),
    ], name="cnn_lstm")


def _resblock(x, filters, k=3, downsample=False):
    stride = 2 if downsample else 1
    shortcut = x
    x = Conv1D(filters, k, strides=stride, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv1D(filters, k, padding="same")(x)
    x = BatchNormalization()(x)
    if downsample or shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, strides=stride, padding="same")(shortcut)
        shortcut = BatchNormalization()(shortcut)
    x = Add()([x, shortcut])
    return Activation("relu")(x)


def build_resnet1d(input_shape, num_classes):
    inp = Input(shape=input_shape)
    x = Conv1D(64, 7, strides=2, padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(3, strides=2, padding="same")(x)
    x = _resblock(x, 64); x = _resblock(x, 64)
    x = _resblock(x, 128, downsample=True); x = _resblock(x, 128)
    x = _resblock(x, 256, downsample=True); x = _resblock(x, 256)
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    return Model(inp, Dense(num_classes, activation="softmax")(x), name="resnet1d")


# ---------------------------------------------------------------------------
# Transformer
#
# The earlier configuration (three encoders with key_dim 256 over a length-350
# sequence at batch 128) exhausted the 6 GB GPU. This version keeps the same
# encoder design but downsamples the beat to length ~87 before attention and
# uses a smaller key dimension, which fits comfortably and trains to
# convergence. Batch size is reduced by the harness for this model only.
# ---------------------------------------------------------------------------
def _tx_encoder(inp, head_size, num_heads, ff_dim, dropout=0.0):
    x = LayerNormalization(epsilon=1e-6)(inp)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = Add()([x, inp])
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Conv1D(ff_dim, 1, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Conv1D(inp.shape[-1], 1)(x)
    x = Dropout(dropout)(x)
    return Add()([x, res])


def build_transformer(input_shape, num_classes):
    inp = Input(shape=input_shape)
    x = Conv1D(128, 11, strides=2, activation="relu", padding="same")(inp)
    x = MaxPooling1D(4)(x)
    for _ in range(3):
        x = _tx_encoder(x, 64, 4, 128, 0.2)
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)
    return Model(inp, Dense(num_classes, activation="softmax")(x), name="transformer")


# ---------------------------------------------------------------------------
# InceptionTime (Ismail Fawaz et al., 2020) -- a top performer on the PTB-XL
# benchmark. Six inception modules with a residual connection every third.
# ---------------------------------------------------------------------------
def _inception_module(x, filters=32, kernel_sizes=(39, 19, 9), bottleneck=32):
    if x.shape[-1] > 1:
        inp = Conv1D(bottleneck, 1, padding="same", use_bias=False)(x)
    else:
        inp = x
    branches = [Conv1D(filters, k, padding="same", use_bias=False)(inp)
                for k in kernel_sizes]
    pooled = MaxPooling1D(3, strides=1, padding="same")(x)
    branches.append(Conv1D(filters, 1, padding="same", use_bias=False)(pooled))
    x = Concatenate(axis=-1)(branches)
    x = BatchNormalization()(x)
    return Activation("relu")(x)


def _shortcut(residual, x):
    s = Conv1D(x.shape[-1], 1, padding="same", use_bias=False)(residual)
    s = BatchNormalization()(s)
    return Activation("relu")(Add()([s, x]))


def build_inceptiontime(input_shape, num_classes, depth=6, filters=32):
    inp = Input(shape=input_shape)
    x = inp
    residual = inp
    for d in range(depth):
        x = _inception_module(x, filters=filters)
        if d % 3 == 2:
            x = _shortcut(residual, x)
            residual = x
    x = GlobalAveragePooling1D()(x)
    return Model(inp, Dense(num_classes, activation="softmax")(x), name="inceptiontime")


# ---------------------------------------------------------------------------
# xresnet1d50 (He et al. "bag of tricks" ResNet, 1D variant) -- the strongest
# single model in the PTB-XL benchmark of Strodthoff et al.
# ---------------------------------------------------------------------------
def _conv_bn(x, filters, k, stride=1, act=True):
    x = Conv1D(filters, k, strides=stride, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    return Activation("relu")(x) if act else x


def _xbottleneck(x, filters, stride=1, expansion=4):
    out = filters * expansion
    shortcut = x
    x = _conv_bn(x, filters, 1)
    x = _conv_bn(x, filters, 3, stride=stride)
    x = _conv_bn(x, out, 1, act=False)
    if shortcut.shape[-1] != out or stride != 1:
        # "Trick D": average-pool the shortcut instead of a strided 1x1 conv.
        if stride != 1:
            shortcut = MaxPooling1D(stride, strides=stride, padding="same")(shortcut)
        shortcut = _conv_bn(shortcut, out, 1, act=False)
    return Activation("relu")(Add()([x, shortcut]))


def build_xresnet1d50(input_shape, num_classes):
    inp = Input(shape=input_shape)
    # "Trick B": three 3-wide convolutions instead of one 7-wide stem.
    x = _conv_bn(inp, 32, 5, stride=2)
    x = _conv_bn(x, 32, 5)
    x = _conv_bn(x, 64, 5)
    x = MaxPooling1D(3, strides=2, padding="same")(x)
    for filters, blocks, stride in [(64, 3, 1), (128, 4, 2), (256, 6, 2), (512, 3, 2)]:
        for b in range(blocks):
            x = _xbottleneck(x, filters, stride=stride if b == 0 else 1)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.3)(x)
    return Model(inp, Dense(num_classes, activation="softmax")(x), name="xresnet1d50")


def build_cnn_tuned(input_shape, num_classes):
    """The configuration selected by the random search in tune_hparams.py."""
    return Sequential([
        Conv1D(128, 9, activation="relu", input_shape=input_shape),
        MaxPooling1D(), Dropout(0.3),
        Conv1D(128, 11, activation="relu"), MaxPooling1D(), Dropout(0.3),
        Conv1D(128, 7, activation="relu"), Flatten(),
        Dense(128, activation="relu"), Dropout(0.4),
        Dense(num_classes, activation="softmax"),
    ], name="cnn_tuned")


BUILDERS = {
    "CNN (ours)": build_cnn_ours,
    "CNN (tuned)": build_cnn_tuned,
    "CNN-LSTM": build_cnn_lstm,
    "ResNet1D": build_resnet1d,
    "Transformer": build_transformer,
    "InceptionTime": build_inceptiontime,
    "xresnet1d50": build_xresnet1d50,
}

# Models that need a smaller batch to fit a 6 GB GPU.
BATCH_OVERRIDE = {"Transformer": 64, "xresnet1d50": 64, "InceptionTime": 64,
                  "CNN (tuned)": 256}
