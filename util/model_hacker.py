import keras
from keras import layers
import numpy as np


def get_modified_conv_weights(layer, eligible_filters, use_bias=True):
    nb_filter = len(eligible_filters)
    if use_bias:
        W, B = layer.get_weights()
    else:
        W = layer.get_weights()[0]
    original_kernel_shape = W.shape
    kernel_shape = (original_kernel_shape[0], original_kernel_shape[1], original_kernel_shape[2], nb_filter)

    n_kernel = np.zeros(shape=kernel_shape)
    if use_bias:
        n_bias = np.zeros(shape=nb_filter)
    tid = 0
    for fn in eligible_filters:
        n_kernel[:, :, :, tid] = W[:, :, :, fn]
        if use_bias:
            n_bias[tid] = B[fn]
        tid += 1

    if use_bias:
        return nb_filter, n_kernel, n_bias
    return nb_filter, n_kernel


def get_modified_batch_weights(layer, eligible_filters):
    nb_filter = len(eligible_filters)
    A, B, C = layer.get_weights()

    kernel_shape = nb_filter
    nA = np.zeros(shape=kernel_shape)
    nB = np.zeros(shape=kernel_shape)
    nC = np.zeros(shape=kernel_shape)

    tid = 0
    for fn in eligible_filters:
        nA[tid] = A[fn]
        nB[tid] = B[fn]
        nC[tid] = C[fn]
        tid += 1

    return [nA, nB, nC]


def hack_model(base_model, layer_name, model_name, eligible_filters):
    layer = base_model.get_layer(layer_name)

    k_shape = layer.kernel_size
    strides = layer.strides
    padding = layer.padding
    use_bias = layer.use_bias
    name = layer.name
    trainable = layer.trainable

    if use_bias:
        nb_filter, n_kernel, n_bias = get_modified_conv_weights(layer, eligible_filters, use_bias)
    else:
        nb_filter, n_kernel = get_modified_conv_weights(layer, eligible_filters, use_bias)

    if model_name == 'inceptionresnetv2':
        batch_layer_name = name + "_bn"
        batch_layer = base_model.get_layer(batch_layer_name)
        axis = batch_layer.axis[0]
        scale = batch_layer.scale

        act_layer_name = name + "_ac"
        act_layer = base_model.get_layer(act_layer_name)
        activation = act_layer.activation.__name__

        x = layers.Conv2D(
            nb_filter,
            k_shape,
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            name=name,
            trainable=False,
            weights=[n_kernel]
        )(base_model.get_layer('block8_10').output)

        x = layers.BatchNormalization(axis=axis, scale=scale, name=batch_layer_name, trainable=False,
                                      weights=get_modified_batch_weights(batch_layer, eligible_filters))(x)
        x = layers.Activation(activation, name=act_layer_name, trainable=False)(x)

    model = keras.Model(base_model.input, x)
    return model
