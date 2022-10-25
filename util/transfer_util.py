import random
import time

from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Lambda

from data_type.enums import LayerType, getLayerType
from keras import initializers, Sequential
import numpy as np
from keras import backend as K


def get_transfer_filters(layerPos, layerNeg, threshold=0.7,
                         relax_relevance=True):
    if layerPos.type != LayerType.Conv2D or layerNeg.type != LayerType.Conv2D:
        raise Exception('Only Conv2D supported')
    if layerPos.filters != layerNeg.filters:
        raise Exception('Filters must be same')

    eligible_filters = []
    for fn in range(layerPos.filters):
        pac = layerPos.active_count_for_filter[fn]
        nac = layerNeg.active_count_for_filter[fn]

        if (relax_relevance and pac >= threshold) or \
                (not relax_relevance and (pac >= threshold or pac > nac)):
            eligible_filters.append(fn)

    return eligible_filters


def get_modified_weights(layerPos, eligible_filters, zero_initialize=False):
    nb_filter = len(eligible_filters)
    nb_filter=layerPos.filters
    kernel_shape = (layerPos.W.shape[0], layerPos.W.shape[1], layerPos.W.shape[2], nb_filter)

    if not zero_initialize:
        kernelInit = layerPos.kernel_initializer
        biasInit = layerPos.bias_initializer
        n_kernel = kernelInit(shape=kernel_shape).numpy()
        n_bias = biasInit(shape=nb_filter).numpy()
        tid = 0
        for fn in eligible_filters:
            n_kernel[:, :, :, tid] = layerPos.W[:, :, :, fn]
            n_bias[tid] = layerPos.B[fn]
            tid += 1
    else:
        zeroInit = initializers.get('zeros')
        n_kernel = zeroInit(shape=kernel_shape).numpy()
        n_bias = zeroInit(shape=nb_filter).numpy()
        for fn in eligible_filters:
            n_kernel[:, :, :, fn] = layerPos.W[:, :, :, fn]
            n_bias[fn] = layerPos.B[fn]

    return nb_filter, n_kernel, n_bias, len(eligible_filters)


def get_worked_out_transfer_filters(layerPos, layerNeg, trainable=False, threshold=0.7,
                                    relax_relevance=True):
    eligible_filters = get_transfer_filters(layerPos, layerNeg,
                                            threshold=threshold, relax_relevance=relax_relevance)
    nb_filter = len(eligible_filters)

    if nb_filter <= 0:
        raise Exception('No eligible filter')

    return get_modified_weights(layerPos, eligible_filters)

    # if trainable:
    #     kernelInit = layerPos.kernel_initializer
    #     biasInit = layerPos.bias_initializer
    #     n_kernel = kernelInit(shape=kernel_shape).numpy()
    #     n_bias = biasInit(shape=nb_filter).numpy()
    #     for fn in eligible_filters:
    #         n_kernel[:, :, :, fn] = layerPos.W[:, :, :, fn]
    #         n_bias[fn] = layerPos.B[fn]
    # else:
    #     zeroInit = initializers.get('zeros')
    #     n_kernel = zeroInit(shape=kernel_shape).numpy()
    #     n_bias = zeroInit(shape=nb_filter).numpy()
    #     for fn in eligible_filters:
    #         n_kernel[:, :, :, fn] = layerPos.W[:, :, :, fn]
    #         n_bias[fn] = layerPos.B[fn]

    # return nb_filter, n_kernel, n_bias, len(eligible_filters)


def construct_target_model(concernPos, concernNeg, originalModel, freezeUntil=-1, thresholds=None,
                           relax_relevance=True,
                           transfer=True, traditional_transfer=False, n_classes=5):
    target_model = Sequential()

    conv_id = 0
    for layerNo, layer in enumerate(concernPos):
        orLayer = originalModel.layers[layerNo]

        if transfer and layer.type == LayerType.Conv2D:

            trainable = False
            if conv_id <= freezeUntil:
                conv_id += 1
            else:
                conv_id += 1
                continue

            if not traditional_transfer:
                n_filter, n_kernel, n_bias, actual_transferred = get_worked_out_transfer_filters(concernPos[layerNo],
                                                                                                 concernNeg[layerNo],
                                                                                                 trainable=trainable,
                                                                                                 threshold=thresholds[
                                                                                                     conv_id],
                                                                                                 relax_relevance=relax_relevance)
                print(str(actual_transferred) + ' transferred out of ' + str(layer.filters) + ' in ' + str(
                    conv_id) + '\'th conv2d layer')
            else:
                n_filter, n_kernel, n_bias = layer.filters, layer.W, layer.B

            if layer.first_layer:
                if not trainable:
                    n_layer = Conv2D(n_filter, orLayer.kernel_size,
                                     padding=orLayer.padding,
                                     input_shape=layer.original_input_shape,
                                     activation=orLayer.activation,
                                     weights=[n_kernel, n_bias],
                                     trainable=trainable)
                else:
                    n_layer = Conv2D(n_filter, orLayer.kernel_size,
                                     padding=orLayer.padding,
                                     input_shape=layer.original_input_shape,
                                     activation=orLayer.activation,
                                     trainable=trainable)
            else:
                if not trainable:
                    n_layer = Conv2D(n_filter, orLayer.kernel_size,
                                     padding=orLayer.padding,
                                     activation=orLayer.activation,
                                     weights=[n_kernel, n_bias],
                                     trainable=trainable)
                else:
                    n_layer = Conv2D(n_filter, orLayer.kernel_size,
                                     padding=orLayer.padding,
                                     activation=orLayer.activation,
                                     trainable=trainable)
            target_model.add(n_layer)
        elif not transfer and layer.type == LayerType.Conv2D:
            trainable = True

            n_layer = Conv2D(layer.filters, orLayer.kernel_size,
                             padding=orLayer.padding,
                             input_shape=layer.original_input_shape,
                             activation=orLayer.activation,
                             trainable=trainable)
            target_model.add(n_layer)

        elif layer.type == LayerType.MaxPooling2D and \
                (getLayerType(target_model.layers[-1]) == LayerType.Conv2D):
            n_layer = MaxPooling2D(pool_size=orLayer.pool_size,
                                   padding=orLayer.padding)
            target_model.add(n_layer)

        elif layer.type == LayerType.Dropout and \
                (getLayerType(target_model.layers[-1]) == LayerType.MaxPooling2D or \
                 getLayerType(target_model.layers[-1]) == LayerType.Dense or \
                 getLayerType(target_model.layers[-1]) == LayerType.Conv2D):
            n_layer = Dropout(orLayer.rate)
            target_model.add(n_layer)

        elif layer.type == LayerType.Flatten:
            n_layer = Flatten()
            target_model.add(n_layer)

        elif layer.type == LayerType.Dense and layer.last_layer:
            n_layer = Dense(n_classes,
                            activation=orLayer.activation)
            target_model.add(n_layer)

        elif layer.type == LayerType.Dense:
            n_layer = Dense(layer.num_node,
                            activation=orLayer.activation)
            target_model.add(n_layer)

    target_model.compile(loss='categorical_crossentropy',
                         optimizer='adam',
                         metrics=['accuracy'])

    return target_model


def construct_target_model_partial_freeze(concernPos, concernNeg, originalModel, thresholds=None,
                                          relax_relevance=True, n_classes=5):
    target_model_a = Sequential()
    target_model_b = Sequential()

    conv_id = 0
    transfer_filters = {}
    for layerNo, layer in enumerate(concernPos):
        orLayer = originalModel.layers[layerNo]

        if layer.type == LayerType.Conv2D:

            eligible_transfer = get_transfer_filters(concernPos[layerNo], concernNeg[layerNo],
                                                     threshold=thresholds[conv_id],
                                                     relax_relevance=relax_relevance)
            print(str(len(eligible_transfer)) + ' transferred out of ' + str(layer.filters) + ' in ' + str(
                conv_id) + '\'th conv2d layer')

            n_filter, n_kernel, n_bias = layer.filters, layer.W, layer.B

            if layer.first_layer:
                n_layer = Conv2D(n_filter, orLayer.kernel_size,
                                 padding=orLayer.padding,
                                 input_shape=layer.original_input_shape,
                                 activation=orLayer.activation,
                                 weights=[n_kernel, n_bias])
            else:
                n_layer = Conv2D(n_filter, orLayer.kernel_size,
                                 padding=orLayer.padding,
                                 activation=orLayer.activation,
                                 weights=[n_kernel, n_bias])

            target_model_a.add(n_layer)
            target_model_b.add(n_layer)

            if len(eligible_transfer) > 0:
                target_model_a.add(Lambda(stopBackprop, arguments={'filters': eligible_transfer}))
                target_model_b.add(Lambda(stopBackpropRandom, arguments={'num_filters':
                                                                             len(eligible_transfer)}))

            conv_id += 1
            transfer_filters[layerNo] = eligible_transfer

        elif layer.type == LayerType.MaxPooling2D:
            n_layer = MaxPooling2D(pool_size=orLayer.pool_size,
                                   padding=orLayer.padding)
            target_model_a.add(n_layer)
            target_model_b.add(n_layer)

        elif layer.type == LayerType.Dropout:
            n_layer = Dropout(orLayer.rate)
            target_model_a.add(n_layer)
            target_model_b.add(n_layer)

        elif layer.type == LayerType.Flatten:
            n_layer = Flatten()
            target_model_a.add(n_layer)
            target_model_b.add(n_layer)

        elif layer.type == LayerType.Dense and layer.last_layer:
            n_layer = Dense(n_classes,
                            activation=orLayer.activation)
            target_model_a.add(n_layer)
            target_model_b.add(n_layer)

        elif layer.type == LayerType.Dense:
            n_layer = Dense(layer.num_node,
                            activation=orLayer.activation)
            target_model_a.add(n_layer)
            target_model_b.add(n_layer)

    target_model_a.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])
    target_model_b.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

    return target_model_a, target_model_b, transfer_filters


def construct_target_model_as_feature_extractor(concernPos, concernNeg, originalModel, thresholds=None,
                                                relax_relevance=True, n_classes=5, target_layer=0):
    target_model_a = Sequential()
    target_model_b = Sequential()

    conv_id = 0
    for layerNo, layer in enumerate(concernPos):
        orLayer = originalModel.layers[layerNo]

        if layer.type == LayerType.Conv2D:
            if conv_id <= target_layer:
                if conv_id == target_layer:
                    n_filter, n_kernel, n_bias, actual_transferred = get_worked_out_transfer_filters(
                        concernPos[layerNo],
                        concernNeg[layerNo],
                        threshold=thresholds[conv_id],
                        relax_relevance=relax_relevance)
                    print(str(n_filter) + ' transferred out of ' + str(layer.filters) + ' in ' + str(
                        conv_id) + '\'th conv2d layer')
                else:
                    n_filter, n_kernel, n_bias = layer.filters, layer.W, layer.B

                if layer.first_layer:
                    n_layer = Conv2D(n_filter, orLayer.kernel_size,
                                     padding=orLayer.padding,
                                     input_shape=layer.original_input_shape,
                                     activation=orLayer.activation,
                                     weights=[n_kernel, n_bias],
                                     trainable=False)
                else:
                    n_layer = Conv2D(n_filter, orLayer.kernel_size,
                                     padding=orLayer.padding,
                                     activation=orLayer.activation,
                                     weights=[n_kernel, n_bias],
                                     trainable=False)

                target_model_a.add(n_layer)
                target_model_b.add(n_layer)
            # else:
            # n_layer = Conv2D(layer.filters, orLayer.kernel_size,
            #                  padding=orLayer.padding,
            #                  activation=orLayer.activation,
            #                  trainable=True)

            conv_id += 1

        elif layer.type == LayerType.MaxPooling2D:
            n_layer = MaxPooling2D(pool_size=orLayer.pool_size,
                                   padding=orLayer.padding)
            target_model_a.add(n_layer)
            target_model_b.add(n_layer)

        elif layer.type == LayerType.Dropout:
            n_layer = Dropout(orLayer.rate)
            target_model_a.add(n_layer)
            target_model_b.add(n_layer)

        elif layer.type == LayerType.Flatten:
            n_layer = Flatten()
            target_model_a.add(n_layer)
            target_model_b.add(n_layer)

        elif layer.type == LayerType.Dense and layer.last_layer:
            n_layer = Dense(n_classes,
                            activation=orLayer.activation)
            target_model_a.add(n_layer)
            target_model_b.add(n_layer)

        elif layer.type == LayerType.Dense:
            n_layer = Dense(layer.num_node,
                            activation=orLayer.activation)
            target_model_a.add(n_layer)
            target_model_b.add(n_layer)

    target_model_a.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])
    target_model_b.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

    return target_model_a, target_model_b


def construct_target_model_approach_2(concernPos, originalModel, chosen_filters, n_classes=5, target_layer=0, zero_initialize=False):
    target_model_a = Sequential()
    conv_id = 0
    for layerNo, layer in enumerate(concernPos):
        orLayer = originalModel.layers[layerNo]

        if layer.type == LayerType.Conv2D:
            if conv_id <= target_layer:
                if conv_id == target_layer:
                    n_filter, n_kernel, n_bias, actual_transferred = get_modified_weights(
                        concernPos[layerNo], chosen_filters, zero_initialize=zero_initialize)
                    print(str(n_filter) + ' transferred out of ' + str(layer.filters) + ' in ' + str(
                        conv_id) + '\'th conv2d layer')
                else:
                    n_filter, n_kernel, n_bias = layer.filters, layer.W, layer.B

                if layer.first_layer:
                    n_layer = Conv2D(n_filter, orLayer.kernel_size,
                                     padding=orLayer.padding,
                                     input_shape=layer.original_input_shape,
                                     activation=orLayer.activation,
                                     weights=[n_kernel, n_bias],
                                     trainable=False)
                else:
                    n_layer = Conv2D(n_filter, orLayer.kernel_size,
                                     padding=orLayer.padding,
                                     activation=orLayer.activation,
                                     weights=[n_kernel, n_bias],
                                     trainable=False)

                target_model_a.add(n_layer)

            conv_id += 1

        elif layer.type == LayerType.MaxPooling2D and getLayerType(target_model_a.layers[-1]) == LayerType.Conv2D:
            n_layer = MaxPooling2D(pool_size=orLayer.pool_size,
                                   padding=orLayer.padding)
            target_model_a.add(n_layer)

        elif layer.type == LayerType.Dropout and \
                (getLayerType(target_model_a.layers[-1]) == LayerType.MaxPooling2D or \
                 getLayerType(target_model_a.layers[-1]) == LayerType.Dense or \
                 getLayerType(target_model_a.layers[-1]) == LayerType.Conv2D):
            n_layer = Dropout(orLayer.rate)
            target_model_a.add(n_layer)

        elif layer.type == LayerType.Flatten:
            n_layer = Flatten()
            target_model_a.add(n_layer)

        elif layer.type == LayerType.Dense and layer.last_layer:
            n_layer = Dense(n_classes,
                            activation=orLayer.activation)
            target_model_a.add(n_layer)

        elif layer.type == LayerType.Dense:
            n_layer = Dense(layer.num_node,
                            activation=orLayer.activation)
            target_model_a.add(n_layer)

    target_model_a.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

    return target_model_a


def train(model, x_train, y_train, x_test, y_test, epochs=50):
    start = time.time()

    history = model.fit(x_train,
                        y_train,
                        epochs=epochs,
                        batch_size=50,
                        verbose=0)
    end = time.time()
    # print("Training time: ", end - start)
    scores = model.evaluate(x_test, y_test, verbose=0)
    # print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    return scores[1] * 100, end - start


def stopBackprop(x, filters):
    mask = np.ones(x.shape[1:])  # 1 means all should learn by default
    for f in filters:
        mask[:, :, f] = 0  # filter f should not be updated, frozen
    mask_l = K.cast(mask, dtype=x.dtype)
    mask_h = K.abs(mask_l - 1)
    mask_h = K.cast(mask_h, dtype=x.dtype)

    return K.stop_gradient(mask_h * x) + mask_l * x


def stopBackpropRandom(x, num_filters):
    mask = np.ones(x.shape[1:])  # 1 means all should learn by default
    filters = random.sample(range(0, x.shape[3]), num_filters)
    for f in filters:
        mask[:, :, f] = 0  # filter f should not be updated, frozen
    mask_l = K.cast(mask, dtype=x.dtype)
    mask_h = K.abs(mask_l - 1)
    mask_h = K.cast(mask_h, dtype=x.dtype)

    return K.stop_gradient(mask_h * x) + mask_l * x
