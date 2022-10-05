from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten

from data_type.enums import LayerType
from keras import initializers, Sequential


def get_transferable_filters(layerPos, layerNeg, trainable=False, threshold=0.7,
                             relax_relevance=False):
    if layerPos.type != LayerType.Conv2D or layerNeg.type != LayerType.Conv2D:
        raise Exception('Only Conv2D supported')
    if layerPos.filters != layerNeg.filters:
        raise Exception('Filters must be same')

    eligible_filters = []
    for fn in range(layerPos.filters):
        pac = layerPos.active_count_for_filter[fn]
        nac = layerNeg.active_count_for_filter[fn]

        if (not relax_relevance and pac >= threshold) or \
                (relax_relevance and pac >= threshold and pac > nac):
            eligible_filters.append(fn)

    nb_filter = len(eligible_filters)

    if nb_filter <= 0:
        raise Exception('No eligible filter')

    # if trainable:
    nb_filter = layerPos.filters

    kernel_shape = (layerPos.W.shape[0], layerPos.W.shape[1], layerPos.W.shape[2], nb_filter)

    if trainable:
        kernelInit = layerPos.kernel_initializer
        biasInit = layerPos.bias_initializer
        n_kernel = kernelInit(shape=kernel_shape).numpy()
        n_bias = biasInit(shape=nb_filter).numpy()
        for fn in eligible_filters:
            n_kernel[:, :, :, fn] = layerPos.W[:, :, :, fn]
            n_bias[fn] = layerPos.B[fn]
    else:
        zeroInit = initializers.get('zeros')
        n_kernel = zeroInit(shape=kernel_shape).numpy()
        n_bias = zeroInit(shape=nb_filter).numpy()
        for fn in eligible_filters:
            n_kernel[:, :, :, fn] = layerPos.W[:, :, :, fn]
            n_bias[fn] = layerPos.B[fn]

    return nb_filter, n_kernel, n_bias, len(eligible_filters)


def construct_target_model(concernPos, concernNeg, originalModel, freezeUntil=-1, thresholds=None,
                           relax_relevance=False,
                           transfer=True, traditional_transfer=False, n_classes=5):
    target_model = Sequential()

    conv_id = 0
    for layerNo, layer in enumerate(concernPos):
        orLayer = originalModel.layers[layerNo]

        if transfer and layer.type == LayerType.Conv2D:

            trainable = True
            if conv_id <= freezeUntil:
                trainable = False

            if not traditional_transfer:
                n_filter, n_kernel, n_bias, actual_transferred = get_transferable_filters(concernPos[layerNo],
                                                                                          concernNeg[layerNo],
                                                                                          trainable=trainable,
                                                                                          threshold=thresholds[conv_id],
                                                                                          relax_relevance=relax_relevance)
                print(str(actual_transferred) + ' transferred out of ' + str(layer.filters) + ' in ' + str(
                    conv_id) + '\'th conv2d layer')
            else:
                n_filter, n_kernel, n_bias = layer.filters, layer.W, layer.B

            if layer.first_layer:
                n_layer = Conv2D(n_filter, orLayer.kernel_size,
                                 padding=orLayer.padding,
                                 input_shape=layer.original_input_shape,
                                 activation=orLayer.activation,
                                 weights=[n_kernel, n_bias],
                                 trainable=trainable)
            else:
                n_layer = Conv2D(n_filter, orLayer.kernel_size,
                                 padding=orLayer.padding,
                                 activation=orLayer.activation,
                                 weights=[n_kernel, n_bias],
                                 trainable=trainable)
            target_model.add(n_layer)
            conv_id += 1
        elif not transfer and layer.type == LayerType.Conv2D:
            trainable = True

            n_layer = Conv2D(layer.filters, orLayer.kernel_size,
                             padding=orLayer.padding,
                             input_shape=layer.original_input_shape,
                             activation=orLayer.activation,
                             trainable=trainable)
            target_model.add(n_layer)

        elif layer.type == LayerType.MaxPooling2D:
            n_layer = MaxPooling2D(pool_size=orLayer.pool_size,
                                   padding=orLayer.padding)
            target_model.add(n_layer)

        elif layer.type == LayerType.Dropout:
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


def get_transfer_model_name(freezeUntil, relax_relevance, threshold):
    name = 'transfer_model/target_'
    name += str(freezeUntil) + '_'
    name += str(int(relax_relevance)) + '_'
    name += str(threshold)
    name += '.h5'
    return name


def train(model, x_train, y_train, x_test, y_test, epochs=50):
    history = model.fit(x_train,
                        y_train,
                        epochs=epochs,
                        batch_size=50,
                        verbose=2)
    scores = model.evaluate(x_test, y_test, verbose=2)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
