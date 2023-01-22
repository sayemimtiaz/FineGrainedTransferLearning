import math
import random
import time

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from models.imagenet import target_dataset

from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Lambda, GlobalAveragePooling2D

from keras import initializers, Sequential, Model, optimizers
import numpy as np
from keras import backend as K
import tensorflow as tf
from util.common import freezeModel
from util.ordinary import dump_as_pickle


def get_modified_weights(layerPos, eligible_filters, zero_initialize=False):
    if not zero_initialize:
        nb_filter = len(eligible_filters)
    else:
        nb_filter = layerPos.filters
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


def get_svm(n_classes=5, shape=None):
    target_model = Sequential()
    target_model.add(Flatten())
    target_model.add(Dense(n_classes, activation='softmax'))
    target_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return target_model


def construct_target_without_inactive_feature(base_model, p_values=None):
    dump_as_pickle(p_values, 'p_values.pickle')

    return freezeModel(base_model)


def construct_reweighted_target(base_model, n_classes=5, p_values=None):
    freezeModel(base_model)
    x = base_model.output
    if p_values is not None:
        # x = Lambda(reweight, arguments={'weights': p_values})(x)
        x = Lambda(discardZeros, arguments={'weights': p_values})(x)

    if target_dataset == 'dog':
        model = Sequential()
        model.add(base_model)
        model.add(GlobalAveragePooling2D())
        model.add(Dropout(0.2))
        model.add(Dense(n_classes, activation='softmax'))
    elif target_dataset == 'bird':
        x = tf.keras.layers.Flatten()(base_model.output)
        x = tf.keras.layers.Dense(256, activation='tanh')(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Dense(200, activation='softmax')(x)

        optimizer = tf.keras.optimizers.Adam()

        model = tf.keras.Model(base_model.input, x)
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )
    else:

        x = Flatten()(x)
        # x = Dense(256, activation='relu')(x)
        # x = Dropout(0.2)(x)
        predictions = Dense(n_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def construct_target_partial_update(base_model, n_classes=5, filters=None):
    x = base_model.output
    x = Lambda(stopBackprop, arguments={'filters': filters})(x)
    x = Flatten()(x)
    # x = Dense(256, activation='relu')(x)
    # x = Dropout(0.2)(x)
    predictions = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train(model, x_train, y_train, x_test, y_test, epochs=50):
    start = time.time()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train,
                        y_train,
                        epochs=epochs,
                        batch_size=50,
                        verbose=0)
    end = time.time()
    # print("Training time: ", end - start)
    scores = model.evaluate(x_test, y_test, verbose=0)
    # print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    return scores[1], end - start


def trainDog(model, train_ds, val_ds, nb_train_samples, nb_valid_samples, epoch=30, batch_size=128, verbose=0):
    start = time.time()
    earlystop = EarlyStopping(
        monitor='val_loss',
        min_delta=0.001,
        patience=3,
        verbose=1,
        mode='auto'
    )
    reduceLR = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=3,
        verbose=1,
        mode='auto'
    )
    callbacks = [earlystop, reduceLR]

    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        train_ds,
        epochs=epoch,
        steps_per_epoch=nb_train_samples // batch_size,
        validation_data=val_ds,
        validation_steps=nb_valid_samples // batch_size,
        callbacks=callbacks,
        shuffle=True
    )
    end = time.time()

    return history.history['val_accuracy'][-1], end - start


def trainBird(model, train_ds, val_ds, epoch=100, batch_size=128):
    start = time.time()
    optimizer = tf.keras.optimizers.Adam()
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    # Adjust learning rate while training with LearningRateScheduler
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10 ** ((100 - epoch) / 20))

    history = model.fit(
        train_ds,
        epochs=epoch,
        shuffle=False,
        batch_size=batch_size,
        validation_data=val_ds,
        callbacks=[lr_scheduler]
    )
    end = time.time()

    return history.history['val_accuracy'][0], end - start


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


def removeInactives(model, x, weights):
    x = tf.reshape(x, [-1, x.shape[0], x.shape[1], x.shape[2]])

    x = model.predict(x, verbose=0)
    x = x.reshape([x.shape[1], x.shape[2], x.shape[3]])
    nz = 0
    for f in weights:
        if weights[f] > 0.0:
            nz += 1

    nx = np.zeros([x.shape[0], x.shape[1], nz])

    nf = 0
    for f in weights:
        if weights[f] > 0.0:
            nx[:, :, nf] = x[:, :, f]
            nf += 1

    return nx


def discardZeros(x, weights):
    includeIndices = []
    for f in weights:
        if weights[f] > 0.0:
            includeIndices.append(f)

    x = tf.gather(params=x, indices=includeIndices, axis=3)
    return x


def reweight(x, weights):
    mask = np.ones(x.shape[1:])
    for f in weights:
        mask[:, :, int(f)] = weights[f]
        # mask[int(f)] = weights[f]
    return mask * x


def plainWeight(pv):
    return pv


def regularWeight(pv):
    return math.exp(pv)


def randomWeight():
    return random.random() + 1.0


def accentedWeight(pv):
    return math.exp(1 + pv)


def binaryWeight(pv):
    if pv > 0.0:
        return 1.0
    return 0.0
