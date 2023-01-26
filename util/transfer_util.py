import math
import random
import time

import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from constants import target_dataset

from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Lambda, GlobalAveragePooling2D, \
    RandomFourierFeatures

from keras import initializers, Sequential, Model, optimizers
import numpy as np
from keras import backend as K
import tensorflow as tf
from util.common import freezeModel
from util.ordinary import dump_as_pickle, get_bottleneck_name, get_transfer_model_name

import warnings

warnings.filterwarnings("ignore")


def compile_svm_classifier(model):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.hinge,
        metrics=[keras.metrics.CategoricalAccuracy(name="acc")],
    )
    return model


def get_svm_classifier(shape, n_classes=5):
    target_model = Sequential()
    target_model.add(Flatten(input_shape=shape))
    target_model.add(RandomFourierFeatures(output_dim=4096, scale=10.0, kernel_initializer="gaussian"))
    target_model.add(Dense(n_classes))

    return target_model


def get_dense_classifier(shape, n_classes=5):
    target_model = Sequential()
    target_model.add(Flatten(input_shape=shape))
    target_model.add(Dense(n_classes, activation='softmax'))

    return target_model


def get_pool_classifier(shape, n_classes=5):
    target_model = Sequential()
    target_model.add(GlobalAveragePooling2D(input_shape=shape))
    target_model.add(Dense(n_classes, activation='softmax'))

    return target_model


def save_filtered_bottleneck_data(base_model, train_generator, nb_train_samples, batch_size, p_values, split):
    predict_size_train = int(math.ceil(nb_train_samples / batch_size))
    bottleneck_features_train = base_model.predict_generator(
        train_generator, predict_size_train)

    # print(bottleneck_features_train.shape)

    np.save(get_bottleneck_name(target_dataset, split, isTafe=False), bottleneck_features_train)

    includeIndices = []
    for f in p_values:
        if p_values[f] > 0.0:
            includeIndices.append(f)

    x = np.take(bottleneck_features_train, includeIndices, axis=3)

    np.save(get_bottleneck_name(target_dataset, split, isTafe=True), x)
    print('Bottleneck feature saved')

    # return original_shape and bottleneck shape
    return bottleneck_features_train.shape[1:], x.shape[1:]


# def construct_reweighted_target(base_model, n_classes=5, p_values=None):
#     freezeModel(base_model)
#     x = base_model.output
#     if p_values is not None and SLICE_MODE == 'online':
#         # x = Lambda(reweight, arguments={'weights': p_values})(x)
#         x = Lambda(discardZeros, arguments={'weights': p_values})(x)
#
#     if target_dataset == 'dog':
#         model = Sequential()
#         model.add(base_model)
#         # model.add(Flatten())
#         model.add(GlobalAveragePooling2D())
#         # model.add(Dropout(0.2))
#         model.add(Dense(n_classes, activation='softmax'))
#     elif target_dataset == 'bird':
#         x = tf.keras.layers.Flatten()(base_model.output)
#         x = tf.keras.layers.Dense(256, activation='tanh')(x)
#         x = tf.keras.layers.Dropout(0.1)(x)
#         x = tf.keras.layers.Dense(200, activation='softmax')(x)
#
#         optimizer = tf.keras.optimizers.Adam()
#
#         model = tf.keras.Model(base_model.input, x)
#         model.compile(
#             loss='sparse_categorical_crossentropy',
#             optimizer=optimizer,
#             metrics=['accuracy']
#         )
#     else:
#
#         x = Flatten()(x)
#         # x = Dense(256, activation='relu')(x)
#         # x = Dropout(0.2)(x)
#         predictions = Dense(n_classes, activation='softmax')(x)
#         model = Model(inputs=base_model.input, outputs=predictions)
#         model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
#     return model

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
        shuffle=True,
        verbose=verbose
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


def discardZeros(x, weights):
    includeIndices = []
    for f in weights:
        if weights[f] > 0.0:
            includeIndices.append(f)

    x = tf.gather(params=x, indices=includeIndices, axis=3)
    # includeIndices = []
    # for f in weights:
    #     if weights[f] > 0.0:
    #         includeIndices.append(x[:,:,:,f])
    # # x = tf.stack(includeIndices, axis=3)
    # x=tf.concat(includeIndices, 1)
    return x


def reweight(x, weights):
    mask = np.ones(x.shape[1:])
    for f in weights:
        mask[:, :, int(f)] = weights[f]
        # mask[int(f)] = weights[f]
    return mask * x
