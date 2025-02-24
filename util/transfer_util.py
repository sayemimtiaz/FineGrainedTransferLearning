import math
import os
import random
import time

import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2
from keras import regularizers

from constants import target_dataset

from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Lambda, GlobalAveragePooling2D, \
    RandomFourierFeatures, Activation

from keras import initializers, Sequential, Model, optimizers
import numpy as np
from keras import backend as K
import tensorflow as tf
from util.common import freezeModel
from util.ordinary import dump_as_pickle, get_bottleneck_name, get_transfer_model_name, load_pickle_file

import warnings

warnings.filterwarnings("ignore")


def compile_svm_classifier(model, lr=0.0001):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.hinge,
        metrics=[keras.metrics.CategoricalAccuracy(name="acc")],
    )
    return model


def compile_dense_classifier(model, lr=0.0001):
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def get_svm_classifier(shape, n_classes=5):
    model = Sequential()
    model.add(GlobalAveragePooling2D(input_shape=shape))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(n_classes, kernel_regularizer=l2(0.01), activation='softmax'))
    model.compile(optimizer='adam', loss='squared_hinge', metrics=['accuracy'])
    return model


def get_dense_classifier(shape, n_classes=5):
    target_model = Sequential()
    target_model.add(Flatten(input_shape=shape))
    # target_model.add(Dropout(0.5))
    # target_model.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.l1(0.01)))
    # target_model.add(Dense(1024, activation='relu'))
    # target_model.add(Dropout(0.5))
    # target_model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l1(0.01)))
    # target_model.add(Dense(512, activation='relu'))
    # target_model.add(Dropout(0.5))
    target_model.add(Dense(n_classes, activation='softmax'))

    target_model.compile(optimizer='rmsprop',
                         loss='categorical_crossentropy', metrics=['accuracy'])

    return target_model


def get_pool_classifier(shape, n_classes=5):
    target_model = Sequential()
    target_model.add(GlobalAveragePooling2D(input_shape=shape))
    # target_model.add(Dense(512, activation='relu'))
    # target_model.add(Dense(256, activation='relu'))
    # target_model.add(Dense(128, activation='relu'))
    target_model.add(Dense(n_classes, activation='softmax'))
    target_model.compile(optimizer='rmsprop',
                         loss='categorical_crossentropy', metrics=['accuracy'])
    return target_model


def save_bottleneck_data(base_model, train_generator, nb_train_samples, batch_size, split, save=True, target_ds=None,
                         model_name=None):
    if target_ds is None:
        target_ds = target_dataset
    fileName = get_bottleneck_name(target_ds, split, isTafe=False, isLabel=False, model_name=model_name)
    if os.path.exists(fileName):
        bottleneck_features_train = load_pickle_file(fileName)
        return bottleneck_features_train

    predict_size_train = int(math.ceil(nb_train_samples / batch_size))
    bottleneck_features_train = base_model.predict_generator(
        train_generator, predict_size_train)

    if save:
        np.save(fileName, bottleneck_features_train)

    return bottleneck_features_train


def save_filtered_bottleneck_data(bottleneck_features, p_values, split, alpha, target_ds=None, model_name=None):
    if target_ds is None:
        target_ds = target_dataset

    includeIndices = []
    for f in p_values:
        if p_values[f] > 0.0:
            includeIndices.append(f)

    x = np.take(bottleneck_features, includeIndices, axis=3)

    np.save(get_bottleneck_name(target_ds, model_name, split, isTafe=True, isLabel=False, alpha=alpha), x)
    print('Bottleneck feature saved')

    # return original_shape and bottleneck shape
    return bottleneck_features.shape[1:], x.shape[1:]


def delete_bottleneck_data(bottleneck_features, p_values, split, deleteRate, target_ds=None):
    if target_ds is None:
        target_ds = target_dataset

    p_values = dict(sorted(p_values.items(), key=lambda item: item[1]))
    delUntil = int(deleteRate * bottleneck_features.shape[3])
    includeIndices = []
    i = 0
    for f in p_values:
        if i > delUntil:
            includeIndices.append(f)
        i += 1

    includeIndices = sorted(includeIndices)
    x = np.take(bottleneck_features, includeIndices, axis=3)

    np.save(get_bottleneck_name(target_ds, split, isTafe=True, isLabel=False, alpha=deleteRate), x)
    print('Bottleneck feature saved')

    return bottleneck_features.shape[1:], x.shape[1:]


def save_random_bottleneck_data(bottleneck_features, p_values, split, alpha, target_ds=None):
    if target_ds is None:
        target_ds = target_dataset

    num_sample = 0
    for f in p_values:
        if p_values[f] > 0.0:
            num_sample += 1

    includeIndices = np.random.choice(range(bottleneck_features.shape[3]), num_sample, replace=False)
    includeIndices = sorted(includeIndices)
    x = np.take(bottleneck_features, includeIndices, axis=3)

    np.save(get_bottleneck_name(target_ds, split, isTafe=True, isLabel=False, alpha=alpha), x)
    print('Random bottleneck feature saved')

    # return original_shape and bottleneck shape
    return bottleneck_features.shape[1:], x.shape[1:]


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


# def trainDog(model, train_ds, val_ds, nb_train_samples, nb_valid_samples, epoch=30, batch_size=128, verbose=0):
#     start = time.time()
#     earlystop = EarlyStopping(
#         monitor='val_loss',
#         min_delta=0.001,
#         patience=3,
#         verbose=1,
#         mode='auto'
#     )
#     reduceLR = ReduceLROnPlateau(
#         monitor='val_loss',
#         factor=0.1,
#         patience=3,
#         verbose=1,
#         mode='auto'
#     )
#     callbacks = [earlystop, reduceLR]
#
#     model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy',
#                   metrics=['accuracy'])
#
#     history = model.fit(
#         train_ds,
#         epochs=epoch,
#         steps_per_epoch=nb_train_samples // batch_size,
#         validation_data=val_ds,
#         validation_steps=nb_valid_samples // batch_size,
#         callbacks=callbacks,
#         shuffle=True,
#         verbose=verbose
#     )
#     end = time.time()
#
#     return history.history['val_accuracy'][-1], end - start


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
