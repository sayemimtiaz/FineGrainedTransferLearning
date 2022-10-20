import math
import random

from keras.datasets import mnist, cifar10, cifar100
from keras.utils.np_utils import to_categorical
import numpy as np
import tensorflow as tf


def getMnistData(one_hot=True):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # x_train = tf.expand_dims(x_train, -1)
    # x_test = tf.expand_dims(x_test, -1)
    # x_train = x_train.numpy()
    # x_test = x_test.numpy()

    x_train /= 255
    x_test /= 255

    if one_hot:
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
    return x_train, y_train, x_test, y_test, 10


def dist_from_mean_image(target, data, h=28, w=28):
    # r = np.zeros((h, w))
    rs = 0
    for r in range(h):
        for c in range(w):
            # r[:, c] = data[:, c].mean()

            rs += ((target[r, c] - data[:, r, c].mean()) * (target[r, c] - data[:, r, c].mean()))

    return math.sqrt(rs)


def mnist_class_data():
    x_train, y_train, x_test, y_test, nb_classes = getMnistData(one_hot=False)
    Y_test = to_categorical(y_test, nb_classes)

    heldOutClass = 0
    while heldOutClass < nb_classes:
        X_train = []
        Y_train = []
        for i, _y in enumerate(y_train):
            if _y == heldOutClass:
                X_train.append(i)

        X_train = x_train[X_train]
        for x in X_train:
            Y_train.append(dist_from_mean_image(x, X_train))

        yield X_train, np.asarray(Y_train), x_test, Y_test, 1, heldOutClass

        heldOutClass += 1
