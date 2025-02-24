import tensorflow as tf
import numpy as np
from keras.utils import to_categorical


def reshape(data, shape=None):
    if shape is None:
        shape = [28, 28]
    data = tf.image.resize(data, list(shape))
    data = data.numpy()
    return data

def transformToGrayAndReshape(data, shape=None):
    if shape is None:
        shape = [28, 28]
    if len(data.shape) == 4 and data.shape[3] > 1:
        data = tf.image.rgb_to_grayscale(data)
    data = tf.image.resize(data, list(shape))
    data = data.numpy()
    return data


def transformToGrayAndReshapeBoth(x_train, x_test, shape=None):
    return transformToGrayAndReshape(x_train, shape=shape), \
           transformToGrayAndReshape(x_test, shape=shape)


# default shape is 28*28
def normalize(data, by=255):
    data /= 255
    return data


def normalizeBoth(x_train, x_test, by=255):
    return normalize(x_train, by), normalize(x_test, by)


def oneEncode(data):
    data = to_categorical(data)
    return data


def oneEncodeBoth(y_train, y_test):
    return oneEncode(y_train), oneEncode(y_test)


def asType(data, as_type='float32'):
    data = data.astype(as_type)
    return data


def asTypeBoth(x_train, x_test, as_type='float32'):
    return asType(x_train, as_type=as_type), asType(x_test, as_type=as_type)


def makeScalar(data):
    new_data = []
    for i in range(len(data)):
        if type(data[i]) == list or type(data[i]) is np.ndarray:
            if len(data[i]) > 1:
                raise Exception('Data should not be hot encoded')

            new_data.append(data[i][0])
        else:
            new_data.append(data[i])

    return np.asarray(new_data)


def getIndexesMatchingSubset(Y, match):
    indexes = []
    for i in range(len(Y)):
        if Y[i] in match:
            indexes.append(i)
    return indexes


def shuffle(x, y):
    x_train_indexs = range(len(x))
    x_train_indexs = np.random.choice(x_train_indexs, len(x), replace=False)
    x = x[x_train_indexs]
    y = y[x_train_indexs]

    return x, y


def relabel(yA, yB):
    mp = {}
    c = 0
    nYa = []
    nYb = []
    for i in range(len(yA)):
        if yA[i] in mp:
            nYa.append(mp[yA[i]])
        else:
            mp[yA[i]] = c
            nYa.append(c)
            c += 1

    for i in range(len(yB)):
        if yB[i] in mp:
            nYb.append(mp[yB[i]])
        else:
            mp[yB[i]] = c
            nYb.append(c)
            c += 1
    nYa = np.asarray(nYa)
    nYb = np.asarray(nYb)

    return nYa, nYb, mp


def concatenateTwoY(yA, yB):
    mp = {'A': {}, 'B': {}}
    c = 0
    nYa = []
    nYb = []
    for i in range(len(yA)):
        if yA[i] in mp['A']:
            nYa.append(mp['A'][yA[i]])
        else:
            mp['A'][yA[i]] = c
            nYa.append(c)
            c += 1

    for i in range(len(yB)):
        if yB[i] in mp['B']:
            nYb.append(mp['B'][yB[i]])
        else:
            mp['B'][yB[i]] = c
            nYb.append(c)
            c += 1
    nYa = np.asarray(nYa)
    nYb = np.asarray(nYb)

    return np.concatenate((nYa, nYb)), mp
