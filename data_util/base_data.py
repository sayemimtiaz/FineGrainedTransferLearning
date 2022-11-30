import os

import keras
from keras.datasets import mnist, cifar10, cifar100
from keras.utils.np_utils import to_categorical
import numpy as np
import tensorflow as tf

from data_util.util import transformToGrayAndReshapeBoth, asTypeBoth, normalizeBoth, oneEncodeBoth


def getKerasDataset(one_hot=True, dataset='cifar100', gray=False, additional_param=None, shape=(28, 28)):
    if dataset == 'cifar100':
        if additional_param is None:
            raise Exception('What is the label_mode for cifar100')
        (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode=additional_param['label_mode'])
    elif dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    elif dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    else:
        raise Exception(dataset + ' not supported by keras')

    x_train, x_test = asTypeBoth(x_train, x_test)

    if len(x_train.shape) < 4:
        x_train = tf.expand_dims(x_train, -1)
        x_test = tf.expand_dims(x_test, -1)

    if gray:
        x_train, x_test = transformToGrayAndReshapeBoth(x_train, x_test, shape=shape)

    x_train, x_test = normalizeBoth(x_train, x_test)

    num_classes = max(y_train.max() + 1, y_test.max() + 1)
    if one_hot:
        y_train, y_test = oneEncodeBoth(y_train, y_test)

    return x_train, y_train, x_test, y_test, num_classes


def loadFromDir(dir, labels=None, label_index=1, shape=(28, 28),
                mode='grayscale', label_mode='int', batch_size=32, shuffle=True):
    root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    dataPath = os.path.join(root, 'data', dir)
    if labels.endswith('.txt'):
        newLabels = []
        with open(os.path.join(root, 'data', labels)) as f:
            lines = f.readlines()
            for l in lines:
                l = l.split()
                if label_mode == 'int':
                    newLabels.append(int(l[label_index]))
                else:
                    newLabels.append(l[label_index])
        labels = newLabels
        batch_size = len(labels)

    ds = keras.utils.image_dataset_from_directory(
        dataPath, labels=labels, label_mode=label_mode,
        color_mode=mode, image_size=shape, batch_size=batch_size, shuffle=shuffle)

    for x, y in ds.take(1):
        # print(x.shape, y)

        cns=ds.class_names

        # from PIL import Image
        # img = Image.fromarray(x.numpy()[0].astype(np.uint8), 'RGB')
        # img.show()

        return x.numpy(), y.numpy(), cns


# loadFromDir('mnist_m/mnist_m_train/', 'mnist_m/mnist_m_train_labels.txt')
# loadFromDir('tiny-imagenet/train/', labels="inferred", mode='rgb', label_mode='int'
#             , shape=(64, 64), batch_size=100000)
