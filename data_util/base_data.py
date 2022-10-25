from keras.datasets import mnist, cifar10, cifar100
from keras.utils.np_utils import to_categorical
import numpy as np

from data_util.util import transformToGrayAndReshape, asType, normalize, oneEncode


def getKerasDataset(one_hot=True, dataset='cifar100', gray=False, additional_param=None):
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

    x_train, x_test = asType(x_train, x_test)

    if gray:
        x_train, x_test = transformToGrayAndReshape(x_train, x_test)

    x_train, x_test = normalize(x_train, x_test)

    if one_hot:
        y_train, y_test = oneEncode(y_train, y_test)

    return x_train, y_train, x_test, y_test, y_train.shape[1]

