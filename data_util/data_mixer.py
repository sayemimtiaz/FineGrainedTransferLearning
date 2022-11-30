from data_util.cifar_specific import getCifar10
from data_util.mnist_specific import getMnist
from data_util.util import makeScalar, getIndexesMatchingSubset, shuffle, oneEncodeBoth, concatenateTwoY
import numpy as np


def mixTwoDataSet(dataA, dataB, one_hot=True, takeFromA=None, takeFromB=None):
    x_train, y_train, x_test, y_test, num_class_a = dataA
    y_train = makeScalar(y_train)
    y_test = makeScalar(y_test)

    x_train_indexs = getIndexesMatchingSubset(y_train, takeFromA)
    x_test_indexes = getIndexesMatchingSubset(y_test, takeFromA)

    x_train_a, y_train_a = x_train[x_train_indexs], y_train[x_train_indexs]
    x_test_a, y_test_a = x_test[x_test_indexes], y_test[x_test_indexes]

    x_train, y_train, x_test, y_test, num_class_b = dataB
    y_train = makeScalar(y_train)
    y_test = makeScalar(y_test)

    x_train_indexs = getIndexesMatchingSubset(y_train, takeFromB)
    x_test_indexes = getIndexesMatchingSubset(y_test, takeFromB)

    x_train_b, y_train_b = x_train[x_train_indexs], y_train[x_train_indexs]
    x_test_b, y_test_b = x_test[x_test_indexes], y_test[x_test_indexes]

    x_train = np.concatenate((x_train_a, x_train_b))
    class_mapper={}
    # y_train, class_mapper = concatenateTwoY(y_train_a, y_train_b) # need to retrieve this mapper when evaluating
    y_train = np.concatenate((y_train_a, y_train_b)) # temporary measure
    x_test = np.concatenate((x_test_a, x_test_b))
    # y_test, _ = concatenateTwoY(y_test_a, y_test_b)
    y_test = np.concatenate((y_test_a, y_test_b))  # temporary measure

    x_train, y_train = shuffle(x_train, y_train)

    if one_hot:
        y_train, y_test = oneEncodeBoth(y_train, y_test)

    return x_train, y_train, x_test, y_test, len(takeFromB) + len(takeFromA), class_mapper


def mixMnistCifar10(takeFromCifar=None, takeFromMnist=None, gray=True, one_hot=True):
    dataMnist = getMnist(one_hot=False, gray=gray)
    dataCifar = getCifar10(one_hot=False, gray=gray)
    return mixTwoDataSet(dataMnist, dataCifar,
                         takeFromA=takeFromMnist, takeFromB=takeFromCifar, one_hot=one_hot)
