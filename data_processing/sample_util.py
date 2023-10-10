import math

from data_processing.data_util import makeScalar, getIndexesMatchingSubset, shuffle, oneEncodeBoth, relabel
import numpy as np


def selectDataMatchingSubset(data, subset):
    x_train, y_train, x_test, y_test, num_class_a = data
    y_train = makeScalar(y_train)
    y_test = makeScalar(y_test)

    x_train_indexs = getIndexesMatchingSubset(y_train, subset)
    x_test_indexes = getIndexesMatchingSubset(y_test, subset)

    x_train, y_train = x_train[x_train_indexs], y_train[x_train_indexs]
    x_test, y_test = x_test[x_test_indexes], y_test[x_test_indexes]

    x_train, y_train = shuffle(x_train, y_train)

    return x_train, y_train, x_test, y_test


def sample(data, num_sample=-1, num_classes=None, balance=True, sample_only_classes=None, seed=None):
    data_x, data_y = data
    data_y = makeScalar(data_y)
    flag = {}
    all_chosen_index = []

    if seed is not None:
        np.random.seed(seed)

    if sample_only_classes is not None:
        num_classes = len(sample_only_classes)
    if balance and num_sample > 0:
        num_sample = int(math.ceil(num_sample / num_classes))
    for y in data_y:
        if y in flag:
            continue
        if sample_only_classes is not None and y not in sample_only_classes:
            continue
        flag[y] = 1

        class_all_index = getIndexesMatchingSubset(data_y, [y])

        if num_sample == -1 or num_sample > len(class_all_index):
            chosen_index = np.random.choice(class_all_index, len(class_all_index), replace=False)
        else:
            chosen_index = np.random.choice(class_all_index, num_sample, replace=False)
        all_chosen_index.extend(chosen_index)

    np.random.shuffle(all_chosen_index)
    data_x, data_y = data_x[all_chosen_index], data_y[all_chosen_index]

    return data_x, data_y


def sampleTrainTest(data, train=True, num_sample=-1, sample_only_classes=None, seed=None, one_hot=False):
    x_train, y_train, x_test, y_test, _ = data
    if not train:
        data = (x_test, y_test)
        x_test, y_test = sample(data, num_sample=num_sample,
                                sample_only_classes=sample_only_classes, seed=seed)
        data = (x_train, y_train)
        x_train, y_train = sample(data, num_sample=-1,
                                  sample_only_classes=sample_only_classes, seed=seed)
    else:
        data = (x_train, y_train)
        x_train, y_train = sample(data, num_sample=num_sample,
                                  sample_only_classes=sample_only_classes, seed=seed)
        data = (x_test, y_test)
        x_test, y_test = sample(data, num_sample=-1,
                                sample_only_classes=sample_only_classes, seed=seed)

    y_train, y_test, _ = relabel(y_train, y_test)
    num_classes = max(y_train.max() + 1, y_test.max() + 1)
    if one_hot:
        y_train, y_test = oneEncodeBoth(y_train, y_test)

    return x_train, y_train, x_test, y_test, num_classes


def sample_for_training(x, y, rate):
    n = int(len(y) * rate)
    random_indices = np.random.permutation(len(y))[:n]

    return x[random_indices], y[random_indices]
