from data_util.util import makeScalar, getIndexesMatchingSubset, shuffle
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


def sample(data, num_sample=-1, num_classes=None, balance=True, sample_only_classes=None):
    data_x, data_y = data
    data_y = makeScalar(data_y)
    flag = {}
    all_chosen_index = []
    if sample_only_classes is not None:
        num_classes = len(sample_only_classes)
    if balance:
        num_sample = int(num_sample / num_classes)
    for y in data_y:
        if y in flag:
            continue
        if sample_only_classes is not None and y not in sample_only_classes:
            continue
        flag[y] = 1

        class_all_index = getIndexesMatchingSubset(data_y, [y])

        if num_sample > len(class_all_index):
            chosen_index = np.random.choice(class_all_index, len(class_all_index), replace=False)
        else:
            chosen_index = np.random.choice(class_all_index, num_sample, replace=False)
        all_chosen_index.extend(chosen_index)

    np.random.shuffle(all_chosen_index)
    data_x, data_y = data_x[all_chosen_index], data_y[all_chosen_index]

    return data_x, data_y
