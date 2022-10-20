import random

from keras.datasets import mnist, cifar10, cifar100
from keras.utils.np_utils import to_categorical
import numpy as np
import tensorflow as tf

cifar100classes = ['aquatic mammals', 'fish', 'flowers', 'food containers', 'fruit and vegetables',
                   'household electrical devices', 'household furniture', 'insects', 'large carnivores',
                   'large man-made outdoor things', 'large natural outdoor scenes', 'large omnivores and herbivores'
    , 'medium-sized mammals', 'non-insect invertebrates', 'people', 'reptiles',
                   'small mammals', 'trees', 'vehicles 1', 'vehicles 2']
cifar10classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def getSuperClassData(one_hot=True, insert_noise=False, dataset='cifar100', gray=False):
    if dataset == 'cifar100':
        (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='coarse')
    else:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    if gray:
        x_train = tf.image.rgb_to_grayscale(x_train)
        x_test = tf.image.rgb_to_grayscale(x_test)
        x_train = tf.image.resize(x_train, [28, 28])
        x_test = tf.image.resize(x_test, [28, 28])
        x_train = x_train.numpy()
        x_test = x_test.numpy()

    x_train /= 255
    x_test /= 255

    if insert_noise:
        for j in range(21, 30):
            for i in range(500):
                x_train = np.append(x_train, np.random.random((1, 32, 32, 3)), axis=0)
                y_train = np.append(y_train, np.array(j).reshape(1, 1), axis=0)
        train_index = list(range(len(x_train)))

        train_index = np.random.choice(train_index, len(x_train), replace=False)
        x_train = x_train[train_index]
        y_train = y_train[train_index]

    if one_hot:
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
    return x_train, y_train, x_test, y_test, y_train.shape[1]


def getCifar10BinaryData(one_hot=True):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    man_made = [0, 1, 8, 9]
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # x_train = tf.image.rgb_to_grayscale(x_train)
    # x_test = tf.image.rgb_to_grayscale(x_test)
    # x_train = tf.image.resize(x_train, [28, 28])
    # x_test = tf.image.resize(x_test, [28, 28])
    # x_train = x_train.numpy()
    # x_test = x_test.numpy()

    x_train /= 255
    x_test /= 255

    for i in range(len(y_train)):
        if y_train[i][0] in man_made:
            y_train[i] = 0
        else:
            y_train[i] = 1

    for i in range(len(y_test)):
        if y_test[i][0] in man_made:
            y_test[i] = 0
        else:
            y_test[i] = 1

    if one_hot:
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
    return x_train, y_train, x_test, y_test, y_train.shape[1]


def getCifar10MnistMixed(one_hot=True, takeFromCifar=None,takeFromMnist=None):
    x_train, y_train_,x_test, y_test_,_ = getSuperClassData(one_hot=False, dataset='cifar10', gray=True)

    x_train_indexs = []
    x_test_indexes = []
    y_train = []
    y_test = []
    for i in range(len(y_train_)):
        y_train.append(y_train_[i][0])
        if y_train_[i][0] in takeFromCifar:
            x_train_indexs.append(i)

    for i in range(len(y_test_)):
        y_test.append(y_test_[i][0])
        if y_test_[i][0] in takeFromCifar:
            x_test_indexes.append(i)

    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    x_train_cifar, y_train_cifar = x_train[x_train_indexs], y_train[x_train_indexs]
    x_test_cifar, y_test_cifar = x_test[x_test_indexes], y_test[x_test_indexes]

    x_train, y_train, x_test, y_test, _ = getMnistData(one_hot=False)

    x_train_indexs = []
    x_test_indexes = []
    for i in range(len(y_train)):
        if y_train[i] in takeFromMnist:
            x_train_indexs.append(i)

    for i in range(len(y_test)):
        if y_test[i] in takeFromMnist:
            x_test_indexes.append(i)

    x_train_mnist, y_train_mnist = x_train[x_train_indexs], y_train[x_train_indexs]
    x_test_mnist, y_test_mnist = x_test[x_test_indexes], y_test[x_test_indexes]

    x_train = np.concatenate((x_train_mnist, x_train_cifar))
    y_train = np.concatenate((y_train_mnist, y_train_cifar))
    x_test = np.concatenate((x_test_mnist, x_test_cifar))
    y_test = np.concatenate((y_test_mnist, y_test_cifar))

    x_train_indexs = range(len(x_train))
    x_train_indexs = np.random.choice(x_train_indexs, len(x_train), replace=False)
    x_train = x_train[x_train_indexs]
    y_train = y_train[x_train_indexs]

    if one_hot:
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
    return x_train, y_train, x_test, y_test, len(takeFromMnist) + len(takeFromCifar)


def getFineGrainedClass(superclass='trees', num_sample=100, seed=None, one_hot=True, gray=False):
    superclass = cifar100classes.index(superclass)

    (_, y_train_coarse), (_, y_test_coarse) = cifar100.load_data(label_mode='coarse')
    (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

    n_x_train = []
    n_y_train = []
    new_y_label = []

    eligible_i = {}
    for i in range(len(y_train_coarse)):
        if y_train_coarse[i][0] == superclass:
            if y_train[i][0] not in eligible_i:
                eligible_i[y_train[i][0]] = []
            eligible_i[y_train[i][0]].append(i)

    random.seed(seed)
    for k in eligible_i.keys():
        if num_sample == -1:
            random_i = random.sample(range(0, len(eligible_i[k])), len(eligible_i[k]))
        else:
            random_i = random.sample(range(0, len(eligible_i[k])), num_sample)

        for ni in random_i:
            i = eligible_i[k][ni]
            n_x_train.append(x_train[i])
            if y_train[i][0] not in new_y_label:
                new_y_label.append(y_train[i][0])
            n_y_train.append(new_y_label.index(y_train[i][0]))

    n_x_train = np.asarray(n_x_train)
    n_y_train = np.asarray(n_y_train)

    n_x_test = []
    n_y_test = []
    for i in range(len(y_test_coarse)):
        if y_test_coarse[i][0] == superclass:
            n_x_test.append(x_test[i])
            n_y_test.append(new_y_label.index(y_test[i][0]))

    n_x_test = np.asarray(n_x_test)
    n_y_test = np.asarray(n_y_test)

    n_x_train = n_x_train.astype('float32')
    n_x_test = n_x_test.astype('float32')
    if gray:
        n_x_train = tf.image.rgb_to_grayscale(n_x_train)
        n_x_test = tf.image.rgb_to_grayscale(n_x_test)
        n_x_train = tf.image.resize(n_x_train, [28, 28])
        n_x_test = tf.image.resize(n_x_test, [28, 28])
        n_x_train = n_x_train.numpy()
        n_x_test = n_x_test.numpy()
    n_x_train /= 255
    n_x_test /= 255
    num_classes = n_y_test.max() + 1
    if one_hot:
        n_y_train = to_categorical(n_y_train)
        n_y_test = to_categorical(n_y_test)
    return n_x_train, n_y_train, n_x_test, n_y_test, num_classes


def sampleForDecomposition(sample=-1, positive_classes=None, dataset='cifar100', gray=False):
    positive_class_indexes = []
    superclasses = cifar10classes
    if dataset == 'cifar100':
        superclasses = cifar100classes
    for pc in positive_classes:
        positive_class_indexes.append(superclasses.index(pc))

    x_train, y_train, _, _, _ = getSuperClassData(one_hot=False, dataset=dataset, gray=gray)
    pos_index = []
    neg_index = []
    for i, _y in enumerate(y_train):
        if _y[0] in positive_class_indexes:
            pos_index.append(i)
        else:
            neg_index.append(i)

    if sample != -1:
        pos_index = np.random.choice(pos_index, sample, replace=False)
        neg_index = np.random.choice(neg_index, sample, replace=False)
    pos_x, neg_x = x_train[pos_index], x_train[neg_index]

    return pos_x, neg_x


def sample(sample=-1, data_x=None, data_y=None, num_classes=None):
    flag = {}
    all_chosen_index = []
    sample = int(sample / num_classes)
    for y in data_y:
        if type(y) == np.ndarray:
            y = y[0]
        if y in flag:
            continue
        flag[y] = 1

        class_all_index = []
        for _i, _y in enumerate(data_y):
            if type(_y) == np.ndarray:
                _y = _y[0]
            if _y == y:
                class_all_index.append(_i)

        if sample > len(class_all_index):
            chosen_index = np.random.choice(class_all_index, len(class_all_index), replace=False)
        else:
            chosen_index = np.random.choice(class_all_index, sample, replace=False)
        all_chosen_index.extend(chosen_index)

    # all_index = list(range(len(data_x)))

    np.random.shuffle(all_chosen_index)
    data_x, data_y = data_x[all_chosen_index], data_y[all_chosen_index]

    return data_x, data_y


# def getCifar10FineGrainedClass(superclass='trees'):
#     superclass = superclasses.index(superclass)
#
#     (_, y_train_coarse), (_, y_test_coarse) = cifar100.load_data(label_mode='coarse')
#     (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
#
#     n_x_train = []
#     n_y_train = []
#     new_y_label = []
#     cnt = 0
#     for i in range(len(y_train_coarse)):
#         if y_train_coarse[i][0] == superclass:
#             # if cnt>200:
#             #     break
#             n_x_train.append(x_train[i])
#             if y_train[i][0] not in new_y_label:
#                 new_y_label.append(y_train[i][0])
#             n_y_train.append(new_y_label.index(y_train[i][0]))
#             cnt += 1
#
#     n_x_train = np.asarray(n_x_train)
#     n_y_train = np.asarray(n_y_train)
#
#     n_x_test = []
#     n_y_test = []
#     for i in range(len(y_test_coarse)):
#         if y_test_coarse[i][0] == superclass:
#             n_x_test.append(x_test[i])
#             n_y_test.append(new_y_label.index(y_test[i][0]))
#
#     n_x_test = np.asarray(n_x_test)
#     n_y_test = np.asarray(n_y_test)
#
#     n_x_train = n_x_train.astype('float32')
#     n_x_test = n_x_test.astype('float32')
#     n_x_train /= 255
#     n_x_test /= 255
#     n_y_train = to_categorical(n_y_train)
#     n_y_test = to_categorical(n_y_test)
#     return n_x_train, n_y_train, n_x_test, n_y_test, n_y_test.shape[1]

# getFineGrainedClass()
# sampleForDecomposition(10)


def getMnistData(one_hot=True):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = tf.expand_dims(x_train, -1)
    x_test = tf.expand_dims(x_test, -1)
    x_train = x_train.numpy()
    x_test = x_test.numpy()

    x_train /= 255
    x_test /= 255

    if one_hot:
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
    return x_train, y_train, x_test, y_test, 10
