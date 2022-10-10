from keras.datasets import fashion_mnist, cifar10, cifar100
from keras.utils.np_utils import to_categorical
import numpy as np

cifar100classes = ['aquatic mammals', 'fish', 'flowers', 'food containers', 'fruit and vegetables',
                'household electrical devices', 'household furniture', 'insects', 'large carnivores',
                'large man-made outdoor things', 'large natural outdoor scenes', 'large omnivores and herbivores'
    , 'medium-sized mammals', 'non-insect invertebrates', 'people', 'reptiles',
                'small mammals', 'trees', 'vehicles 1', 'vehicles 2']
cifar10classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def getSuperClassData(one_hot=True, insert_noise=False, dataset='cifar100'):
    if dataset == 'cifar100':
        (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='coarse')
    else:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
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


def getFineGrainedClass(superclass='trees'):
    superclass = cifar100classes.index(superclass)

    (_, y_train_coarse), (_, y_test_coarse) = cifar100.load_data(label_mode='coarse')
    (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

    n_x_train = []
    n_y_train = []
    new_y_label = []
    cnt = 0
    for i in range(len(y_train_coarse)):
        if y_train_coarse[i][0] == superclass:
            # if cnt>200:
            #     break
            n_x_train.append(x_train[i])
            if y_train[i][0] not in new_y_label:
                new_y_label.append(y_train[i][0])
            n_y_train.append(new_y_label.index(y_train[i][0]))
            cnt += 1

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
    n_x_train /= 255
    n_x_test /= 255
    n_y_train = to_categorical(n_y_train)
    n_y_test = to_categorical(n_y_test)
    return n_x_train, n_y_train, n_x_test, n_y_test, n_y_test.shape[1]


def sampleForDecomposition(sample=-1, positive_classes=None, dataset='cifar100'):
    positive_class_indexes = []
    superclasses = cifar10classes
    if dataset=='cifar100':
        superclasses=cifar100classes
    for pc in positive_classes:
        positive_class_indexes.append(superclasses.index(pc))

    x_train, y_train, _, _, _ = getSuperClassData(one_hot=False, dataset=dataset)
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
