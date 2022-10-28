from data_util.base_data import getKerasDataset
from data_util.util import makeScalar


def getCifar100CoarseClasses():
    cifar100classes = ['aquatic mammals', 'fish', 'flowers', 'food containers', 'fruit and vegetables',
                       'household electrical devices', 'household furniture', 'insects', 'large carnivores',
                       'large man-made outdoor things', 'large natural outdoor scenes', 'large omnivores and herbivores'
        , 'medium-sized mammals', 'non-insect invertebrates', 'people', 'reptiles',
                       'small mammals', 'trees', 'vehicles 1', 'vehicles 2']
    return cifar100classes


def getCifar10Classes():
    cifar10classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    return cifar10classes


def getCifar10(gray=True, shape=(28, 28), one_hot=True):
    return getKerasDataset(one_hot=one_hot, dataset='cifar10', gray=gray, shape=shape)


def getCifar100Fine(gray=True, shape=(28, 28), one_hot=True):
    additional_param = {'label_mode': 'fine'}
    return getKerasDataset(dataset='cifar100', one_hot=one_hot, gray=gray,
                           additional_param=additional_param, shape=shape)


def getCifar100Coarse(gray=True, shape=(28, 28), one_hot=True):
    additional_param = {'label_mode': 'coarse'}
    return getKerasDataset(dataset='cifar100', one_hot=one_hot, gray=gray,
                           additional_param=additional_param, shape=shape)


def cifar100FineClassIndexes(superclasses=None):
    superclass = []
    cifar100classes = getCifar100CoarseClasses()
    for pc in superclasses:
        superclass.append(cifar100classes.index(pc))

    (_, y_train_coarse), (_, _) = getCifar100Coarse(one_hot=False, gray=False)
    (_, y_train_fine), (_, _) = getCifar100Fine(one_hot=False, gray=False)

    y_train_coarse = makeScalar(y_train_coarse)
    y_train_fine = makeScalar(y_train_fine)

    fine_classes = []
    for i in range(len(y_train_coarse)):
        if y_train_coarse[i] in superclass:
            fine_classes.append(y_train_fine[i])

    return fine_classes


def cifar100CoarseClassIndexes(superclasses=None):
    superclass = []
    cifar100classes = getCifar100CoarseClasses()
    for pc in superclasses:
        superclass.append(cifar100classes.index(pc))

    return superclass


def cifar10ClassIndexes(superclasses=None):
    superclass = []
    cifar100classes = getCifar10Classes()
    for pc in superclasses:
        superclass.append(cifar100classes.index(pc))

    return superclass
