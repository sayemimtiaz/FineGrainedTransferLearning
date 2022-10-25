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


def cifar100FineClassIndexes(superclasses=None):
    superclass = []
    cifar100classes = getCifar100CoarseClasses()
    for pc in superclasses:
        superclass.append(cifar100classes.index(pc))

    additional_param = {'label_mode': 'coarse'}
    (_, y_train_coarse), (_, _) = getKerasDataset(dataset='cifar100', one_hot=False, gray=False,
                                                  additional_param=additional_param)
    additional_param['label_mode'] = 'fine'
    (_, y_train_fine), (_, _) = getKerasDataset(dataset='cifar100', one_hot=False, gray=False,
                                                additional_param=additional_param)

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
