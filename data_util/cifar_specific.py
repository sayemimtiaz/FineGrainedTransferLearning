import random

from data_util.base_data import getKerasDataset
from data_util.sample_util import sample, sampleTrainTest
from data_util.util import makeScalar, oneEncodeBoth


class Cifar10:
    data = None
    gray = True
    shape = (28, 28)

    def __init__(self, one_hot=False, shape=(28, 28), gray=True, load=True):
        self.gray = gray
        self.shape = shape
        if load:
            self.data = self.getCifar10(one_hot=one_hot, shape=shape, gray=gray)

    def getCifar10(self, gray=True, shape=(28, 28), one_hot=True):
        return getKerasDataset(one_hot=one_hot, dataset='cifar10', gray=gray, shape=shape)

    def getClasses(self):
        return list(range(200))

    def getClassName(self, idx):
        return getCifar10Classes()[idx]

    def getBirdIndex(self):
        return getCifar10Classes().index('bird')

    def sample(self, sample_only_classes=None, train=True, num_sample=-1, seed=None, one_hot=False):
        return sampleTrainTest(self.data, num_sample=num_sample, sample_only_classes=sample_only_classes,
                               seed=seed,
                               one_hot=one_hot, train=train)


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

    _, y_train_coarse, _, _, _ = getCifar100Coarse(one_hot=False, gray=False)
    _, y_train_fine, _, _, _ = getCifar100Fine(one_hot=False, gray=False)

    y_train_coarse = makeScalar(y_train_coarse)
    y_train_fine = makeScalar(y_train_fine)

    fine_classes = set()
    for i in range(len(y_train_coarse)):
        if y_train_coarse[i] in superclass:
            fine_classes.add(y_train_fine[i])

    return list(fine_classes)


def cifar100CoarseClassIndexes(superclasses=None):
    if superclasses is None or len(superclasses) == 0:
        return list(range(len(getCifar100CoarseClasses())))
    superclass = []
    cifar100classes = getCifar100CoarseClasses()
    for pc in superclasses:
        superclass.append(cifar100classes.index(pc))

    return superclass


def cifar10ClassIndexes(superclasses=None):
    if superclasses is None or len(superclasses) == 0:
        return list(range(len(getCifar10Classes())))
    superclass = []
    cifar100classes = getCifar10Classes()
    for pc in superclasses:
        superclass.append(cifar100classes.index(pc))

    return superclass


def sampleCifar10(superclasses=None, train=True, num_sample=-1, gray=True, seed=None, one_hot=False):
    sc = cifar10ClassIndexes(superclasses)
    data = getCifar10(one_hot=False, gray=gray)

    return sampleTrainTest(data, num_sample=num_sample, sample_only_classes=sc, seed=seed,
                           one_hot=one_hot, train=train)


def sampleCifar100Fine(superclasses=None, train=True, num_sample=-1, gray=True,
                       seed=None, one_hot=False, max_class=None, shape=(28,28)):
    sc = cifar100FineClassIndexes(superclasses)

    if max_class is not None:
        sc = random.sample(sc, max_class)

    data = getCifar100Fine(one_hot=False, gray=gray, shape=shape)

    return sampleTrainTest(data, num_sample=num_sample, sample_only_classes=sc, seed=seed,
                           one_hot=one_hot, train=train)
