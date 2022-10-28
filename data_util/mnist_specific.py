from data_util.base_data import loadFromDir, getKerasDataset
from data_util.util import asTypeBoth, normalizeBoth, oneEncodeBoth


def getMnistM(gray=True, shape=(28, 28), one_hot=True):
    if gray:
        gray = 'grayscale'
    else:
        gray = 'rgb'

    x_train, y_train = loadFromDir('mnist_m/mnist_m_train/', 'mnist_m/mnist_m_train_labels.txt',
                                   shape=shape, mode=gray)
    x_test, y_test = loadFromDir('mnist_m/mnist_m_test/', 'mnist_m/mnist_m_test_labels.txt',
                                 shape=shape, mode=gray)
    x_train, x_test = asTypeBoth(x_train, x_test)

    x_train, x_test = normalizeBoth(x_train, x_test)

    if one_hot:
        y_train, y_test = oneEncodeBoth(y_train, y_test)

    return x_train, y_train, x_test, y_test, y_train.shape[1]


def getMnist(gray=True, shape=(28, 28), one_hot=True):
    return getKerasDataset(one_hot=one_hot, dataset='mnist', gray=gray, shape=shape)
