from keras.datasets import fashion_mnist, cifar10,cifar100
from keras.preprocessing import utils


def getdata(normalize=True):
    # the data, split between train and test sets
    # (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    if normalize:
        x_train /= 255
        x_test /= 255
    # x_train=x_train.reshape(x_train.shape[0],28,28,1)
    # x_test = x_test.reshape(x_test.shape[0],28,28, 1)
    y_train = utils.to_categorical(y_train)
    y_test = utils.to_categorical(y_test)
    return x_train, y_train, x_test, y_test,y_test.shape[1]


# getdata()