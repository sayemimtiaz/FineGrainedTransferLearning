import os

import keras
from keras.datasets import mnist, cifar10, cifar100
from keras.utils.np_utils import to_categorical
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from glob import glob

from data_processing.data_util import transformToGrayAndReshapeBoth, asTypeBoth, normalizeBoth, oneEncodeBoth, reshape
from util.common import cropImg, displayImg


def getKerasDataset(one_hot=True, dataset='cifar100', gray=False, additional_param=None, shape=(28, 28)):
    if dataset == 'cifar100':
        if additional_param is None:
            raise Exception('What is the label_mode for cifar100')
        (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode=additional_param['label_mode'])
    elif dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    elif dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    else:
        raise Exception(dataset + ' not supported by keras')

    x_train, x_test = asTypeBoth(x_train, x_test)

    if len(x_train.shape) < 4:
        x_train = tf.expand_dims(x_train, -1)
        x_test = tf.expand_dims(x_test, -1)

    if gray:
        x_train, x_test = transformToGrayAndReshapeBoth(x_train, x_test, shape=shape)
    else:
        x_train = reshape(x_train, shape=shape)
        x_test = reshape(x_test, shape=shape)
        # x_train = x_train.repeat(3, -1)
        # x_test = x_test.repeat(3, -1)

        # x_train = tf.image.grayscale_to_rgb(x_train,name=None)
        # x_test = tf.image.grayscale_to_rgb(x_test,name=None)

    x_train, x_test = normalizeBoth(x_train, x_test)

    num_classes = max(y_train.max() + 1, y_test.max() + 1)
    if one_hot:
        y_train, y_test = oneEncodeBoth(y_train, y_test)

    return x_train, y_train, x_test, y_test, num_classes


def loadTensorFlowDataset(datasetName, one_hot=True, shape=(28, 28), gray=False):
    (x_train, y_train), \
    (x_test, y_test) = \
        tfds.as_numpy(tfds.load(datasetName, split=['train', 'test'], batch_size=-1, as_supervised=True))

    # img = Image.fromarray(x_train[5].astype(np.uint8), 'RGB')
    # img.show()

    if gray:
        x_train, x_test = transformToGrayAndReshapeBoth(x_train, x_test, shape=shape)
        x_train, x_test = normalizeBoth(x_train, x_test)
    else:
        x_train = reshape(x_train, shape=shape)
        x_test = reshape(x_test, shape=shape)

    # img = Image.fromarray(x_train[5].astype(np.uint8), 'RGB')
    # img.show()

    num_classes = max(y_train.max() + 1, y_test.max() + 1)
    if one_hot:
        y_train, y_test = oneEncodeBoth(y_train, y_test)

    return x_train, y_train, x_test, y_test, num_classes


def loadFromDir(dir, labels=None, label_index=1, shape=(28, 28),
                mode='grayscale', label_mode='int', batch_size=32, shuffle=True):
    root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    dataPath = os.path.join(root, 'data', dir)
    if labels.endswith('.txt'):
        newLabels = []
        with open(os.path.join(root, 'data', labels)) as f:
            lines = f.readlines()
            for l in lines:
                l = l.split()
                if label_mode == 'int':
                    newLabels.append(int(l[label_index]))
                else:
                    newLabels.append(l[label_index])
        labels = newLabels
        batch_size = len(labels)

    ds = keras.utils.image_dataset_from_directory(
        dataPath, labels=labels, label_mode=label_mode,
        color_mode=mode, image_size=shape, batch_size=batch_size, shuffle=shuffle)

    for x, y in ds.take(1):
        # print(x.shape, y)

        cns = ds.class_names

        return x.numpy(), y.numpy(), cns


def sampleFromDir(dir, shape=(224, 224),
                  mode='rgb', sample_size=32, ext='JPEG', seed=None, crop=False):
    root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    dataPath = os.path.join(root, 'data', dir)

    if seed is not None:
        np.random.seed(seed)

    imageFiles = [y for x in os.walk(dataPath) for y in glob(os.path.join(x[0], '*.' + ext))]
    chosen_index = np.random.choice(len(imageFiles), sample_size, replace=False)

    chosenImages = []
    for idx, iF in enumerate(imageFiles):
        if idx not in chosen_index:
            continue
        image = tf.keras.utils.load_img(
            iF,
            color_mode=mode,
            target_size=shape,
            interpolation="nearest",
            keep_aspect_ratio=False,
        )
        # displayImg(tf.keras.utils.img_to_array(image))
        if crop:
            image = cropImg(image)
            # displayImg(tf.keras.utils.img_to_array(image))
        input_arr = tf.keras.utils.img_to_array(image)
        chosenImages.append(input_arr)
    chosenImages = np.asarray(chosenImages)

    return chosenImages


def sampleFromClassesInDir(dir, shape=(224, 224),
                           mode='rgb', sample_size_per_class=32, ext='JPEG',
                           shuffle=True, seed=None, crop=False):
    root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    dataPath = os.path.join(root, 'data', dir)

    images = []
    for dn in os.listdir(dataPath):
        fdn = os.path.join(dataPath, dn)
        if os.path.isdir(fdn):
            img = sampleFromDir(fdn, shape=shape, mode=mode,
                                sample_size=sample_size_per_class, ext=ext, seed=seed, crop=crop)
            images.extend(img)
    images = np.asarray(images)
    if shuffle:
        np.random.shuffle(images)

    return images

# loadFromDir('mnist_m/mnist_m_train/', 'mnist_m/mnist_m_train_labels.txt')
# loadFromDir('tiny-imagenet/train/', labels="inferred", mode='rgb', label_mode='int'
#             , shape=(64, 64), batch_size=100000)

# loadTensorFlowDataset('caltech_birds2011')

# sampleFromDir('tiny-imagenet/train/n01443537', sample_size=5)
# sampleFromClassesInDir('tiny-imagenet/train/', sample_size_per_class=1)
