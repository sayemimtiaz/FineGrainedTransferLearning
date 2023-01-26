import random

from data_processing.base_data import loadTensorFlowDataset
from data_processing.sample_util import sampleTrainTest
import cv2
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf

from data_processing.data_util import shuffle


class Bird:
    data = None
    one_hot = False
    gray = False

    def __init__(self, one_hot=False, gray=False, load_data=True, interpolation=False, shape=(128, 128)):
        self.one_hot = one_hot
        self.gray = gray
        if load_data:
            self.data = self.getBirds(shape=shape)

    def getClasses(self):
        return list(range(self.data[4]))

    def getBirds(self, shape=(64, 64)):
        x_train, y_train, x_test, y_test, num_classes = \
            loadTensorFlowDataset('caltech_birds2011', shape=shape, one_hot=self.one_hot, gray=self.gray)

        return x_train, y_train, x_test, y_test, num_classes

    def sample(self, sample_only_classes=None, num_sample=-1, seed=None, one_hot=False, train=True):
        return sampleTrainTest(self.data, num_sample=num_sample,
                               sample_only_classes=sample_only_classes, seed=seed,
                               one_hot=one_hot, train=train)


def getBirdTrainingData(shape=(128,128)):
    # load Caltech Birds dataset from 2011
    (train_ds, val_ds), ds_info = tfds.load('caltech_birds2011',
                                            split=['train', 'test'],
                                            shuffle_files=True, as_supervised=True, with_info=True)

    # Set batch size and image dimensions allowed by your memory resources
    batch_size = 128
    image_height = shape[0]
    image_width = shape[1]


    def format_dataset(data):
        norm = lambda image, label: (tf.cast(image, tf.float32) / 255., label)
        pad = lambda image, label: (tf.image.resize_with_pad(image, image_height, image_width), label)
        data = data.map(norm, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        data = data.map(pad, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        data = data.batch(batch_size)
        data = data.prefetch(tf.data.experimental.AUTOTUNE)
        return data

    train_ds, val_ds = tuple(map(format_dataset, [train_ds, val_ds]))

    return train_ds,val_ds,200
# Bird(interpolation=True, one_hot=True)
