import csv
import numpy as np

from constants import SHAPE
from data_processing.base_data import loadFromDir, sampleFromClassesInDir
import os

from data_processing.sample_util import sample
from data_processing.data_util import oneEncodeBoth, oneEncode
from util.common import displayImg


class TinyImageNet:

    def __init__(self):
        pass

    def getClasses(self):
        return 200

    def sampleFromDir(self, sample_size_per_class=20, seed=None, ext='JPEG'):
        return sampleFromClassesInDir('tiny-imagenet/train/',
                                      sample_size_per_class=sample_size_per_class, seed=seed, ext=ext,
                                      shape=SHAPE)


# tiny = TinyImageNet()
# print(tiny.data[3])
# tiny.sampleTinyImageNet(sample_only_classes=[5], num_sample=10)
