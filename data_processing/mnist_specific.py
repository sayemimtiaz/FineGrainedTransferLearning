import os

from keras.utils import to_categorical
from keras_preprocessing.image import ImageDataGenerator

from constants import SHAPE
from data_processing.base_data import loadFromDir, getKerasDataset, sampleFromClassesInDir
from util.common import displayImg, get_project_root


#
# def getMnist(gray=True, shape=(28, 28), one_hot=True):
#     return getKerasDataset(one_hot=one_hot, dataset='mnist', gray=gray, shape=shape)


class Mnist:
    data = None
    data_path = None

    def __init__(self):
        self.data_path = get_project_root()
        self.data_path = os.path.join(self.data_path, 'data', 'mnist', 'train')

    def getClasses(self):
        return 10

    def sampleFromDir(self, sample_size_per_class=20, seed=None, ext='jpg', crop=False):
        return sampleFromClassesInDir(self.data_path,
                                      sample_size_per_class=sample_size_per_class, seed=seed, ext=ext,
                                      shape=SHAPE, crop=crop)


# mn = Mnist()
# data = mn.sampleFromDir()
# displayImg(data[0][0])
