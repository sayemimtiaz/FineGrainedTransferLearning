
from constants import SHAPE
from data_processing.base_data import sampleFromClassesInDir
import os
from util.common import get_project_root


class ImageNet:
    data_path = None

    def __init__(self):
        self.data_path = get_project_root()
        self.data_path = os.path.join(self.data_path, 'data', 'imagenet2012_subset', '10pct')

    def getClasses(self):
        return 1000

    def sampleFromDir(self, sample_size_per_class=5, seed=None, ext='JPEG'):
        return sampleFromClassesInDir(self.data_path,
                                      sample_size_per_class=sample_size_per_class, seed=seed, ext=ext,
                                      shape=SHAPE)


# tiny = ImageNet()
# images=tiny.sampleFromDir()
# print(images.shape)

