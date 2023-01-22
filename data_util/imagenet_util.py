import csv
import numpy as np
from data_util.base_data import loadFromDir, sampleFromClassesInDir
import os

from data_util.sample_util import sample
from data_util.util import oneEncodeBoth, oneEncode
from util.common import displayImg


class TinyImageNet:
    data = None
    shape=(224,224)

    def __init__(self, train_data=False, shape=(224,224)):
        self.shape=shape
        if train_data:
            self.data = self.getTinyImageNetForTrain()
        # else:
        #     self.data = self.getTinyImageNet(shape=shape)

    def getClasses(self):
        return self.data[2]

    def getBirdIndex(self):
        return self.getClassIndex('bird')

    def getPaperIndex(self):
        return self.getClassIndex('paper')

    # {'fish': 0, 'salamander': 1, 'frog': 2, 'crocodile': 3, 'snake': 4, 'trilobite': 5, 'arachnid': 6, 'bug': 7,
    #  'bird': 8, 'marsupial': 9, 'coral': 10, 'mollusk': 11, 'crustacean': 12, 'marine mammals': 13, 'dog': 14,
    #  'cat': 15, 'wild cat': 16, 'bear': 17, 'butterfly': 18, 'echinoderms': 19, 'rodent': 20, 'hog': 21,
    #  'ungulate': 22,
    #  'primate': 23, 'technology': 24, 'clothing': 25, 'furniture': 26, 'accessory': 27, 'building': 28,
    #  'container': 29,
    #  'ball': 30, 'vehicle': 31, 'lab equipment': 32, 'food': 33, 'tool': 34, 'outdoor scene': 35, 'decor': 36,
    #  'train': 37, 'weapon': 38, 'electronics': 39, 'instrument': 40, 'cooking': 41, 'boat': 42, 'fence': 43,
    #  'sports equipment': 44, 'hat': 45, 'toy': 46, 'paper': 47, 'vegetable': 48, 'fungus': 49, 'fruit': 50,
    #  'plant': 51}
    def getClassIndex(self, className):
        return self.data[3][className]

    def getClassName(self, idx):
        for k in self.data[3].keys():
            if self.data[3][k] == idx:
                return k
        return 'NotFound'

    def getClassData(self, clsIdx):
        data = self.data[0][self.data[1] == clsIdx]
        # displayImg(data[3])
        return data

    def getTinyImageNet(self):
        x, y, class_names = loadFromDir('tiny-imagenet/train/', labels="inferred", mode='rgb', label_mode='int'
                                        , shape=self.shape, batch_size=100000, shuffle=True)
        class_idx, cat_idx = self.get_high_level_categories(class_names)
        new_y = []
        for _y in y:
            new_y.append(class_idx[class_names[_y]])
        new_y = np.asarray(new_y)

        return x, new_y, list(set(new_y)), cat_idx

    def getTinyImageNetForTrain(self, one_hot=True):
        x_train, y_train, _ = loadFromDir('tiny-imagenet/train/', labels="inferred", mode='rgb', label_mode='int'
                                          , shape=self.shape, batch_size=100000, shuffle=True)

        if one_hot:
            y_train = oneEncode(y_train)

        return x_train, y_train, 200

    def sampleFromDir(self, sample_size_per_class=20, seed=None, ext='JPEG'):
        return sampleFromClassesInDir('tiny-imagenet/train/',
                                      sample_size_per_class=sample_size_per_class, seed=seed, ext=ext,
                                      shape=self.shape)


    def sample(self, sample_only_classes=None, num_sample=-1, seed=None):
        nd = self.data[0], self.data[1]
        if sample_only_classes is not None:
            x_train, y_train = sample(nd, num_sample=num_sample,
                                      sample_only_classes=sample_only_classes, seed=seed)
        else:
            x_train, y_train = sample(nd, num_sample=num_sample,
                                      num_classes=len(self.getClasses()), seed=seed)
        return x_train, y_train, None, None, None

    def get_high_level_categories(self, class_names):
        root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        file = os.path.join(root_dir, 'data', 'tiny-imagenet', 'imagenet_categories_synset.csv')

        class_map = {}
        class_idx = {}
        cat_idx = {}
        idx = 0
        with open(file, 'r') as input:
            reader = csv.reader(input)
            next(reader)
            for line in reader:
                cName = line[1].strip()
                cat = line[3].strip()
                if cName not in class_names:
                    continue
                # class_map[cName] = cat
                if cat not in cat_idx:
                    class_idx[cName] = idx
                    cat_idx[cat] = idx
                    idx += 1
                else:
                    class_idx[cName] = cat_idx[cat]

        # x, y, class_names = loadFromDir('tiny-imagenet/train/', labels="inferred", mode='rgb', label_mode='int'
        #                                 , shape=(64, 64), batch_size=100000)
        # high_level = set()
        # for c in class_names:
        #     high_level.add(class_map[c])

        # print(high_level)
        # print(len(high_level))
        return class_idx, cat_idx

# tiny = TinyImageNet()
# print(tiny.data[3])
# tiny.sampleTinyImageNet(sample_only_classes=[5], num_sample=10)
