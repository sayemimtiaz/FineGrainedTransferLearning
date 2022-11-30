import csv
import numpy as np
from data_util.base_data import loadFromDir
import os

from data_util.sample_util import sample


class TinyImageNet:
    data = None

    def __init__(self):
        self.data = self.getTinyImageNet()

    def getClasses(self):
        return self.data[2]

    def getTinyImageNet(self, shape=(64, 64)):
        x, y, class_names = loadFromDir('tiny-imagenet/train/', labels="inferred", mode='rgb', label_mode='int'
                                        , shape=shape, batch_size=100000, shuffle=True)
        class_idx, _ = self.get_high_level_categories(class_names)
        new_y = []
        for _y in y:
            new_y.append(class_idx[class_names[_y]])
        new_y = np.asarray(new_y)
        return x, new_y, list(set(new_y))

    def sampleTinyImageNet(self, sample_only_classes=None, num_sample=-1, seed=None):
        nd = self.data[0], self.data[1]
        x_train, y_train = sample(nd, num_sample=num_sample,
                                  sample_only_classes=sample_only_classes, seed=seed)
        return x_train, y_train

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
        return class_idx, idx


# tiny = TinyImageNet()
# tiny.sampleTinyImageNet(sample_only_classes=[5], num_sample=10)