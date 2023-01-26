import math
import numpy as np

from data_processing.tiny_imagenet_util import TinyImageNet
from models.imagenet import getSourceModel, freezeModel, target_dataset, MODE, getTargetNumClass, getSourceData
from util.hypothesis_testing import getPValue, isSameDistribution
from util.ordinary import load_pickle_file, get_transfer_filter_name, get_transfer_model_name
from util.transfer_util import construct_reweighted_target

sourceRate = load_pickle_file(get_transfer_filter_name(mode=MODE, end='source'))
targetRate = load_pickle_file(get_transfer_filter_name(mode=MODE, end=target_dataset))
target_num_class = getTargetNumClass()
alpha = 0.01

numFilter = int(sourceRate['numFilter'])

tiny = getSourceData(shape=(64, 64), gray=False)

rnks = {}
for source_c in sourceRate['class']:
    # if source_c != tiny.getBirdIndex():
    #     continue
    compat = 0
    nonCompat = 0
    for filterNo in range(numFilter):
        sourceFilter = sourceRate['class'][source_c][filterNo]
        targetFilter = targetRate[filterNo]
        if type(sourceFilter) == list:
            sourceFilter = np.asarray(sourceFilter)
        if type(targetFilter) == list:
            targetFilter = np.asarray(targetFilter)

        sourceFilter = sourceFilter.flatten()
        targetFilter = targetFilter.flatten()
        sourceActiveFilter = sourceFilter[sourceFilter > 0]
        targetActiveFilter = targetFilter[targetFilter > 0]

        if sourceFilter.mean() > 0 and targetFilter.mean() > 0 and isSameDistribution(sourceActiveFilter,
                                                                                      targetActiveFilter,
                                                                                      alpha=alpha):
            compat += 1
        else:
            nonCompat += 1
        #
        # if isSameDistribution(sourceFilter, targetFilter, alpha=alpha):
        #     compat += 1
        # else:
        #     nonCompat += 1

    rnks[tiny.getClassName(source_c)] = compat / (compat + nonCompat)

rnks = dict(sorted(rnks.items(), key=lambda item: item[1], reverse=True))

import json

print(json.dumps(rnks, indent=4, sort_keys=False))
