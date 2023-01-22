import math
import numpy as np

from models.cifar100 import MODE, target_dataset
from models.imagenet import getTargetNumClass
from util.hypothesis_testing import getPValue, isSameDistribution
from util.ordinary import load_pickle_file, get_transfer_filter_name, get_transfer_model_name, dump_as_pickle
from util.transfer_util import construct_reweighted_target

sourceRate = load_pickle_file(get_transfer_filter_name(mode=MODE, end='source'))
targetRate = load_pickle_file(get_transfer_filter_name(mode=MODE, end=target_dataset))
target_num_class = getTargetNumClass()
alpha = 0.01

numFilter = int(sourceRate['numFilter'])

sourceRate = sourceRate['class'][9]

rnks = {}
compat = 0
nonCompat = 0
for filterNo in range(numFilter):
    sourceFilter = sourceRate[filterNo]
    targetFilter = targetRate[filterNo]
    if type(sourceFilter) == list:
        sourceFilter = np.asarray(sourceFilter)
    if type(targetFilter) == list:
        targetFilter = np.asarray(targetFilter)

    sourceFilter = sourceFilter.flatten()
    targetFilter = targetFilter.flatten()
    sourceActiveFilter = sourceFilter[sourceFilter > 0]
    targetActiveFilter = targetFilter[targetFilter > 0]

    try:
        rnks[int(filterNo)] = getPValue(sourceActiveFilter, targetActiveFilter)
    except:
        rnks[int(filterNo)] = 0.0

rnks = dict(sorted(rnks.items(), key=lambda item: item[1], reverse=True))

# import json
#
# print(json.dumps(rnks, indent=4, sort_keys=False))

dump_as_pickle(rnks, 'transfer_model/filter_ranked.pickle')
