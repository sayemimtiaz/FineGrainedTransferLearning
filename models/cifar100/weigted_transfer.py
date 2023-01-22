import math
from random import random

import numpy as np

from models.cifar100 import getSourceModel, getTargetNumClass, MODE, target_dataset
from util.common import freezeModel
from util.hypothesis_testing import getPValue
from util.ordinary import load_pickle_file, get_transfer_filter_name, get_transfer_model_name
from util.transfer_util import construct_reweighted_target, regularWeight, plainWeight, binaryWeight


def getWeigtedTransferModel(weighting_scheme=regularWeight, targetIndex=None):
    sourceRate = load_pickle_file(get_transfer_filter_name(mode=MODE, end='source'))
    targetRate = load_pickle_file(get_transfer_filter_name(mode=MODE, end=target_dataset))
    model = getSourceModel()
    target_num_class = getTargetNumClass()

    numFilter = int(sourceRate['numFilter'])

    p_values = {}

    for filterNo in range(numFilter):
        if weighting_scheme == plainWeight or weighting_scheme == binaryWeight:
            p_values[filterNo] = 0.0
        else:
            p_values[filterNo] = 1.0

    for source_c in sourceRate['class']:
        if targetIndex is not None and source_c != targetIndex:
            continue
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

            try:
                pv = getPValue(sourceActiveFilter, targetActiveFilter)
                # pv = getPValue(sourceFilter, targetFilter)

                pv = weighting_scheme(
                    pv)  # in partial compat, 1+p working better, in usual case, just pv is working best

                p_values[filterNo] = max(pv, p_values[filterNo])
            except:
                pass

    # print(len(p_values.keys()))

    if weighting_scheme == binaryWeight:
        z = 0
        for k in p_values:
            if p_values[k] == 0:
                z += 1
        print('Zeroed: ', ((z / len(p_values)) * 100.0))

    target_model = construct_reweighted_target(freezeModel(model), n_classes=target_num_class,
                                               p_values=p_values)
    target_model.save(get_transfer_model_name(prefix='weighted', model_name=target_dataset))

# getWeigtedTransferModel(weighting_scheme=regularWeight)
