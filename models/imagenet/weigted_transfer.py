import math
import numpy as np

from models.imagenet import getSourceModel, target_dataset, MODE, getTargetNumClass, SHAPE, source_model_name
from util.common import freezeModel
from util.hypothesis_testing import getPValue
from util.ordinary import load_pickle_file, get_transfer_filter_name, get_transfer_model_name
from util.transfer_util import construct_reweighted_target, regularWeight, plainWeight, binaryWeight, \
    construct_target_without_inactive_feature


def getWeigtedTransferModel(weighting_scheme=regularWeight, targetIndex=None, alpha=0.00):
    sourceRate = load_pickle_file(get_transfer_filter_name(mode=MODE, end=source_model_name))
    targetRate = load_pickle_file(get_transfer_filter_name(mode=MODE, end=target_dataset))
    model = getSourceModel(shape=SHAPE)
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

                if weighting_scheme != binaryWeight:
                    pv = weighting_scheme(
                        pv)  # in partial compat, 1+p working better, in usual case, just pv is working best

                p_values[filterNo] = max(pv, p_values[filterNo])
            except:
                pass

    # print(len(p_values.keys()))
    if weighting_scheme == binaryWeight:
        for k in p_values:
            if p_values[k] <= alpha:
                p_values[k] = 0.0
        z = 0
        for k in p_values:
            if p_values[k] == 0:
                z += 1
        print('Deleted: '+ str(round(((z / len(p_values)) * 100.0), 2))+'%')

    target_model = construct_reweighted_target(freezeModel(model), n_classes=target_num_class,
                                               p_values=p_values)
    # target_model = construct_target_without_inactive_feature(freezeModel(model),
    #                                            p_values=p_values)
    target_model.save(get_transfer_model_name(prefix='weighted', model_name=target_dataset))

# getWeigtedTransferModel(weighting_scheme=regularWeight)
