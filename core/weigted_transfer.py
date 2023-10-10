import math
import numpy as np

from constants import source_model_name, target_dataset, source_dataset, NUM_SOURCE_SAMPLE
from core import getSourceModel, getTargetNumClass
from util.common import freezeModel
from util.hypothesis_testing import getPValue,isSameDistribution
from util.model_hacker import hack_model
from util.ordinary import load_pickle_file, get_transfer_filter_name, get_transfer_model_name


# def getWeigtedTransferModel(alpha=0.00):
#     sourceRate = load_pickle_file(get_transfer_filter_name(source_model_name))
#     targetRate = load_pickle_file(get_transfer_filter_name(target_dataset))
#     model = getSourceModel()
#     target_num_class = getTargetNumClass()
#
#     numFilter = int(sourceRate['numFilter'])
#
#     p_values = {}
#
#     for filterNo in range(numFilter):
#         p_values[filterNo] = 0.0
#
#     for source_c in sourceRate['class']:
#         for filterNo in range(numFilter):
#             sourceFilter = sourceRate['class'][source_c][filterNo]
#             targetFilter = targetRate[filterNo]
#             if type(sourceFilter) == list:
#                 sourceFilter = np.asarray(sourceFilter)
#             if type(targetFilter) == list:
#                 targetFilter = np.asarray(targetFilter)
#
#             sourceFilter = sourceFilter.flatten()
#             targetFilter = targetFilter.flatten()
#             sourceActiveFilter = sourceFilter[sourceFilter > 0]
#             targetActiveFilter = targetFilter[targetFilter > 0]
#
#             try:
#                 pv = getPValue(sourceActiveFilter, targetActiveFilter)
#                 p_values[filterNo] = max(pv, p_values[filterNo])
#             except:
#                 pass
#
#     # print(len(p_values.keys()))
#     for k in p_values:
#         if p_values[k] <= alpha:
#             p_values[k] = 0.0
#     z = 0
#     for k in p_values:
#         if p_values[k] == 0:
#             z += 1
#     print('Deleted: ' + str(round(((z / len(p_values)) * 100.0), 2)) + '%')
#
#     if SLICE_MODE == 'offline':
#         filters = []
#         for k in p_values:
#             if p_values[k] != 0:
#                 filters.append(k)
#         model = hack_model(model, 'conv_7b', 'inceptionresnetv2', filters)
#
#     target_model = construct_reweighted_target(freezeModel(model), n_classes=target_num_class,
#                                                p_values=p_values)
#
#     target_model.save(get_transfer_model_name(prefix='weighted', model_name=target_dataset))


def getPValues(alpha=0.00, target_ds=None, parent_model=None):
    if target_ds is None:
        target_ds = target_dataset
    if parent_model is None:
        parent_model = source_model_name
    sourceRate = load_pickle_file(get_transfer_filter_name(parent_model, source_dataset, NUM_SOURCE_SAMPLE))
    targetRate = load_pickle_file(get_transfer_filter_name(target_ds))

    numFilter = int(sourceRate['numFilter'])

    p_values = {}

    for filterNo in range(numFilter):
        p_values[filterNo] = 0.0

    for source_c in sourceRate['class']:
        for filterNo in range(numFilter):
            sourceFilter = sourceRate['class'][source_c][filterNo]
            targetFilter = targetRate['class'][source_c][filterNo]
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
                p_values[filterNo] = max(pv, p_values[filterNo])
            except:
                pass

    # print(len(p_values.keys()))
    for k in p_values:
        if p_values[k] <= alpha:
            p_values[k] = 0.0
    z = 0
    for k in p_values:
        if p_values[k] == 0:
            z += 1
    delRate = round(((z / len(p_values)) * 100.0), 2)
    print('Deleted: ' + str(delRate) + '%')

    return p_values, delRate


def getPValuesNoAlpha(target_ds=None, parent_model=None):
    if target_ds is None:
        target_ds = target_dataset
    if parent_model is None:
        parent_model = source_model_name
    sourceRate = load_pickle_file(get_transfer_filter_name(parent_model, source_dataset, NUM_SOURCE_SAMPLE))
    targetRate = load_pickle_file(get_transfer_filter_name(target_ds))

    numFilter = int(sourceRate['numFilter'])

    p_values = {}

    for filterNo in range(numFilter):
        p_values[filterNo] = 0.0

    for source_c in sourceRate['class']:
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
                pv = getPValue(sourceFilter, targetFilter)
                p_values[filterNo] = max(pv, p_values[filterNo])
            except:
                pass

    return p_values


def similarFeatureRate(alpha=0.00, sourceRate=None, targetRate=None, numFilter=None):
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

        if sourceFilter.mean() > 0 and targetFilter.mean() > 0 and isSameDistribution(sourceActiveFilter,
                                                                                  targetActiveFilter,
                                                                                  alpha=alpha):
            compat += 1
        else:
            nonCompat += 1

    rnk = round(((compat / (compat + nonCompat))*100.0),2)

    return rnk