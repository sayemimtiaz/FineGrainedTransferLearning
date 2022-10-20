from typing import Optional, Any
import numpy as np
from util.hypothesis_testing import isSameDistribution
from util.ordinary import dump_as_pickle, load_pickle_file

MODE = 'val'
alpha = 0.01
sourceRate = load_pickle_file('transfer_model/source_filter_active_' + MODE + '.pickle')
targetRate = load_pickle_file('transfer_model/target_filter_active_' + MODE + '.pickle')


def compute_filter_active_rate_by_pool_val(data):
    res = []
    for filterNo in data.keys():
        dataFilter = data[filterNo]
        if type(dataFilter) == list:
            dataFilter = np.asarray(dataFilter)
        allObsLen = len(dataFilter)
        activeFilter = dataFilter[dataFilter > 0]
        activeFilter = len(activeFilter) / allObsLen
        res.append(activeFilter)

    return res


def pool_filter_across_samples(data):
    res = []
    for filterNo in data.keys():
        dataFilter = data[filterNo]
        if type(dataFilter) == list:
            dataFilter = np.asarray(dataFilter)
        res.append(dataFilter.mean())

    return res


all_rate = []
for layerNo in sourceRate.keys():
    compat = 0
    nonCompat = 0

    # sourceFilter = sourceRate[layerNo]
    # targetFilter = targetRate[layerNo]
    # filterCovSrc = compute_filter_active_rate_by_pool_val(sourceFilter)
    # filterCovTar = compute_filter_active_rate_by_pool_val(targetFilter)
    # filterPoolSrc = pool_filter_across_samples(sourceFilter)
    # filterPoolTar = pool_filter_across_samples(targetFilter)
    # if  \
    #         isSameDistribution(filterPoolSrc, filterPoolTar, alpha=alpha):
    #     compat += 1
    # else:
    #     nonCompat += 1

    for filterNo in sourceRate[layerNo].keys():
        sourceFilter = sourceRate[layerNo][filterNo]
        targetFilter = targetRate[layerNo][filterNo]
        if type(sourceFilter) == list:
            sourceFilter = np.asarray(sourceFilter)
        if type(targetFilter) == list:
            targetFilter = np.asarray(targetFilter)

        sourceFilter = sourceFilter.flatten()
        targetFilter = targetFilter.flatten()

        # allSourceObsLen = len(sourceFilter)
        # allTargetObsLen = len(targetFilter)
        sourceActiveFilter = sourceFilter[sourceFilter > 0]
        targetActiveFilter = targetFilter[targetFilter > 0]
        # # # # print(len(sourceFilter), len(targetFilter))
        # sourceFilterActiveRate = len(sourceFilter) / allSourceObsLen
        # targetFilterActiveRate = len(targetFilter) / allTargetObsLen
        # # #
        # if sourceFilterActiveRate < 0.05 or targetFilterActiveRate < 0.1:
        #     continue

        # sourceZero = len(sourceFilter[sourceFilter == 0])
        # targetZero = len(targetFilter[targetFilter == 0])
        # sourceFilter = np.append(sourceFilter, sourceZero / len(sourceFilter))
        # targetFilter = np.append(targetFilter, targetZero / len(targetFilter))

        # print(len(sourceFilter))

        if sourceFilter.mean() > 0 and targetFilter.mean() > 0 and isSameDistribution(sourceActiveFilter, targetActiveFilter,
                                                                                      alpha=alpha):
            compat += 1
        else:
            nonCompat += 1

    rate = (compat / (compat + nonCompat)) * 100.0
    all_rate.append(rate)

    print('Layer: ' + str(layerNo) + ', Compatibility: ' + str(rate))

print('Mean compatibility: ', np.asarray(all_rate).mean())
