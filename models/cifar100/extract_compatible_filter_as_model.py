import math
from typing import Optional, Any
import numpy as np
from keras.saving.save import load_model

from util.common import initModularLayers
from util.hypothesis_testing import isSameDistribution, isSameDistributionZero, getPValue
from util.ordinary import dump_as_pickle, load_pickle_file, get_transfer_model_name, get_transfer_filter_name
from util.transfer_util import construct_target_model_approach_2

freezeUntil = 3
model_name = 'h5/source_model_mixed_cat.h5'
MODE = 'val'
globalPoolThreshold = 0
alpha = 0.0000001
sourceRate = load_pickle_file(get_transfer_filter_name(model_name, MODE, end='source'))
targetRate = load_pickle_file(get_transfer_filter_name(model_name, MODE, end='target'))

chosen_filters = []
p_values = {}
for serial, layerNo in enumerate(sourceRate.keys()):
    compat = 0
    nonCompat = 0

    if serial != freezeUntil:
        continue

    for filterNo in sourceRate[layerNo].keys():
        sourceFilter = sourceRate[layerNo][filterNo]
        targetFilter = targetRate[layerNo][filterNo]
        if type(sourceFilter) == list:
            sourceFilter = np.asarray(sourceFilter)
        if type(targetFilter) == list:
            targetFilter = np.asarray(targetFilter)

        sourceFilter = sourceFilter.flatten()
        targetFilter = targetFilter.flatten()
        sourceActiveFilter = sourceFilter[sourceFilter > 0]
        targetActiveFilter = targetFilter[targetFilter > 0]

        if sourceFilter.mean() > globalPoolThreshold and targetFilter.mean() > globalPoolThreshold:
            if alpha == 0.0 and isSameDistributionZero(
                    sourceActiveFilter,
                    targetActiveFilter):
                chosen_filters.append(filterNo)
            if alpha > 0.0 and isSameDistribution(
                    sourceActiveFilter,
                    targetActiveFilter, alpha=alpha):
                chosen_filters.append(filterNo)
        try:
            # pv= getPValue(sourceFilter, targetFilter)
            pv = getPValue(sourceActiveFilter, targetActiveFilter)
            # p_values[filterNo] = 1.0+pv # or math.exp(pv) only
            p_values[filterNo] = math.exp(pv) # in partial compat, 1+p working better, in usual case, just pv is working best
        except:
            p_values[filterNo] = 1.0  # exp(0)

        # if sourceFilter.mean() > globalPoolThreshold and targetFilter.mean() > globalPoolThreshold:
        # chosen_filters.append(filterNo)

print(len(p_values.keys()))
print(len(chosen_filters))
model = load_model(model_name)

positiveConcern = initModularLayers(model.layers)

target_model = construct_target_model_approach_2(positiveConcern, model,
                                                 chosen_filters,
                                                 target_layer=freezeUntil, n_classes=5,
                                                 zero_initialize=True,
                                                 partial_freeze=False,
                                                 p_values=p_values)
target_model.save(get_transfer_model_name(freezeUntil, model_name, prefix='target'))
