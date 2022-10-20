from typing import Optional, Any
import numpy as np
from keras.saving.save import load_model

from util.common import initModularLayers
from util.hypothesis_testing import isSameDistribution
from util.ordinary import dump_as_pickle, load_pickle_file
from util.transfer_util import construct_target_model_approach_2, get_transfer_model_name

freezeUntil = 3
model_name = 'h5/source_model_mixed.h5'
MODE = 'val'
globalPoolThreshold = 0
alpha = 10 ** -29
sourceRate = load_pickle_file('transfer_model/source_filter_active_' + MODE + '.pickle')
targetRate = load_pickle_file('transfer_model/target_filter_active_' + MODE + '.pickle')

chosen_filters = []
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

        if sourceFilter.mean() > globalPoolThreshold and targetFilter.mean() > globalPoolThreshold and isSameDistribution(
                sourceActiveFilter,
                targetActiveFilter,
                alpha=alpha):
            chosen_filters.append(filterNo)

        # if sourceFilter.mean() > globalPoolThreshold and targetFilter.mean() > globalPoolThreshold:
            # chosen_filters.append(filterNo)

model = load_model(model_name)

positiveConcern = initModularLayers(model.layers)

target_model = construct_target_model_approach_2(positiveConcern, model,
                                                 chosen_filters,
                                                 target_layer=freezeUntil)
target_model.save('transfer_model/transfer_' + str(freezeUntil) + '.h5')
