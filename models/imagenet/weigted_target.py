import math
import numpy as np

from models.imagenet import getSourceModel, source_model_name, freezeSource
from util.hypothesis_testing import getPValue
from util.ordinary import load_pickle_file, get_transfer_filter_name
from util.transfer_util import construct_target_imagenet

MODE = 'val'
sourceRate = load_pickle_file(get_transfer_filter_name(source_model_name, MODE, end='source'))
targetRate = load_pickle_file(get_transfer_filter_name(source_model_name, MODE, end='target'))
model = getSourceModel()
target_name = ''
target_num_class = 5

numFilter = int(sourceRate['numFilter'])

chosen_filters = []
p_values = {}

for filterNo in range(numFilter):
    p_values[filterNo] = 1.0

for source_c in sourceRate['class']:
    for filterNo in range(numFilter):
        sourceFilter = sourceRate[source_c][filterNo]
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
            pv = math.exp(pv)  # in partial compat, 1+p working better, in usual case, just pv is working best

            p_values[filterNo] = max(pv, p_values[filterNo])
        except:
            pass

print(len(p_values.keys()))

target_model = construct_target_imagenet(freezeSource(model), n_classes=target_num_class,
                                         p_values=p_values)
target_model.save('target_' + target_name + '.h5')
