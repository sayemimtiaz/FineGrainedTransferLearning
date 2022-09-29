from data_type.constants import Constants
from modularization.concern.concern_identification import *

from keras.models import load_model

import pickle

from mining.ruler_extractor import find_local_feature
from util.data_interface import from_criteria_interface, get_train_df_interface, get_target_and_drop_vars_interface, \
    to_criteria_interface
from util.ordinary import get_feature_module_name, get_semi_feature_name

# all_model_names = ['model_b1', 'model_b2', 'model_b3', 'model_b4', 'model_b5', 'model_b6']
from util.sampling_util import sample


def update_and_rules(active, interaction_counter):
    for f1 in active:
        for f2 in active:
            _f1 = f1[0]
            _f2 = f2[0]
            if _f1 == _f2:
                continue
            interaction_counter[_f1][_f2]['and'] += 1


def update_not_rules(inactive, interaction_counter):
    for f1 in inactive:
        _f1 = f1[0]
        interaction_counter[_f1][_f1]['not'] += 1


def update_or_rules(active, inactive, interaction_counter, or_support):
    for f1 in active:
        for f2 in inactive:
            _f1 = f1[0]
            _f2 = f2[0]
            if _f1 == _f2:
                continue
            or_support[_f1] += 1
            interaction_counter[_f1][_f2]['or'] += 1


def normalize_interaction(interaction_counter, c):
    for f1 in interaction_counter.keys():
        for f2 in interaction_counter[f1].keys():
            for f3 in interaction_counter[f1][f2].keys():
                interaction_counter[f1][f2][f3] /= c


def filter_interaction(interaction_counter, thres, PRINT=True):
    filtered = {}
    for f1 in interaction_counter.keys():
        for f2 in interaction_counter[f1].keys():
            for f3 in interaction_counter[f1][f2].keys():
                if interaction_counter[f1][f2][f3] < thres:
                    if f1 not in filtered:
                        filtered[f1] = {}
                    if f2 not in filtered[f1]:
                        filtered[f1][f2] = {}
                    filtered[f1][f2][f3] = interaction_counter[f1][f2][f3]

                    if f3 == 'not':
                        print('(NOT ' + f1 + ') = ' + str(interaction_counter[f1][f2][f3]))
                    else:
                        print('(' + f1 + ' ' + f3.upper() + ' ' + f2 + ') = ' + str(interaction_counter[f1][f2][f3]))


def top_n_interaction(interaction_counter, n, or_support, PRINT=True):
    tmp = []
    for f1 in interaction_counter.keys():
        for f2 in interaction_counter[f1].keys():
            for f3 in interaction_counter[f1][f2].keys():
                if f3 == 'or':
                    if or_support[f2] == 0 or or_support[f1] == 0:
                        continue
                    if or_support[f1]/or_support[f2]>4 or or_support[f2]/or_support[f1]>4:
                        continue

                tmp.append((f1, f2, f3, interaction_counter[f1][f2][f3]))

    tmp = sorted(tmp, key=lambda item: item[3], reverse=True)
    i = 0
    filtered = {}
    for (f1, f2, f3, v) in tmp:
        if i < n:
            i += 1
            if f1 not in filtered:
                filtered[f1] = {}
            if f2 not in filtered[f1]:
                filtered[f1][f2] = {}
            filtered[f1][f2][f3] = v
            if PRINT:
                if f3 == 'not':
                    print('(NOT ' + f1 + ') = ' + str(v))
                else:
                    print('(' + f1 + ' ' + f3.upper() + ' ' + f2 + ') = ' + str(v))


base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

datasetName = 'adult'
# datasetName = 'titanic'
# datasetName = 'bank'

datasetName = os.path.join(base_path, 'models', datasetName)
target_var, drop_vars = get_target_and_drop_vars_interface(datasetName)

all_model_names = ['model_b2']

neuronSimilarityThreshold = 0.65

take_top_samples = True
top_sample_to_take = 50
data_sampling_rate = 0.05
correctSamples = True
PRINT = True

featureNames, featureValues = to_criteria_interface(datasetName)

for m in all_model_names:
    model_name = os.path.join(datasetName, 'h5', m + '.h5')
    # print(model_name)
    with open(get_feature_module_name(model_name, datasetName), 'rb') as handle:
        feature_neurons = pickle.load(handle)

    concernIdentifier = ConcernIdentification()

    model = load_model(model_name)
    concern = initModularLayers(model.layers)

    for lb in [0, 1]:

        interaction_counter = {}
        or_support = {}
        for f1 in featureNames:
            if f1 not in interaction_counter:
                interaction_counter[f1] = {}
                # or_support[f1] = {}

            for f2 in featureNames:
                if f2 not in interaction_counter[f1]:
                    interaction_counter[f1][f2] = {}
                    # or_support[f1][f2] = {}
                if f1 != f2:
                    interaction_counter[f1][f2]['and'] = 0
                    interaction_counter[f1][f2]['or'] = 0
                    or_support[f1] = 0
                    or_support[f2] = 0

            interaction_counter[f1][f1]['not'] = 0

        print('Interaction for class ', lb)
        sx, sy, _ = sample(column_name=target_var, value=lb,
                           data_acquisition=get_train_df_interface(datasetName), frac_sample=data_sampling_rate)

        nx = 0
        for x in sx:
            x_t = x

            hidden_val = {}

            for layerNo, _layer in enumerate(concern):
                x_t = concernIdentifier.propagateThroughLayer(_layer, x_t, apply_activation=True)

                if _layer.type == LayerType.Dense and not _layer.last_layer:
                    hidden_val[layerNo] = x_t

            if correctSamples:
                if x_t < 0.5 and lb != 0:
                    continue
                if x_t >= 0.5 and lb != 1:
                    continue
            else:
                if x_t < 0.5 and lb == 0:
                    continue
                if x_t >= 0.5 and lb == 1:
                    continue

            nx += 1
            active_features, inactive_features, _ = \
                find_local_feature(hidden_val, feature_neurons, similarity_threshold=neuronSimilarityThreshold)

            update_and_rules(active_features, interaction_counter)
            update_not_rules(inactive_features, interaction_counter)
            update_or_rules(active_features, inactive_features, interaction_counter, or_support)

        normalize_interaction(interaction_counter, nx)

        # filtered_features = {}

        top_n_interaction(interaction_counter, top_sample_to_take, or_support)

        # with open(get_semi_feature_name(model_name, lb, datasetName), 'wb') as handle:
        #     pickle.dump(filtered_features, handle, protocol=pickle.HIGHEST_PROTOCOL)
