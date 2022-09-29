from data_type.constants import Constants
from modularization.concern.concern_identification import *

from keras.models import load_model

import pickle

from mining.ruler_extractor import find_local_feature
from util.data_interface import get_target_and_drop_vars_interface, from_criteria_interface, get_train_df_interface, \
    get_test_df_interface
from util.ordinary import get_model_name, get_feature_module_name, get_global_feature_name
from util.sampling_util import sample

base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# datasetName = 'adult'
datasetName = 'titanic'
# datasetName = 'bank'

datasetName = os.path.join(base_path, 'models', datasetName)
target_var, drop_vars = get_target_and_drop_vars_interface(datasetName)

all_model_names = ['model_b4']
lowerThreshold = 0.00
upperThreshold = 1.0
sampling_rate = 1.0
activeFeatureOnly = True
# activeFeatureOnly=False
neuronSimilarityThreshold = 0.8

correctSamples = True

for mi, m in enumerate(all_model_names):
    model_name = os.path.join(datasetName, 'h5', m + '.h5')
    with open(get_feature_module_name(model_name, datasetName), 'rb') as handle:
        feature_neurons = pickle.load(handle)

    concernIdentifier = ConcernIdentification()

    model = load_model(model_name)
    concern = initModularLayers(model.layers)

    sx, sy, _ = sample(data_acquisition=get_train_df_interface(datasetName), frac_sample=sampling_rate)
    # sx, sy, _ = sample(data_acquisition=get_test_df_interface(datasetName), num_sample=1000, random_state=19)

    feature_count = {}
    nx = 0
    for idx in range(len(sx)):
        x_t = sx[idx]

        hidden_val = {}

        for layerNo, _layer in enumerate(concern):
            x_t = concernIdentifier.propagateThroughLayer(_layer, x_t, apply_activation=True)

            if _layer.type == LayerType.Dense and not _layer.last_layer:
                hidden_val[layerNo] = x_t

        if correctSamples:
            if x_t < 0.5 and sy[idx] != 0:
                continue
            if x_t >= 0.5 and sy[idx] != 1:
                continue
        else:
            if x_t < 0.5 and sy[idx] == 0:
                continue
            if x_t >= 0.5 and sy[idx] == 1:
                continue

        nx += 1
        active_features, inactive_features,_ = \
            find_local_feature(hidden_val, feature_neurons, similarity_threshold=neuronSimilarityThreshold)

        if not activeFeatureOnly:
            active_features = inactive_features
        for f in active_features:
            f = (f[0], f[1])
            if f not in feature_count:
                feature_count[f] = 0
            feature_count[f] += 1

    for f in feature_count.keys():
        feature_count[f] /= nx

    feature_count = dict(sorted(feature_count.items(), key=lambda item: item[1], reverse=True))

    if activeFeatureOnly:
        print('Most active features ')
    else:
        print('Most inactive features ')

    filtered_features = {}
    for f in feature_count.keys():
        if lowerThreshold <= feature_count[f] <= upperThreshold:
            print(from_criteria_interface(f[0], f[1], datasetName), feature_count[f])

            filtered_features[f] = feature_count[f]

    with open(get_global_feature_name(model_name, datasetName), 'wb') as handle:
        pickle.dump(filtered_features, handle, protocol=pickle.HIGHEST_PROTOCOL)
