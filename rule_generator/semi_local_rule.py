from data_type.constants import Constants
from modularization.concern.concern_identification import *

from keras.models import load_model

import pickle

from mining.ruler_extractor import find_local_feature
from util.data_interface import from_criteria_interface, get_train_df_interface, get_target_and_drop_vars_interface
from util.ordinary import get_feature_module_name, get_semi_feature_name

# all_model_names = ['model_b1', 'model_b2', 'model_b3', 'model_b4', 'model_b5', 'model_b6']
from util.sampling_util import sample

base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# datasetName = 'adult'
datasetName='titanic'
# datasetName = 'bank'

datasetName = os.path.join(base_path, 'models', datasetName)
target_var, drop_vars = get_target_and_drop_vars_interface(datasetName)

all_model_names = ['model_b1']

neuronSimilarityThreshold = 0.3

take_top_samples = True
top_sample_to_take = 15
data_sampling_rate = 1.0
lowerThreshold = 0.3
upperThreshold = 1.0
# activeFeatureOnly=False
activeFeatureOnly = True
correctSamples = True
PRINT = True

for m in all_model_names:
    model_name = os.path.join(datasetName, 'h5', m + '.h5')
    # print(model_name)
    with open(get_feature_module_name(model_name, datasetName), 'rb') as handle:
        feature_neurons = pickle.load(handle)

    concernIdentifier = ConcernIdentification()

    model = load_model(model_name)
    concern = initModularLayers(model.layers)

    for lb in [0,1]:
        sx, sy, _ = sample(column_name=target_var, value=lb,
                           data_acquisition=get_train_df_interface(datasetName), frac_sample=data_sampling_rate)

        feature_count = {}
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

        if PRINT:
            if activeFeatureOnly:
                print('Most active features for class ' + str(lb))
            else:
                print('Most inactive features for class ' + str(lb))
        filtered_features = {}
        cntr = 0
        for f in feature_count.keys():
            if lowerThreshold <= feature_count[f] <= upperThreshold:
                if PRINT:
                    print(from_criteria_interface(f[0], f[1], datasetName), feature_count[f])

                if (take_top_samples and cntr < top_sample_to_take) or (not take_top_samples):
                    filtered_features[f] = feature_count[f]
                    cntr += 1
                if take_top_samples and cntr >= top_sample_to_take:
                    break

        with open(get_semi_feature_name(model_name, lb, datasetName), 'wb') as handle:
            pickle.dump(filtered_features, handle, protocol=pickle.HIGHEST_PROTOCOL)
