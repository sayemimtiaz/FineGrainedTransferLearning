from modularization.concern.concern_identification import *

from keras.models import load_model

import pickle

from mining.ruler_extractor import find_local_feature
from util.data_interface import get_train_df_interface, get_target_and_drop_vars_interface, get_test_df_interface
from util.ordinary import get_feature_module_name, get_local_feature_name
from util.sampling_util import sample_df

base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# datasetName = 'adult'
datasetName='titanic'
# datasetName = 'bank'

datasetName = os.path.join(base_path, 'models', datasetName)

model_name = os.path.join(datasetName, 'h5', 'model_b1.h5')

with open(get_feature_module_name(model_name, datasetName), 'rb') as handle:
    feature_neurons = pickle.load(handle)

neuronSimilarityThreshold = 0.80
targetLabel = -1
PRINT = False
activeFeature = True
# activeFeature = False
concernIdentifier = ConcernIdentification()

# df = sample_df(num_sample=500, data_acquisition=get_train_df_interface(datasetName), random_state=17)
# df = sample_df(num_sample=1000, data_acquisition=get_test_df_interface(datasetName), random_state=29)
df = sample_df(num_sample=300, data_acquisition=get_train_df_interface(datasetName), random_state=17)

model = load_model(model_name)
concern = initModularLayers(model.layers)

target_var, drop_vars = get_target_and_drop_vars_interface(datasetName)
all_dump = []
for i in range(len(df)):
    x_t = df.iloc[[i]].drop(drop_vars, axis=1).values
    if len(x_t.shape) > 1:
        x_t = x_t.flatten()
    hidden_val = {}

    for layerNo, _layer in enumerate(concern):
        x_t = concernIdentifier.propagateThroughLayer(_layer, x_t, apply_activation=True)

        if _layer.type == LayerType.Dense and not _layer.last_layer:
            hidden_val[layerNo] = x_t

    # print(df.iloc[[i]].to_string())
    if PRINT:
        if activeFeature:
            print('Active features: ')
        else:
            print('Inactive features: ')

    important_features, unimportant_features, all_features = find_local_feature(hidden_val, feature_neurons,
                                                                  similarity_threshold=neuronSimilarityThreshold)
    important_features=all_features[:int(len(all_features)/2)]
    unimportant_features=all_features[int(len(all_features)/2):]

    if not activeFeature:
        important_features = unimportant_features
    if PRINT:
        print(important_features)

        print('Predicted level', (x_t >= 0.5))

    fSet = []
    for (f0, f1, _) in important_features:
        fSet.append((f0, f1))

    if len(fSet) > 1 and \
            ((targetLabel == -1) or (targetLabel == 1 and x_t >= 0.5) or
             (targetLabel == 0 and x_t < 0.5)):
        # (data, features)
        all_dump.append((df.iloc[[i]], fSet))

print(len(all_dump))
with open(get_local_feature_name(model_name, targetLabel, datasetName, activeFeature), 'wb') as handle:
    pickle.dump(all_dump, handle, protocol=pickle.HIGHEST_PROTOCOL)
