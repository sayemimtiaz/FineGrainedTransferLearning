from datetime import datetime

from models.adult.data_util import to_criteria, get_train_df, get_whole_df
from rule_generator.evaluator import evaluate_contrast
from modularization.concern.concern_identification import *

from keras.models import load_model

from modularization.concern.util import removeNeurons2, removeNeurons3
from util.dimensionality_reduction import standardize_hidden_values, get_discrimanting_neurons
from util.ordinary import get_feature_module_name
from util.sampling_util import sample_near_adversarial, sample

model_name = 'h5/model_b1.h5'

firstModel = load_model(model_name)
concernIdentifier = ConcernIdentification()

targetFeature, targetValues = to_criteria()

print("Start Time:" + datetime.now().strftime("%H:%M:%S"))

# summaryOut = open(os.path.join("result.csv"), "w")
# summaryOut.write('Feature,Feature Value,Numper of Training Sample, Number of Testing Sample,Feature Value Important '
#                  'Set,Feature Value Unimportant Set\n')
feature_neurons = {}
aLoss = 0.0
bLoss = 0.0
total = 0
for feature in targetFeature:
    if feature not in feature_neurons:
        feature_neurons[feature] = {}
    for featureValue in targetValues[feature]:
        model = load_model(model_name)

        sx, sy, bx = sample(column_name=feature, value=featureValue, data_acquisition=get_train_df)
        # tx, _, _ = sample(column_name=feature, value=negate_condition(featureValue), num_sample=len(sx),
        #                   data_acquisition=get_train_df)
        # bx = np.concatenate((bx, tx), axis=0)

        if len(sx > 2000):
            sx = sx[0:2000]
            sy = sy[0:2000]
            bx = bx[0:2000]

        print('Testing removal of feature: ' + str(feature) + ', and value: ' + str(featureValue))
        print('Number of sample: ', len(sx))
        concernPos = initModularLayers(model.layers)
        concernNeg = initModularLayers(model.layers)

        hidden_values_pos = {}
        all_hidden = set()
        for x in sx:
            x_t = x

            for layerNo, _layer in enumerate(concernPos):
                x_t = concernIdentifier.propagateThroughLayer(_layer, x_t, apply_activation=True)

                if _layer.type == LayerType.Dense and not _layer.last_layer:
                    if layerNo not in hidden_values_pos:
                        hidden_values_pos[layerNo] = []
                    hidden_values_pos[layerNo].append(x_t)
                    for nodeNum in range(_layer.num_node):
                        all_hidden.add(x_t[nodeNum])

        if len(sx) < 10:
            print('No examples found')
            continue
        hidden_values_neg = {}
        for x in bx:
            x_t = x

            for layerNo, _layer in enumerate(concernNeg):
                x_t = concernIdentifier.propagateThroughLayer(_layer, x_t, apply_activation=True)

                if _layer.type == LayerType.Dense and not _layer.last_layer:
                    if layerNo not in hidden_values_neg:
                        hidden_values_neg[layerNo] = []
                    hidden_values_neg[layerNo].append(x_t)
                    for nodeNum in range(_layer.num_node):
                        all_hidden.add(x_t[nodeNum])

        # hidden_values_pos, hidden_values_neg = standardize_hidden_values(np.asarray(list(all_hidden)),
        #                                                                  hidden_values_pos, hidden_values_neg)
        discrimanting = get_discrimanting_neurons(concernPos, hidden_values_pos, hidden_values_neg)

        for layerNo, _layer in enumerate(concernPos):
            if _layer.type == LayerType.Dense and not _layer.last_layer:
                calculate_percentile_of_nodes(hidden_values_pos[layerNo], concernPos[layerNo], percentile=50.0)
                calculate_percentile_of_nodes(hidden_values_neg[layerNo], concernNeg[layerNo], percentile=50.0)
                calculateActivePercentage(hidden_values_pos[layerNo], concernPos[layerNo])
                calculateActivePercentage(hidden_values_neg[layerNo], concernNeg[layerNo])

        frequent_set_pos = set()
        for layerNo, _layer in enumerate(concernPos):
            if shouldRemove(_layer):
                if _layer.type == LayerType.Dense and not _layer.last_layer:
                    _layer.DW = _layer.W
                    _layer.DB = _layer.B
                    rs = removeNeurons3(layerNo, concernPos[layerNo],
                                        concernNeg[layerNo], discrimnating_set=discrimanting)
                    frequent_set_pos = frequent_set_pos.union(rs)

        # ad_x, ad_y, b_x, b_y, _ = sample_near_adversarial(model, feature, featureValue)

        for layerNo, _layer in enumerate(concernPos):
            if _layer.last_layer:
                model.layers[layerNo].set_weights(model.layers[layerNo].get_weights())
                # getDeadNodePercent(_layer)

            elif _layer.type == LayerType.Dense:
                model.layers[layerNo].set_weights([_layer.DW, _layer.DB])
                # getDeadNodePercent(_layer)

        moduleName = 'modules/module.h5'
        model.save(moduleName)

        diff = evaluate_contrast(firstModel, load_model(moduleName), feature, featureValue,
                                 data_acquisition=get_whole_df)

        # if diff[list(diff.keys())[0]]>=diff[list(diff.keys())[1]]:
        #     print('No significant difference observed')
        #     continue
        # summaryOut.write(feature + ',' + str(featureValue) + ',' + str(len(sx)) + ',' + str(len(ad_x)) + ',' +
        #                  str(diff[list(diff.keys())[0]])
        #                  + ',' + str(diff[list(diff.keys())[1]]) + '\n')

        for k in diff.keys():
            print(k, diff[k])

        if type(diff[list(diff.keys())[0]]) is not str and type(diff[list(diff.keys())[1]]) is not str:
            aLoss += diff[list(diff.keys())[0]]
            bLoss += diff[list(diff.keys())[1]]
            total += 1
        # frequent_set_pos = sorted(frequent_set_pos, key=lambda tup: tup[2],reverse=True)
        print('Size of neuron set', len(frequent_set_pos))
        feature_neurons[feature][featureValue] = frequent_set_pos

aLoss = aLoss / total
bLoss = bLoss / total
print('Total A loss, B loss', aLoss, bLoss)
import pickle

datasetPath = os.path.dirname(os.path.realpath(__file__))

with open(get_feature_module_name(model_name, datasetPath), 'wb') as handle:
    pickle.dump(feature_neurons, handle, protocol=pickle.HIGHEST_PROTOCOL)
