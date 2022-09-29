import pickle

from scipy import spatial
import numpy as np
from data_type.constants import Constants


def get_dice_similarity(feature_set, data_set):
    sim = (2 * len(feature_set.intersection(data_set))) / (
            len(feature_set) + len(data_set))  # dice simialrity coeff
    return sim


def get_ji_similarity(feature_set, data_set):
    sim = len(feature_set.intersection(data_set)) / len(feature_set.union(data_set))  # jaccard
    return sim


def get_data_set(featureSet, hidden_val):
    data_set = set()
    for (layerNo, nodeNo, _) in featureSet:
        if hidden_val[layerNo][nodeNo] > 0:
            data_set.add((layerNo, nodeNo))
    return data_set


def custom_weighted_jaccard_similarity(featureSet, hidden_val):
    data_set = get_data_set(featureSet, hidden_val)

    neum = 0.0
    denom = 0.0
    for (layerNo, nodeNo, weight) in featureSet:
        if (layerNo, nodeNo) in data_set:
            neum += weight
        denom += weight

    return (2 * neum) / (denom + neum)
    # return neum / denom


def custom_weighted_cosine_similarity(featureSet, hidden_val):
    data_set = get_data_set(featureSet, hidden_val)

    neum = []
    denom = []
    for (layerNo, nodeNo, weight) in featureSet:
        # if (layerNo, nodeNo) in data_set:
        #     neum.append(weight)
        #     # neum.append(hidden_val[layerNo][nodeNo])
        # else:
        #     neum.append(0.0)
        # neum.append(-weight)
        neum.append(hidden_val[layerNo][nodeNo])

        if type(weight) is tuple:
            minDis = 0.0
            minW = None
            for w in weight:
                dis = abs(hidden_val[layerNo][nodeNo] - w)
                if minW is None:
                    minDis = dis
                    minW = w
                elif dis < minDis:
                    minDis = dis
                    minW = w
            denom.append(minW)
        else:
            denom.append(weight)

    a = np.asarray(neum)
    b = np.asarray(denom)
    denom.append(0.0)
    neum.append(len(a[a == 0]))
    # if len(neum[neum == 0]) == len(neum):
    #     return 0.0

    sim = 1 - spatial.distance.cosine(neum, denom)
    return sim


def find_local_feature(hidden_val, feature_neurons, similarity_threshold=0.9):
    active_features = []
    inactive_features = []
    all_features=[]
    for feature in feature_neurons.keys():
        maxSim = 0.0
        maxFeatureValues = []
        maxFeatureValue=0.0
        featureIgnored = True
        for featureValue in feature_neurons[feature].keys():
            featureSet = feature_neurons[feature][featureValue]

            if len(featureSet) == 0:
                continue
            featureIgnored = False

            fL = []
            dL = []
            data_set = set()
            feature_set = set()
            for (layerNo, nodeNo, _) in featureSet:
                fL.append(1)
                feature_set.add((layerNo, nodeNo))
                if hidden_val[layerNo][nodeNo] > 0:
                    data_set.add((layerNo, nodeNo))
                    dL.append(1)
                else:
                    dL.append(0)

            # sim = len(feature_set.intersection(data_set)) / len(feature_set.union(data_set))  # jaccard
            # sim = get_dice_similarity(feature_set, data_set)  # dice simialrity coeff
            # sim = 1 - spatial.distance.hamming(fL, dL)
            # sim = custom_weighted_jaccard_similarity(featureSet, hidden_val)
            sim = custom_weighted_cosine_similarity(featureSet, hidden_val)

            # print(sim)

            if sim == maxSim:
                # maxFeatureValue = featureValue
                maxFeatureValues.append(featureValue)
            elif sim > maxSim:
                maxSim = sim
                # maxFeatureValue = featureValue
                maxFeatureValues = [featureValue]

        for f in maxFeatureValues:
            all_features.append((feature, f, maxSim))

        activeFeatureFlag = set()
        if maxSim >= similarity_threshold:
            # important_features.append((feature, maxFeatureValue, maxSim))
            # for featureValue1 in feature_neurons[feature].keys():
            #     if are_features_similar(feature, maxFeatureValue, feature, featureValue1, feature_neurons):
            #         active_features.append((feature, featureValue1, maxSim))
            #         activeFeatureFlag.add(featureValue1)

            for f in maxFeatureValues:
                active_features.append((feature, f, maxSim))
                activeFeatureFlag.add(f)

        elif not featureIgnored:
            for featureValue in feature_neurons[feature].keys():
                if featureValue not in activeFeatureFlag:
                    inactive_features.append((feature, featureValue, 0.0))
    active_features.sort(key=lambda a: a[2], reverse=True)
    all_features.sort(key=lambda a: a[2], reverse=True)

    return active_features, inactive_features,all_features


def is_all_inactive_feature(rule):
    count = 0
    for r in rule:
        if r.startswith('-'):
            count += 1

    return (count == len(rule))


def are_features_similar(feature1, featureValue1, feature2, featureValue2, feature_neurons,
                         thres=0.9):
    featureSet1 = feature_neurons[feature1][featureValue1]

    featureSet2 = feature_neurons[feature2][featureValue2]

    if len(featureSet1) == 0 or len(featureSet2) == 0:
        return False
    neum = []
    denom = []

    all_nodes = set()
    f_map1 = {}
    for (layerNo, nodeNo, weight) in featureSet1:
        if layerNo not in f_map1:
            f_map1[layerNo] = {}
        f_map1[layerNo][nodeNo] = weight
        all_nodes.add((layerNo, nodeNo))

    f_map2 = {}
    for (layerNo, nodeNo, weight) in featureSet2:
        if layerNo not in f_map2:
            f_map2[layerNo] = {}
        f_map2[layerNo][nodeNo] = weight
        all_nodes.add((layerNo, nodeNo))

    dZero = 0
    nZero = 0
    for (layerNo, nodeNo) in all_nodes:
        if layerNo in f_map1 and nodeNo in f_map1[layerNo]:

            if layerNo in f_map2 and nodeNo in f_map2[layerNo]:
                minW = None
                minW1 = None
                minW2 = None
                for w1 in f_map1[layerNo][nodeNo]:
                    for w2 in f_map2[layerNo][nodeNo]:
                        if minW is None:
                            minW = abs(w1 - w2)
                            minW1 = w1
                            minW2 = w2
                        elif abs(w1 - w2) < minW:
                            minW = abs(w1 - w2)
                            minW1 = w1
                            minW2 = w2
                neum.append(minW1)
                denom.append(minW2)
            else:
                neum.append(max(f_map1[layerNo][nodeNo]))
                denom.append(0)
                dZero += 1
        else:
            denom.append(max(f_map2[layerNo][nodeNo]))
            neum.append(0)
            nZero += 1

    neum.append(nZero)
    denom.append(dZero)

    sim = 1 - spatial.distance.cosine(neum, denom)

    if sim >= thres:
        return True
    return False


def find_similar_features(feature_neurons, similarity_threshold=0.9, set_comparison=True):
    similar_features = []
    vis = []
    for feature1 in feature_neurons.keys():
        for featureValue1 in feature_neurons[feature1].keys():
            featureSet1 = feature_neurons[feature1][featureValue1]

            for feature2 in feature_neurons.keys():
                for featureValue2 in feature_neurons[feature2].keys():

                    if feature1 == feature2 and featureValue1 == featureValue2:
                        continue

                    vS = set(sorted([featureValue1, featureValue2, feature1, feature2]))
                    if vS in vis:
                        continue
                    vis.append(vS)

                    featureSet2 = feature_neurons[feature2][featureValue2]

                    # if len(featureSet1) == 0 and len(featureSet2) == 0:
                    #     similar_features.append((feature1, (featureValue1, featureValue2), 1.0))
                    #     continue

                    if set_comparison:
                        neum = set()
                        denom = set()
                    else:
                        neum = []
                        denom = []
                    for (layerNo, nodeNo, weight) in featureSet1:
                        if set_comparison:
                            neum.add((layerNo, nodeNo))
                        else:
                            neum.append(weight)

                    for (layerNo, nodeNo, weight) in featureSet2:
                        if not set_comparison:
                            denom.append(weight)
                        else:
                            denom.add((layerNo, nodeNo))

                    if not set_comparison:
                        sim = 1 - spatial.distance.cosine(neum, denom)
                    else:
                        sim = (2 * len(neum.intersection(denom))) / (
                                len(neum) + len(denom))  # dice simialrity coeff

                    if sim >= similarity_threshold:
                        similar_features.append(((feature1, featureValue1), (feature2,
                                                                             featureValue2), sim))
                        continue

    similar_features = sorted(similar_features, key=lambda tup: tup[2], reverse=True)

    return similar_features

# model_name='model_b4'
# with open('../models/adult/modules/feature_neurons_' + model_name+'_'+
#           str(Constants.MODE) + '.pickle', 'rb') as handle:
#     feature_neurons = pickle.load(handle)
#
# print(find_similar_features(feature_neurons, similarity_threshold=0.0,set_comparison=True))
