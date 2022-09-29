import math
import os

import numpy as np
import scipy
from keras.layers import Dense, TimeDistributed, RepeatVector
from keras.models import load_model
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from data_type.constants import Constants
from data_type.enums import ActivationType, LayerType, getLayerType
from data_type.modular_layer_type import ModularLayer


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference
    # return scipy.special.softmax(x)


def sigmoid(x):
    return scipy.special.expit(x)


def tanh(x):
    return np.tanh(x)


def relu(x_t):
    x_t[x_t < 0] = 0
    return x_t


def initModularLayers(layers, timestep=None):
    myLayers = []
    first = True
    for serial, _layer in enumerate(layers):
        l = ModularLayer(_layer, timestep=timestep)

        if l.type == LayerType.TimeDistributed:
            if myLayers[serial - 1].timestep is None:
                l.initTimeDistributedWeights(timestep)
            else:
                l.initTimeDistributedWeights(myLayers[serial - 1].timestep)
            l.setHiddenState()

        l.first_layer = first
        l.layer_serial = serial

        if not first:
            myLayers[len(myLayers) - 1].next_layer = l

        if len(layers) == serial + 1:
            l.last_layer = True
            l.DW = l.W
            l.DB = l.B

        myLayers.append(l)
        first = False

    return myLayers


def isNodeActive(layer, nodeNum, threshold=None, timestep=None):
    hs = 0
    if timestep is not None:
        hs = layer.hidden_state[timestep][0, nodeNum]
    else:
        hs = layer.hidden_state[0, nodeNum]

    if layer.activation == ActivationType.Relu:
        return not (hs <= 0)
    if layer.activation == ActivationType.Sigmoid:
        return not (hs < 10)
    if layer.activation == ActivationType.Tanh:
        return not (-0.1 <= hs <= 0.1)


def getDeadNodePercent(layer, timestep=None):
    W = layer.DW
    if timestep is not None:
        W = layer.DW[timestep]

    totalDeadPercdnt = 0.0
    print('Dead Node in ' + str(layer.type) + ':')
    if type(W) == list:
        for ts, w in enumerate(W):
            # print('Timestep: '+str(ts))
            alive = 0
            dead = 0
            for r in range(w.shape[0]):
                for c in range(w.shape[1]):
                    if w[r][c] == 0.0:
                        dead += 1
                    else:
                        alive += 1
            p = 0.0
            if alive + dead != 0:
                p = (dead / (alive + dead)) * 100.0
            totalDeadPercdnt += p
        avgDeadPercent = totalDeadPercdnt / (len(W) + 1)
        print('Average dead node: ' + str(avgDeadPercent) + '%')

    else:
        alive = 0
        dead = 0
        for r in range(W.shape[0]):
            for c in range(W.shape[1]):
                if W[r][c] == 0.0:
                    dead += 1
                else:
                    alive += 1

        p = 0.0
        if alive + dead != 0:
            p = (dead / (alive + dead)) * 100.0
        print('Dead node: ' + str(p) + '%')


def areArraysSame(a, b):
    for i in range(len(a)):
        if a[i].argmax() != b[i].argmax():
            print(a[i].argmax(), b[i].argmax())
            return False
    return True


def shouldRemove(_layer):
    if _layer.last_layer:
        return False
    if _layer.type == LayerType.Embedding \
            or _layer.type == LayerType.RepeatVector \
            or _layer.type == LayerType.Flatten or _layer.type == LayerType.Dropout:
        return False
    return True


def isIntrinsicallyTrainableLayer(_layer):
    if _layer.type == LayerType.Embedding \
            or _layer.type == LayerType.RepeatVector \
            or _layer.type == LayerType.Flatten \
            or _layer.type == LayerType.Dropout \
            or _layer.type == LayerType.Input \
            or _layer.type == LayerType.Activation:
        return False
    return True


def repopulateModularWeights(modularLayers, module_dir, moduleNo, only_decoder=False):
    from modularization.concern.concern_identification_encoder_decoder import ConcernIdentificationEnDe
    # module=module_dir
    module = load_model(
        os.path.join(module_dir, 'module' + str(moduleNo) + '.h5'))
    for layerNo, _layer in enumerate(modularLayers):
        if _layer.type == LayerType.RepeatVector \
                or _layer.type == LayerType.Flatten \
                or _layer.type == LayerType.Input \
                or _layer.type == LayerType.Dropout:
            continue
        if _layer.type == LayerType.RNN:
            if only_decoder and not ConcernIdentificationEnDe.is_decoder_layer(_layer):
                modularLayers[layerNo].DW, modularLayers[layerNo].DU, \
                modularLayers[layerNo].DB = module.layers[layerNo].get_weights()

            elif Constants.UNROLL_RNN:
                for ts in range(_layer.timestep):
                    tempModel = load_model(
                        os.path.join(module_dir, 'module' + str(moduleNo) + '_layer' + str(layerNo) + '_timestep' + str(
                            ts) + '.h5'))
                    modularLayers[layerNo].DW[ts], modularLayers[layerNo].DU[ts], \
                    modularLayers[layerNo].DB[ts] = tempModel.layers[layerNo].get_weights()
            else:
                modularLayers[layerNo].DW, modularLayers[layerNo].DU, \
                modularLayers[layerNo].DB = module.layers[layerNo].get_weights()

        elif _layer.type == LayerType.Embedding:
            modularLayers[layerNo].DW = module.layers[layerNo].get_weights()[0]
        else:
            modularLayers[layerNo].DW, \
            modularLayers[layerNo].DB = module.layers[layerNo].get_weights()


def calculate_percentile_of_nodes(observed_values, refLayer, normalize=True, percentile=100.0):
    for nodeNum in range(refLayer.num_node):
        tl = []
        for o in observed_values:
            tl.append(math.fabs(o[nodeNum]))
        tl = np.asarray(tl).flatten()
        # median = np.percentile(tl, percentile)
        # median = get_mean_minus_outliersn_minus_outliers(tl)
        median = np.mean(tl)
        refLayer.median_node_val[:, nodeNum] = median

    if normalize:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(refLayer.median_node_val.reshape(-1, 1))
        scaled = scaled.reshape(1, -1)
        refLayer.median_node_val = scaled


def remove_outliers(data, m=2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / mdev if mdev else 0.
    d = data[s < m]

    return d

def get_mean_minus_outliers(data, m=2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / mdev if mdev else 0.
    d = data[s < m]
    from scipy import stats

    return np.mean(d)
    # return stats.mode(d.flatten())[0][0]


def minWeightsToRemoveToTurnInactive(layer, x, currentNode):
    d = {}
    # total = layer.B[currentNode]
    total = 0.0
    for prevNode in range(layer.W.shape[0]):
        d[prevNode] = layer.W[prevNode, currentNode] * x[prevNode]
        total += d[prevNode]

    d = {k: v for k, v in
         sorted(d.items(), key=lambda item: item[1], reverse=True)}

    edgesToRemove = []
    for prevNode in d.keys():
        total -= d[prevNode]
        edgesToRemove.append((prevNode, currentNode))
        if total <= 0.0:
            break
    return edgesToRemove


def calculateActivePercentage(observed_values, refLayer):
    for nodeNum in range(refLayer.num_node):
        ac = 0
        for o in observed_values:
            if o[nodeNum] > 0.0:
                ac += 1

        median = ac / len(observed_values)
        refLayer.active_count[:, nodeNum] = median


def filter_dataframe(df, column_name=None, value=None):
    if type(value) == tuple:
        for c in value:
            df = filter_dataframe(df, column_name, c)
        return df

    if type(value) == str:
        if value[0] == '<' and value[1] == '=':
            v = float(value[2:])
            df = df[df[column_name] < v]
        elif value[0] == '<':
            v = float(value[1:])
            df = df[df[column_name] < v]
        elif value[0] == '>' and value[1] == '=':
            v = float(value[2:])
            df = df[df[column_name] >= v]
        elif value[0] == '>':
            v = float(value[1:])
            df = df[df[column_name] > v]
        elif value[0] == '=':
            v = float(value[1:])
            df = df[df[column_name] == v]
        elif value[0] == '!':
            v = float(value[1:])
            df = df[df[column_name] != v]
    else:
        df = df[df[column_name] == value]
    return df


def filter_dataframe_by_multiple_condition(df, condtions):
    for (col, cond) in condtions:
        df = filter_dataframe(df, col, cond)

    return df


def modify_dataframe(df, column_name=None, column_value=None, value=None):
    if type(column_value) == tuple:
        print(column_value)
    if type(column_value) == str:
        if column_value[0] == '<' and column_value[1] == '=':
            v = float(column_value[2:])
            df.loc[df[column_name] < v, column_name] = value
        elif column_value[0] == '<':
            v = float(column_value[1:])
            df.loc[df[column_name] < v, column_name] = value
        elif column_value[0] == '>' and column_value[1] == '=':
            v = float(column_value[2:])
            df.loc[df[column_name] >= v, column_name] = value
        elif column_value[0] == '>':
            v = float(column_value[1:])
            df.loc[df[column_name] > v, column_name] = value
        elif column_value[0] == '=':
            v = float(column_value[1:])
            df.loc[df[column_name] == v, column_name] = value
        elif column_value[0] == '!':
            v = float(column_value[1:])
            df.loc[df[column_name] != v, column_name] = value
    else:
        df.loc[df[column_name] == column_value, column_name] = value
    return df


def negate_condition(value):
    if type(value) == str:
        if value[0] == '<' and value[1] == '=':
            return '>' + value[2:]
        elif value[0] == '<':
            v = float(value[1:])
            return '>=' + value[1:]
        elif value[0] == '>' and value[1] == '=':
            return '<' + value[2:]
        elif value[0] == '>':
            return '<=' + value[1:]
        elif value[0] == '=':
            return '!' + value[1:]
        elif value[0] == '!':
            return '=' + value[1:]
    else:
        return '!' + str(value)
