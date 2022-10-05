import tensorflow as tf

# assumes Relu
from data_type.enums import LayerType
import numpy as np

from modularization.concern.concern_identification import ConcernIdentification


def activeRateFilterEachObs(hidden_val, layer):
    hidden_val = tf.squeeze(hidden_val)  # remove first dimension
    hidden_val = hidden_val.numpy()
    active_count_for_filter = {}
    for fn in range(layer.filters):
        inactive = 0
        active = 0
        for ri in range(hidden_val.shape[0]):
            for ci in range(hidden_val.shape[1]):
                if hidden_val[ri][ci][fn] <= 0:  # inactive node
                    inactive += 1
                else:
                    active += 1

        active_count_for_filter[fn] = (active / (active + inactive))
    return active_count_for_filter


def observe_cnn(concern, data, individualStatCb, allStatCb):
    concernIdentifier = ConcernIdentification()
    all_stats = {}
    for x in data:
        x_t = tf.reshape(x, [-1, x.shape[0], x.shape[1], x.shape[2]])

        for layerNo, _layer in enumerate(concern):
            x_t = concernIdentifier.propagateThroughLayer(_layer, x_t, apply_activation=True)

            if _layer.type == LayerType.Conv2D:
                active_count_for_filter = individualStatCb(x_t, _layer)
                if layerNo not in all_stats:
                    all_stats[layerNo] = []
                all_stats[layerNo].append(active_count_for_filter)

    for layerNo, _layer in enumerate(concern):
        if _layer.type == LayerType.Conv2D:
            allStatCb(all_stats[layerNo], _layer)


def activeRateFilterAllObs(hidden_val, layer):
    for fn in range(layer.filters):
        data = []
        for obs in hidden_val:
            data.append(obs[fn])
        data = np.asarray(data)
        layer.active_count_for_filter[fn] = np.mean(data)
