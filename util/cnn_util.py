import tensorflow as tf

# assumes Relu
import numpy as np
from data_processing.data_util import makeScalar


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


def observe_feature(model, data):
    all_stats = []
    active_count_for_filter = {}
    numFilter = None
    for x in data:
        x_t = tf.reshape(x, [-1, x.shape[0], x.shape[1], x.shape[2]])

        x_t = model.predict(x_t, verbose=0)
        # tmp = makeScalar(activeValNodeEachObs(x_t))
        tmp = activeValNodeEachObs(x_t)

        all_stats.append(tmp)
        # numFilter = len(tmp)
        numFilter = tmp.shape[2]

    # for fn in range(numFilter):
    #     ac = []
    #     for obs in all_stats:
    #         ac.append(obs[fn])
    #
    #     active_count_for_filter[fn] = ac

    for fn in range(numFilter):
        ac = []
        for obs in all_stats:
            ac.append(obs[:, :, fn].mean())

        active_count_for_filter[fn] = ac

    return active_count_for_filter, numFilter


def activeRateFilterAllObs(hidden_val, layer):
    for fn in range(layer.filters):
        data = []
        for obs in hidden_val:
            data.append(obs[fn])
        data = np.asarray(data)
        layer.active_count_for_filter[fn] = np.mean(data)


def activeRateNodeEachObs(hidden_val, layer):
    hidden_val = tf.squeeze(hidden_val)  # remove first dimension
    hidden_val = hidden_val.numpy()
    active_count_for_filter = np.zeros(hidden_val.shape)
    for fn in range(layer.filters):
        for ri in range(hidden_val.shape[0]):
            for ci in range(hidden_val.shape[1]):
                if hidden_val[ri][ci][fn] > 0:  # active node
                    active_count_for_filter[ri][ci][fn] += 1

    return active_count_for_filter


def activeRateNodeAllObs(hidden_val, layer):
    for fn in range(layer.filters):
        ac = np.zeros((hidden_val[0].shape[0], hidden_val[0].shape[1]))
        for obs in hidden_val:
            for ri in range(obs.shape[0]):
                for ci in range(obs.shape[1]):
                    ac[ri][ci] += obs[ri][ci][fn]

        for ri in range(ac.shape[0]):
            for ci in range(ac.shape[1]):
                ac[ri][ci] /= len(hidden_val)

        layer.active_count_for_filter[fn] = ac


def activeValNodeEachObs(hidden_val, layer=None):
    hidden_val = tf.squeeze(hidden_val)  # remove first dimension
    hidden_val = hidden_val.numpy()
    return hidden_val


def activeRawValNodeEachObs(hidden_val, layer):
    hidden_val = layer.raw_hidden_state
    hidden_val = tf.squeeze(hidden_val)  # remove first dimension
    hidden_val = hidden_val.numpy()
    return hidden_val


def activeValNodeAllObs(hidden_val, layer=None):
    for fn in range(layer.filters):
        ac = []
        for obs in hidden_val:
            # for ri in range(obs.shape[0]):
            #     for ci in range(obs.shape[1]):
            #         ac.append(obs[ri][ci][fn])
            ac.append(obs[:, :, fn].mean())

        layer.active_count_for_filter[fn] = ac


