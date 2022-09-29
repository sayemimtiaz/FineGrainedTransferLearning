from scipy import spatial
from scipy.spatial import distance
from scipy.stats import stats
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import MeanShift, estimate_bandwidth

from data_type.enums import LayerType
from util.common import get_mean_minus_outliers, remove_outliers


def standardize_hidden_values(all_hidden, hidden_pos, hidden_neg):
    scaler = MinMaxScaler()
    all_hidden = all_hidden.reshape(-1, 1)
    scaler.fit(all_hidden)
    transformed = scaler.transform(all_hidden)
    st_map = {}
    for i in range(len(all_hidden)):
        st_map[all_hidden[i][0]] = transformed[i][0]

    for layer in hidden_pos:
        for x in range(len(hidden_pos[layer])):
            for nodeNum in range(len(hidden_pos[layer][x])):
                hidden_pos[layer][x][nodeNum] = st_map[hidden_pos[layer][x][nodeNum]]

    for layer in hidden_neg:
        for x in range(len(hidden_neg[layer])):
            for nodeNum in range(len(hidden_neg[layer][x])):
                hidden_neg[layer][x][nodeNum] = st_map[hidden_neg[layer][x][nodeNum]]

    return hidden_pos, hidden_neg


def get_observations_for_neuron(hidden, layer, nodeNum, onlyActive=True):
    obs = []
    for x in range(len(hidden[layer])):
        if onlyActive and hidden[layer][x][nodeNum] > 0.0:
            obs.append(hidden[layer][x][nodeNum])
        if not onlyActive:
            obs.append(hidden[layer][x][nodeNum])
    return obs


def get_discrimanting_neurons(concernPos, hidden_pos, hidden_neg, alpha=0.01, min_cluster_size=30):
    discrimanting = {}
    for layerNo, _layer in enumerate(concernPos):
        if _layer.type == LayerType.Dense and not _layer.last_layer:
            for nodeNum in range(_layer.num_node):
                pos_obs = get_observations_for_neuron(hidden_pos, layerNo, nodeNum, onlyActive=True)
                neg_obs = get_observations_for_neuron(hidden_neg, layerNo, nodeNum, onlyActive=True)

                pos_obs_all = get_observations_for_neuron(hidden_pos, layerNo, nodeNum, onlyActive=False)
                neg_obs_all = get_observations_for_neuron(hidden_neg, layerNo, nodeNum, onlyActive=False)

                if len(pos_obs) == 0:
                    # if layerNo not in discrimanting:
                    #     discrimanting[layerNo] = {}
                    #
                    # discrimanting[layerNo][nodeNum] = [0]
                    continue

                if len(pos_obs) > 0 and len(neg_obs) == 0:
                    if layerNo not in discrimanting:
                        discrimanting[layerNo] = {}

                    discrimanting[layerNo][nodeNum] = [np.mean(np.asarray(pos_obs))]
                    continue

                # pos_obs = np.asarray(pos_obs)
                # neg_obs = np.asarray(neg_obs)
                # # dis_sim = distance.jensenshannon(pos_obs, neg_obs)
                # sim = 1-distance.cosine(pos_obs, neg_obs)
                # if sim > 0.5:
                #     continue
                # discrimanting.add((layerNo, nodeNum))

                # if not isSameDistribution(pos_obs, neg_obs, alpha=0.01):
                #     if layerNo not in discrimanting:
                #         discrimanting[layerNo] = {}
                #
                #     discrimanting[layerNo][nodeNum] = 1 - getPValue(pos_obs, neg_obs)

                if layerNo not in discrimanting:
                    discrimanting[layerNo] = {}
                if not isSameDistribution(pos_obs_all, neg_obs_all, alpha=alpha):

                    pos_clusters = getClustersWithOutliers(pos_obs, min_samples=min_cluster_size, quantile=0.1)
                    neg_clusters = getClustersWithOutliers(neg_obs, min_samples=min_cluster_size, quantile=0.1)
                    for pc in pos_clusters:
                        sameFlag = False
                        for nc in neg_clusters:
                            if isSameDistribution(pc, nc, alpha=alpha):
                                sameFlag = True
                                break
                        if not sameFlag:
                            if nodeNum not in discrimanting[layerNo]:
                                discrimanting[layerNo][nodeNum] = []
                            discrimanting[layerNo][nodeNum].append(np.mean(np.asarray(pc)))

    # print('Discrimnating set length: ', len(discrimanting))
    return discrimanting


def standardize(hidden_pos, hidden_neg):
    scaler = MinMaxScaler()
    # data = np.concatenate((hidden_pos.reshape(-1,1),hidden_neg.reshape(-1,1)), axis=0)

    pos_shape = hidden_pos.shape
    neg_shape = hidden_neg.shape
    scaler.fit(hidden_neg.reshape(-1, 1))
    scaler.fit(hidden_pos.reshape(-1, 1))

    hidden_pos = scaler.transform(hidden_pos.reshape(-1, 1)).reshape(pos_shape)
    hidden_neg = scaler.transform(hidden_neg.reshape(-1, 1)).reshape(neg_shape)

    return hidden_pos, hidden_neg


def get_n_component(data, ratio=0.95):
    pca = PCA(n_components=ratio)
    pca.fit(data)
    return pca.n_components_


def get_pca_transformed_data(data, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(data)
    return pca.transform(data)


def apply_pca(hidden_pos, hidden_neg):
    hidden_pos, hidden_neg = standardize(hidden_pos, hidden_neg)
    pos_n = get_n_component(hidden_pos)
    neg_n = get_n_component(hidden_neg)
    nc = max(pos_n, neg_n)
    # nc = min(hidden_pos.shape[1], hidden_neg.shape[1])
    nc = 3

    hidden_pos = get_pca_transformed_data(hidden_pos, nc)
    hidden_neg = get_pca_transformed_data(hidden_neg, nc)

    return hidden_pos, hidden_neg


# non-parameteric  2-sided Kolmogorov-Smirnov test
def isSameDistribution(a, b, alpha=0.01):
    if getPValue(a, b) < alpha:
        return False  # different distribution
    return True


def getPValue(a, b):
    res = stats.kstest(a, b)

    return res.pvalue


def getClusters(x, quantile):
    X = np.array(list(zip(x, np.zeros(len(x)))), dtype=float)
    bandwidth = estimate_bandwidth(X, quantile=quantile)
    if bandwidth <= 0.0:
        return [x]

    # print(quantile)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    clusters = []
    for k in range(n_clusters_):
        my_members = labels == k
        clusters.append(X[my_members, 0])
        # print("cluster {0}: {1}".format(k, X[my_members, 0]))
    return clusters


def getClustersWithOutliers(x, quantile=0.3, min_samples=30):
    y = remove_outliers(np.asarray(x))
    # outliers = list(set(x) - set(y.tolist()))
    # x = y.tolist()
    clusters = getClusters(x, quantile)
    # clusters.append(outliers)

    if min_samples == -1:
        return clusters

    filteredClusters = []
    for c in clusters:
        if len(c) >= min_samples:
            filteredClusters.append(c)
    return filteredClusters

# a = np.array([[1, 100, 20, 2, 656, 78, 122], [3, 30, 40, 60, 89, 1000, 488]])
# b = np.array([[-6000, -8090, -20, -90], [45, 12, 100, 60]])
#
# a, b = standardize(a, b)
#
# print(isSameDistribution([1,2,3],[]))
# print(apply_pca(a, b))
# print(distance.jensenshannon([1,2,3], [1,2]))


# x = [1, 1, 5, 6, 1, 5, 10, 22, 23, 23, 50, 51, 51, 52, 100, 112, 130, 500, 512, 600, 12000, 12230]

# print(getClustersWithOutliers(x))
