from scipy import spatial
from scipy.spatial import distance
from scipy.stats import stats
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import MeanShift, estimate_bandwidth


def standardize(data):
    scaler = MinMaxScaler()

    pos_shape = data.shape
    scaler.fit(data.reshape(-1, 1))

    data = scaler.transform(data.reshape(-1, 1)).reshape(pos_shape)

    return data


def get_n_component(data, ratio):
    pca = PCA(n_components=ratio)
    pca.fit(data)
    return pca.n_components_


def get_pca_transformed_data(data, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(data)
    return pca.transform(data)


def apply_pca(data, ratio=0.95):
    data=np.reshape(data, (data.shape[0],data.shape[1]*data.shape[2]*data.shape[3]))
    data = standardize(data)
    print(data)
    nc = get_n_component(data, ratio=ratio)

    data = get_pca_transformed_data(data, nc)

    return data


# a = np.array([[[[1, 100, 20, 2, 656, 78, 122], [3, 30, 40, 60, 89, 1000, 488]]],
#               [[[1, 100, 20, 2, 656, 78, 122], [3, 30, 40, 60, 89, 1000, 488]]]])
# # a=np.reshape(a, (2,1*2*7))
# print(a.shape)
# b=apply_pca(a)
# print(b.shape)
# # print(b)
