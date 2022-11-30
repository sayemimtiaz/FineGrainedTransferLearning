from scipy import spatial
from scipy.spatial import distance
from scipy.stats import stats
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import MeanShift, estimate_bandwidth

from data_type.enums import LayerType
from util.common import get_mean_minus_outliers, remove_outliers


# non-parameteric  2-sided Kolmogorov-Smirnov test
def isSameDistributionZero(a, b):
    if getPValue(a, b) > 0.00:
        return True  # different distribution
    return False


# non-parameteric  2-sided Kolmogorov-Smirnov test
def isSameDistribution(a, b, alpha=0.01):
    if getPValue(a, b) < alpha:
        return False  # different distribution
    return True


def getPValue(a, b):
    res = stats.kstest(a, b)

    return res.pvalue


def jenShanDistance(a, b):
    return distance.jensenshannon(a, b)

# a = np.array([[1, 100, 20, 2, 656, 78, 122], [3, 30, 40, 60, 89, 1000, 488]])
# b = np.array([[-6000, -8090, -20, -90], [45, 12, 100, 60]])
#
# a, b = standardize(a, b)
#
# print(isSameDistribution([1,2,3],[]))
# print(apply_pca(a, b))
# print(distance.jensenshannon([1,2,3], [1,2,6]))


# x = [1, 1, 5, 6, 1, 5, 10, 22, 23, 23, 50, 51, 51, 52, 100, 112, 130, 500, 512, 600, 12000, 12230]

# print(getClustersWithOutliers(x))
