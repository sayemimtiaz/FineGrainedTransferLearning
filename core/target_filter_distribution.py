from constants import target_dataset
from core import getSourceModel
from util.cnn_util import observe_feature
from util.ordinary import dump_as_pickle, get_transfer_filter_name

model = getSourceModel()


def calculateTargetDistribution(pos_x):

    print('Num target samples: ', len(pos_x))
    obs, _ = observe_feature(model, pos_x)

    dump_as_pickle(obs, get_transfer_filter_name(target_dataset))

    return obs
