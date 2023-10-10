import os

from constants import target_dataset, source_model_name
from core import getSourceModel
from util.cnn_util import observe_feature
from util.ordinary import dump_as_pickle, get_transfer_filter_name, load_pickle_file


def calculateTargetDistribution(pos_x, target_ds=None, parent_model=None):
    if target_ds is None:
        target_ds = target_dataset
    if parent_model is None:
        parent_model = source_model_name

    fileName=get_transfer_filter_name(parent_model, target_ds)
    if os.path.exists(fileName):
        targetRate = load_pickle_file(fileName)
        numFilter = int(targetRate['numFilter'])
        return targetRate, numFilter

    model = getSourceModel(parent_model)

    print('Num target samples: ', len(pos_x))

    obs = {'class': {}, 'numFilter': None}

    obs['class'][0], obs['numFilter'] = observe_feature(model, pos_x)

    dump_as_pickle(obs, fileName)

    return obs, obs['numFilter']
