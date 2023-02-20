from constants import target_dataset,source_model_name
from core import getSourceModel
from util.cnn_util import observe_feature
from util.ordinary import dump_as_pickle, get_transfer_filter_name



def calculateTargetDistribution(pos_x, target_ds=None, parent_model=None):
    if target_ds is None:
        target_ds = target_dataset
    if parent_model is None:
        parent_model = source_model_name
    
    model = getSourceModel(parent_model)

    print('Num target samples: ', len(pos_x))

    obs, _ = observe_feature(model, pos_x)

    dump_as_pickle(obs, get_transfer_filter_name(target_ds))

    return obs
