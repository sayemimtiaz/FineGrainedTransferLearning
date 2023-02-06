from constants import CLASS_WISE, SHAPE, source_model_name, NUM_SOURCE_SAMPLE, source_dataset, pretrained_architecures
from core import getSourceModel, sampleSourceData
from util.cnn_util import observe_feature
from util.ordinary import dump_as_pickle, get_transfer_filter_name


def gen_source_dist(model_name=None):
    if model_name is None:
        model_name = source_model_name

    model = getSourceModel(model_name)

    pos_x = sampleSourceData(num_sample=NUM_SOURCE_SAMPLE)

    obs = {'class': {}, 'numFilter': None}

    obs['class'][0], obs['numFilter'] = observe_feature(model, pos_x)

    dump_as_pickle(obs, get_transfer_filter_name(model_name, source_dataset, NUM_SOURCE_SAMPLE))


# gen_source_dist()

for pa in pretrained_architecures:
    gen_source_dist(pa)
