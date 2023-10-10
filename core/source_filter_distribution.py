from constants import CLASS_WISE, SHAPE, source_model_name, NUM_SOURCE_SAMPLE, source_dataset, pretrained_architecures
from core import getSourceModel, sampleSourceData
from util.cnn_util import observe_feature
from util.ordinary import dump_as_pickle, get_transfer_filter_name


def gen_source_dist(pos_x, model_name=None):
    if model_name is None:
        model_name = source_model_name
    print('Generating source filter distribution for: ',model_name)

    model = getSourceModel(model_name)

    obs = {'class': {}, 'numFilter': observe_feature(model, pos_x)[1]}

    dump_as_pickle(obs, get_transfer_filter_name(model_name, source_dataset, NUM_SOURCE_SAMPLE))


# gen_source_dist()

pos_x = sampleSourceData(num_sample=NUM_SOURCE_SAMPLE)
print('Sampled ',source_dataset)
print('Sample size: ',len(pos_x))

for pa in pretrained_architecures:
    gen_source_dist(pos_x,model_name=pa)
