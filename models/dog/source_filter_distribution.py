from models.imagenet import getSourceModel, source_model_name, MODE, getSourceData, SHAPE
from util.cnn_util import observe_feature
from util.ordinary import dump_as_pickle, get_transfer_filter_name

model = getSourceModel(shape=SHAPE)

pos_x = getSourceData(shape=SHAPE, gray=False, num_sample_per_class=20)

print('Num samples loaded: ', len(pos_x))

obs = {'class': {}, 'numFilter': None}

obs['class'][0], obs['numFilter'] = observe_feature(model, pos_x)

dump_as_pickle(obs, get_transfer_filter_name(mode=MODE, end=source_model_name))
