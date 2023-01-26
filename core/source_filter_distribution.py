from constants import CLASS_WISE, SHAPE, source_sample_size, source_model_name
from core import getSourceModel, sampleSourceData
from util.cnn_util import observe_feature
from util.ordinary import dump_as_pickle, get_transfer_filter_name

model = getSourceModel()

pos_x = sampleSourceData(num_sample_per_class=1)

obs = {'class': {}, 'numFilter': None}

obs['class'][0], obs['numFilter'] = observe_feature(model, pos_x)

dump_as_pickle(obs, get_transfer_filter_name(source_model_name))
