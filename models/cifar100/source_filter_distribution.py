from models.cifar100 import getSourceModel, getSourceData, MODE
from modularization.concern.cnn_util import observe_resnet
from util.ordinary import dump_as_pickle, get_transfer_filter_name

numSample = 1000
model = getSourceModel()

tiny = getSourceData()
tiny_classes = tiny.getClasses()

obs = {'class': {}, 'numFilter': None}
for tiny_c in tiny_classes:
    # if tiny_c != 9:  # automobile
    #     continue
    if tiny_c != 4:  # deer
        continue
    pos_x, _, _, _, _ = tiny.sample(sample_only_classes=[tiny_c], num_sample=numSample)

    obs['class'][tiny_c], obs['numFilter'] = observe_resnet(model, pos_x)

dump_as_pickle(obs, get_transfer_filter_name(mode=MODE, end='source'))
