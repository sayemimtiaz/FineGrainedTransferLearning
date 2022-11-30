from data_util.imagenet_util import TinyImageNet
from models.imagenet import getSourceModel, source_model_name
from modularization.concern.cnn_util import observe_resnet
from util.ordinary import dump_as_pickle, get_transfer_filter_name

MODE = 'val'
numSample = 1
model = getSourceModel()

tiny = TinyImageNet()
tiny_classes = tiny.getClasses()

obs = {'class': {}, 'numFilter': None}
for tiny_c in tiny_classes:
    pos_x, _ = tiny.sampleTinyImageNet(sample_only_classes=[tiny_c], num_sample=numSample)

    obs['class'][tiny_c], obs['class']['numFilter'] = observe_resnet(model, pos_x)

dump_as_pickle(obs, get_transfer_filter_name(source_model_name, MODE, end='source'))
