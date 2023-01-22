from data_util.cifar_specific import Cifar10
from data_util.imagenet_util import TinyImageNet
from models.imagenet import getSourceModel, source_model_name, MODE, getSourceData, SHAPE
from util.cnn_util import observe_resnet
from util.ordinary import dump_as_pickle, get_transfer_filter_name

CLASS_WISE = False
numSample = 4000
model = getSourceModel(shape=SHAPE)

tiny = getSourceData(shape=SHAPE, gray=False)
tiny_classes = tiny.getClasses()

obs = {'class': {}, 'numFilter': None}

if CLASS_WISE:
    for tiny_c in tiny_classes:
        pos_x, _, _, _, _ = tiny.sample(sample_only_classes=[tiny_c], num_sample=numSample)

        obs['class'][tiny_c], obs['numFilter'] = observe_resnet(model, pos_x)

else:
    pos_x, _, _, _, _ = tiny.sample(num_sample=numSample)

    obs['class'][0], obs['numFilter'] = observe_resnet(model, pos_x)

dump_as_pickle(obs, get_transfer_filter_name(mode=MODE, end=source_model_name))
