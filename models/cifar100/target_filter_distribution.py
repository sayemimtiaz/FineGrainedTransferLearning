from data_processing.sample_util import sample
from models.cifar100 import getSourceModel, getTargetData, MODE, target_dataset, getTargetNumClass
from modularization.concern.cnn_util import observe_resnet
from util.ordinary import dump_as_pickle, get_transfer_filter_name

model = getSourceModel()


def calculateTargetDistribution(data, numSample=1000):

    pos_x, _ = sample(data, num_sample=numSample, num_classes=getTargetNumClass())

    obs, _ = observe_resnet(model, pos_x)

    dump_as_pickle(obs, get_transfer_filter_name(mode=MODE, end=target_dataset))

    return obs


# x_train, y_train, x_test, y_test, num_classes = getTargetData(sample_rate=1.0, one_hot=False, gray=True)
#
# calculateTargetDistribution((x_train, y_train))

