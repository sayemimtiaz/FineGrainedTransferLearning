from data_util.bird_util import Bird
from data_util.sample_util import sample
from models.imagenet import getSourceModel, source_model_name, target_dataset, MODE, getSourceData, getTargetNumClass, \
    SHAPE
from util.cnn_util import observe_resnet
from util.ordinary import dump_as_pickle, get_transfer_filter_name

model = getSourceModel(shape=SHAPE)


# def calculateTargetDistribution(bird, numSample=2000):
#     pos_x, _, _, _, _ = bird.sample(num_sample=numSample, train=True, one_hot=False,
#                                     sample_only_classes=bird.getClasses())
#
#     obs, _ = observe_resnet(model, pos_x)
#
#     dump_as_pickle(obs, get_transfer_filter_name(mode=MODE, end=target_dataset))
#
#     return obs

def calculateTargetDistribution(data, numSample=3000):

    pos_x, _ = sample(data, num_sample=numSample, num_classes=getTargetNumClass())
    print('Num target samples: ', len(pos_x))
    obs, _ = observe_resnet(model, pos_x)

    dump_as_pickle(obs, get_transfer_filter_name(mode=MODE, end=target_dataset))

    return obs

def calculateTargetDistributionAlreadySampled(pos_x):

    print('Num target samples: ', len(pos_x))
    obs, _ = observe_resnet(model, pos_x)

    dump_as_pickle(obs, get_transfer_filter_name(mode=MODE, end=target_dataset))

    return obs

# bird = Bird(gray=False)
# calculateTargetDistribution(bird)

# tiny = getSourceData(shape=(64, 64), gray=False)
# data = tiny.getClassData(tiny.getBirdIndex())
# data = tiny.getClassData(tiny.getPaperIndex())
# data = data[0:50]
# obs, _ = observe_resnet(model, data)
#
# dump_as_pickle(obs, get_transfer_filter_name(mode=MODE, end=target_dataset))