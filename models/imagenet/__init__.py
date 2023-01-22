from keras import Model
from keras.applications import ResNet50, InceptionV3, InceptionResNetV2
from keras.saving.save import load_model

from data_util.bird_util import Bird, getBirdTrainingData
from data_util.cifar_specific import Cifar10, sampleCifar100Fine
from data_util.dog_util import Dog
from data_util.imagenet_util import TinyImageNet

SHAPE=(224,224)
# target_dataset = 'bird'
target_dataset = 'dog'
# target_dataset = 'cifar100'
# source_model_name = 'resnet50'
# source_model_name = 'inceptionv3'
source_model_name = 'inceptionresnetv2'

MODE = 'val'


# conv5_block3_out layer before avg_pool for resnet50
def getSourceModel(out_layer_name='avg_pool', shape=(224, 224), channel=3):
    global source_model_name

    if source_model_name == 'resnet50':
        return ResNet50(weights='imagenet', input_shape=(shape[0], shape[1], channel), include_top=False)
    elif source_model_name == 'inceptionv3':
        return InceptionV3(input_shape=(shape[0], shape[1], 3),
                           include_top=False,
                           weights='imagenet')
    elif source_model_name == 'inceptionresnetv2':
        return InceptionResNetV2(input_shape=(shape[0], shape[1], 3),
                                 include_top=False,
                                 weights='imagenet')

    base_model = load_model(source_model_name)

    model = Model(inputs=base_model.input, outputs=base_model.get_layer(out_layer_name).output)

    return model


def getSourceData(shape=(224, 224), gray=False, num_sample_per_class=20):
    if source_model_name in ['resnet50', 'inceptionv3', 'inceptionresnetv2']:
        return TinyImageNet(shape=shape).sampleFromDir(sample_size_per_class=num_sample_per_class)
    if 'cifar10' in source_model_name:
        return Cifar10(shape=shape, gray=gray)


def getTargetNumClass():
    global target_dataset

    if target_dataset == 'bird':
        return 200
    if target_dataset == 'dog':
        return 120
    if 'cifar100' in target_dataset:
        return 5


def getTargetSampleSize(rate):
    if target_dataset == 'bird':
        return int(6000 * rate)
    if target_dataset == 'cifar100':
        return int(500 * getTargetNumClass() * rate)


def getTargetDataForTraining():
    if target_dataset == 'dog':
        return Dog(train_data=True).data

    if target_dataset == 'bird':
            return getBirdTrainingData()

    # if target_dataset == 'cifar100':
    #     numSample = getTargetSampleSize(sample_rate)
    #     return sampleCifar100Fine(superclasses=[task], num_sample=numSample,
    #                               seed=seed, gray=gray,
    #                               one_hot=one_hot, train=True, shape=(64, 64)
    #                               )
