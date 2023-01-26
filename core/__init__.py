from keras.applications import ResNet50, InceptionV3, InceptionResNetV2, VGG16, MobileNet
from constants import pretrained_architecures, source_model_name, SHAPE, source_dataset, target_dataset
from data_processing.bird_util import Bird, getBirdTrainingData
from data_processing.cifar_specific import Cifar10, sampleCifar100Fine
from data_processing.dog_util import Dog
from data_processing.tiny_imagenet_util import TinyImageNet


# conv5_block3_out layer before avg_pool for resnet50
def getSourceModel(model_name=None):
    if model_name is None:
        model_name=source_model_name

    if model_name == 'resnet50':
        return ResNet50(weights='imagenet', input_shape=SHAPE, include_top=False)
    elif model_name == 'inceptionv3':
        return InceptionV3(input_shape=SHAPE,
                           include_top=False,
                           weights='imagenet')
    elif model_name == 'inceptionresnetv2':
        return InceptionResNetV2(input_shape=SHAPE,
                                 include_top=False,
                                 weights='imagenet')
    elif model_name == 'vgg16':
        return VGG16(input_shape=SHAPE,
                     include_top=False,
                     weights='imagenet')
    elif model_name == 'inceptionv3':
        return MobileNet(input_shape=SHAPE,
                         include_top=False,
                         weights='imagenet')


def sampleSourceData(num_sample_per_class=20):
    if source_model_name in pretrained_architecures:
        if source_dataset == 'tiny':
            return TinyImageNet().sampleFromDir(sample_size_per_class=num_sample_per_class)

    # if 'cifar10' in source_model_name:
    #     return Cifar10(shape=shape, gray=gray)


def getTargetNumClass():
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


def smapleTargetData(sample_size_per_class=20):
    if target_dataset == 'dog':
        dog = Dog(shape=SHAPE, train_data=False)
        target_sample = dog.sampleFromDir(sample_size_per_class=sample_size_per_class, ext='jpg')
        return target_sample
