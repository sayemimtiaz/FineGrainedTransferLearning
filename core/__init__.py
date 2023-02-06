import math

from keras.applications import ResNet50, InceptionV3, InceptionResNetV2, VGG16, MobileNet, Xception, DenseNet121, \
    DenseNet201
from constants import pretrained_architecures, source_model_name, SHAPE, source_dataset, target_dataset
from data_processing.bird_util import Bird
from data_processing.dog_util import Dog
from data_processing.imagenet_util import ImageNet
from data_processing.tiny_imagenet_util import TinyImageNet


# conv5_block3_out layer before avg_pool for resnet50
def getSourceModel(model_name=None):
    if model_name is None:
        model_name = source_model_name

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
    elif model_name == 'xception':
        return Xception(input_shape=SHAPE,
                        include_top=False,
                        weights='imagenet')
    elif model_name == 'densenet121':
        return DenseNet121(input_shape=SHAPE,
                           include_top=False,
                           weights='imagenet')
    elif model_name == 'densenet201':
        return DenseNet201(input_shape=SHAPE,
                           include_top=False,
                           weights='imagenet')


def sampleSourceData(num_sample=5000, model_name=None):
    if model_name is None:
        model_name = source_model_name
    if model_name in pretrained_architecures:
        obj = None
        if source_dataset == 'tiny':
            obj = TinyImageNet()
        if source_dataset == 'imagenet':
            obj = ImageNet()

        num_sample_per_class = int(math.ceil(num_sample / obj.getClasses()))

        return obj.sampleFromDir(sample_size_per_class=num_sample_per_class)

    # if 'cifar10' in source_model_name:
    #     return Cifar10(shape=shape, gray=gray)


def getTargetNumClass(target_ds=None):
    if target_ds is None:
        target_ds = target_dataset
    if target_ds == 'bird':
        return 200
    if target_ds == 'dog':
        return 120
    if 'cifar100' in target_ds:
        return 5


def getTargetDataForTraining(batch_size=128, shuffle=False, target_ds=None):
    if target_ds is None:
        target_ds = target_dataset

    if target_ds == 'dog':
        return Dog().getTrainingDogs(batch_size=batch_size, shuffle=shuffle)

    if target_ds == 'bird':
        return Bird().getTrainingBirds(batch_size=batch_size, shuffle=shuffle)

    # if target_dataset == 'cifar100':
    #     numSample = getTargetSampleSize(sample_rate)
    #     return sampleCifar100Fine(superclasses=[task], num_sample=numSample,
    #                               seed=seed, gray=gray,
    #                               one_hot=one_hot, train=True, shape=(64, 64)
    #                               )


def smapleTargetData(sample_size_per_class=20, target_ds=None):
    if target_ds is None:
        target_ds = target_dataset

    if target_ds == 'dog':
        dog = Dog()
        target_sample = dog.sampleFromDir(sample_size_per_class=sample_size_per_class, ext='jpg')
        return target_sample

    if target_ds == 'bird':
        bird = Bird()
        target_sample = bird.sampleFromDir(sample_size_per_class=sample_size_per_class, ext='jpg')
        return target_sample
