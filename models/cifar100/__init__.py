from keras import Model
from keras.saving.save import load_model

from data_util.cifar_specific import Cifar10, sampleCifar100Fine
from data_util.data_mixer import mixMnistCifar10

target_dataset = 'cifar100'
# source_model_name = 'h5/source_model_mixed_vehicle.h5'
source_model_name = 'h5/source_model_mixed_deer.h5'

MODE = 'val'


# conv5_block3_out layer before avg_pool
def getSourceModel(out_layer_name='avg_pool'):
    global source_model_name

    base_model = load_model(source_model_name)

    model = Model(inputs=base_model.input, outputs=base_model.get_layer(out_layer_name).output)

    return model


def getSourceData():
    takeFromCifar = []
    takeFromMnist = []
    if 'vehicle' in source_model_name:
        takeFromCifar = [1, 9]
        takeFromMnist = [0, 2, 3, 4, 5, 6, 7, 8]
    if 'deer' in source_model_name:
        takeFromCifar = [4]
        takeFromMnist = [0, 1, 2, 3, 5, 6, 7, 8, 9]

    x_train, y_train, x_test, y_test, num_classes, _ = mixMnistCifar10(takeFromMnist=takeFromMnist,
                                                                       takeFromCifar=takeFromCifar,
                                                                       one_hot=False)

    obj = Cifar10(load=False)
    obj.data = x_train, y_train, x_test, y_test, num_classes

    return obj


def getTargetData(sample_rate=1.0, seed=None, one_hot=True, gray=True):
    numSample = getTargetSampleSize(sample_rate)
    superclasses=[]
    if target_dataset == 'cifar100':
        if 'vehicle' in source_model_name:
            superclasses = ['vehicles 1']
        if 'deer' in source_model_name:
            superclasses = ['large carnivores']

        return sampleCifar100Fine(superclasses=superclasses, num_sample=numSample,
                                  seed=seed, gray=gray,
                                  one_hot=one_hot, train=True
                                  )


def getTargetNumClass():
    if 'cifar100' in target_dataset:
        return 5


def getTargetSampleSize(rate):
    if target_dataset == 'cifar100':
        return int(500 * getTargetNumClass() * rate)
