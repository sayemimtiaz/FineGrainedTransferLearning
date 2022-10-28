from typing import Optional, Any
import numpy as np
from keras import Input, Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, concatenate, Flatten, Dense
from keras.saving.save import load_model

from data_type.enums import getLayerType, LayerType
from models.cifar100.data_util_ import getCifar10MnistMixed
from util.common import initModularLayers
from util.hypothesis_testing import isSameDistribution
from util.ordinary import dump_as_pickle, load_pickle_file, get_transfer_model_name
from util.transfer_util import construct_target_model_approach_2

freezeUntil = 3
moduleNames = ['h5/source_model_partial_mixed_1.h5', 'h5/source_model_partial_mixed_2.h5']


def getInputModule(module):
    module=initModularLayers(module.layers)
    n_layer = Input(shape=module[0].original_input_shape)
    return n_layer


def getFeatureModule(modules, inputModule, stopLayer=LayerType.Flatten):
    featureModules = []
    for module in modules:
        featureModule = inputModule
        for layerNo, layer in enumerate(module.layers):
            if getLayerType(layer) == stopLayer:
                break
            if getLayerType(layer) == LayerType.Conv2D:
                featureModule = Conv2D(layer.filters, layer.kernel_size,
                                       padding=layer.padding,
                                       activation=layer.activation,
                                       weights=layer.get_weights(),
                                       trainable=layer.trainable)(featureModule)
            elif getLayerType(layer) == LayerType.MaxPooling2D:
                featureModule = MaxPooling2D(pool_size=layer.pool_size,
                                             padding=layer.padding)(featureModule)
            elif getLayerType(layer) == LayerType.Dropout:
                featureModule = Dropout(layer.rate)(featureModule)
        featureModules.append(featureModule)
    return concatenate(featureModules)


def getClassifyModule(module, featureModule, fromLayer=LayerType.Flatten):
    classifyModule = featureModule
    classifyFlag = False
    for layerNo, layer in enumerate(module.layers):
        if getLayerType(layer) == fromLayer:
            classifyFlag = True
        if not classifyFlag:
            continue

        if getLayerType(layer) == LayerType.Flatten:
            classifyModule = Flatten()(classifyModule)
        elif getLayerType(layer) == LayerType.Dense:
            classifyModule = Dense(layer.units, activation=layer.activation)(classifyModule)

    return classifyModule


def compile_model(inputModule, classifyModule):
    model = Model(inputs=inputModule, outputs=classifyModule)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


modules = []
for moduleName in moduleNames:
    modules.append(load_model(get_transfer_model_name(freezeUntil, moduleName, prefix='target')))

inputModule = getInputModule(modules[0])
featureModule = getFeatureModule(modules, inputModule)
classifyModule = getClassifyModule(modules[0], featureModule)
model = compile_model(inputModule, classifyModule)

model.save('transfer_model/transfer_combined_' + str(freezeUntil) + '.h5')