import pickle
from datetime import datetime
from models.cifar100.data_util import getSuperClassData, sampleForDecomposition, getCifar10BinaryData, sample, \
    getFineGrainedClass, getMnistData
from modularization.concern.cnn_util import activeRateFilterEachObs, observe_cnn, activeRateFilterAllObs, \
    activeRateNodeEachObs, activeRateNodeAllObs, extractActiveCountFromObj, activeValNodeEachObs, activeValNodeAllObs, \
    activeRawValNodeEachObs
from keras.models import load_model
from util.common import initModularLayers
from util.ordinary import dump_as_pickle, load_pickle_file
from util.transfer_util import construct_target_model, get_transfer_model_name, construct_target_model_partial_freeze, \
    get_transfer_filter_name, construct_target_model_as_feature_extractor

MODE = 'val'  # rate or val
numSample = 2000

eachFun = activeRateNodeEachObs
allFun = activeRateNodeAllObs
if MODE == 'val':
    eachFun = activeValNodeEachObs
    # eachFun=activeRawValNodeEachObs
    allFun = activeValNodeAllObs

# x_train, y_train, _, _, num_classes = getSuperClassData(insert_noise=False, dataset='cifar10')
# x_train, y_train, _, _, num_classes = getCifar10BinaryData(one_hot=False)
# num_classes += 1
positive_classes = ['truck', 'automobile']
dataset = 'cifar10'
pos_x, neg_x = sampleForDecomposition(sample=numSample, positive_classes=positive_classes,
                                      dataset=dataset, gray=True)
# pos_x, _ = sample(numSample, data_x=x_train, data_y=y_train, num_classes=num_classes)

model_name = 'h5/source_model_mixed.h5'

model = load_model(model_name)

positiveConcern = initModularLayers(model.layers)

observe_cnn(positiveConcern, pos_x, eachFun, allFun)
sourceRate = extractActiveCountFromObj(positiveConcern)
dump_as_pickle(sourceRate, 'transfer_model/source_filter_active_' + MODE + '.pickle')

# _, _, x_train, y_train, num_classes = getSuperClassData(insert_noise=False, dataset='cifar10')
# _, _, x_train, y_train, num_classes = getCifar10BinaryData(one_hot=False)
# pos_x, _ = sample(numSample, data_x=x_test, data_y=y_test, num_classes=num_classes + 1)
x_train, y_train, _, _, num_classes = getFineGrainedClass(superclass='vehicles 1', num_sample=-1,
                                                          one_hot=False, gray=True)
# x_train, y_train, _, _, num_classes = getMnistData(one_hot=False)
pos_x, _ = sample(numSample, data_x=x_train, data_y=y_train, num_classes=num_classes)

positiveConcern = initModularLayers(model.layers)

observe_cnn(positiveConcern, pos_x, eachFun, allFun)
targetRate = extractActiveCountFromObj(positiveConcern)
dump_as_pickle(targetRate, 'transfer_model/target_filter_active_' + MODE + '.pickle')

# newCon=load_pickle_file('transfer_model/source_filter_active_rate.pickle')
# print(newCon)
