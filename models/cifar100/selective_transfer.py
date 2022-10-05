from datetime import datetime
from models.cifar100.data_util import getSuperClassData, sampleForDecomposition
from modularization.concern.cnn_util import activeRateFilterEachObs, observe_cnn, activeRateFilterAllObs
from keras.models import load_model
from util.common import initModularLayers
from util.transfer_util import construct_target_model, get_transfer_model_name

numSample = 300
freezeUntil = 3
thresholds = [0.01, 0.05, 0.08, 0.1, 0.2]
relax_relevance = False

positive_classes = ['automobile', 'truck']
dataset = 'cifar10'
pos_x, neg_x = sampleForDecomposition(sample=numSample, positive_classes=positive_classes, dataset=dataset)

model_name = 'h5/source_model.h5'

model = load_model(model_name)

print("Start Time:" + datetime.now().strftime("%H:%M:%S"))

positiveConcern = initModularLayers(model.layers)
negativeConcern = initModularLayers(model.layers)

observe_cnn(positiveConcern, pos_x, activeRateFilterEachObs, activeRateFilterAllObs)
observe_cnn(negativeConcern, neg_x, activeRateFilterEachObs, activeRateFilterAllObs)

target_model = construct_target_model(positiveConcern, negativeConcern, model, freezeUntil=freezeUntil,
                                      thresholds=thresholds, relax_relevance=relax_relevance)

target_name = get_transfer_model_name(freezeUntil, relax_relevance, thresholds[0])
target_model.save(target_name)
