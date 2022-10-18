import pickle
from datetime import datetime
from models.cifar100.data_util import getSuperClassData, sampleForDecomposition
from modularization.concern.cnn_util import activeRateFilterEachObs, observe_cnn, activeRateFilterAllObs
from keras.models import load_model
from util.common import initModularLayers
from util.transfer_util import construct_target_model, get_transfer_model_name, construct_target_model_partial_freeze, \
    get_transfer_filter_name, construct_target_model_as_feature_extractor

numSample = 200
freezeUntil = -1
thresholds = [0.00, 0.00, 0.05, 0.03, 0.008]
relax_relevance = False

positive_classes = ['automobile', 'truck']
dataset = 'cifar10'
pos_x, neg_x = sampleForDecomposition(sample=numSample, positive_classes=positive_classes, dataset=dataset)

model_name = 'h5/source_model_binary.h5'

model = load_model(model_name)

print("Start Time:" + datetime.now().strftime("%H:%M:%S"))

positiveConcern = initModularLayers(model.layers)
negativeConcern = initModularLayers(model.layers)

observe_cnn(positiveConcern, pos_x, activeRateFilterEachObs, activeRateFilterAllObs)
observe_cnn(negativeConcern, neg_x, activeRateFilterEachObs, activeRateFilterAllObs)

# target_model = construct_target_model(positiveConcern, negativeConcern, model, freezeUntil=freezeUntil,
#                                       thresholds=thresholds, relax_relevance=relax_relevance)
# target_model_a, target_model_b, transfer_filter = construct_target_model_partial_freeze(positiveConcern,
#                                                                                         negativeConcern, model,
#                                                                                         thresholds=thresholds,
#                                                                                         relax_relevance=relax_relevance)

target_model_a, target_model_b = construct_target_model_as_feature_extractor(positiveConcern,
                                                                             negativeConcern, model,
                                                                             thresholds=thresholds,
                                                                             relax_relevance=relax_relevance,
                                                                             target_layer=3)
target_name = get_transfer_model_name(freezeUntil, relax_relevance, thresholds[0])
target_model_a.save(target_name + '.h5')
target_model_b.save(target_name + '_random.h5')

# target_name = get_transfer_filter_name(freezeUntil, relax_relevance, thresholds[0])
#
# with open(target_name, 'wb') as handle:
#     pickle.dump(transfer_filter, handle, protocol=pickle.HIGHEST_PROTOCOL)
