from datetime import datetime
from models.cifar100.data_util import getSuperClassData, sampleForDecomposition
from modularization.concern.cnn_util import activeRateFilterEachObs, observe_cnn, activeRateFilterAllObs
from keras.models import load_model
from util.common import initModularLayers
from util.ordinary import get_transfer_model_name
from util.transfer_util import construct_target_model

freezeUntil = 3

model_name = 'h5/source_model_mixed.h5'

model = load_model(model_name)

print("Start Time:" + datetime.now().strftime("%H:%M:%S"))

positiveConcern = initModularLayers(model.layers)
negativeConcern = initModularLayers(model.layers)

target_model = construct_target_model(positiveConcern, negativeConcern, model, freezeUntil=freezeUntil,
                                      traditional_transfer=True, n_classes=5)

target_model.save(get_transfer_model_name(freezeUntil, model_name, prefix='traditional_transfer'))

model = load_model(model_name)
positiveConcern = initModularLayers(model.layers)
negativeConcern = initModularLayers(model.layers)
no_transfer_model = construct_target_model(positiveConcern, negativeConcern, model, transfer=False, n_classes=5)

no_transfer_model.save(get_transfer_model_name(None, model_name, prefix='no_transfer'))
