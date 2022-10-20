from datetime import datetime
from models.cifar100.data_util import getSuperClassData, sampleForDecomposition
from modularization.concern.cnn_util import activeRateFilterEachObs, observe_cnn, activeRateFilterAllObs
from keras.models import load_model
from util.common import initModularLayers
from util.transfer_util import construct_target_model, get_transfer_model_name

freezeUntil = 3

model_name = 'h5/source_model_mixed.h5'

model = load_model(model_name)

print("Start Time:" + datetime.now().strftime("%H:%M:%S"))

positiveConcern = initModularLayers(model.layers)
negativeConcern = initModularLayers(model.layers)

target_model = construct_target_model(positiveConcern, negativeConcern, model, freezeUntil=freezeUntil,
                                      traditional_transfer=True)

target_name = 'transfer_model/traditional_transfer_'+str(freezeUntil)+'.h5'
target_model.save(target_name)

model = load_model(model_name)
positiveConcern = initModularLayers(model.layers)
negativeConcern = initModularLayers(model.layers)
no_transfer_model = construct_target_model(positiveConcern, negativeConcern, model, transfer=False)

no_transfer_name = 'transfer_model/no_transfer.h5'
no_transfer_model.save(no_transfer_name)
