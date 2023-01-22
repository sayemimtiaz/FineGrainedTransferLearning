from models.imagenet import getSourceModel, target_dataset, getTargetNumClass, SHAPE
from util.common import freezeModel
from util.ordinary import get_transfer_model_name
from util.transfer_util import construct_reweighted_target

model = getSourceModel(shape=SHAPE)
target_num_class = getTargetNumClass()


target_model = construct_reweighted_target(freezeModel(model), n_classes=target_num_class)
target_model.save(get_transfer_model_name(prefix='traditional', model_name=target_dataset))
