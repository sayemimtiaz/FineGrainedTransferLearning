import random
import numpy as np
from keras.models import load_model

from constants import SHAPE, target_dataset
from core import getTargetDataForTraining, getSourceModel, smapleTargetData, getTargetNumClass
from core.target_filter_distribution import calculateTargetDistribution
from core.weigted_transfer import getPValues
from util.ordinary import get_transfer_model_name
from util.transfer_util import save_filtered_bottleneck_data, get_svm_classifier, get_dense_classifier, \
    get_pool_classifier

alpha_values = [0.0, 0.0001, 0.001, 0.01]
classfiers = [get_svm_classifier, get_dense_classifier, get_pool_classifier]
targetClass = getTargetNumClass()

train_generator, valid_generator, nb_train_samples, nb_valid_samples, num_classes, batch_size = \
    getTargetDataForTraining()

for alpha in alpha_values:
    print('>> alpha rate: ', alpha)

    target_sample = smapleTargetData(sample_size_per_class=1)

    calculateTargetDistribution(target_sample)

    p_values = getPValues(alpha=alpha)

    original_input_shape, tafe_input_shape = save_filtered_bottleneck_data(getSourceModel(),
                                                                           train_generator, nb_train_samples,
                                                                           batch_size, p_values, split='train')

    _, _ = save_filtered_bottleneck_data(getSourceModel(),
                                         valid_generator, nb_train_samples,
                                         batch_size, p_values, split='valid')

    for cn in classfiers:
        target_model = cn(tafe_input_shape, targetClass)
        target_model.save(get_transfer_model_name(prefix='weighted', alpha=alpha, model_name=target_dataset))

        target_model = cn(original_input_shape, targetClass)
        target_model.save(get_transfer_model_name(prefix='baseline', alpha=alpha, model_name=target_dataset))
