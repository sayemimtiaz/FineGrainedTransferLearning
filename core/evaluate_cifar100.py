import random
import numpy as np
from constants import source_model_name, pretrained_architecures
from core import getSourceModel
from core.evaluate import evaluate
from core.target_filter_distribution import calculateTargetDistribution
from core.weigted_transfer import getPValues
from data_processing.cifar_specific import getCifar100CoarseClasses, sampleCifar100Fine
from util.ordinary import get_bottleneck_name, dump_as_pickle, get_delete_rate_name
from util.transfer_util import save_filtered_bottleneck_data


def save_cifar100_feature(base_model, data, split):
    bottleneck_features_train = base_model.predict(data)

    np.save(get_bottleneck_name('cifar100', split, isTafe=False, isLabel=False),
            bottleneck_features_train)

    return bottleneck_features_train


def acquire_cifar100(parent_model=None, target_ds=None):
    if parent_model is None:
        parent_model = source_model_name

    alpha_values = [0.0, 1e-25, 1e-15, 1e-5, 0.01, 0.05]

    x_train, y_train, x_test, y_test, num_classes = sampleCifar100Fine(superclasses=[task], num_sample=2500,
                                                                       gray=False,
                                                                       one_hot=True, train=True, shape=(224, 224))

    np.save(get_bottleneck_name(target_ds, 'train', isLabel=True), y_train)
    np.save(get_bottleneck_name(target_ds, 'valid', isLabel=True), y_test)

    bottleneck_features_train = save_cifar100_feature(getSourceModel(parent_model),
                                                      x_train, split='train')
    bottleneck_features_valid = save_cifar100_feature(getSourceModel(parent_model),
                                                      x_test, split='valid')

    target_sample, _, _, _, _ = sampleCifar100Fine(superclasses=[task], num_sample=1000,
                                                   gray=False, shape=(224, 224))

    calculateTargetDistribution(target_sample, target_ds=target_ds, parent_model=parent_model)

    delete_rates = {}

    for alpha in alpha_values:
        print('>> alpha rate: ', alpha)

        p_values, delRate = getPValues(alpha=alpha, target_ds=target_ds, parent_model=parent_model)

        delete_rates[str(alpha)] = delRate

        _, _ = save_filtered_bottleneck_data(bottleneck_features_train,
                                             p_values,
                                             split='train',
                                             alpha=alpha,
                                             target_ds=target_ds)

        _, _ = save_filtered_bottleneck_data(bottleneck_features_valid,
                                             p_values, split='valid', alpha=alpha,
                                             target_ds=target_ds)

    dump_as_pickle(delete_rates, get_delete_rate_name(target_ds))


for task in getCifar100CoarseClasses():
    for pa in pretrained_architecures:
        acquire_cifar100(parent_model=pa, target_ds=task)
        evaluate(target_ds=task, parent_model=pa)
