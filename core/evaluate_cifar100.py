import random
import numpy as np
from constants import source_model_name, pretrained_architecures, CURRENT_ACQUIRE
from core import getSourceModel
from core.evaluate import evaluate
from core.target_filter_distribution import calculateTargetDistribution
from core.weigted_transfer import getPValues
from data_processing.cifar_specific import getCifar100CoarseClasses, sampleCifar100Fine
from util.ordinary import get_bottleneck_name, dump_as_pickle, get_delete_rate_name
from util.transfer_util import save_filtered_bottleneck_data


def save_cifar100_feature(base_model, data, split, target_ds):
    bottleneck_features_train = base_model.predict(data)

    np.save(get_bottleneck_name(target_ds, split, isTafe=False, isLabel=False),
            bottleneck_features_train)

    return bottleneck_features_train


def acquire_cifar100(parent_model=None, target_ds=None, task=None):
    if parent_model is None:
        parent_model = source_model_name

    if parent_model in CURRENT_ACQUIRE and target_ds in CURRENT_ACQUIRE[parent_model]:
        return

    alpha_values = [0.0, 1e-45, 1e-25, 1e-15, 1e-5]

    x_train, y_train, x_test, y_test, num_classes = sampleCifar100Fine(superclasses=[task], num_sample=2500,
                                                                       gray=False,
                                                                       one_hot=True, train=True, shape=(224, 224))

    np.save(get_bottleneck_name(target_ds, 'train', isLabel=True), y_train)
    np.save(get_bottleneck_name(target_ds, 'valid', isLabel=True), y_test)

    bottleneck_features_train = save_cifar100_feature(getSourceModel(parent_model),
                                                      x_train, split='train', target_ds=target_ds)
    bottleneck_features_valid = save_cifar100_feature(getSourceModel(parent_model),
                                                      x_test, split='valid', target_ds=target_ds)

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


done = []
# done=['aquaticmammals', 'fish', 'flowers', 'foodcontainers','fruitandvegetables', 'householdelectricaldevices',
#  'householdfurniture','insects','largecarnivores', 'largeman-madeoutdoorthings', 'largenaturaloutdoorscenes', 'largeomnivoresandherbivores',
#       'medium-sizedmammals', 'non-insectinvertebrates']
# , 'people', 'reptiles', 'trees', 'vehicles1', 'smallmammals']
for task in getCifar100CoarseClasses():
    fds = task.replace(' ', '')
    if fds not in done:
        for pa in pretrained_architecures:
            acquire_cifar100(parent_model=pa, target_ds=fds, task=task)
            evaluate(target_ds=fds, parent_model=pa)
