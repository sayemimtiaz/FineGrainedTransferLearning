import numpy as np
from constants import target_dataset, source_model_name, CURRENT_ACQUIRE, target_datasets, pretrained_architecures, \
    DONE, NUM_CLASSIFIER
from core import (
    getTargetDataForTraining,
    getSourceModel,
    smapleTargetData,
)
from core.evaluate_cifar100 import save_cifar100_feature
from core.evaluate_helper import repeater
from core.target_filter_distribution import calculateTargetDistribution
from core.weigted_transfer import getPValues, getPValuesNoAlpha
from data_processing.cifar_specific import sampleCifar100Fine,getCifar100CoarseClasses
from util.ordinary import (
    get_bottleneck_name,
    dump_as_pickle,
    get_delete_rate_name,
    get_summary_out_name, load_pickle_file)
from util.transfer_util import (
    get_pool_classifier,
    get_dense_classifier,
    save_bottleneck_data,
    delete_bottleneck_data,
save_filtered_bottleneck_data)


def cross_acquire(target_ds=None, ref_ds=None, parent_model=None):
    if target_ds is None:
        target_ds = target_dataset
    if parent_model is None:
        parent_model = source_model_name

    if parent_model in CURRENT_ACQUIRE and target_ds in CURRENT_ACQUIRE[parent_model]:
        return

    # delete_rate = [0.2, 0.5, 0.8, 0.9, 0.95, 0.99]
    delete_rate = [ 0.0, 1e-45, 1e-25, 1e-15, 1e-5, 0.01]


    (
        train_generator,
        valid_generator,
        nb_train_samples,
        nb_valid_samples,
        num_classes,
        batch_size,
        train_labels,
        valid_labels,
    ) = getTargetDataForTraining(batch_size=32, shuffle=False, target_ds=target_ds)

    np.save(get_bottleneck_name(target_ds, "train", isLabel=True), train_labels)
    np.save(get_bottleneck_name(target_ds, "valid", isLabel=True), valid_labels)

    bottleneck_features_train = save_bottleneck_data(
        getSourceModel(parent_model),
        train_generator,
        nb_train_samples,
        batch_size,
        split="train",
        target_ds=target_ds,
    )
    bottleneck_features_valid = save_bottleneck_data(
        getSourceModel(parent_model),
        valid_generator,
        nb_valid_samples,
        batch_size,
        split="valid",
        target_ds=target_ds,
    )

    sample_size_per_class = 30
    if ref_ds == 'bird':
        sample_size_per_class = 20
    if ref_ds == "cats_vs_dogs":
        sample_size_per_class = 1500
    if ref_ds == "stl10":
        sample_size_per_class = 300
    if ref_ds == "mnist":
        sample_size_per_class = 200

    target_sample = smapleTargetData(
        sample_size_per_class=sample_size_per_class, target_ds=ref_ds, crop=False
    )

    calculateTargetDistribution(
        target_sample, target_ds=target_ds, parent_model=parent_model
    )

    delete_rates = {}
    
    # p_values = getPValuesNoAlpha(target_ds=target_ds, parent_model=parent_model)
    # print(p_values)

    for alpha in delete_rate:
        print(">> delete rate: ", alpha)

#         delete_rates[str(alpha)] = alpha

#         o, d = delete_bottleneck_data(
#             bottleneck_features_train,
#             p_values,
#             split="train",
#             deleteRate=alpha,
#             target_ds=target_ds,
#         )

#         _, _ = delete_bottleneck_data(
#             bottleneck_features_valid,
#             p_values,
#             split="valid",
#             deleteRate=alpha,
#             target_ds=target_ds,
#         )
        
        p_values, delRate = getPValues(
            alpha=alpha, target_ds=target_ds, parent_model=parent_model
        )
        delete_rates[str(alpha)] = delRate


        _, _ = save_filtered_bottleneck_data(
            bottleneck_features_train,
            p_values,
            split="train",
            alpha=alpha,
            target_ds=target_ds,
        )

        _, _ = save_filtered_bottleneck_data(
            bottleneck_features_valid,
            p_values,
            split="valid",
            alpha=alpha,
            target_ds=target_ds,
        )
        # print(o, d)

    dump_as_pickle(delete_rates, get_delete_rate_name(target_ds))


def cross_acquire_cifar100(parent_model=None, target_ds=None, task=None, ref_task=None):
    print(task, ref_task)
    if parent_model is None:
        parent_model = source_model_name

    if parent_model in CURRENT_ACQUIRE and target_ds in CURRENT_ACQUIRE[parent_model]:
        return

    delete_rate = [0.2, 0.5, 0.8, 0.90, 0.95, 0.99]
    # delete_rate = [ 0.95]

    x_train, y_train, x_test, y_test, num_classes = sampleCifar100Fine(superclasses=[task], num_sample=2500,
                                                                       gray=False,
                                                                       one_hot=True, train=True, shape=(224, 224))

    np.save(get_bottleneck_name(target_ds, 'train', isLabel=True), y_train)
    np.save(get_bottleneck_name(target_ds, 'valid', isLabel=True), y_test)

    bottleneck_features_train = save_cifar100_feature(getSourceModel(parent_model),
                                                      x_train, split='train', target_ds=target_ds)
    bottleneck_features_valid = save_cifar100_feature(getSourceModel(parent_model),
                                                      x_test, split='valid', target_ds=target_ds)

    target_sample, _, _, _, _ = sampleCifar100Fine(superclasses=[ref_task], num_sample=1000,
                                                   gray=False, shape=(224, 224))

    calculateTargetDistribution(target_sample, target_ds=target_ds, parent_model=parent_model)

    delete_rates = {}

    for alpha in delete_rate:
        print(">> delete rate: ", alpha)

        p_values = getPValuesNoAlpha(target_ds=target_ds, parent_model=parent_model)

        delete_rates[str(alpha)] = alpha

        _, _ = delete_bottleneck_data(bottleneck_features_train,
                                             p_values,
                                             split='train',
                                             deleteRate=alpha,
                                             target_ds=target_ds)

        _, _ = delete_bottleneck_data(bottleneck_features_valid,
                                             p_values, split='valid', deleteRate=alpha,
                                             target_ds=target_ds)

    dump_as_pickle(delete_rates, get_delete_rate_name(target_ds))


def cross_evaluate(target_ds=None, parent_model=None, study_type=None, ref_ds=None):
    if target_ds is None:
        target_ds = target_dataset
    if parent_model is None:
        parent_model = source_model_name

    summaryOut = open(get_summary_out_name(target_ds), "a")
    summaryOut.write(
        "Study Type,"+ref_ds+",Source,Target,Transfer Type,Classifier Type,Alpha,Epoch,Repeat,Accuracy,STD,Minimum"
        " Accuracy,Maximum Accuracy,Elapsed,Delete Rate\n")
    summaryOut.close()

    epoch = 30
    REPEAT = 10
    batch_size = 32
    delete_rate = [ 0.0, 1e-45, 1e-25, 1e-15, 1e-5, 0.01]
    # delete_rate = [0.2, 0.5, 0.8, 0.9, 0.95, 0.99]

    classfiers = {'pool': get_pool_classifier}

    delete_rates = load_pickle_file(get_delete_rate_name(target_ds))

    for cn in classfiers.keys():
        if cn == 'svm':
            epoch = 100

        if parent_model in DONE and target_ds in DONE[parent_model] and cn in DONE[parent_model][target_ds]:
            continue

        print('> Evaluating TAFE: ')
        for alpha in delete_rate:
            print('>> Evaluating alpha rate: ', alpha)

            delRate = delete_rates[str(alpha)]

            if delRate < 100.0:
                repeater(REPEAT, get_classifier=classfiers[cn], alpha=alpha, isBaseline=False,
                         batch_size=batch_size, epoch=epoch, classifierType=cn,
                         delRate=delRate,
                         target_ds=target_ds, parent_model=parent_model, study_type=study_type)


# tds = ['dog', 'bird']
# rds = ['bird', 'dog']

if __name__ == "__main__":
    # total_task=20
    # all_cifar100_tasks=getCifar100CoarseClasses()
    # tds=set()
    # while len(tds)<total_task:
    #     includeIndices = np.random.choice(range(len(all_cifar100_tasks)), 2, replace=False)
    #     tds.add((all_cifar100_tasks[includeIndices[0]], all_cifar100_tasks[includeIndices[1]]))
    # tds = [('trees', 'vehicles 1'), ('vehicles 1', 'trees'), ('fish','large man-made outdoor things'),
    #       ('large man-made outdoor things', 'fish'), ('people', 'reptiles')]
    # tds=list(tds)
    # print(tds)

    tds=[('dog', 'mnist'), ('bird', 'mnist')]

    for (ts, rts) in tds:
        fds=ts.replace(' ', '')
        for pa in pretrained_architecures:
            cross_acquire(target_ds=fds, parent_model=pa, ref_ds=rts)
            cross_evaluate(target_ds=fds, parent_model=pa, study_type='cross', ref_ds=rts)

            cross_acquire(target_ds=fds, parent_model=pa, ref_ds=ts)
            cross_evaluate(target_ds=fds, parent_model=pa, study_type='regular', ref_ds=ts)
