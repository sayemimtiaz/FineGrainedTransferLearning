import numpy as np
from constants import target_dataset, source_model_name, CURRENT_ACQUIRE
from core import (
    getTargetDataForTraining,
    getSourceModel,
    smapleTargetData,
    getTargetNumClass,
)
from core.target_filter_distribution import calculateTargetDistribution
from core.weigted_transfer import getPValues
from util.ordinary import (
    get_transfer_model_name,
    get_bottleneck_name,
    dump_as_pickle,
    get_delete_rate_name,
)
from util.transfer_util import (
    save_filtered_bottleneck_data,
    get_svm_classifier,
    get_dense_classifier,
    get_pool_classifier,
    save_bottleneck_data,
    save_random_bottleneck_data
)


def acquire(target_ds=None, parent_model=None, ablation=False):
    if target_ds is None:
        target_ds = target_dataset
    if parent_model is None:
        parent_model = source_model_name

    if parent_model in CURRENT_ACQUIRE and target_ds in CURRENT_ACQUIRE[parent_model]:
        return

    alpha_values = [0.0, 1e-45, 1e-25, 1e-15, 1e-5]

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
    if target_ds == "cats_vs_dogs":
        sample_size_per_class = 1500
    if target_ds == "stl10":
        sample_size_per_class = 300

    target_sample = smapleTargetData(
        sample_size_per_class=sample_size_per_class, target_ds=target_ds
    )

    calculateTargetDistribution(
        target_sample, target_ds=target_ds, parent_model=parent_model
    )

    delete_rates = {}

    for alpha in alpha_values:
        print(">> alpha rate: ", alpha)

        p_values, delRate = getPValues(
            alpha=alpha, target_ds=target_ds, parent_model=parent_model
        )

        delete_rates[str(alpha)] = delRate

        if not ablation:
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
        else:
            _, _ = save_random_bottleneck_data(
                bottleneck_features_train,
                p_values,
                split="train",
                alpha=alpha,
                target_ds=target_ds,
            )

            _, _ = save_random_bottleneck_data(
                bottleneck_features_valid,
                p_values,
                split="valid",
                alpha=alpha,
                target_ds=target_ds,
            )

    dump_as_pickle(delete_rates, get_delete_rate_name(target_ds))
