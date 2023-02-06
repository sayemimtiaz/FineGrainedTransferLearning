import gc
import time
import numpy as np

from constants import target_dataset, source_model_name
from core import getTargetNumClass
from util.common import init_gpu
from util.ordinary import get_bottleneck_name, get_summary_out_name, load_pickle_file, get_delete_rate_name
from keras import backend as K


def load_classifier(input_shape, get_classifier=None, target_ds=None):
    # classifier = load_model(get_transfer_model_name(isBaseline=isBaseline, alpha=alpha, type=type,
    #                                                 model_name=target_dataset))
    if target_ds is None:
        target_ds = target_dataset
    targetClass = getTargetNumClass(target_ds=target_ds)
    classifier = get_classifier(input_shape, targetClass)

    return classifier


def getInputShape(alpha, isBaseline, target_ds=None):
    if target_ds is None:
        target_ds = target_dataset

    if not isBaseline:
        train_ds_tafe = np.load(get_bottleneck_name(target_ds, 'train', isTafe=True,
                                                    isLabel=False, alpha=alpha))
        return train_ds_tafe.shape[1:]

    train_ds_baseline = np.load(get_bottleneck_name(target_ds, 'train', isTafe=False, isLabel=False))

    return train_ds_baseline.shape[1:]


def trainTafe(model, alpha, epoch=30, batch_size=128, verbose=0, target_ds=None):
    if target_ds is None:
        target_ds = target_dataset

    train_ds = np.load(get_bottleneck_name(target_ds, 'train', isTafe=True,
                                           isLabel=False, alpha=alpha))
    valid_ds = np.load(get_bottleneck_name(target_ds, 'valid', isTafe=True,
                                           isLabel=False, alpha=alpha))

    train_labels = np.load(get_bottleneck_name(target_ds, 'train', isLabel=True))
    valid_labels = np.load(get_bottleneck_name(target_ds, 'valid', isLabel=True))

    acc, elapse = trainDog(model, train_ds, valid_ds, train_labels, valid_labels,
                           epoch=epoch, batch_size=batch_size, verbose=verbose)

    return acc, elapse


def trainBaseline(model, epoch=30, batch_size=128, verbose=0, target_ds=None):
    if target_ds is None:
        target_ds = target_dataset

    train_ds = np.load(get_bottleneck_name(target_ds, 'train', isTafe=False, isLabel=False))
    valid_ds = np.load(get_bottleneck_name(target_ds, 'valid', isTafe=False, isLabel=False))

    train_labels = np.load(get_bottleneck_name(target_ds, 'train', isLabel=True))
    valid_labels = np.load(get_bottleneck_name(target_ds, 'valid', isLabel=True))

    acc, elapse = trainDog(model, train_ds, valid_ds, train_labels, valid_labels,
                           epoch=epoch, batch_size=batch_size, verbose=verbose)

    return acc, elapse


def trainDog(model, train_ds, val_ds, train_labels, validation_labels, epoch=30, batch_size=128, verbose=0):
    start = time.time()

    history = model.fit(train_ds, train_labels,
                        epochs=epoch,
                        batch_size=batch_size,
                        validation_data=(val_ds, validation_labels))
    end = time.time()

    return history.history['val_accuracy'][-1], end - start


def repeater(num_repeat, get_classifier=None, alpha=None, isBaseline=False, batch_size=128, epoch=30,
             classifierType=None, delRate=0.0, target_ds=None, parent_model=None):
    if target_ds is None:
        target_ds = target_dataset

    if parent_model is None:
        parent_model = source_model_name

    summaryOut = open(get_summary_out_name(target_ds), "a")
    all_acc = []
    all_elaps = []

    input_shape = getInputShape(alpha, isBaseline, target_ds=target_ds)
    for r in range(num_repeat):
        classifier = load_classifier(input_shape, get_classifier, target_ds=target_ds)

        if not isBaseline:
            acc, elpase = trainTafe(classifier, alpha,
                                    epoch=epoch, batch_size=batch_size, target_ds=target_ds)
        else:
            acc, elpase = trainBaseline(classifier,
                                        epoch=epoch, batch_size=batch_size, target_ds=target_ds)
        acc = acc * 100.0
        all_acc.append(acc)
        all_elaps.append(elpase)

        K.clear_session()
        try:
            del classifier
        except:
            pass
        gc.collect()
        init_gpu()

    all_acc = np.asarray(all_acc)
    all_elaps = np.asarray(all_elaps)
    print('(accuracy, std, low, high, elapse)', round(all_acc.mean(), 2), round(all_acc.std(), 2),
          round(all_acc.min(), 2), round(all_acc.max(), 2),
          round(all_elaps.mean(), 2))

    if isBaseline:
        transferType = 'Baseline'
    else:
        transferType = 'tafe'

    summaryOut.write(parent_model + ',' + target_ds + ',' + classifierType + ',' + transferType + ',' +
                     str(alpha) + ',' + str(epoch) + ',' + str(num_repeat) + ',' +
                     str(round(all_acc.mean(), 2)) + ',' + str(round(all_acc.std(), 2)) + ','
                     + str(round(all_acc.min(), 2)) + ',' + str(round(all_acc.max(), 2)) +
                     ',' + str(round(all_elaps.mean(), 2)) + ',' +
                     str(delRate) +
                     '\n')

    summaryOut.close()
