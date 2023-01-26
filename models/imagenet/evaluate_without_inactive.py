import random
import numpy as np
from keras.models import load_model

from data_processing.bird_util import Bird
from data_processing.cifar_specific import getCifar100CoarseClasses
from data_processing.data_util import oneEncodeBoth
from models.imagenet import target_dataset, getTargetDataForTraining, getSourceData
from models.imagenet.target_filter_distribution import calculateTargetDistribution
from models.imagenet.weigted_transfer import getWeigtedTransferModel
from util.ordinary import get_transfer_model_name, load_pickle_file
from util.transfer_util import train, binaryWeight, removeInactives, get_svm
import tensorflow as tf

epoch = 15
REPEAT = 20
sample_rates = [1.0, 0.5, 0.15, 0.3]

weighting_schemes = [binaryWeight]

tiny = getSourceData(shape=(64, 64), gray=False)
# targetIndex = tiny.getBirdIndex()
targetIndex = None

tasks = getCifar100CoarseClasses()

for scheme in weighting_schemes:
    print('> Evaluating scheme: ', str(scheme))
    for sr in sample_rates:

        print('>> Evaluating sampling rate: ', sr)

        for _task in tasks:
            print('>>> Task: ', _task)
            x_train, y_train, x_test, y_test, num_classes = getTargetDataForTraining(sample_rate=sr,
                                                                                     one_hot=False, gray=False,
                                                                                     task=_task)
            calculateTargetDistribution((x_train, y_train))

            getWeigtedTransferModel(weighting_scheme=scheme, targetIndex=targetIndex)

            p_values = load_pickle_file('p_values.pickle')

            target_names = [get_transfer_model_name(prefix='weighted', model_name=target_dataset),
                            get_transfer_model_name(prefix='traditional', model_name=target_dataset)]

            y_train, y_test = oneEncodeBoth(y_train, y_test)

            feature_model = load_model(target_names[0])
            nx_train = []
            nx_test = []
            # i = 0
            for x in x_train:
                nx_train.append(removeInactives(feature_model, x, p_values))
                # if i >= 4:
                #     break
                # i += 1
            # y_train = y_train[:i+1]
            # i = 0
            for x in x_test:
                nx_test.append(removeInactives(feature_model, x, p_values))
            #     if i >= 4:
            #         break
            #     i += 1
            # y_test = y_test[:i+1]

            nx_train = np.asarray(nx_train)
            nx_test = np.asarray(nx_test)

            for target_name in target_names:
                print('>>>> Evaluating ' + target_name)

                sacc = 0
                selapse = 0
                all_acc = []
                all_elaps = []
                for r in range(REPEAT):
                    if 'weighted' not in target_name:
                        model = load_model(target_name)
                        acc, elpase = train(model, x_train, y_train, x_test, y_test, epochs=epoch)
                    else:
                        model = get_svm()
                        acc, elpase = train(model, nx_train, y_train, nx_test, y_test, epochs=epoch)

                    # print(model.summary())
                    sacc += acc
                    selapse += elpase
                    all_acc.append(acc)
                    all_elaps.append(elpase)

                result = (sacc / REPEAT, selapse / REPEAT)
                all_acc = np.asarray(all_acc)
                all_elaps = np.asarray(all_elaps)
                print('(accuracy, std, low, high, elapse)', round(all_acc.mean(), 2), round(all_acc.std(), 2),
                      round(all_acc.min(), 2), round(all_acc.max(), 2),
                      round(all_elaps.mean(), 2))
