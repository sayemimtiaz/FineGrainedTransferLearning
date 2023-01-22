import random
import numpy as np
from keras.models import load_model

from data_util.bird_util import Bird
from data_util.cifar_specific import getCifar100CoarseClasses
from data_util.util import oneEncodeBoth
from models.imagenet import target_dataset, getTargetDataForTraining, getSourceData
from models.imagenet.target_filter_distribution import calculateTargetDistribution
from models.imagenet.weigted_transfer import getWeigtedTransferModel
from util.ordinary import get_transfer_model_name
from util.transfer_util import train, binaryWeight

epoch = 5
REPEAT = 20
sample_rates = [1.0]

weighting_schemes = [binaryWeight]

tiny = getSourceData(shape=(64, 64), gray=False)
# targetIndex = tiny.getBirdIndex()
targetIndex = None

tasks=getCifar100CoarseClasses()

for scheme in weighting_schemes:
    print('> Evaluating scheme: ', str(scheme))
    for sr in sample_rates:

        print('>> Evaluating sampling rate: ', sr)

        for _task in tasks:
            print('>>> Task: ', _task)
            x_train, y_train, x_test, y_test, num_classes = getTargetDataForTraining(sample_rate=sr,
                                                                                     one_hot=False, gray=False,
                                                                                     task=_task)
            calculateTargetDistribution((x_train, y_train), numSample=10)
            # bird = Bird(load_data=False)
            # bird.data = x_train, y_train, x_test, y_test, num_classes
            # calculateTargetDistribution(bird=bird)

            getWeigtedTransferModel(weighting_scheme=scheme, targetIndex=targetIndex)

            target_names = [get_transfer_model_name(prefix='weighted', model_name=target_dataset),
                            get_transfer_model_name(prefix='traditional', model_name=target_dataset)]

            y_train, y_test = oneEncodeBoth(y_train, y_test)

            for target_name in target_names:
                print('>>>> Evaluating ' + target_name)

                sacc = 0
                selapse = 0
                all_acc = []
                all_elaps = []
                for r in range(REPEAT):
                    model = load_model(target_name)

                    # print(model.summary())
                    acc, elpase = train(model, x_train, y_train, x_test, y_test, epochs=epoch)
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
