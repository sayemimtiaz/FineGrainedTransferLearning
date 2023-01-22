import random
import numpy as np
from keras.models import load_model

from data_util.util import oneEncodeBoth
from models.cifar100 import getSourceData, getTargetData, target_dataset
from models.cifar100.target_filter_distribution import calculateTargetDistribution
from models.cifar100.weigted_transfer import getWeigtedTransferModel
from util.ordinary import get_transfer_model_name
from util.transfer_util import train, binaryWeight

epoch = 10
REPEAT = 20
sample_rates = [0.15, 0.30, 0.5, 1.0]
# sample_rates = [1.0]

weighting_schemes = [binaryWeight]

# targetIndex = 9
targetIndex = 3

for scheme in weighting_schemes:
    print('> Evaluating scheme: ', str(scheme))
    for sr in sample_rates:

        print('>> Evaluating sampling rate: ', sr)
        x_train, y_train, x_test, y_test, num_classes = getTargetData(sample_rate=sr, one_hot=False, gray=True)

        calculateTargetDistribution((x_train, y_train))
        getWeigtedTransferModel(weighting_scheme=scheme, targetIndex=targetIndex)

        target_names = [get_transfer_model_name(prefix='weighted', model_name=target_dataset),
                        get_transfer_model_name(prefix='traditional', model_name=target_dataset)]

        y_train, y_test = oneEncodeBoth(y_train, y_test)

        for target_name in target_names:
            print('>>> Evaluating ' + target_name)

            sacc = 0
            selapse = 0
            all_acc=[]
            all_elaps=[]
            for r in range(REPEAT):
                model = load_model(target_name)

                # print(model.summary())
                acc, elpase = train(model, x_train, y_train, x_test, y_test, epochs=epoch)
                sacc += acc
                selapse += elpase
                all_acc.append(acc)
                all_elaps.append(elpase)

            result = (sacc / REPEAT, selapse / REPEAT)
            all_acc=np.asarray(all_acc)
            all_elaps=np.asarray(all_elaps)
            print('(accuracy, std, low, high, elapse)', round(all_acc.mean(),2), round(all_acc.std(),2),
                  round(all_acc.min(), 2), round(all_acc.max(), 2),
                  round(all_elaps.mean(),2))
