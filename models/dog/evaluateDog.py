import random
import numpy as np
from keras.models import load_model

from data_util.dog_util import Dog
from models.imagenet import target_dataset, getTargetDataForTraining, SHAPE
from models.imagenet.target_filter_distribution import calculateTargetDistribution, \
    calculateTargetDistributionAlreadySampled
from models.imagenet.weigted_transfer import getWeigtedTransferModel
from util.ordinary import get_transfer_model_name
from util.transfer_util import binaryWeight, trainDog

epoch = 10
REPEAT = 3
alpha_values = [0.0, 0.01, 0.05]

weighting_schemes = [binaryWeight]
targetIndex = None

for scheme in weighting_schemes:
    print('> Evaluating scheme: ', str(scheme))
    for alpha in alpha_values:

        print('>> Evaluating alpha rate: ', alpha)

        train_generator, valid_generator, nb_train_samples, nb_valid_samples, num_classes, batch_size = \
            getTargetDataForTraining()

        dog = Dog(shape=SHAPE, train_data=False)
        target_sample = dog.sampleFromDir(sample_size_per_class=20, ext='jpg')
        calculateTargetDistributionAlreadySampled(target_sample)

        getWeigtedTransferModel(weighting_scheme=scheme, targetIndex=targetIndex, alpha=alpha)

        target_names = [get_transfer_model_name(prefix='weighted', model_name=target_dataset),
                        get_transfer_model_name(prefix='traditional', model_name=target_dataset)]

        for target_name in target_names:
            print('>>>> Evaluating ' + target_name)

            sacc = 0
            selapse = 0
            all_acc = []
            all_elaps = []
            for r in range(REPEAT):
                model = load_model(target_name)

                # print(model.summary())
                acc, elpase = trainDog(model, train_generator, valid_generator, nb_train_samples, nb_valid_samples,
                                       epoch=epoch, batch_size=batch_size)
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
