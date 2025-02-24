import random

from keras import Sequential
from keras.layers import GlobalAveragePooling2D, Dense

from constants import target_dataset
from core import getSourceModel
from data_processing.dog_util import Dog
from data_processing.sample_util import sample_for_training
from util.ordinary import get_bottleneck_name
from util.transfer_util import save_bottleneck_data
import numpy as np


def np_load_test():
    train_generator, valid_generator, nb_train_samples, nb_valid_samples, num_classes, batch_size, train_labels, valid_labels = \
        Dog().getTrainingDogs(shuffle=True, batch_size=32)

    # train_data_before = save_bottleneck_data(getSourceModel(model_name='inceptionv3'),
    #                                          train_generator, nb_train_samples,
    #                                          batch_size, split='train', save=True, target_ds='dog')

    np.save(get_bottleneck_name('dog', "train", isLabel=True), train_labels)
    train_data_label = np.load(get_bottleneck_name('dog', 'train', isLabel=True))


    train_data_after = np.load(get_bottleneck_name('dog', 'train', isTafe=False, isLabel=False))

    random_sample_id = random.randint(0, train_data_after.shape[0])

    train_ds, valid_ds = sample_for_training(train_data_after, train_data_label, 0.1)

    # for i in range(train_data_after.shape[1]):
    #     for j in range(train_data_after.shape[2]):
    #         for k in range(train_data_after.shape[3]):
    #             if train_data_before[random_sample_id, i, j, k] != train_data_after[random_sample_id, i, j, k]:
    #                 print('Test failed: np loaded data corrupted')
    #                 return
    print('Test successful: np data loaded properly')


np_load_test()
