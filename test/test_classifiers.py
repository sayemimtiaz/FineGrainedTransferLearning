from sklearn import svm

from constants import target_dataset
from core import getSourceModel
from data_processing.dog_util import Dog
from util.ordinary import get_bottleneck_name
from util.transfer_util import get_svm_classifier, save_bottleneck_data
import numpy as np


def svm_test():
    train_generator, valid_generator, nb_train_samples, nb_valid_samples, num_classes, batch_size, train_labels, valid_labels = \
        Dog().getTrainingDogs(shuffle=False, batch_size=32, class_mode='categorical')

    # train_data = save_bottleneck_data(getSourceModel(),
    #                                   train_generator, nb_train_samples,
    #                                   batch_size, split='train', save=True)
    # print(train_data.shape)
    # val_data = save_bottleneck_data(getSourceModel(),
    #                                 valid_generator, nb_valid_samples,
    #                                 batch_size, split='valid', save=True)

    train_data = np.load(get_bottleneck_name(target_dataset, 'train', isTafe=False, isLabel=False))
    val_data = np.load(get_bottleneck_name(target_dataset, 'valid', isTafe=False, isLabel=False))

    model=get_svm_classifier(train_data.shape[1:], 120)

    history = model.fit(train_data, train_labels,
                        epochs=50,
                        batch_size=batch_size,
                        validation_data=(val_data, valid_labels))



svm_test()
