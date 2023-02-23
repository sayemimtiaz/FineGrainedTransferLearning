from keras.utils import to_categorical
from keras_preprocessing.image import ImageDataGenerator

from constants import SHAPE
from data_processing.base_data import loadTensorFlowDataset, sampleFromClassesInDir
import os

from util.common import get_project_root


class Cat:
    data = None
    classes = None

    data_path = None

    def __init__(self):
        self.data_path = get_project_root()
        self.data_path = os.path.join(self.data_path, 'data', 'cats_vs_dogs')

    def getClasses(self):
        return self.classes

    def getTrainingCats(self, batch_size=32, shuffle=True):
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            # shear_range=0.2,
            # zoom_range=0.2,
            # horizontal_flip=True,
            # rotation_range=20,
            # width_shift_range=0.2,
            # height_shift_range=0.2,
            validation_split=0.2
        )

        valid_datagen = ImageDataGenerator(
            rescale=1. / 255,
            validation_split=0.2,
        )

        train_generator = train_datagen.flow_from_directory(
            self.data_path,
            target_size=(SHAPE[0], SHAPE[1]),
            color_mode='rgb',
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=shuffle,
            seed=1337
        )

        valid_generator = valid_datagen.flow_from_directory(
            self.data_path,
            target_size=(SHAPE[0], SHAPE[1]),
            color_mode='rgb',
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=shuffle,
            seed=1337
        )

        num_classes = len(train_generator.class_indices)
        train_labels = train_generator.classes
        train_labels = to_categorical(train_labels, num_classes=num_classes)
        valid_labels = valid_generator.classes
        valid_labels = to_categorical(valid_labels, num_classes=num_classes)
        nb_train_samples = len(train_generator.filenames)
        nb_valid_samples = len(valid_generator.filenames)

        self.classes = list(range(num_classes))

        return train_generator, valid_generator, nb_train_samples, nb_valid_samples, num_classes, batch_size, \
               train_labels, valid_labels

    def sampleFromDir(self, sample_size_per_class=20, seed=None, ext='jpg', crop=False):
        return sampleFromClassesInDir(self.data_path,
                                      sample_size_per_class=sample_size_per_class, seed=seed, ext=ext,
                                      shape=SHAPE, crop=crop)
