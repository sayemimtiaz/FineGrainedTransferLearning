from keras.utils import to_categorical
from keras_preprocessing.image import ImageDataGenerator

from data_processing.base_data import loadTensorFlowDataset, sampleFromClassesInDir
import os


class Dog:
    data = None
    classes = None
    shape = (224, 224)

    def __init__(self, train_data=False, shape=(224, 224)):
        self.shape = shape
        if train_data:
            self.data = self.getTrainingDogs(shape=shape)
        # else:
        #     self.data = self.getDogs(shape=shape)

    def getClasses(self):
        return self.classes

    def getTrainingDogs(self, shape=(224, 224), batch_size=128, shuffle=False):
        root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        img_path = os.path.join(root, 'data', 'dogs', 'images')

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            validation_split=0.2,

        )

        valid_datagen = ImageDataGenerator(
            rescale=1. / 255,
            validation_split=0.2,
        )

        train_generator = train_datagen.flow_from_directory(
            img_path,
            target_size=(shape[0], shape[1]),
            color_mode='rgb',
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=shuffle,
            seed=1337
        )

        valid_generator = valid_datagen.flow_from_directory(
            img_path,
            target_size=(shape[0], shape[1]),
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

    def getDogs(self, shape=(224, 224)):
        x_train, y_train, x_test, y_test, num_classes = \
            loadTensorFlowDataset('stanford_dogs', shape=shape, one_hot=False)
        self.classes = list(range(num_classes))
        return x_train, y_train, x_test, y_test, num_classes

    def sampleFromDir(self, sample_size_per_class=20, seed=None, ext='JPEG'):
        return sampleFromClassesInDir('dogs/images/',
                                      sample_size_per_class=sample_size_per_class, seed=seed, ext=ext,
                                      shape=self.shape)
