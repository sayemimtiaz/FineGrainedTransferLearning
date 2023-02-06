import os
from keras.utils import to_categorical
from keras_preprocessing.image import ImageDataGenerator

from constants import SHAPE
from data_processing.base_data import loadTensorFlowDataset, sampleFromClassesInDir


# class Bird:
#     data = None
#     one_hot = False
#     gray = False
#
#     def __init__(self, one_hot=False, gray=False, load_data=True, interpolation=False, shape=(128, 128)):
#         self.one_hot = one_hot
#         self.gray = gray
#         if load_data:
#             self.data = self.getBirds(shape=shape)
#
#     def getClasses(self):
#         return list(range(self.data[4]))
#
#     def getBirds(self, shape=(64, 64)):
#         x_train, y_train, x_test, y_test, num_classes = \
#             loadTensorFlowDataset('caltech_birds2011', shape=shape, one_hot=self.one_hot, gray=self.gray)
#
#         return x_train, y_train, x_test, y_test, num_classes
#
#     def sample(self, sample_only_classes=None, num_sample=-1, seed=None, one_hot=False, train=True):
#         return sampleTrainTest(self.data, num_sample=num_sample,
#                                sample_only_classes=sample_only_classes, seed=seed,
#                                one_hot=one_hot, train=train)


# def getBirdTrainingData(shape=(128,128)):
#     # load Caltech Birds dataset from 2011
#     (train_ds, val_ds), ds_info = tfds.load('caltech_birds2011',
#                                             split=['train', 'test'],
#                                             shuffle_files=True, as_supervised=True, with_info=True)
#
#     # Set batch size and image dimensions allowed by your memory resources
#     batch_size = 128
#     image_height = shape[0]
#     image_width = shape[1]
#
#
#     def format_dataset(data):
#         norm = lambda image, label: (tf.cast(image, tf.float32) / 255., label)
#         pad = lambda image, label: (tf.image.resize_with_pad(image, image_height, image_width), label)
#         data = data.map(norm, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#         data = data.map(pad, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#         data = data.batch(batch_size)
#         data = data.prefetch(tf.data.experimental.AUTOTUNE)
#         return data
#
#     train_ds, val_ds = tuple(map(format_dataset, [train_ds, val_ds]))
#
#     return train_ds,val_ds,200
from util.common import get_project_root


class Bird:
    data = None
    data_path = None

    def __init__(self):
        self.data_path = get_project_root()
        self.data_path = os.path.join(self.data_path, 'data', 'CUB_200_2011','CUB_200_2011', 'images')

    def getClasses(self):
        return 200

    def getTrainingBirds(self, batch_size=128, shuffle=True):
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

        return train_generator, valid_generator, nb_train_samples, nb_valid_samples, num_classes, batch_size, \
               train_labels, valid_labels

    def sampleFromDir(self, sample_size_per_class=20, seed=None, ext='jpg'):
        return sampleFromClassesInDir(self.data_path,
                                      sample_size_per_class=sample_size_per_class, seed=seed, ext=ext,
                                      shape=SHAPE)
