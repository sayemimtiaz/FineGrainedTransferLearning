import keras
from keras import Sequential, optimizers
from keras.applications import InceptionV3, ResNet50, VGG16, InceptionResNetV2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import GlobalAveragePooling2D, Dropout, Dense
from keras.utils import to_categorical
from keras_preprocessing.image import ImageDataGenerator
import tensorflow_datasets as tfds
import tensorflow as tf
from data_processing.bird_util import getBirdTrainingData

import tensorflow as tf

batch_size = 128
image_height = 224
image_width = 224
img_path='../../data/dogs/images'
# img_path='../../data/CUB_200_2011/CUB_200_2011/images'

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
    target_size=(image_width, image_height),
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=1337
)

valid_generator = valid_datagen.flow_from_directory(
    img_path,
    target_size=(image_width, image_height),
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=True,
    seed=1337
)

num_classes = len(train_generator.class_indices)
# train_labels = train_generator.classes
# train_labels = to_categorical(train_labels, num_classes=num_classes)
# valid_labels = valid_generator.classes
# valid_labels = to_categorical(valid_labels, num_classes=num_classes)
nb_train_samples = len(train_generator.filenames)
nb_valid_samples = len(valid_generator.filenames)

pretrained_model = InceptionV3(input_shape=(image_height, image_width, 3),
                               include_top=False,
                               weights='imagenet')
# pretrained_model = VGG16(input_shape=(image_height, image_width, 3),
#                                include_top=False,
#                                weights='imagenet')

# pretrained_model = InceptionResNetV2(input_shape=(image_height, image_width, 3),
#                                include_top=False,
#                                weights='imagenet')
#
# pretrained_model = ResNet50(weights='imagenet', input_shape=(image_height, image_width, 3), include_top=False)

# Iterate through layers and make untrainable
for layer in pretrained_model.layers:
    layer.trainable = False

model = Sequential()

for layer in pretrained_model.layers:
    layer.trainable = False
#     print(layer,layer.trainable)

model.add(pretrained_model)
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
# model.summary()

earlystop = EarlyStopping(
    monitor='val_loss',
    min_delta=0.001,
    patience=3,
    verbose=1,
    mode='auto'
)
reduceLR = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=3,
    verbose=1,
    mode='auto'
)

callbacks = [earlystop, reduceLR]

model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    epochs=30,
    steps_per_epoch=nb_train_samples // batch_size,
    validation_data=valid_generator,
    validation_steps=nb_valid_samples // batch_size,
    callbacks=callbacks,
    shuffle=True
)
