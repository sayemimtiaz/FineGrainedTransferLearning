from keras import Sequential, Model, regularizers
from keras.applications import ResNet50, InceptionV3
from keras.layers import Flatten, Dense, Activation, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D, \
    BatchNormalization, MaxPool2D
from tensorflow import keras
import tensorflow as tf
from data_util.bird_util import Bird

h=64
w=64
bird = Bird(gray=False, one_hot=True, interpolation=True, shape=(h,w))
x_train, y_train, x_test, y_test, num_classes = bird.data

print('Train size: ', x_train.shape)

# base_model = ResNet50(weights='imagenet', input_shape=(64, 64, 3),include_top=False)
base_model = InceptionV3(input_shape=(h,w, 3),
                               include_top=False,
                               weights='imagenet')

# x = base_model.output
# x = Flatten()(x)
# predictions = Dense(200, activation='softmax')(x)
x = tf.keras.layers.Flatten()(base_model.output)
x = tf.keras.layers.Dense(256, activation='tanh')(x)
x = tf.keras.layers.Dropout(0.1)(x)
x = tf.keras.layers.Dense(200, activation='softmax')(x)

for layer in base_model.layers:
    layer.trainable = False

model = Model(inputs=base_model.input, outputs=x)

# epochs = 5
#
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
#
# # saver = CustomSaver()
# history = model.fit(x_train,
#                     y_train,
#                     epochs=epochs,
#                     batch_size=64,
#                     verbose=2)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10 ** ((100 - epoch) / 20))
optimizer = tf.keras.optimizers.Adam()

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

scores = model.evaluate(x_test, y_test, verbose=2)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
