from keras.applications import InceptionV3, ResNet50

from data_util.bird_util import getBirdTrainingData

import tensorflow as tf

batch_size = 128
image_height = 128
image_width = 128
# Prepare validation data
train_ds, val_ds, _ = getBirdTrainingData(shape=(image_height, image_width))

pretrained_model = InceptionV3(input_shape=(image_height, image_width, 3),
                               include_top=False,
                               weights='imagenet')

# pretrained_model = ResNet50(weights='imagenet', input_shape=(image_height, image_width, 3), include_top=False)

# Iterate through layers and make untrainable
for layer in pretrained_model.layers:
    layer.trainable = False

# Flatten the output, and add fully connected layer with a node for each class
x = tf.keras.layers.Flatten()(pretrained_model.output)
x = tf.keras.layers.Dense(256, activation='tanh')(x)
x = tf.keras.layers.Dropout(0.1)(x)
x = tf.keras.layers.Dense(200, activation='softmax')(x)

# Adjust learning rate while training with LearningRateScheduler
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10 ** ((100 - epoch) / 20))
optimizer = tf.keras.optimizers.Adam()

model = tf.keras.Model(pretrained_model.input, x)
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    epochs=100,
    shuffle=False,
    batch_size=batch_size,
    validation_data=val_ds,
    callbacks=[lr_scheduler]
)

# print(history)
