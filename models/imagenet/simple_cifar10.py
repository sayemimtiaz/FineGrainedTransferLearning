from keras import Sequential
from keras.applications import ResNet50
from keras.layers import Flatten, Dense, Activation, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow import keras

from data_util.bird_util import Bird
from data_util.cifar_specific import getCifar10
from data_util.imagenet_util import TinyImageNet


class CustomSaver(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        self.model.save('h5/simple_bird.h5')


# x_train, y_train, x_test, y_test, num_classes = getCifar10(shape=(64, 64), gray=True)

model = ResNet50(weights=None,input_shape=(64,64),include_top=False, classes=num_classes, classifier_activation='softmax')

bird = Bird(gray=True, one_hot=True)
x_train, y_train, x_test, y_test, num_classes=bird.data

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))
model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))
model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))
model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu', name='conv_out'))

model.add(GlobalAveragePooling2D(name='avg_pool'))

model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

epochs = 20

saver = CustomSaver()
history = model.fit(x_train,
                    y_train,
                    epochs=epochs,
                    batch_size=50,
                    verbose=2,
                    callbacks=[saver])

scores = model.evaluate(x_test, y_test, verbose=2)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
