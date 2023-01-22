from keras import Sequential
from keras.layers import Flatten, Dense, Activation, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow import keras

from data_util.cifar_specific import getCifar10
from data_util.imagenet_util import TinyImageNet


class CustomSaver(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        self.model.save('h5/simple.h5')


tiny = TinyImageNet(train_data=True)
x_train, y_train, num_classes = tiny.data

# x_train, y_train, x_test, y_test, num_classes=getCifar10(shape=(64,64), gray=False)

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
model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(GlobalAveragePooling2D(name='avg_pool'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
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
                    callbacks=[saver],
                    validation_split=0.1
                    )

# scores = model.evaluate(x_test, y_test, verbose=2)
# print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))