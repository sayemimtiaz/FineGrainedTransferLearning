from keras import Sequential
from keras.layers import Flatten, Dense, Activation, Dropout, Conv2D, MaxPooling2D
from tensorflow import keras

from models.cifar100.data_util import getSuperClassData, getCifar10BinaryData, getCifar10MnistMixed


class CustomSaver(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        self.model.save('h5/source_model_mixed.h5')


takeFromCifar = [1, 9]
takeFromMnist = [0, 2, 3, 4, 5, 6, 7, 8]
x_train, y_train, x_test, y_test, num_classes = getCifar10MnistMixed(takeFromMnist=takeFromMnist,
                                                                     takeFromCifar=takeFromCifar)

model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3), padding='valid', activation='relu', input_shape=x_train.shape[1:]))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))
model.add(Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))
model.add(Conv2D(64, kernel_size=(3, 3), padding='valid', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))
model.add(Conv2D(128, kernel_size=(3, 3), padding='valid', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
# model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))
model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.2))
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
                    callbacks=[saver]
                    )
scores = model.evaluate(x_test, y_test, verbose=2)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

model.save('h5/source_model_mixed.h5')
