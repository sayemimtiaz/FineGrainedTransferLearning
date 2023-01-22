from keras import Sequential
from keras.layers import Flatten, Dense, Activation, Dropout, Conv2D, MaxPooling2D
from tensorflow import keras

from models.mnist.data_util import getMnistData, mnist_class_data

# class CustomSaver(keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs={}):
#         self.model.save('h5/model.h5')

x_train, y_train, x_test, y_test, nb_classes=getMnistData()

model = Sequential()
model.add(Flatten(input_shape=(x_train.shape[1:])))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(nb_classes))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

epochs = 3

# saver = CustomSaver()
history = model.fit(x_train,
                    y_train,
                    epochs=epochs,
                    batch_size=50,
                    verbose=2,
                    validation_split=0.1
                    )
print(history.history['val_accuracy'][0])

# for x_train, y_train, x_test, y_test, nb_classes, heldOutClass in mnist_class_data():
#     print('Class: ', heldOutClass)
#     model = Sequential()
#     model.add(Flatten(input_shape=(x_train.shape[1:])))
#     model.add(Dense(100, activation='relu'))
#     model.add(Dense(100, activation='relu'))
#     model.add(Dense(nb_classes))
#
#     model.compile(loss='categorical_crossentropy',
#                   optimizer='adam',
#                   metrics=['accuracy'])
#
#     epochs = 2
#
#     # saver = CustomSaver()
#     history = model.fit(x_train,
#                         y_train,
#                         epochs=epochs,
#                         batch_size=50,
#                         verbose=2,
#                         validation_split=0.1
#                         )
#     print(history.history['val_accuracy'][0])
#     # scores = model.evaluate(x_test, y_test, verbose=2)
#     # print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
#
#     # model.save('h5/model' + str(heldOutClass) + '.h5')
