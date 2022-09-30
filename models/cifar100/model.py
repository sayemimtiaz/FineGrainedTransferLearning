from keras import Sequential
from keras.layers import Flatten, Dense, Activation, Dropout, Conv2D, MaxPooling2D
from tensorflow import keras

from models.cifar100.data_util import getdata

x_train, y_train, x_test, y_test, num_classes=getdata()

model = Sequential()

model.add(Conv2D(64, (5, 5), padding='same', input_shape=x_train.shape[1:], activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

epochs = 5

history = model.fit(x_train,
                    y_train,
                    epochs=epochs,
                    batch_size=128,
                    verbose=2)
scores = model.evaluate(x_test, y_test, verbose=2)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

model.save('h5/original.h5')