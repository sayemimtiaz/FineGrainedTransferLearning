from keras import Sequential
from keras.layers import Flatten, Dense, Activation, Dropout, Conv2D, MaxPooling2D, Lambda
from models.cifar100.data_util_ import getSuperClassData
from util.transfer_util import stopBackprop
import numpy as np

x_train, y_train, x_test, y_test, num_classes = getSuperClassData(insert_noise=False, dataset='cifar10')
filters=[1, 3]


model = Sequential()
model.add(Conv2D(8, kernel_size=(3, 3), padding='same', activation='relu', input_shape=x_train.shape[1:]))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))

model.add(Conv2D(8, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Lambda(stopBackprop, arguments={'filters': filters}))

# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))


model.add(Conv2D(8, kernel_size=(3, 3), padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

weightBefore1 = model.layers[0].get_weights()
weightBefore2 = model.layers[1].get_weights()
weightBefore3 = model.layers[3].get_weights()

epochs = 2

history = model.fit(x_train,
                    y_train,
                    epochs=epochs,
                    batch_size=50,
                    verbose=2
                    )
scores = model.evaluate(x_test, y_test, verbose=2)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

model.save('h5/stop_test.h5')

weightAfter1 = model.layers[0].get_weights()
weightAfter2 = model.layers[1].get_weights()
weightAfter3 = model.layers[3].get_weights()

print(weightAfter1)
#
# modelBefore = load_model('h5/stop_test.h5')
# modelAfter = load_model('h5/stop_test.h5')
