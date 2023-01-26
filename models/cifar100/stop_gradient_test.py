import time

from keras import Sequential, optimizers
from keras.applications import InceptionResNetV2
from keras.layers import Flatten, Dense, Activation, Dropout, Conv2D, MaxPooling2D, Lambda, GlobalAveragePooling2D
from models.cifar100.data_util_ import getSuperClassData
from models.imagenet import getTargetDataForTraining
from util.common import freezeModel
from util.model_hacker import hack_model
from util.transfer_util import stopBackprop, reweight, discardZeros, trainDog
import numpy as np

x_train, y_train, x_test, y_test, num_classes = getSuperClassData(insert_noise=False, dataset='cifar10')
filters = [1, 3]

weight_map = {}
includes=[]
for i in range(1024):
    weight_map[i] = 0
    if i > 1000:
        weight_map[i] = 1
        includes.append(i)

model = Sequential()
model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu',
                 input_shape=x_train.shape[1:], trainable=False))

# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))

model.add(Conv2D(1024, kernel_size=(3, 3), padding='same', activation='relu', trainable=False))
# model.add(Lambda(stopBackprop, arguments={'filters': filters}))
# model.add(Lambda(reweight, arguments={'weights': weight_map}))


# model.add(Lambda(discardZeros, arguments={'weights': weight_map}))

# model.add(Conv2D(8, kernel_size=(3, 3), padding='same', activation='relu'))

model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# train_generator, valid_generator, nb_train_samples, nb_valid_samples, num_classes, batch_size = \
#     getTargetDataForTraining()
#
# inModel=InceptionResNetV2(input_shape=(224,224, 3),
#                                  include_top=False,
#                                  weights='imagenet')
# inModel=hack_model(inModel, 'conv_7b', 'inceptionresnetv2', [0,1])
# inModel=freezeModel(inModel)
#
# model = Sequential()
# model.add(inModel)
# model.add(GlobalAveragePooling2D())
# model.add(Dropout(0.2))
# model.add(Dense(120, activation='softmax'))
#
# start = time.time()
#
# trainDog(model, train_generator, valid_generator, nb_train_samples, nb_valid_samples,
#                                        epoch=1, batch_size=batch_size)
#
# end = time.time()
#
# print('Elapsed time: ', (end - start))

weightBefore1 = model.layers[0].get_weights()
weightBefore2 = model.layers[1].get_weights()
weightBefore3 = model.layers[3].get_weights()

epochs = 2


history = model.fit(x_train,
                    y_train,
                    epochs=epochs,
                    batch_size=50,
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
