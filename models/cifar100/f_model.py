import keras
from keras import Sequential, Input, Model
from keras.layers import Flatten, Dense, Activation, Dropout, Conv2D, MaxPooling2D, concatenate

from models.cifar100.data_util import getSuperClassData


def compose_conv_layers(num_layer, in_layer):
    cl = []
    for i in range(num_layer):
        cl.append(Conv2D(1, kernel_size=(3, 3), padding='same', activation='relu')(in_layer))

    return concatenate(cl)


class CustomSaver(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        self.model.save('h5/source_f_model.h5')


x_train, y_train, x_test, y_test, num_classes = getSuperClassData(insert_noise=False, dataset='cifar10')

inputs = Input(shape=x_train.shape[1:])

y = compose_conv_layers(64, inputs)
y = MaxPooling2D(pool_size=(2, 2))(y)
y = Dropout(0.2)(y)

y = compose_conv_layers(64, y)
y = MaxPooling2D(pool_size=(2, 2))(y)
y = Dropout(0.2)(y)

y = compose_conv_layers(64, y)
y = MaxPooling2D(pool_size=(2, 2))(y)
y = Dropout(0.2)(y)

y = compose_conv_layers(64, y)
y = MaxPooling2D(pool_size=(2, 2))(y)
y = Dropout(0.2)(y)

y = Flatten()(y)
y = Dense(256, activation='relu')(y)
y = Dropout(0.2)(y)
outputs = Dense(num_classes, activation='softmax')(y)

model = Model(inputs=inputs, outputs=outputs)

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

model.save('h5/source_f_model.h5')
