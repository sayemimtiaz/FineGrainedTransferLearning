from datetime import datetime
from keras.models import load_model
import tensorflow as tf

from models.cifar100.data_util import getdata
from modularization.concern.concern_identification import ConcernIdentification
from util.common import initModularLayers
from util.layer_propagator import LayerPropagator

x_train, y_train, x_test, y_test, num_classes=getdata()

model_name = 'h5/original.h5'

model = load_model(model_name)
concernIdentifier = LayerPropagator()

print("Start Time:" + datetime.now().strftime("%H:%M:%S"))

concern = initModularLayers(model.layers)
print(concern)


for x in x_test:
    x_t = tf.reshape(x, [-1, x.shape[0], x.shape[1], x.shape[2]])

    print(model.predict(x_t))

    for layerNo, _layer in enumerate(concern):
        x_t = concernIdentifier.propagateThroughLayer(_layer, x_t, apply_activation=True)
    print(x_t)

# input_shape = (4, 28, 28, 3)
# x = tf.random.normal(input_shape)
# y = tf.keras.layers.Conv2D(2, 3, activation='relu',padding='same', input_shape=input_shape[1:])(x)
# print(y.shape)