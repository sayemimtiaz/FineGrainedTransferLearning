from datetime import datetime
from keras.models import load_model
import tensorflow as tf

from data_util.cifar_specific import sampleCifar10

from util.common import initModularLayers
from util.layer_propagator import LayerPropagator

x_train, y_train, x_test, y_test, num_classes = sampleCifar10(superclasses=[], num_sample=10)

model_name = 'h5/source_model_cifar10.h5'
model = load_model(model_name)
# model.pop()
# model.pop()
concernIdentifier = LayerPropagator()

print("Start Time:" + datetime.now().strftime("%H:%M:%S"))

concern = initModularLayers(model.layers)
print(concern)

for x in x_test:
    x_t = tf.reshape(x, [-1, x.shape[0], x.shape[1], x.shape[2]])

    print("Keras prediction: ", model.predict(x_t))

    for layerNo, _layer in enumerate(concern):
        x_t = concernIdentifier.propagateThroughLayer(_layer, x_t, apply_activation=True)
    print("Our prediction: ", x_t)

# input_shape = (4, 28, 28, 3)
# x = tf.random.normal(input_shape)
# y = tf.keras.layers.Conv2D(2, 3, activation='relu',padding='same', input_shape=input_shape[1:])(x)
# print(y.shape)
