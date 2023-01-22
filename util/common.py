import math
import os

import numpy as np
import scipy
from PIL import Image
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def freezeModel(model):
    for layer in model.layers:
        layer.trainable = False
    return model

def displayImg(arr):
    img = Image.fromarray(arr.astype(np.uint8), 'RGB')
    img.show()
