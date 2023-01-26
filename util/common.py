import math
import os

import numpy as np
import scipy
from PIL import Image
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pathlib import Path
import tensorflow_datasets as tfds


def freezeModel(model):
    for layer in model.layers:
        layer.trainable = False
    return model


def displayImg(arr):
    img = Image.fromarray(arr.astype(np.uint8), 'RGB')
    img.show()


def saveImg(arr, path, format='JPEG'):
    file_name = path[path.rindex('/') + 1:]
    path = path[:path.rindex('/')]
    Path(path).mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(arr.astype(np.uint8), 'RGB')

    img.save(os.path.join(path, file_name), format=format)


def save_tf_dataset_as_image(datasetName, split='train',
                             saveDir='/Users/sayem/Documents/Research/FineGrainedTransferLearning/data/'):
    all_exs = list(tfds.as_numpy(tfds.load(datasetName, split=split)))

    saveDir = os.path.join(saveDir, datasetName)
    # print(all_exs)

    for sl in all_exs:
        try:
            imgFileName = sl['file_name'].decode("utf-8")
        except:
            imgFileName = sl['image/filename'].decode("utf-8")
        className=str(sl['label'])
        imgFileName = os.path.join(saveDir, className, imgFileName)

        saveImg(sl['image'], imgFileName)


# save_tf_dataset_as_image('imagenet2012_subset/10pct')