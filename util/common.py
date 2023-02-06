import os
import numpy as np
from PIL import Image
from pathlib import Path
import tensorflow_datasets as tfds
import tensorflow as tf


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


def save_tf_dataset_as_image(datasetName, split=None,
                             saveDir='/Users/sayem/Documents/Research/FineGrainedTransferLearning/data/'):
    all_exs = list(tfds.as_numpy(tfds.load(datasetName, split=split)))

    saveDir = os.path.join(saveDir, datasetName)
    # print(all_exs)

    numId = 0
    for spl in all_exs:
        for sl in spl:
            if 'file_name' in sl:
                imgFileName = sl['file_name'].decode("utf-8")
            elif 'image/filename' in sl:
                imgFileName = sl['image/filename'].decode("utf-8")
            else:
                imgFileName = str(numId)+'.jpg'

            className = str(sl['label'])
            imgFileName = os.path.join(saveDir, className, imgFileName)

            # displayImg(sl['image'])
            saveImg(sl['image'], imgFileName)

            numId += 1


def get_project_root():
    file_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.dirname(file_path)
    return file_path


def init_gpu():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


# save_tf_dataset_as_image('cats_vs_dogs', split=['train'])
