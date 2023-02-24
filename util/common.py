import os
import random

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


def cropImg(image, width=224, height=224, crop_width=64, crop_height=64):
    left_corner = int(round(width / 2)) - int(round(crop_width / 2))
    top_corner = int(round(height / 2)) - int(round(crop_height / 2))
    cropped = image.crop((left_corner, top_corner, left_corner + crop_width, top_corner + crop_height))
    cropped = cropped.resize((width, height))
    return cropped


def cropImgRandom(image, width=224, height=224, crop_width=64, crop_height=64):
    left_shift = random.randint(0, int((width - crop_width)))
    down_shift = random.randint(0, int((height - crop_height)))
    cropped = image.crop((left_shift, down_shift, crop_width + left_shift, crop_height + down_shift))
    cropped = cropped.resize((width, height))
    return cropped


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
        className = str(sl['label'])
        imgFileName = os.path.join(saveDir, className, imgFileName)

        saveImg(sl['image'], imgFileName)


def get_project_root():
    file_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.dirname(file_path)
    return file_path


def init_gpu():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

# save_tf_dataset_as_image('imagenet2012_subset/1pct')
