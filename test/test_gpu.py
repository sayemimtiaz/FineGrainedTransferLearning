from keras import Sequential
from keras.layers import GlobalAveragePooling2D, Dense
# from multiprocessing import Process
import multiprocessing as mp
from core import getSourceModel
from data_processing.bird_util import Bird
from data_processing.dog_util import Dog
from util.common import clear_gpu_memory, reinit_gpu
from util.transfer_util import save_bottleneck_data
from numba import cuda
from constants import target_dataset, source_model_name
import numpy as np
from util.ordinary import get_bottleneck_name, get_summary_out_name, load_pickle_file, get_delete_rate_name
import tensorflow as tf

train_generator, valid_generator, nb_train_samples, nb_valid_samples, num_classes, batch_size, train_labels, valid_labels = \
    Dog().getTrainingDogs(shuffle=False, batch_size=32)

train_data = np.load(get_bottleneck_name(target_dataset, 'train', isTafe=False, isLabel=False))
val_data = np.load(get_bottleneck_name(target_dataset, 'valid', isTafe=False, isLabel=False))

batch_size = 32
epochs = 10


def model_runner(model):
    print(epochs)
    # config = tf.compat.v1.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    # config.gpu_options.allow_growth = True
    # tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_data, train_labels,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(val_data, valid_labels))
    reinit_gpu(model)


# def clear_gpu_test():

#     print('testing clear gpu test')

#     for i in range(4):
#         model = Sequential()
#         model.add(GlobalAveragePooling2D(input_shape=train_data.shape[1:]))
#         # model.add(Dense(256, activation='relu'))
#         # model.add(Dropout(0.5))
#         model.add(Dense(120, activation='softmax'))
#         print('memory before: ',cuda.current_context().get_memory_info())
#         # p = Process(target=model_runner, args=(model,))
#         # p.start()
#         # p.join()

#         ctx = mp.get_context('spawn')
#         q = ctx.Queue()
#         p = ctx.Process(target=model_runner, args=(model,))
#         p.start()
#         print(q.get())
#         p.join()

#         print('memory after: ', cuda.current_context().get_memory_info())


# print(cuda.current_context().get_memory_info())

# clear_gpu_test()


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    model = Sequential()
    model.add(GlobalAveragePooling2D(input_shape=train_data.shape[1:]))
    # model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(120, activation='softmax'))

    p = mp.Process(target=model_runner, args=(model,), daemon=True)
    p.start()
    p.join()
