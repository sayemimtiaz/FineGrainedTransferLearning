from keras import Sequential
from keras.layers import GlobalAveragePooling2D, Dense

from core import getSourceModel
from data_processing.bird_util import Bird
from data_processing.dog_util import Dog
from util.transfer_util import save_bottleneck_data


def dog_train_test():
    train_generator, valid_generator, nb_train_samples, nb_valid_samples, num_classes, batch_size, train_labels, valid_labels = \
        Dog().getTrainingDogs(shuffle=False, batch_size=32)

    train_data = save_bottleneck_data(getSourceModel(),
                                      train_generator, nb_train_samples,
                                      batch_size, split='train', save=False)
    val_data = save_bottleneck_data(getSourceModel(),
                                    valid_generator, nb_train_samples,
                                    batch_size, split='valid', save=False)

    batch_size = 32
    epochs = 50

    model = Sequential()
    model.add(GlobalAveragePooling2D(input_shape=train_data.shape[1:]))
    # model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(120, activation='softmax'))

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_data, train_labels,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(val_data, valid_labels))

def bird_train_test():
    train_generator, valid_generator, nb_train_samples, nb_valid_samples, num_classes, batch_size, train_labels, valid_labels = \
        Bird().getTrainingBirds(shuffle=False, batch_size=32)
    print(nb_train_samples)

    train_data = save_bottleneck_data(getSourceModel(),
                                      train_generator, nb_train_samples,
                                      batch_size, split='train', save=False)
    print(train_data.shape)
    val_data = save_bottleneck_data(getSourceModel(),
                                    valid_generator, nb_train_samples,
                                    batch_size, split='valid', save=False)

    batch_size = 32
    epochs = 50

    model = Sequential()
    model.add(GlobalAveragePooling2D(input_shape=train_data.shape[1:]))
    # model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(200, activation='softmax'))

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_data, train_labels,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(val_data, valid_labels))


# dog_train_test()
bird_train_test()
