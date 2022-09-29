from keras import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.saving.save import load_model

from models.adult.data_util import get_train_test, get_test_df
from rule_generator.evaluator import class_wise_accuracy

x_train, y_train, x_test,y_test, num_classes = get_train_test()

model = Sequential()

model.add(Flatten(input_shape=(x_train.shape[1:])))
model.add(Dense(256, activation='relu', kernel_initializer='random_normal'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu',kernel_initializer='random_normal'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu',kernel_initializer='random_normal'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu',kernel_initializer='random_normal'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu',kernel_initializer='random_normal'))
model.add(Dropout(0.2))

model.add(Dense(num_classes, activation='sigmoid',kernel_initializer='random_normal'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

epochs = 20
history = model.fit(x_train,
                    y_train,
                    epochs=epochs,
                    batch_size=128,
                    verbose=2)
scores = model.evaluate(x_test, y_test, verbose=2)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

model.save('h5/model_b5.h5')
# class_wise_accuracy(load_model('h5/model_b4.h5'), data_acquisition=get_test_df)
