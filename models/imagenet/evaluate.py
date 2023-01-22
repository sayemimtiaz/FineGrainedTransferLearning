import random

from keras.models import load_model

from models.imagenet import target_dataset, getTargetDataForTraining
from util.ordinary import get_transfer_model_name
from util.transfer_util import train

epoch = 10
REPEAT = 5
numSample = 1000

target_names = [get_transfer_model_name(prefix='weighted', model_name=target_dataset),
                get_transfer_model_name(prefix='traditional', model_name=target_dataset)]

seeds = random.sample(range(1, 10000), REPEAT)
for target_name in target_names:
    print('Evaluating ' + target_name)

    sacc = 0
    selapse = 0
    for r in range(REPEAT):
        model = load_model(target_name)

        x_train, y_train, x_test, y_test, num_classes = getTargetDataForTraining(
            numSample=numSample, seed=seeds[r])

        # print(model.summary())
        acc, elpase = train(model, x_train, y_train, x_test, y_test, epochs=epoch)
        sacc += acc
        selapse += elpase

    result = (sacc / REPEAT, selapse / REPEAT)
    print(result)
