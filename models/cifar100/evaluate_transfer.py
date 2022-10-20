import random
from datetime import datetime
from models.cifar100.data_util import getFineGrainedClass
from keras.models import load_model
from util.transfer_util import train

epoch = 20
freezeUntil = 3
REPEAT = 20
num_sample = 50
target_names = []
# target_names.append('target_-1_0_0.0')
# target_names.append(target_names[0] + '_random')
target_names.append('traditional_transfer_' + str(freezeUntil))
target_names.append('transfer_' + str(freezeUntil))
target_names.append('no_transfer')
# target_names = ['traditional_transfer', 'no_transfer']


seeds = random.sample(range(1, 10000), REPEAT)
# seeds = [19]
# result={}
for target_name in target_names:
    print('Evaluating ' + target_name)
    target_name = 'transfer_model/' + target_name + '.h5'

    sacc = 0
    selapse = 0
    for r in range(REPEAT):
        model = load_model(target_name)

        x_train, y_train, x_test, y_test, num_classes = getFineGrainedClass(superclass='vehicles 1',
                                                                            num_sample=num_sample,
                                                                            seed=seeds[r], gray=True)

        # print(model.summary())
        acc, elpase = train(model, x_train, y_train, x_test, y_test, epochs=epoch)
        sacc += acc
        selapse += elpase

    result = (sacc / REPEAT, selapse / REPEAT)
    print(result)

# print(result)
