import random
from datetime import datetime
from models.cifar100.data_util_ import getFineGrainedClass
from keras.models import load_model

from util.ordinary import get_transfer_model_name
from util.transfer_util import train

epoch = 20
freezeUntil = 3
REPEAT = 30
num_sample = 50
source_model_name = 'h5/source_model_mixed.h5'

target_names=[]
target_names.append(get_transfer_model_name(freezeUntil, source_model_name, prefix='traditional_transfer'))
target_names.append(get_transfer_model_name(freezeUntil, source_model_name, prefix='target'))

# source_model_name = 'h5/source_model_partial_mixed_2.h5'
# target_names.append(get_transfer_model_name(freezeUntil, source_model_name, prefix='target'))
# target_names.append(get_transfer_model_name(freezeUntil, source_model_name, prefix='traditional_transfer'))

target_names.append(get_transfer_model_name(None, source_model_name, prefix='no_transfer'))

# target_names.append('transfer_model/transfer_combined_' + str(freezeUntil) + '.h5')

seeds = random.sample(range(1, 10000), REPEAT)
print('Number sample per class: ', num_sample)
for target_name in target_names:
    print('Evaluating ' + target_name)

    sacc = 0
    selapse = 0
    for r in range(REPEAT):
        model = load_model(target_name)

        x_train, y_train, x_test, y_test, num_classes = getFineGrainedClass(superclasses=['vehicles 1'],
                                                                            num_sample=num_sample,
                                                                            seed=seeds[r], gray=True)

        # print(model.summary())
        acc, elpase = train(model, x_train, y_train, x_test, y_test, epochs=epoch)
        sacc += acc
        selapse += elpase

    result = (sacc / REPEAT, selapse / REPEAT)
    print(result)

# print(result)
