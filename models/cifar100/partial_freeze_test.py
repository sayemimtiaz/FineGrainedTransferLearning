import math
import numpy as np
from keras import Model

from models.cifar100 import MODE, target_dataset, getSourceModel, getTargetData, getTargetNumClass
from util.hypothesis_testing import getPValue, isSameDistribution
from util.ordinary import load_pickle_file, get_transfer_filter_name, get_transfer_model_name, dump_as_pickle
from util.transfer_util import construct_reweighted_target, construct_target_partial_update, train

rnks = load_pickle_file('transfer_model/filter_ranked.pickle')
target_num_class = getTargetNumClass()

freeze_rates = [0.05, 0.15, 0.30, 0.5, 0.75]
sample_rates = [0.25, 0.5, 0.75, 1.0]
epoch = 10
REPEAT = 10


def getN(rnks, p, top=True):
    rnks = dict(sorted(rnks.items(), key=lambda item: item[1], reverse=top))

    p = int(p * len(rnks))
    i = 0
    r = []
    for k in rnks.keys():
        if rnks[k]==0:
            continue
        if i >= p:
            break
        r.append(k)
        i += 1
    return r


def eval(model, sample_rate=1.0):
    x_train, y_train, x_test, y_test, num_classes = getTargetData(sample_rate=sample_rate,
                                                                  one_hot=True, gray=True)

    sacc = 0
    selapse = 0
    for r in range(REPEAT):
        acc, elpase = train(model, x_train, y_train, x_test, y_test, epochs=epoch)
        sacc += acc
        selapse += elpase

    result = sacc / REPEAT
    print(result)
    return result


def situation(fls, sr):
    model = getSourceModel()
    model = Model(inputs=model.input, outputs=model.layers[-2].output)
    for layer in model.layers:
        layer.trainable = False
    model.layers[-1].trainable = True
    target_model = construct_target_partial_update(model, n_classes=target_num_class, filters=fls)

    return eval(target_model, sample_rate=sr)


out = open("result/partial_freeze_test.csv", "w")
out.write('Sample Rate,Freeze Rate,Accuracy (Best), Accuracy (Worst)\n')
delta = 0
cnt = 0
for _v in freeze_rates:
    for sr in sample_rates:
        print('Evaluating for sample rate: ' + str(sr))

        # best filters frozen
        fls = getN(rnks, _v, top=True)

        print('Freezing best ' + str(_v * 100) + '% filters')
        aBest = situation(fls, sr)

        # worst filters frozen
        fls = getN(rnks, _v, top=False)

        print('Freezing worst ' + str(_v * 100) + '% filters')
        aWorst = situation(fls, sr)

        out.write(str(sr * 100.0) + ',' + str(_v * 100.0) + ',' + str(aBest) + ',' + str(aWorst) + '\n')
        delta += (aBest - aWorst)
        cnt += 1

print('Average delta', (delta / cnt))
