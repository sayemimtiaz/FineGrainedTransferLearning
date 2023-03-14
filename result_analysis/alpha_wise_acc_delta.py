import csv
import os
import numpy as np
from data_processing.cifar_specific import getCifar100CoarseClasses
from result_analysis.result_util import parseLine, getAllArchitectures, getBaseline, getTafeObservation
from glob import glob

from util.common import get_project_root

classifierType = 'new'
regularizerType = 'Linear'
# regularizerType = 'dropout'
# regularizerType = 'L2'
# regularizerType = 'L1'
# regularizerType = 'L1_L2'
skipArchs = ['vgg16']

result_path = os.path.join(get_project_root(), 'final_results', classifierType, regularizerType)

csvFiles = [y for x in os.walk(result_path) for y in
            glob(os.path.join(x[0], '*.csv'))]

accs={}
dels={}
for idx, iF in enumerate(csvFiles):

    for arch in getAllArchitectures(iF):
        if arch in skipArchs:
            continue
        baseline = getBaseline(iF, arch)
        if baseline is None:
            continue

        task = baseline.task

        flag = False
        for obs in getTafeObservation(iF, arch):
            accDelta = obs.accuracy - baseline.accuracy

            if obs.alpha not in accs:
                accs[obs.alpha]=[]
                dels[obs.alpha]=[]

            accs[obs.alpha].append(accDelta)
            dels[obs.alpha].append(obs.delRate)


summaryOut = open(os.path.join(get_project_root(), 'final_results', 'processed', 'alpha_wise_delta.csv'), "w")

summaryOut.write("Alpha,Mean accuracy delta,Mean delete rate\n")

for alpha in accs:
    summaryOut.write(alpha+','+str(round(np.asarray(accs[alpha]).mean(), 2))+','+
                     str(round(np.asarray(dels[alpha]).mean(), 2))+'\n')

summaryOut.close()






