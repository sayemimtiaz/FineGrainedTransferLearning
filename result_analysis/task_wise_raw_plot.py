import csv
import os

from data_processing.cifar_specific import getCifar100CoarseClasses
from result_analysis.result_util import parseLine, getAllArchitectures, getBaseline, getTafeObservation
from glob import glob

from util.common import get_project_root
import numpy as np

target = 'std'  # std
baselines = ['Linear', 'L1', 'L2', 'L1_L2']
archs = ['inceptionv3', 'xception', 'densenet201', 'vgg16']
tasks = ['dog', 'bird', 'mit67', 'pet', 'stl10']

data = {}
for baselineName in baselines:

    result_path = os.path.join(get_project_root(), 'final_results', 'new', baselineName)

    csvFiles = [y for x in os.walk(result_path) for y in
                glob(os.path.join(x[0], '*.csv'))]

    if baselineName not in data:
        data[baselineName] = {}
        data['TAS - ' + baselineName] = {}

    for idx, iF in enumerate(csvFiles):

        for arch in getAllArchitectures(iF):
            if arch not in archs:
                continue
            baseline = getBaseline(iF, arch)
            if baseline is None:
                continue

            if arch not in data[baselineName]:
                data[baselineName][arch] = {}
                data['TAS - ' + baselineName][arch] = {}

            task = baseline.task

            if target == 'accuracy':
                data[baselineName][arch][task] = baseline.accuracy

            if target == 'std':
                data[baselineName][arch][task] = baseline.std

            flag = False
            bestAcc = 0.0
            chosenStd = 0.0
            for obs in getTafeObservation(iF, arch):
                if not flag or bestAcc < obs.accuracy:
                    bestAcc = obs.accuracy
                    chosenStd = obs.std
                    flag = True

            if target == 'accuracy':
                data['TAS - ' + baselineName][arch][task] = bestAcc
            if target == 'std':
                data['TAS - ' + baselineName][arch][task] = chosenStd

for arch in archs:
    summaryOut = open(os.path.join(get_project_root(), "final_results", "processed",
                                   "raw_" + target + "_" + arch + ".csv"), "w")
    summaryOut.write("Type,Stanford Dogs,Caltech Birds,MIT Indoor,Oxford Pets,STL-10,Mean\n")

    for baselineName in baselines:
        if baselineName == 'Linear':
            tmp = baselineName
        else:
            tmp = 'Linear - ' + baselineName

        summaryOut.write(tmp)

        ss = []
        for task in tasks:
            if task in data[baselineName][arch]:
                ss.append(data[baselineName][arch][task])
                summaryOut.write("," + str(data[baselineName][arch][task]))
            else:
                summaryOut.write(",")

        summaryOut.write("," + str(round(np.asarray(ss).mean(), 2)) + '\n')

    for baselineName in baselines:

        baselineName = 'TAS - ' + baselineName

        summaryOut.write(baselineName)

        ss = []
        for task in tasks:
            if task in data[baselineName][arch]:
                ss.append(data[baselineName][arch][task])
                summaryOut.write("," + str(data[baselineName][arch][task]))
            else:
                summaryOut.write(",")
        summaryOut.write("," + str(round(np.asarray(ss).mean(), 2)) + '\n')
