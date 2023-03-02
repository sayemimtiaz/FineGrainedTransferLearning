import csv
import os

from data_processing.cifar_specific import getCifar100CoarseClasses
from result_analysis.result_util import parseLine, getAllArchitectures, getBaseline, getTafeObservation
from glob import glob

from util.common import get_project_root

classifierType = 'pool'
skipArchs = ['resnet50']
cifar100Classes = []
for c in getCifar100CoarseClasses():
    cifar100Classes.append(c.replace(' ', ''))

result_path = os.path.join(get_project_root(), 'final_results', classifierType)

csvFiles = [y for x in os.walk(result_path) for y in
            glob(os.path.join(x[0], '*.csv'))]

overallDelta = {}
overallDelRate = {}
overallElapsed = {}
improve_stat = {}
same_or_better = {}
numObs = {}
for idx, iF in enumerate(csvFiles):

    for arch in getAllArchitectures(iF):
        if arch in skipArchs:
            continue
        baseline = getBaseline(iF, arch)
        if baseline is None:
            continue

        task = baseline.task
        if task in cifar100Classes:
            task = 'cifar100'

        flag = False
        bestDelta = 0.0
        delRate = 0.0
        elapsed = 0.0
        for obs in getTafeObservation(iF, arch):
            accDelta = obs.accuracy - baseline.accuracy
            if not flag or bestDelta < accDelta:
                bestDelta = accDelta
                delRate = obs.delRate
                elapsed = obs.elapsed
                flag = True

        if task not in overallDelta:
            overallDelta[task] = 0.0
            overallDelRate[task] = 0.0
            overallElapsed[task] = 0.0
            numObs[task] = 0
            improve_stat[task] = 0.0
            same_or_better[task] = 0.0

        overallDelta[task] += bestDelta
        overallDelRate[task] += delRate
        overallElapsed[task] += ((baseline.elapsed - elapsed) / baseline.elapsed)
        numObs[task] += 1
        if bestDelta > 0:
            improve_stat[task] += 1
        if bestDelta >= 0:
            same_or_better[task] += 1

oAd = 0
oAdr = 0
oAe = 0
oIp = 0
oSb = 0
aCount = 0
for arch in numObs.keys():
    aD = round(overallDelta[arch] / numObs[arch], 2)
    aDr = round(overallDelRate[arch] / numObs[arch], 2)
    aE = round((overallElapsed[arch] / numObs[arch]) * 100.0, 2)
    iP = round((improve_stat[arch] / numObs[arch]) * 100, 2)
    sB = round((same_or_better[arch] / numObs[arch]) * 100, 2)

    oAd += aD
    oAdr += aDr
    oAe += aE
    oIp += iP
    oSb += sB
    aCount += 1

    print('Task: ', arch)
    print('Number observatios: ', numObs[arch])
    print('Improved in ' + str(iP) + '% cases')
    print('Same or better in ' + str(sB) + '% cases')
    print('Average accuracy delta: ' + str(aD) + '%')
    print('Average delete rate: ' + str(aDr) + '%')
    print('Convergance accelration: ' + str(aE) + '%')

    print('\n')

oAd = round(oAd / aCount, 2)
oAdr = round(oAdr / aCount, 2)
oAe = round(oAe / aCount, 2)
oIp = round(oIp / aCount, 2)
oSb = round(oSb / aCount, 2)

print('Overall Improved in ' + str(oIp) + '% cases')
print('Overall Same or better in ' + str(oSb) + '% cases')
print('Overall Average accuracy delta: ' + str(oAd) + '%')
print('Overall Average delete rate: ' + str(oAdr) + '%')
print('Overall Convergance accelration: ' + str(oAe) + '%')
