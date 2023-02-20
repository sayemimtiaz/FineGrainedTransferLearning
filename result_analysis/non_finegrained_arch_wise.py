import csv
import os

from result_analysis.result_util import parseLine, getAllArchitectures, getBaseline, getTafeObservation
from glob import glob

from util.common import get_project_root

classifierType = 'pool'
exclude=['pet', 'cats_vs_dogs','dog', 'bird']
targetArchiteture='vgg16'

cifar100tasks=['aquaticmammals', 'fish', 'flowers', 'foodcontainers','fruitandvegetables', 'householdelectricaldevices','householdfurniture',
      'insects','largecarnivores', 'largeman-madeoutdoorthings', 'largenaturaloutdoorscenes', 'largeomnivoresandherbivores',
     'medium-sizedmammals', 'non-insectinvertebrates',
      'people', 'reptiles', 'trees', 'vehicles1', 'smallmammals', 'vehicles2']

result_path = os.path.join(get_project_root(), 'final_results', classifierType)

csvFiles = [y for x in os.walk(result_path) for y in
            glob(os.path.join(x[0], '*.csv'))]

overallDelta = {}
overallDelRate = {}
overallElapsed = {}
numObs = {}
for idx, iF in enumerate(csvFiles):

    taskName=iF[iF.rfind('/')+1:-4]
    if taskName in exclude:
        continue

    for arch in getAllArchitectures(iF):
        if arch !=targetArchiteture:
            continue
        baseline = getBaseline(iF, arch)

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

        if baseline.task not in overallDelta:
            overallDelta[baseline.task]={}
            overallDelRate[baseline.task]={}
            overallElapsed[baseline.task]={}
            numObs[baseline.task]={}
        if arch not in overallDelta[baseline.task]:
            overallDelta[baseline.task][arch]=0.0
            overallDelRate[baseline.task][arch]=0.0
            overallElapsed[baseline.task][arch]=0.0
            numObs[baseline.task][arch]=0

        overallDelta[baseline.task][arch] += bestDelta
        overallDelRate[baseline.task][arch] += delRate
        overallElapsed[baseline.task][arch] += (elapsed-baseline.elapsed)
        numObs[baseline.task][arch] += 1

for task in numObs.keys():
    for arch in numObs[task].keys():
        aD = overallDelta[task][arch] / numObs[task][arch]
        aDr = overallDelRate[task][arch] / numObs[task][arch]
        aE = overallElapsed[task][arch] / numObs[task][arch]

        print('(task, Architecture): ', task,arch)
        print('Number observatios: ', numObs[task][arch])
        print('Average accuracy delta: ', aD)
        print('Average delete rate: ', aDr)
        print('Average elapsed time: ', aE)
        print('\n')
