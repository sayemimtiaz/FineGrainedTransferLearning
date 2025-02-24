import csv
import os

from result_analysis.result_util import parseLine, getAllArchitectures, getBaseline, getTafeObservation
from glob import glob

from util.common import get_project_root

classifierType = 'pool'

result_path = os.path.join(get_project_root(), 'final_results', classifierType)

csvFiles = [y for x in os.walk(result_path) for y in
            glob(os.path.join(x[0], '*.csv'))]

overallDelta = {}
overallDelRate = {}
overallElapsed = {}
numObs = {}
for idx, iF in enumerate(csvFiles):

    for arch in getAllArchitectures(iF):
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

        if arch not in overallDelta:
            overallDelta[arch]=0.0
            overallDelRate[arch]=0.0
            overallElapsed[arch]=0.0
            numObs[arch]=0

        overallDelta[arch] += bestDelta
        overallDelRate[arch] += delRate
        overallElapsed[arch] += (elapsed-baseline.elapsed)
        numObs[arch] += 1

for arch in numObs.keys():
    aD = overallDelta[arch] / numObs[arch]
    aDr = overallDelRate[arch] / numObs[arch]
    aE = overallElapsed[arch] / numObs[arch]

    print('Architecture: ', arch)
    print('Number observatios: ', numObs[arch])
    print('Average accuracy delta: ', aD)
    print('Average delete rate: ', aDr)
    print('Average elapsed time: ', aE)
    print('\n')
