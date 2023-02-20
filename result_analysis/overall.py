import csv
import os

from result_analysis.result_util import parseLine, getAllArchitectures, getBaseline, getTafeObservation
from glob import glob

from util.common import get_project_root

classifierType = 'pool'

result_path = os.path.join(get_project_root(), 'final_results', classifierType)

csvFiles = [y for x in os.walk(result_path) for y in
            glob(os.path.join(x[0], '*.csv'))]
overallDelta = 0.0
overallDelRate = 0.0
overallElapsed = 0.0
numObs = 0
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

        overallDelta += bestDelta
        overallDelRate += delRate
        overallElapsed += (elapsed-baseline.elapsed)
        numObs += 1

overallDelta = overallDelta / numObs
overallDelRate = overallDelRate / numObs
overallElapsed = overallElapsed / numObs

print('Number observatios: ', numObs)
print('Average accuracy delta: ', overallDelta)
print('Average delete rate: ', overallDelRate)
print('Average elapsed time: ', overallElapsed)
