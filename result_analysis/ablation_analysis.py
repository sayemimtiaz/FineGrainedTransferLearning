import csv
import os

from result_analysis.result_util import parseLine, getAllArchitectures, getBaseline, getTafeObservation, getAllAplha, \
    getObsForAblationStudy
from glob import glob

from util.common import get_project_root

result_path = os.path.join(get_project_root(), 'final_results', 'ablation')

csvFiles = [y for x in os.walk(result_path) for y in
            glob(os.path.join(x[0], '*.csv'))]

for idx, iF in enumerate(csvFiles):

    for arch in getAllArchitectures(iF):

        summaryOut = open(os.path.join(get_project_root(), 'result_analysis', 'processed',
                                       iF[iF.rindex('/')+1:-4] + '_' + arch + '.csv'), "w")

        summaryOut.write("Dataset,Architecture,Alpha,Deletion Rate,TAFS,Random\n")

        for alpha in getAllAplha(iF):
            abOb = getObsForAblationStudy(iF, arch, alpha, type='ablation')
            nonAbOb = getObsForAblationStudy(iF, arch, alpha, type='non-ablation')
            if abOb is None:
                print(1)
            delRate = abOb.delRate
            # if abOb.delRate is None:
            #     delRate=0.0

            summaryOut.write(iF[iF.rindex('/')+1:-4]+','+arch+','+alpha+','+str(delRate)+','+
                             str(nonAbOb.accuracy)+','+str(abOb.accuracy)+'\n')

        summaryOut.close()
