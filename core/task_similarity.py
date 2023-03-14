import numpy as np
from constants import target_dataset, source_model_name, CURRENT_ACQUIRE, target_datasets, pretrained_architecures, \
    DONE, NUM_CLASSIFIER
from core import (
    getSourceModel,
    smapleTargetData,
)
from core.target_filter_distribution import calculateTargetDistribution
from core.weigted_transfer import similarFeatureRate
from util.common import get_project_root
import os

def get_sample_size(ref_ds):
    sample_size_per_class = 20
    if ref_ds == 'bird':
        sample_size_per_class = 10
    if ref_ds == "stl10":
        sample_size_per_class = 100
    if ref_ds == "mnist":
        sample_size_per_class = 100
    return sample_size_per_class
        
def measure_task_similarity(ds_a, ds_b, parent_model, alpha=0.05):
        
    sample_a = smapleTargetData(
        sample_size_per_class=get_sample_size(ds_a), target_ds=ds_a, crop=False, seed=1
    )
    sample_b = smapleTargetData(
        sample_size_per_class=get_sample_size(ds_b), target_ds=ds_b, crop=False, seed=19
    )

    dist_a, numFilter=calculateTargetDistribution(
        sample_a, target_ds=ds_a, parent_model=parent_model
    )
    
    dist_b, _=calculateTargetDistribution(
        sample_b, target_ds=ds_b, parent_model=parent_model
    )

    simScore = similarFeatureRate(alpha=alpha, sourceRate=dist_a, targetRate=dist_b, numFilter=numFilter)
    print(ds_a, ds_b, simScore)
    return simScore

    


if __name__ == "__main__":
    
    summaryOut = open(os.path.join(get_project_root(), "results", "summary", "task_similarity.csv"), "a")
    summaryOut.write("Architecture,Alpha,Task 1,Task 2,Score\n")
    summaryOut.close()
    tds=['dog', 'bird', 'pet', 'stl10', 'mit67', 'mnist']
    alphas=[0.05, 0.01]
    
    for alpha in alphas:
        for ts in tds:
            for rts in tds:
                for pa in pretrained_architecures:
                    ss=measure_task_similarity(ts, rts, pa,alpha)
                    
                    summaryOut = open(os.path.join(get_project_root(), "results", "summary", "task_similarity.csv"), "a")
                    summaryOut.write(pa+","+str(alpha)+","+ts+","+rts+","+str(ss)+"\n")
                    summaryOut.close()
