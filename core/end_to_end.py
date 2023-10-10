from multiprocessing import Process

from constants import target_dataset, source_model_name, pretrained_architecures, target_datasets, DONE, NUM_CLASSIFIER
from core.acquire_transfer_models import acquire
from core.evaluate import evaluate
from core.evaluate_helper import repeater
from util.common import init_gpu
from util.ordinary import get_summary_out_name, load_pickle_file, get_delete_rate_name
from util.transfer_util import get_dense_classifier, get_svm_classifier, get_pool_classifier

import tensorflow as tf

# init_gpu()

data_sample_rate = 0.25
alpha_value = 0.0

for ts in target_datasets:
    for pa in pretrained_architecures:

        if pa in DONE and ts in DONE[pa] and len(DONE[pa][ts]) == NUM_CLASSIFIER:
            continue
        acquire(target_ds=ts, parent_model=pa)
        evaluate(target_ds=ts, parent_model=pa, alpha=alpha_value, data_sample_rate=data_sample_rate)
