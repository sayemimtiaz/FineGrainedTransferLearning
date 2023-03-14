from constants import target_dataset, source_model_name, DONE
from core.evaluate_helper import repeater
from util.ordinary import get_summary_out_name, load_pickle_file, get_delete_rate_name
from util.transfer_util import get_dense_classifier, get_svm_classifier, get_pool_classifier


def evaluate(target_ds=None, parent_model=None):
    if target_ds is None:
        target_ds = target_dataset
    if parent_model is None:
        parent_model = source_model_name

    summaryOut = open(get_summary_out_name(target_ds), "a")
    summaryOut.write(
        "Source,Target,Transfer Type,Classifier Type,Alpha,Epoch,Repeat,Accuracy,STD,Minimum"
        " Accuracy,Maximum Accuracy,Elapsed,Delete Rate\n")
    summaryOut.close()

    epoch = 30
    REPEAT = 10
    batch_size = 32
    alpha_values = [0.0, 1e-45, 1e-35, 1e-25, 1e-15, 1e-5, 0.01, 0.05]
    
    # classfiers = {'pool': get_pool_classifier, 'svm': get_svm_classifier,  'dense': get_dense_classifier}
    # classfiers = {'dense': get_dense_classifier}
    classfiers = {'pool': get_pool_classifier}

    delete_rates = load_pickle_file(get_delete_rate_name(target_ds))

    for cn in classfiers.keys():
        if cn == 'svm':
            epoch = 100

        if parent_model in DONE and target_ds in DONE[parent_model] and cn in DONE[parent_model][target_ds]:
            continue

        print('> Evaluating baseline: ')
        repeater(REPEAT, get_classifier=classfiers[cn], isBaseline=True,
                 batch_size=batch_size, epoch=epoch, classifierType=cn,
                 target_ds=target_ds, parent_model=parent_model)

        print('> Evaluating TAFE: ')
        for alpha in alpha_values:
            print('>> Evaluating alpha rate: ', alpha)
            
            delRate=delete_rates[str(alpha)]
            
            if delRate<100.0:
                repeater(REPEAT, get_classifier=classfiers[cn], alpha=alpha, isBaseline=False,
                         batch_size=batch_size, epoch=epoch, classifierType=cn,
                         delRate=delRate,
                         target_ds=target_ds, parent_model=parent_model)
