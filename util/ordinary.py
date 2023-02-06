import os
import pickle
from pathlib import Path


def get_model_name(model_name):
    if model_name.find('/') >= 0:
        model_name = model_name[model_name.rindex('/'):]
    return model_name.replace('h5', '').replace('/', '').replace('.', '')


def get_delete_rate_name(model_name=None):
    root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    name = root + '/results/observation/'

    if model_name is not None:
        name += '_' + get_model_name(model_name)

    name += '.pickle'
    return name

def get_transfer_model_name(model_name=None, isBaseline=True, alpha=None, type=None):
    root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    name = root + '/results/transfer_model/'
    if isBaseline:
        name += 'baseline'
    else:
        name += 'weighted'

    if model_name is not None:
        name += '_' + get_model_name(model_name)
    if alpha is not None and not isBaseline:
        name += '_' + get_model_name(str(alpha))
    if type is not None:
        name += '_' + type

    name += '.h5'
    return name


def get_transfer_filter_name(model_name, dataset=None, num_obs=None):
    root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    save_file = root + '/results/observation/'
    save_file += get_model_name(model_name)
    if dataset is not None:
        save_file += '_'+get_model_name(dataset)
    if num_obs is not None:
        save_file += '_'+str(num_obs)

    save_file += '.pickle'

    return save_file


def load_pickle_file(file_name):
    data = None
    with open(file_name, 'rb') as handle:
        data = pickle.load(handle)
    return data


def dump_as_pickle(data, file_name):
    with open(file_name, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_bottleneck_name(dataset, split, isTafe=True, isLabel=False, alpha=None):
    root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    name = root + '/results/bottleneck/' + get_model_name(dataset) + '/'

    Path(name).mkdir(parents=True, exist_ok=True)

    name += split
    if isLabel:
        name += '_label'
    else:
        name += '_feature'

        if alpha is not None and isTafe:
            name += '_' + get_model_name(str(alpha))

        if isTafe:
            name += '_tafe'
        else:
            name += '_baseline'

    name += '.npy'
    return name


def get_summary_out_name(dataset):
    root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    save_file = root + '/results/summary/'
    save_file += dataset
    save_file += '.csv'

    return save_file
