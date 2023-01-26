import os
import pickle


def get_model_name(model_name):
    if model_name.find('/') >= 0:
        model_name = model_name[model_name.rindex('/'):]
    return model_name.replace('h5', '').replace('/', '').replace('.', '')


def get_transfer_model_name(model_name=None, prefix=None, alpha=None):
    root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    name = root + '/results/transfer_model/' + prefix

    if model_name is not None:
        name += '_' + get_model_name(model_name)
    if alpha is not None:
        name += '_' + get_model_name(str(alpha))

    name += '.h5'
    return name


def get_transfer_filter_name(model_name):
    root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    save_file = root + '/results/observation/'
    save_file += get_model_name(model_name)
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


def get_bottleneck_name(model_name, split, isTafe=True):
    root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    name = root + '/results/bottleneck/'

    name += get_model_name(model_name) + '_' + split
    if isTafe:
        name += '_tafe'
    else:
        name += '_baseline'
    name += '.npy'
    return name
