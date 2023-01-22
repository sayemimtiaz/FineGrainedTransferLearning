import os
import pickle


def get_model_name(model_name):
    if model_name.find('/') >= 0:
        model_name = model_name[model_name.rindex('/'):]
    return model_name.replace('h5', '').replace('/', '').replace('.', '')


def get_transfer_model_name(freezeUntil=None, model_name=None, prefix=None):
    root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    name = root+'/transfer_model/'+prefix

    if model_name is not None:
        name += '_' + get_model_name(model_name)
    if freezeUntil is not None:
        name+= '_' + str(freezeUntil)
    name += '.h5'
    return name


def get_transfer_filter_name(model_name=None, mode='val', end=''):
    root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    save_file = root+'/transfer_model/' + end
    if model_name is not None:
        save_file += '_' + get_model_name(model_name)
    save_file += '_' + str(mode)
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
