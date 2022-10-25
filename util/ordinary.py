import pickle

from data_type.constants import Constants


def get_model_name(model_name):
    if model_name.find('/') >= 0:
        model_name = model_name[model_name.rindex('/'):]
    return model_name.replace('h5', '').replace('/', '').replace('.', '')


def get_transfer_model_name(freezeUntil, model_name, prefix):
    name = 'transfer_model/'+prefix
    name += '_' + get_model_name(model_name)
    if freezeUntil is not None:
        name+= '_' + str(freezeUntil)
    name += '.h5'
    return name


def get_transfer_filter_name(model_name, mode, end):
    save_file = 'transfer_model/' + end
    save_file += '_' + get_model_name(model_name) + '_' + str(mode)
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
