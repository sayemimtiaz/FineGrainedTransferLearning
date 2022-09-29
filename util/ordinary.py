import pickle

from data_type.constants import Constants


def get_model_name(model_name):
    if model_name.find('/') >= 0:
        model_name = model_name[model_name.rindex('/'):]
    return model_name.replace('h5', '').replace('/', '').replace('.', '')


def get_feature_module_name(model_name, datasetName):
    nm = datasetName + '/modules/feature_neurons_' + get_model_name(model_name)
    nm += '_' + str(Constants.MODE) + '.pickle'
    return nm


def get_global_feature_name(model_name, datasetName, activeFeatureOnly=True):
    save_file = datasetName + '/result/global/'
    if activeFeatureOnly:
        save_file += 'active_'
    else:
        save_file += 'inactive_'

    save_file += get_model_name(model_name)
    save_file += '_' + str(Constants.MODE) + '.pickle'

    return save_file


def get_local_feature_name(model_name, cls, datasetName, activeFeatureOnly=True):
    save_file = datasetName + '/result/local/'
    if activeFeatureOnly:
        save_file += 'active_'
    else:
        save_file += 'inactive_'

    save_file += get_model_name(model_name) + '_' + str(cls)
    save_file += '_' + str(Constants.MODE) + '.pickle'

    return save_file


def get_semi_feature_name(model_name, cls, datasetName, activeFeatureOnly=True):
    save_file = datasetName + '/result/semi/'
    if activeFeatureOnly:
        save_file += 'active_'
    else:
        save_file += 'inactive_'

    save_file += get_model_name(model_name) + '_' + str(cls)
    save_file += '_' + str(Constants.MODE) + '.pickle'

    return save_file


def load_pickle_file(file_name):
    data = None
    with open(file_name, 'rb') as handle:
        data = pickle.load(handle)
    return data


def dump_as_pickle(data, file_name):
    with open(file_name, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
