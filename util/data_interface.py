from models.adult import data_util as adult_util

from models.titanic import data_util as titanic_util

from models.bank import data_util as bank_util


def from_criteria_interface(column_name, criteria, datasetName):
    if 'adult' in datasetName:
        return adult_util.from_criteria(column_name, criteria)
    if 'titanic' in datasetName:
        return titanic_util.from_criteria(column_name, criteria)
    if 'bank' in datasetName:
        return bank_util.from_criteria(column_name, criteria)


def get_train_df_interface(datasetName):
    if 'adult' in datasetName:
        return adult_util.get_train_df
    if 'titanic' in datasetName:
        return titanic_util.get_train_df
    if 'bank' in datasetName:
        return bank_util.get_train_df


def get_test_df_interface(datasetName):
    if 'adult' in datasetName:
        return adult_util.get_test_df
    if 'titanic' in datasetName:
        return titanic_util.get_test_df
    if 'bank' in datasetName:
        return bank_util.get_test_df


def get_whole_df_interface(datasetName):
    if 'adult' in datasetName:
        return adult_util.get_whole_df
    if 'titanic' in datasetName:
        return titanic_util.get_whole_df
    if 'bank' in datasetName:
        return bank_util.get_whole_df


def get_target_and_drop_vars_interface(datasetName, asVar=False):
    if 'adult' in datasetName:
        if asVar:
            return adult_util.get_target_and_drop_vars
        else:
            return adult_util.get_target_and_drop_vars()
    if 'titanic' in datasetName:
        if asVar:
            return titanic_util.get_target_and_drop_vars
        else:
            return titanic_util.get_target_and_drop_vars()
    if 'bank' in datasetName:
        if asVar:
            return bank_util.get_target_and_drop_vars
        else:
            return bank_util.get_target_and_drop_vars()


def to_criteria_interface(datasetName):
    if 'adult' in datasetName:
        return adult_util.to_criteria()
    if 'titanic' in datasetName:
        return titanic_util.to_criteria()
