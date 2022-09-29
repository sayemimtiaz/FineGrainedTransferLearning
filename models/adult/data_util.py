import os
import random

import numpy as np
import pandas as pd
import warnings

import sklearn
from keras.saving.save import load_model
from sklearn.ensemble import RandomForestRegressor

from data_type.constants import Constants
from util.common import filter_dataframe
from util.ordinary import dump_as_pickle, load_pickle_file

warnings.filterwarnings("ignore")


def get_country_index(country_name):
    top_countries = ['N/A', '?', 'United-States', 'England', 'Puerto-Rico', 'Canada',
                     'Germany',
                     'India', 'South', 'China', 'Cuba',
                     'Philippines', 'Italy', 'Jamaica', 'Mexico', 'Dominican-Republic',
                     'El-Salvador']
    if country_name in top_countries:
        return top_countries.index(country_name), country_name
    return len(top_countries), 'Other'


def get_country_name(index):
    top_countries = ['N/A', '?', 'United-States', 'England', 'Puerto-Rico', 'Canada',
                     'Germany',
                     'India', 'South', 'China', 'Cuba',
                     'Philippines', 'Italy', 'Jamaica', 'Mexico', 'Dominican-Republic',
                     'El-Salvador']
    if index < len(top_countries):
        return top_countries[index]
    return 'Other'


def get_education_index(education):
    top_education = ['N/A', '?', 'Bachelors', 'Some-college', 'HS-grad', 'Prof-school', 'Assoc-acdm',
                     'Assoc-voc', 'Masters', 'Doctorate']

    if education in top_education:
        return top_education.index(education), education

    if education in ['Preschool', '1st-4th', '5th-6th']:
        return len(top_education), 'Upto 6th'
    if education in ['7th-8th', '9th']:
        return len(top_education) + 1, 'Upto 9th'
    if education in ['11th', '10th', '12th']:
        return len(top_education) + 2, 'Upto 12th'


def get_education_name(index):
    top_education = ['N/A', '?', 'Bachelors', 'Some-college', 'HS-grad', 'Prof-school', 'Assoc-acdm',
                     'Assoc-voc', 'Masters', 'Doctorate']
    if index < len(top_education):
        return top_education[index]
    if index == len(top_education):
        return 'Upto 6th'
    if index == len(top_education) + 1:
        return 'Upto 9th'
    if index == len(top_education) + 2:
        return 'Upto 12th'


def get_workclass_index(workcls):
    ws = ['N/A', '?', 'Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov',
          'State-gov', 'Without-pay',
          'Never-worked']
    return ws.index(workcls), workcls


def get_workclass_name(index):
    ws = ['N/A', '?', 'Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov',
          'State-gov', 'Without-pay',
          'Never-worked']
    return ws[index]


def get_marital_index(marital_status):
    ws = ['N/A', '?', 'Married-civ-spouse', 'Divorced', 'Never-married', 'Separated',
          'Widowed',
          'Married-spouse-absent', 'Married-AF-spouse']
    return ws.index(marital_status), marital_status


def get_marital_name(index):
    ws = ['N/A', '?', 'Married-civ-spouse', 'Divorced', 'Never-married', 'Separated',
          'Widowed',
          'Married-spouse-absent', 'Married-AF-spouse']
    return ws[index]


def get_occupation_index(occ):
    ws = ['N/A', '?', 'Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial',
          'Prof-specialty',
          'Handlers-cleaners',
          'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving',
          'Priv-house-serv',
          'Protective-serv', 'Armed-Forces']
    return ws.index(occ), occ


def get_occupation_name(index):
    ws = ['N/A', '?', 'Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial',
          'Prof-specialty',
          'Handlers-cleaners',
          'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving',
          'Priv-house-serv',
          'Protective-serv', 'Armed-Forces']
    return ws[index]


def get_relationship_index(rel):
    ws = ['N/A', '?', 'Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative',
          'Unmarried']
    return ws.index(rel), rel


def get_relationship_name(index):
    ws = ['N/A', '?', 'Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative',
          'Unmarried']
    return ws[index]


def get_race_index(rel):
    ws = ['N/A', '?', 'White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
    return ws.index(rel), rel


def get_race_name(index):
    ws = ['N/A', '?', 'White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
    return ws[index]


def get_gender_index(rel):
    ws = ['N/A', '?', 'Female', 'Male']
    return ws.index(rel), rel


def get_gender_name(index):
    ws = ['N/A', '?', 'Female', 'Male']
    return ws[index]


def get_income_bracket_index(rel):
    ws = ['<=50K', '>50K']
    return ws.index(rel), rel


def get_income_bracket_name(index):
    ws = ['<=50K', '>50K']
    return ws[index]


def get_age_index(age):
    normalized = age / 90.0
    if age < 13:
        return normalized, 'Child'
    if age < 19:
        return normalized, 'Teenager'
    if age < 25:
        return normalized, 'Young Adult'
    if age < 36:
        return normalized, 'Adult'
    if age < 51:
        return normalized, 'Mid-age'
    if age < 66:
        return normalized, 'Elder'
    return normalized, 'Senior'


def get_age_name(index):
    age = index * 90.0
    if age < 13:
        return 'Child'
    if age < 19:
        return 'Teenager'
    if age < 25:
        return 'Young Adult'
    if age < 36:
        return 'Adult'
    if age < 51:
        return 'Mid-age'
    if age < 66:
        return 'Elder'
    return 'Senior'


def get_fnwgt_index(v):
    return v / 1484705, v


def get_fnwgt_name(v):
    return v * 1484705


def get_education_num_index(v):
    return v / 16.0, v


def get_education_num_name(v):
    return v * 16.0


def get_capital_gain_index(v):
    return v / 9999.0, v


def get_capital_gain_name(v):
    return v * 9999.0


def get_capital_loss_index(v):
    return v / 4356.0, v


def get_capital_loss_name(v):
    return v * 4356.0


def get_hours_per_week_index(v):
    return v / 99.0, v


def get_hours_per_week_name(v):
    return v * 99.0


COLUMNS = {
    "workclass": ('categorical', 'x', get_workclass_index, get_workclass_name),
    "education": ('categorical', 'x', get_education_index, get_education_name),
    "marital_status": ('categorical', 'x', get_marital_index, get_marital_name),
    "occupation": ('categorical', 'x', get_occupation_index, get_occupation_name),
    "relationship": ('categorical', 'x', get_relationship_index, get_relationship_name),
    "race": ('categorical', 'x', get_race_index, get_race_name),
    "gender": ('categorical', 'x', get_gender_index, get_gender_name),
    "native_country": ('categorical', 'x', get_country_index, get_country_name),
    'income_bracket': ('categorical', 'y', get_income_bracket_index, get_income_bracket_name),

    'fnlwgt': ('continuous', 'x', get_fnwgt_index, get_fnwgt_name),
    "age": ('continuous', 'x', get_age_index, get_age_name),
    "education_num": ('continuous', 'x', get_education_num_index, get_education_num_name),
    "capital_gain": ('continuous', 'x', get_capital_gain_index, get_capital_gain_name),
    "capital_loss": ('continuous', 'x', get_capital_loss_index, get_capital_loss_name),
    "hours_per_week": ('continuous', 'x', get_hours_per_week_index, get_hours_per_week_name)
}


def preprocess():
    all_cols = [
        "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
        "occupation", "relationship", "race", "gender", "capital_gain", "capital_loss",
        "hours_per_week", "native_country", "income_bracket"
    ]
    base_path = os.path.dirname(os.path.realpath(__file__))
    df_train = pd.read_csv(base_path + '/data/adult.data', names=all_cols)
    df_train.dropna(how='any', axis=0)
    # print(df_train['native_country'].value_counts())
    df_test = pd.read_csv(base_path + '/data/adult.test', skiprows=1, names=all_cols)
    df_test.dropna(how='any', axis=0)

    df = df_train.append(df_test, ignore_index=True)

    CONTINUOUS_COLUMNS = [
        'fnlwgt', "age", "education_num", "capital_gain", "capital_loss", "hours_per_week"
    ]
    df[CONTINUOUS_COLUMNS] = df[CONTINUOUS_COLUMNS].astype('float32')

    for i, row in df.iterrows():

        for cc in COLUMNS.keys():
            if COLUMNS[cc][0] == 'categorical':
                v = df.at[i, cc].strip().replace('.', '')
                df.at[i, cc] = COLUMNS[cc][2](v)[0]
            else:
                df.at[i, cc] = COLUMNS[cc][2](df.at[i, cc])[0]

    CATEGORICAL_COLUMNS = [
        "workclass", "education", "marital_status", "occupation", "relationship",
        "race", "gender", "native_country", 'income_bracket'
    ]
    df[CATEGORICAL_COLUMNS] = df[CATEGORICAL_COLUMNS].astype(int)

    dump_as_pickle(df, base_path + '/data/processed.pickle')

    return df


def load():
    base_path = os.path.dirname(os.path.realpath(__file__))
    df = load_pickle_file(base_path + '/data/processed.pickle')
    return df


def get_train_test(features=None):
    df = load()
    if features is not None:
        df = df[features]
    y_test = df[32561:].income_bracket.values
    X_test = df[32561:].drop(['income_bracket'], axis=1).values

    df = df[0:32561]
    df = sklearn.utils.shuffle(df)
    y_train = df['income_bracket'].values
    X_train = df.drop(['income_bracket'], axis=1).values

    # print(type(X_train))
    return X_train, y_train, X_test, y_test, 1


def get_train_df():
    df = load()

    df = df[0:32561]

    vs = get_target_and_drop_vars()
    return df, vs[0], vs[1]


def get_test_df():
    df = load()

    df = df[32561:]

    vs = get_target_and_drop_vars()
    return df, vs[0], vs[1]


def get_whole_df():
    df = load()

    vs = get_target_and_drop_vars()
    return df, vs[0], vs[1]


def get_target_and_drop_vars():
    return 'income_bracket', ['income_bracket']


def to_criteria():
    criteria_columns = [
        "age", "workclass", "education", "marital_status",
        "occupation", "relationship", "race", "gender", "native_country", "education_num",
        "hours_per_week", "capital_gain", "capital_loss", "fnlwgt"]
    # criteria_columns = ["workclass", "education", "marital_status",
    #                     "occupation", "relationship", "race", "gender", "native_country", "education_num"]
    value_map = {}
    if Constants.MODE == 1:
        value_map['workclass'] = ['=1', '=2', '=3', '=4', '=5', '=6', '=7', '=8', '=9']

        value_map['education'] = ['=1', '=2', '=3', '=4', '=5', '=7', '=8', '=9',
                                  '=10', '=11', '=12']
        value_map['marital_status'] = ['=1', '=2', '=3', '=4', '=5', '=6', '=7', '=8']
        value_map['occupation'] = ['=1', '=2', '=3', '=4', '=5', '=7', '=8', '=9',
                                   '=10', '=11', '=12', '=13', '=14', '=15']
        value_map['relationship'] = ['=1', '=2', '=3', '=4', '=5', '=6', '=7']
        value_map['race'] = ['=1', '=2', '=3', '=4', '=5', '=6']
        value_map['gender'] = ['=1', '=2', '=3']
        value_map['native_country'] = ['=1', '=2', '=3', '=4', '=5', '=7', '=8', '=9',
                                       '=10', '=11', '=12', '=13', '=14', '=15', '=16', '=17']
        # value_map['age'] = [('>0.0', '<0.14'), ('>=0.14', '<0.21'), ('>=0.21', '<0.28'),
        #                     ('>=0.28', '<0.4'), ('>=0.4', '<0.57'),
        #                     ('>=0.57', '<0.73'), ('>=0.73', '<=1.0')]
        value_map['age'] = ['>0.0']
        value_map['education_num'] = ['>0.0']
        value_map['hours_per_week'] = ['>0.0']

        value_map['capital_gain'] = ['>0.0']
        value_map['capital_loss'] = ['>0.0']
        value_map['fnlwgt'] = ['>0.0']
    else:
        value_map['workclass'] = ['>0']
        value_map['education'] = ['>0']
        value_map['marital_status'] = ['>0']
        value_map['occupation'] = ['>0']
        value_map['relationship'] = ['>0']
        value_map['race'] = ['>0']
        value_map['gender'] = ['>0']
        value_map['native_country'] = ['>0']
        value_map['age'] = ['>0']
        value_map['education_num'] = ['>0']
        value_map['hours_per_week'] = ['>0']
        value_map['capital_gain'] = ['>0']
        value_map['capital_loss'] = ['>0']
        value_map['fnlwgt'] = ['>0']

    return criteria_columns, value_map


def from_criteria(column_name, criteria):
    if type(criteria) != tuple:

        if criteria.startswith('='):
            v = int(criteria[1:])

            return column_name, COLUMNS[column_name][3](v)
    else:
        if criteria[1][0] != '=':
            return column_name, criteria[1]

        v = float(criteria[1][1:])

        return column_name, COLUMNS[column_name][3](v)

    return column_name, None

# print(get_train_test())
# preprocess()
