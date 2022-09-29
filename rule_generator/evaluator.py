#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 21:38:03 2020

@author:
"""
import os
import numpy as np
import shap
from keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report

from util.common import filter_dataframe
from util.sampling_util import sample_near_adversarial
import pandas as pd


def evaluate_contrast(model, module, feature, featureValue, coverageFlag={}, data_acquisition=None):
    ad_x, ad_y, b_x, b_y, _ = sample_near_adversarial(model, feature, featureValue, data_acquisition=data_acquisition,
                                                      densityChange=False)
    test_data = [(ad_x, ad_y, 'accuracy supposed to decrease for this set'),
                 (b_x, b_y, 'accuracy supposed to remain same for this set')]

    diff = {}
    for (xt, yt, annot) in test_data:
        if len(xt) < 1:
            diff[annot] = 'Too few test sample'
            continue
        xt = np.asarray(xt)
        yt = np.asarray(yt)
        # print(annot)

        if annot not in coverageFlag:
            coverageFlag[annot] = set()

        length = len(yt)
        modulePred = module.predict(xt[:length], verbose=0)
        modulePred = [[1 * (x[0] >= 0.5)] for x in modulePred]

        modelPred = model.predict(xt[:length], verbose=0)
        modelPred = [[1 * (x[0] >= 0.5)] for x in modelPred]

        from sklearn.metrics import accuracy_score

        score1 = accuracy_score(yt[:length], modelPred)

        score2 = accuracy_score(yt[:length], modulePred)

        diff[annot] = round(score2 - score1, 2)

        for i in range(length):
            if 'decrease' in annot and modulePred[i][0] != modelPred[i][0]:
                coverageFlag[annot].add(i)
            elif 'same' in annot and modulePred[i][0] == modelPred[i][0]:
                coverageFlag[annot].add(i)

    return diff


def shap_values(explainer, x_test, feature_names, plot=False):
    shap_values = explainer.shap_values(x_test)

    if plot:
        shap.summary_plot(shap_values[0], plot_type='bar', feature_names=feature_names)

    shap_df = pd.DataFrame(shap_values[0], columns=feature_names)
    vals = np.abs(shap_df.values).mean(0)
    shap_importance = pd.DataFrame(list(zip(feature_names, vals)), columns=['col_name', 'feature_importance_vals'])
    shap_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
    print(shap_importance)


def feature_importance_shap(model, train_data=None, test_data=None, classWise=True, num_test_sample=-1, num_train_sample=-1):
    train_df, target, drop = train_data()
    if num_train_sample != -1:
        train_df = train_df.sample(num_train_sample)
    explainer = shap.DeepExplainer(model, train_df.drop(drop, axis=1).values)
    nb_class=2
    if classWise:
        for c in range(nb_class):
            test_df, target, drop = test_data()
            test_df = filter_dataframe(test_df, target, c)
            x_test = test_df.drop(drop, axis=1).values
            feature_names = test_df.drop(drop, axis=1).columns

            shap_values(explainer, x_test, feature_names)
    else:
        test_df, target, drop = test_data()
        if num_test_sample != -1:
            test_df = test_df.sample(num_test_sample, random_state=19)
        x_test = test_df.drop(drop, axis=1).values
        feature_names = test_df.drop(drop, axis=1).columns
        shap_values(explainer, x_test, feature_names, plot=True)


def permutation_importance(model, features, data_acquisition=None, num_sample=-1, random_state=19, repeat=5):
    acc = []
    for f in features:
        df,target,drop = data_acquisition()
        if num_sample != -1:
            df = df.sample(num_sample, random_state=random_state)

        xor = df.drop(drop, axis=1).values
        yor = df[target].values

        p = model.predict(xor, verbose=0)
        p = [[1 * (x[0] >= 0.5)] for x in p]
        scoreBase = accuracy_score(yor, p)

        score = 0.0
        for r in range(repeat):
            df[f] = df[f].sample(frac=1).values

            xad = df.drop(drop, axis=1).values

            p = model.predict(xad, verbose=0)
            p = [[1 * (x[0] >= 0.5)] for x in p]

            score += accuracy_score(yor, p)

        score /= repeat
        acc.append((f, score - scoreBase))

    acc.sort(key=lambda a: a[1])

    print('Important features (most to least important)')
    for (k, s) in acc:
        print(k, s)


# def removal_importance(model_name, cls):
#     targetFeature = ['Age-bin', 'Sex', 'Pclass', 'Family', 'Miss', 'Master', 'Mr', 'Mrs', 'Others', 'Fare-bin',
#                      'Ticket_1', 'Ticket_2', 'Ticket_3', 'Ticket_C', 'Ticket_P', 'Ticket_S']
#
#     model_path = os.path.dirname(os.path.realpath(__file__))
#
#     model_name = os.path.join(model_path, model_name)
#     model = load_model(model_name)
#
#     acc = []
#     for f in targetFeature:
#         df = load()
#         df = df[0:891]  # test set only
#         df = filter_dataframe(df, 'Survived', cls)
#         xor = df.drop(['Survived', 'PassengerId'], axis=1).values
#         yor = df['Survived'].values
#
#         p = model.predict(xor, verbose=0)
#         p = [[1 * (x[0] >= 0.5)] for x in p]
#         scoreBase = accuracy_score(yor, p)
#
#         df[f] = 0
#
#         xad = df.drop(['Survived', 'PassengerId'], axis=1).values
#
#         p = model.predict(xad, verbose=0)
#         p = [[1 * (x[0] >= 0.5)] for x in p]
#
#         score = accuracy_score(yor, p)
#
#         acc.append((f, score - scoreBase))
#
#     acc.sort(key=lambda a: a[1])
#
#     print('Important features for class ' + str(cls) + '(most to least important)')
#     for (k, s) in acc:
#         print(k, s)
#
#
# def removal_importance_value(model_name, cls):
#     targetFeature = ['Age-bin', 'Sex', 'Pclass', 'Family', 'Miss', 'Master', 'Mr', 'Mrs', 'Others', 'Fare-bin',
#                      'Ticket_1', 'Ticket_2', 'Ticket_3', 'Ticket_C', 'Ticket_P', 'Ticket_S']
#     targetValues = {'Sex': [1, 2], 'Pclass': [1, 2, 3], 'Family': [1, 2, 3], 'Miss': [1],
#                     'Master': [1], 'Mr': [1], 'Mrs': [1], 'Others': [1],
#                     'Fare-bin': ['<3', '>=3'], 'Age-bin': ['<4', '>=4'],
#                     'Ticket_1': [1], 'Ticket_2': [1], 'Ticket_3': [1],
#                     'Ticket_C': [1], 'Ticket_P': [1], 'Ticket_S': [1]}
#     model_path = os.path.dirname(os.path.realpath(__file__))
#
#     model_name = os.path.join(model_path, model_name)
#     model = load_model(model_name)
#
#     acc = []
#     for f in targetFeature:
#         for v in targetValues[f]:
#             df = load()
#             df = df[0:891]  # test set only
#             df = filter_dataframe(df, 'Survived', cls)
#             df = filter_dataframe(df, f, v)
#             xor = df.drop(['Survived', 'PassengerId'], axis=1).values
#             yor = df['Survived'].values
#
#             p = model.predict(xor, verbose=0)
#             p = [[1 * (x[0] >= 0.5)] for x in p]
#             scoreBase = accuracy_score(yor, p)
#
#             df[f] = 0
#
#             xad = df.drop(['Survived', 'PassengerId'], axis=1).values
#
#             p = model.predict(xad, verbose=0)
#             p = [[1 * (x[0] >= 0.5)] for x in p]
#
#             score = accuracy_score(yor, p)
#
#             acc.append((f, v, score - scoreBase))
#
#     acc.sort(key=lambda a: a[2])
#
#     print('Important features for class ' + str(cls) + '(most to least important)')
#     for (k, v, s) in acc:
#         print(k, v, s)
#
#
# def removal_test_for_incorrect_rules(model_name, cls, feature, featureValue):
#     targetFeature = ['Age-bin', 'Sex', 'Pclass', 'Family', 'Miss', 'Master', 'Mr', 'Mrs', 'Others', 'Fare-bin',
#                      'Ticket_1', 'Ticket_2', 'Ticket_3', 'Ticket_C', 'Ticket_P', 'Ticket_S']
#     targetValues = {'Sex': [1, 2], 'Pclass': [1, 2, 3], 'Family': [1, 2, 3], 'Miss': [1],
#                     'Master': [1], 'Mr': [1], 'Mrs': [1], 'Others': [1],
#                     'Fare-bin': ['<3', '>=3'], 'Age-bin': ['<4', '>=4'],
#                     'Ticket_1': [1], 'Ticket_2': [1], 'Ticket_3': [1],
#                     'Ticket_C': [1], 'Ticket_P': [1], 'Ticket_S': [1]}
#     model_path = os.path.dirname(os.path.realpath(__file__))
#
#     model_name = os.path.join(model_path, model_name)
#     model = load_model(model_name)
#
#     df = load()
#     df = df[0:891]  # test set only
#     df = filter_dataframe(df, 'Survived', cls)
#     df = filter_dataframe(df, feature, featureValue)
#
#     xor = df.drop(['Survived', 'PassengerId'], axis=1).values
#     yor = df['Survived'].values
#
#     fIdx = df.columns.get_loc(feature)
#
#     p = model.predict(xor, verbose=0)
#     p = [[1 * (x[0] >= 0.5)] for x in p]
#     xad = []
#     yad = []
#     acc = []
#     for i in range(len(yor)):
#         if (cls == 1 and not p[i][0]) or (cls == 0 and p[i][0]):
#             xad.append(xor[i])
#             yad.append(cls)
#     xad = np.asarray(xad)
#     yad = np.asarray(yad)
#     for v in targetValues[feature]:
#         if v == featureValue:
#             continue
#         for i in range(len(xad)):
#             xad[i][fIdx] = v
#
#         p = model.predict(xad, verbose=0)
#         p = [[1 * (x[0] >= 0.5)] for x in p]
#
#         score = accuracy_score(yad, p)
#
#         acc.append((feature, v, score))
#
#     acc.sort(key=lambda a: a[2])
#
#     for (k, v, s) in acc:
#         print(k, v, s)


def class_wise_accuracy(model, df=None, more_rows_to_drop=None, data_acquisition=None, just_accuracy=False):
    if df is None:
        df, target_var, drop_vars = data_acquisition()
    else:
        target_var, drop_vars = data_acquisition()
    dr = drop_vars
    if more_rows_to_drop is not None:
        dr.extend(more_rows_to_drop)
    xor = df.drop(dr, axis=1).values
    yor = df[target_var].values

    p = model.predict(xor, verbose=0)
    p = [[1 * (x[0] >= 0.5)] for x in p]
    if not just_accuracy:
        print(classification_report(yor, p))

    print('Accuracy', accuracy_score(yor, p))


def class_wise_accuracy_from_path(model_name):
    model_path = os.path.dirname(os.path.realpath(__file__))

    model_name = os.path.join(model_path, model_name)
    model = load_model(model_name)
    class_wise_accuracy(model)
