import tensorflow as tf

tf.compat.v1.disable_v2_behavior()
from keras.models import load_model

from models.adult.data_util import get_train_test, get_test_df, get_train_df, to_criteria
from rule_generator.evaluator import feature_importance_shap, permutation_importance

model_name = 'h5/model_b4.h5'
model = load_model(model_name)

feature_importance_shap(model, train_data=get_train_df, test_data=get_test_df, classWise=False, num_test_sample=1000, num_train_sample=5000)
features, _ = to_criteria()
permutation_importance(model, features, data_acquisition=get_test_df, num_sample=1000)
