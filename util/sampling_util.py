import numpy as np

from util.common import filter_dataframe


def sample(data_acquisition=None,column_name=None, value=None, frac_sample=1.0, random_state=None, num_sample=None):
    df, target_var, drop_vars = data_acquisition()

    if column_name is not None:
        df = filter_dataframe(df, column_name, value)

    if num_sample is not None:
        df = df.sample(num_sample, random_state=random_state)
    else:
        df = df.sample(frac=frac_sample, random_state=random_state)

    y = df[target_var].values
    x = df.drop(drop_vars, axis=1).values

    xad = None
    if column_name is not None:
        df[column_name] = 0

        xad = df.drop(drop_vars, axis=1).values

    return x, y, xad


def sample_df(data_acquisition=None, num_sample=1.0, random_state=None):
    df, target_var, drop_vars = data_acquisition()

    df = df.sample(num_sample, random_state=random_state)

    return df


def sample_near_adversarial(model, column_name=None, value=None, frac_sample=1.0, random_state=None, densityChange=True,data_acquisition=None):
    df, target_var, drop_vars = data_acquisition()

    df = filter_dataframe(df, column_name, value)

    df = df.sample(frac=frac_sample, random_state=random_state)

    xor = df.drop(drop_vars, axis=1).values

    if len(xor)==0:
        return np.array([]),np.array([]),np.array([]),np.array([]),np.array([])
    p1Raw = model.predict(xor, verbose=0)
    p1 = [[1 * (x[0] >= 0.5)] for x in p1Raw]

    flip_value = 0
    df[column_name] = flip_value
    xad = df.drop(drop_vars, axis=1).values

    p2Raw = model.predict(xad, verbose=0)
    p2 = [[1 * (x[0] >= 0.5)] for x in p2Raw]

    adv_x = []
    adv_y = []
    ben_x = []
    ben_y = []
    t_x = []
    for i in range(len(xor)):
        if p1[i][0] != p2[i][0] or (densityChange and abs(p1Raw[i][0] - p2Raw[i][0]) > 0.3):
            adv_x.append(xor[i])
            adv_y.append(p1[i][0])
            t_x.append(xad[i])
        else:
            ben_x.append(xor[i])
            ben_y.append(p1[i][0])

    return np.asarray(adv_x), np.asarray(adv_y), np.asarray(ben_x), np.asarray(ben_y), \
           np.asarray(t_x)
