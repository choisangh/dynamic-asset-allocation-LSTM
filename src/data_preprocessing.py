import numpy as np
from src.config import N_TIME, N_FUTURE


def make_sequence(x):
    T = N_TIME + N_FUTURE
    x_seq = np.expand_dims(np.array(x.iloc[0:T, :]), 0)

    for i in range(1, len(x) - T + 1):
        d = np.expand_dims(np.array(x.iloc[i:(i + T), :]), 0)
        x_seq = np.concatenate((x_seq, d))
    return x_seq


def divide_train_test_set(df, test_ratio):
    test_row = int(df.shape[0] * test_ratio)
    train_df = df.iloc[:test_row]
    test_df = df.iloc[test_row:]
    return train_df, test_df


def data_preprocessing(train_df, test_df):
    rtn_train = make_sequence(train_df)
    rtn_test = make_sequence(test_df)

    xc_train = np.array([x[:N_TIME] for x in rtn_train])
    xf_train = np.array([x[-N_FUTURE:] for x in rtn_train])
    xc_test = np.array([x[:N_TIME] for x in rtn_test])
    xf_test = np.array([x[-N_FUTURE:] for x in rtn_test])

    test_date = test_df[N_TIME:].index

    xc_train = xc_train.astype('float32') * N_TIME
    xf_train = xf_train.astype('float32') * N_TIME
    xc_test = xc_test.astype('float32') * N_TIME
    xf_test = xf_test.astype('float32') * N_TIME

    return xc_train, xf_train, xc_test, xf_test, test_date
