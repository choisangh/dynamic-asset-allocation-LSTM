import os
import numpy as np
from src.config import N_TIME, N_FUTURE
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model

from keras.utils import custom_object_scope
from src.model import markowitz_objective
from src.dataset import test_df, xc_test, xf_test


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
SAVE_MODEL = './models/model_weights.h5'

with custom_object_scope({'markowitz_objective': markowitz_objective}):
    model = load_model(SAVE_MODEL)
test_date = test_df.index
N_STOCKS = xc_test.shape[2]


def back_testing():
    lstm_model = model

    lstm_value = [10000]  # portfolio의 초기 value

    w_hist_lstm = []
    for i in range(0, xc_test.shape[0], N_FUTURE):
        x = xc_test[i][np.newaxis, :, :]
        w_lstm = lstm_model.predict(x)[0]
        w_hist_lstm.append(w_lstm)
        m_rtn = np.sum(xf_test[i] / 100, axis=0)
        lstm_value.append(lstm_value[-1] * np.exp(np.dot(w_lstm, m_rtn)))

    print('Back test를 완료했습니다.')

    idx = np.arange(0, xc_test.shape[0], N_FUTURE)
    perf_df = pd.DataFrame({'lstm_markowitz': lstm_value[:-1]},
                           index=test_date[idx])
    p = perf_df.plot(figsize=(12, 7), style='-', fontsize=12)
    p.legend(fontsize=12)
    plt.show()
