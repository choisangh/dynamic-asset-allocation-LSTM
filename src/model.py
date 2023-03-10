import os
import tensorflow as tf
import tensorflow_probability as tfp
from keras.layers import Activation, Conv1D, MaxPooling1D, Dense, LSTM
from keras.models import Sequential
from keras.optimizers import Adam
from keras import layers, models
from keras.utils import plot_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
GAMMA_CONST = 0.15
REG_CONST = 0.0


def markowitz_objective(y_true, y_pred):
    W = y_pred
    xf_rtn = y_true
    W = tf.expand_dims(W, axis=1)  # W = (None, 1, 50)
    R = tf.expand_dims(tf.reduce_mean(xf_rtn, axis=1), axis=2)  # R = (None, 50, 1)
    C = tfp.stats.covariance(xf_rtn, sample_axis=1)
    rtn = tf.matmul(W, R)
    vol = tf.matmul(W, tf.matmul(C, tf.transpose(W, perm=[0, 2, 1]))) * GAMMA_CONST
    reg = tf.reduce_sum(tf.square(W), axis=-1) * REG_CONST
    objective = rtn - vol - reg

    return -tf.reduce_sum(objective, axis=0)


def create_model(N_TIME, N_STOCKS):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(N_TIME, N_STOCKS)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(units=32, return_sequences=True))
    model.add(LSTM(units=16, return_sequences=False))
    model.add(Dense(units=N_STOCKS, activation='tanh'))
    model.add(Activation('softmax'))
    return model




model = create_model()
with tf.device("/device:GPU:0"):
    model.compile(loss = markowitz_objective,
                optimizer = Adam(learning_rate = 1e-5),
                )
    model.summary()

    hist = model.fit(xc_train, xf_train, epochs=1000, batch_size = 64,
                    validation_data = (xc_test, xf_test))

model.save(SAVE_MODEL)