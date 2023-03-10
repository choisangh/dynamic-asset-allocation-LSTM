import os
import tensorflow as tf
import tensorflow_probability as tfp
from keras.layers import Activation, Conv1D, MaxPooling1D, Dense, LSTM
from keras.models import Sequential
from src.config import N_TIME, GAMMA_CONST, REG_CONST


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def markowitz_objective(y_true, y_pred):
    """
    y_true: actual returns of assets
    y_pred: predicted portfolio weights
    """

    # y_pred is the portfolio weights we're optimizing for
    W = y_pred

    # xf_rtn is the actual returns of the assets
    xf_rtn = y_true

    # add a dimension to W to make it (None, 1, 50)
    W = tf.expand_dims(W, axis=1)

    # compute the average return of the assets and add a dimension to make it (None, 50, 1)
    R = tf.expand_dims(tf.reduce_mean(xf_rtn, axis=1), axis=2)

    # compute the covariance matrix of the asset returns
    C = tfp.stats.covariance(xf_rtn, sample_axis=1)

    # compute the portfolio return
    rtn = tf.matmul(W, R)

    # compute the portfolio volatility
    vol = tf.matmul(W, tf.matmul(C, tf.transpose(W, perm=[0, 2, 1]))) * GAMMA_CONST

    # compute the L2 regularization term
    reg = tf.reduce_sum(tf.square(W), axis=-1) * REG_CONST

    # compute the Markowitz objective function
    objective = rtn - vol - reg

    # we want to maximize the objective function, but TensorFlow's optimizer minimizes loss functions,
    # so we need to negate the objective function to turn it into a loss function
    return -tf.reduce_sum(objective, axis=0)


def create_model(N_STOCKS):
    """
    Create and return a Keras Sequential model for portfolio optimization using LSTM.

    Args:
    - N_STOCKS (int): The number of stocks in the portfolio

    Returns:
    - model (keras.Sequential): The created Keras Sequential model
    """
    # Create the Sequential model
    model = Sequential()

    # Add a 1D convolutional layer with 64 filters, kernel size 3, and ReLU activation function
    # Input shape is (N_TIME, N_STOCKS)
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(N_TIME, N_STOCKS)))

    # Add a max pooling layer with pool size 2
    model.add(MaxPooling1D(pool_size=2))

    # Add a LSTM layer with 32 units and return sequences
    model.add(LSTM(units=32, return_sequences=True))

    # Add a LSTM layer with 16 units and do not return sequences
    model.add(LSTM(units=16, return_sequences=False))

    # Add a dense layer with N_STOCKS units and hyperbolic tangent activation function
    model.add(Dense(units=N_STOCKS, activation='tanh'))

    # Add a softmax activation layer
    model.add(Activation('softmax'))

    # Return the created model
    return model
