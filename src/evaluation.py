import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from .model import create_model

SAVE_MODEL = '../models/data/model_weights.h5'


model = SAVE_MODEL

with tf.device("/device:GPU:0"):
    model.compile(loss = markowitz_objective,
                optimizer = Adam(learning_rate = 1e-5),
                )
    stop = EarlyStopping(patience=10, monitor='val_loss')

    model.summary()

    hist = model.fit(xc_train, xf_train, epochs=1000, batch_size = 64,
                     #callbacks=[stop],
                    validation_data = (xc_test, xf_test))

model.save(SAVE_MODEL)