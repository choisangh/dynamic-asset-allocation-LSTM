import os
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import models
from .model import create_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
SAVE_MODEL = '../models/data/model_weights.h5'


model = models.load_model(SAVE_MODEL)

