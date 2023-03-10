from src.model import create_model, markowitz_objective
from src.dataset import xc_train, xf_train, xc_test, xf_test, N_STOCKS
from keras.optimizers import Adam
import tensorflow as tf
from src.evaluation import back_testing


if __name__ == "__main__":
    model = create_model(N_STOCKS)
    print('Start training the model')
    with tf.device("/device:GPU:0"):
        model.compile(loss=markowitz_objective,
                      optimizer=Adam(learning_rate=1e-5),
                      )
        model.summary()

        hist = model.fit(xc_train, xf_train, epochs=1000, batch_size=64,
                         validation_data=(xc_test, xf_test))

    model.save('./models/model_weights.h5')
    print('Model training completed')

    back_testing()
