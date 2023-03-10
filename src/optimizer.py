import itertools
from keras.optimizers import Adam
from .model import create_model, markowitz_objective
from config import N_TIME


param_grid = {'lstm_units': [16, 32, 64],
              'batch_size': [32, 64, 128]}

param_combinations = list(itertools.product(*(param_grid[param] for param in param_grid)))

best_params = None
best_val_loss = float('inf')


def optimizer(xc_train, xf_train, xc_test, xf_test):
    """
    A function to optimize hyperparameters by training multiple models with different parameter combinations.

    :param xc_train: Training input data (company features)
    :param xf_train: Training output data (future stock returns)
    :param xc_test: Testing input data (company features)
    :param xf_test: Testing output data (future stock returns)
    :return: None
    """
    # iterate over parameter combinations
    for params in param_combinations:
        # create a model with the current parameter combination
        model = create_model(N_TIME, xc_train.shape[2])

        # compile the model with the custom loss function and Adam optimizer
        model.compile(loss=markowitz_objective, optimizer=Adam(learning_rate=1e-5))

        # fit the model to the training data and validate on the testing data
        hist = model.fit(xc_train, xf_train, epochs=1000, batch_size=params[1], validation_data=(xc_test, xf_test))

        # get the validation loss of the trained model
        val_loss = hist.history['val_loss'][-1]

        # if the current model has the best validation loss so far, save the model's parameters and validation loss
        if val_loss < best_val_loss:
            best_params = params
            best_val_loss = val_loss

        # print the best parameters so far
        print('Best params:', best_params)

