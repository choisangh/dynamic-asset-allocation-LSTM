import itertools
from keras.optimizers import Adam
from .model import create_model, markowitz_objective


param_grid = {'lstm_units': [16, 32, 64],
              'batch_size': [32, 64, 128]}

param_combinations = list(itertools.product(*(param_grid[param] for param in param_grid)))

best_params = None
best_val_loss = float('inf')


def optimizer(N_TIME, N_STOCKS, xc_train, xf_train):
    for params in param_combinations:
        model = create_model(N_TIME, N_STOCKS)

        model.compile(loss=markowitz_objective,
                      optimizer=Adam(learning_rate=1e-5))

        hist = model.fit(xc_train, xf_train, epochs=1000, batch_size=params[1], validation_data=(xc_test, xf_test))

        val_loss = hist.history['val_loss'][-1]

        if val_loss < best_val_loss:
            best_params = params
            best_val_loss = val_loss
        print('Best params:', best_params)

