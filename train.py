import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM
from sklearn.model_selection import train_test_split

import data_wrangling as dw
import evaluate_results as er
from options import args

def prepare_data(args):
    daily_data = pd.read_csv(args.daily_data_path)
    timeframe = args.timeframe

    if timeframe == 'hour':
        data = pd.read_csv(args.hourly_data_path)
    elif timeframe == 'minute':
        data = pd.read_csv(args.minute_data_path)
    else:
        data = pd.read_csv(args.daily_data_path)

    x, y = dw.x_y_split(args.x_range, args.y_range, data, daily_data,
                        timeframe, args.time_column, args.values_column,
                        market_hours=True)

    x_train, x_test = train_test_split(x, test_size=args.test_size, shuffle=False)
    y_train, y_test = train_test_split(y, test_size=args.test_size, shuffle=False)

    return x_train, y_train, x_test, y_test


def train_model(args):
    x_train, y_train, x_test, y_test = prepare_data(args)
    print("Data loaded and properly prepared for training")

    x_train, y_train, max_train, min_train = dw.scale_data(x_train, y_train, 0, 1)
    x_test, y_test, max_test, min_test = dw.scale_data(x_test, y_test, 0, 1)

    # Reshapes the data for LSTM training
    # LSTM expects output in the form of (n_samples, timesteps, features)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],  1) 
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1) 
    y_train = y_train.reshape(y_train.shape[0], y_train.shape[1],  1) 
    y_test = y_test.reshape(y_test.shape[0], y_test.shape[1], 1) 

    # Creates model
    model = Sequential()
    model.add(LSTM(args.y_range, stateful=True))
    model.compile(optimizer='adam', loss='mse')

    # Trains model
    model.fit(x_train, y_train, epochs=args.epochs, batch_size=1, shuffle=False)
    
    tf.keras.models.save_model(model, 
                              f'{args.results_path}/{args.timeframe}_{args.values_column}.h5')
    print("Model sucessfully saved.")
    
    predictions = []
    for i in range(0, len(x_test)):
        predictions.append(model.predict(tf.convert_to_tensor(x_test[i:i+1]), verbose=0)) 
    predictions = np.array(predictions)
    predictions = predictions.reshape((predictions.shape[0], predictions.shape[-1], predictions.shape[1]))

    y, predictions = dw.get_predictions_from_sliding_window(y_test, predictions, args.y_range)

    er.get_acc_metrics(y, predictions, verbose=1)

    y = dw.reverse_scale(y, max_test, min_test)
    predictions = dw.reverse_scale(predictions, max_test, min_test)
    er.plot_preds(y, predictions, args.timeframe, args.values_column, args.results_path)



if __name__ == '__main__':
    prepare_data(args)
    train_model(args)
