import os
import pandas as pd
import numpy as np

# Function made to pad missing values on *minute data* only
def pad_missing_values(datapath, timestamp='timestamp', verbose=0):

    data = pd.read_csv(datapath)
    data = data.set_index(timestamp)
    padded_data = data.reindex(range(data.index[0], data.index[-1]+60, 60), method='pad')
    if verbose != 0:
        print("Current amount of missing values.")
        print(data.index[1:]-data.index[:-1].value_counts())
        print("Fixed data: ")
        print(padded_data.index[1:]-padded_data.index[:-1].value_counts())

    return padded_data


# Function made to get minute, hourly, or daily data from a higher frequency data
# date_column arg is the column corresponding to the timestamp of the values
# aggregated keys is a *dictionary* containing the columns and the method they should be aggregated
# like {'price':'last, 'timeframe':last}
def get_lower_data_frequency(datapath, timeframe, aggregated_keys, date_column, save_path=False):

    data = pd.read_csv(datapath)

    if timeframe == 'minute':
        timekey = '%Y-%m-%d %H:%MM'
    elif timeframe == 'hour':
        timekey = '%Y-%m-%d %H'
    elif timeframe == 'day':
        timeframe == '%Y-%m-%d'

    groupkey = pd.to_datetime(data[date_column].dt.strftime(timekey))
    new_data = data.groupby(groupkey).agg(aggregated_keys)

    if save_path:
        new_data.to_csv(save_path)

    return new_data


# Function splits a given dataset into (x,y) value pairs, used for the lstm model
# The function automatically applies sliding window
# The y values returned will always be on the daily frame
def x_y_split(x_range, y_range, data, daily_data, timeframe, datetime, column, market_hours=False):
  
  if timeframe == 'day':
    x_len = x_range
  elif timeframe == 'hour':
    if market_hours:
      x_len = 8 * x_range
    else:
      x_len = 24 * x_range
  elif timeframe == 'minute':
    if market_hours:
      x_len = 60 * 8 * x_range
    else:
      x_len = 60 * 24 * x_range
  
  x = np.zeros((len(data)-1-x_len, x_len))
  y = np.zeros((len(data)-1-x_len, y_range))
  y_date = pd.to_datetime(data[datetime][x_len])
  for i in range(0, len(data)-x_len-1):
    temp_x, temp_y = [], []
    temp_x = data[column][i:i+x_len]
    y_date = pd.to_datetime(data[datetime][i+x_len])
    
    while len(temp_y) < y_range:
      temp = daily_data[daily_data[datetime] == y_date.strftime("%Y-%m-%d")][column]
      temp = np.array(temp)
      if len(temp) != 0:
        temp_y.append(temp[0])
      if y_date < pd.to_datetime(daily_data[datetime][len(daily_data)-1]):
        y_date += pd.Timedelta(days=1)
      else:
        if len(temp_y) < y_range:
          temp_y.append(daily_data[column][len(daily_data)-1])

    x[i] += temp_x
    y[i] += temp_y

  return x, y

# Given an x and y set, scales the data between a and b
# Return the scaled x and y, as well as their maximun and minum values (used for rescaling)
def scale_data(x, y, a, b):
  min_ = min(np.amin(x), np.amin(y))
  max_ = max(np.amax(x), np.amax(y))
  x_scaled = (x - min_) / (max_ - min_) * (b - a) + a
  y_scaled = (y - min_) / (max_ - min_) * (b - a) + a
  return x_scaled, y_scaled, max_, min_

# Reverse the data back to normal
def reverse_scale(x_scaled, min_, max_):
  x = x_scaled * (max_ - min_) + min_
  return x

# The predictions are made on x_test, which is slided
# This functiongets the actual predictions, ignoring repeated values
def get_predictions_from_sliding_window(data, predictions_slided, y_range):
    y, predictions = [], []
    for i in range(0, len(data)-1, y_range):
        if not np.array_equal(data[i], data[i+1]):
          y.append(data[i])
          predictions.append(predictions_slided[i])
    y = np.array(y)
    predictions = np.array(predictions)
    return y.flatten(), predictions.flatten()
