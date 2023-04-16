import matplotlib.pyplot as plt
import numpy as np
import math
import sklearn


# Simple plotting function
def plot_preds(actual, pred, timeframe, column, path):
  plt.figure(figsize=(16, 8))
  plt.grid(False)
  plt.plot(actual, label='Actual')
  plt.plot(pred, label='Forecast')
  plt.legend()
  plt.xlabel('Days')
  plt.ylabel('Price ($)')
  plt.title(f'Actual vs forecast price of {column} ({timeframe})')
  plt.savefig(f'{path}/{column}_{timeframe}_plot.png')
  # plt.show()

# Defines all the accuracy metrics to be used in the analysis
def smape(a, f):
    return 1/len(a) * np.sum(2 * np.abs(f-a) / (np.abs(a) + np.abs(f))*100)

def mse(a, f):
  return math.sqrt(sklearn.metrics.mean_squared_error(a, f))

def mae(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.abs(y_true - predictions))

def getClassSequence(pred, actual):
  signals = []
  i, j = 0, 1
  while j < len(pred):
    if actual[j] > pred[i]:
      signals.append(1)
    elif actual[j] < pred[i]:
      signals.append(-1)
    else:
      signals.append(0)
    j += 1
    i += 1

  return signals


def getTrendSimilarity(actual, forecast):
    actual_signals = getClassSequence(actual, actual)
    forecast_signals = getClassSequence(forecast, forecast)
    trend_similarity = 0
    for i in range(len(actual_signals)):
      if actual_signals[i] == forecast_signals[i]:
        trend_similarity += 1

    trend_similarity = abs(len(actual_signals) - trend_similarity)
    trend_similarity = (1 - trend_similarity/len(actual_signals)) * 100
    return trend_similarity


def get_acc_metrics(actual, pred, verbose=0):
  list_acc = []
  list_acc.append(mae(actual, pred))
  list_acc.append(mse(actual, pred))
  list_acc.append(smape(actual, pred))
  list_acc.append(getTrendSimilarity(actual, pred))
  if verbose != 0:
    print('MAE', mae(actual, pred))
    print('MSE', mse(actual, pred))
    print('sMAPE', smape(actual, pred))
    print('Trend Similarity', getTrendSimilarity(actual, pred))
  return list_acc


