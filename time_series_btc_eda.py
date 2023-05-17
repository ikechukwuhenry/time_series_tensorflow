# Import and format historical Bitcoin data with Python
import csv
from datetime import datetime
import imp
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
import os
from MASE_with_tensorflow import evaluate_preds
from predictions import make_preds

# from the tutorial below
# https://www.mlq.ai/time-series-with-tensorflow-downloading-bitcoin-data/

timesteps = []
btc_price = []

tf.random.set_seed(42)

btc_csv_path = '/Users/mac/Desktop/DATASETS/TIME_SERIES_DATA/BTC-USD.csv'
with open(btc_csv_path, "r") as f:
  csv_reader = csv.reader(f, delimiter=",")
  next(csv_reader) # skipe the column titles line
  for line in csv_reader:
    timesteps.append(datetime.strptime(line[0], "%Y-%m-%d")) # get the dates as dates, not strings
    btc_price.append(float(line[4])) # get the closing price as a float

# View first 10 of each
time_info, btc_info = timesteps[:10], btc_price[:10]
# print((time_info, btc_info))

# Importing time series data with Pandas
# import with pandas
import pandas as pd
# Read in Bitcoin data and parse the dates
df = pd.read_csv(btc_csv_path,
                 parse_dates=["Date"],
                 index_col=["Date"])
print(df.head())

print((df.info()))
print((len(df)))
bitcoin_prices = pd.DataFrame(df["Close"]).rename(columns={"Close": "Price"})


import matplotlib.pyplot as plt
bitcoin_prices.plot(figsize=(10, 7))
plt.ylabel("BTC Price")
plt.title("Bitcoin Price from Sept 14, 20XX to Oct 09, 2022", fontsize=16)
plt.legend(fontsize=14)
# plt.show()

# Get bitcoin date array
timesteps = bitcoin_prices.index.to_numpy()
prices = bitcoin_prices["Price"].to_numpy()
# print(prices)

# split the data into 80% for training and 20% for testing:

# Create train and test sets
split_size = int(0.8 * len(prices)) # 80% train, 20% test

# Create train data splits
X_train, y_train = timesteps[:split_size], prices[:split_size]

# Create test data splits
X_test, y_test = timesteps[split_size:], prices[split_size:]

# Create train and test sets
split_size = int(0.8 * len(prices)) # 80% train, 20% test

# Create train data splits
X_train, y_train = timesteps[:split_size], prices[:split_size]

# Create test data splits
X_test, y_test = timesteps[split_size:], prices[split_size:]

# print(split_size)
#  visualize our training and test data:

# Plot train and test split
plt.figure(figsize=(10,7))
plt.scatter(X_train, y_train, s=5, label="Train data")
plt.scatter(X_test, y_test, s=5, label="Test data")
plt.xlabel("Date")
plt.ylabel("BTC Price")
plt.legend(fontsize=14)
plt.show()

# create a plotting function to make visualizing our time series data less tedious

# Create a function to plot time series data
def plot_time_series(timesteps, values, format=".", start=0, end=None, label=None):
  # plot the series
  plt.plot(timesteps[start:end], values[start:end], format, label=label)
  plt.xlabel("Time")
  plt.ylabel("BTC Price")
  if label:
    plt.legend(fontsize=14)
  plt.grid(True)
  # plt.figure(figsize=(10,7))
  plt.show()


# test our plotting function

plot_time_series(timesteps=X_train, values=y_train, label="Train Data")
plot_time_series(timesteps=X_test, values=y_test, label="Test Data")

# Create naive forecast
naive_forecast = y_test[:-1]

print(f"We want to use {btc_price[:7]} to predict this {btc_price[7]}")

# global variables for window and horizon size
HORIZON = 1
WINDOW_SIZE = 7

# Create a function to label windowed data
def get_labelled_window(x, horizon=HORIZON):
  """
  Create labels for windowed dataset
  E.g if horizon = 1
  Input: [0, 1,  2, 3, 4, 5, 6, 7] -> Output: ([0, 1,  2, 3, 4, 5, 6], [7])
  """
  return x[:, :-horizon], x[:, -horizon]


# Test our window labelling function
test_window, test_label = get_labelled_window(tf.expand_dims(tf.range(8), axis=0))
print(f"Window: {tf.squeeze(test_window).numpy()} -> Label: {tf.squeeze(test_label).numpy()}")

# Create a function to view NumPy arrays as windows
def make_windows(x, window_size=WINDOW_SIZE, horizon=HORIZON):
  """
  Turns a 1D array into a 2D array of sequential labelled windows of window_size with horizon size label.
  """
  # 1. Create a window of specific window_size (add the horizon on the end for labelling later)
  window_step = np.expand_dims(np.arange(window_size+horizon), axis=0)

  # 2. Create a 2D array of multiple window steps (minus 1 to account for 0 indexing)
  window_indexes = window_step + np.expand_dims(np.arange(len(x)-(window_size+horizon-1)), axis=0).T # create 2D array of windows of window size

  print(f"Window indexex:\n {window_indexes, window_indexes.shape}")
  
  # 3. Index on the target array with 2D array of multiple window sets
  windowed_array = x[window_indexes]

  #4. Get the labelled windows
  windows, labels = get_labelled_window(windowed_array, horizon=horizon)
  return windows, labels


make_windows(prices, window_size=WINDOW_SIZE, horizon=HORIZON)

full_windows, full_labels = make_windows(prices, window_size=WINDOW_SIZE, horizon=HORIZON)

# View the first 3 windows & labels
for i in range(3):
  print(f"Window: {full_windows[i]} -> Label: {full_labels[i]}")

# Make train/test splits
def make_train_test_splits(windows, labels, test_split=0.2):
  """
  Splits matching pairs of windows and labels into train and test splits.
  """
  split_size = int(len(windows) * (1-test_split)) # this will default to 80% train, 20% test
  train_windows = windows[:split_size]
  train_labels = labels[:split_size]
  test_windows = windows[split_size:]
  test_labels = labels[split_size:]
  return train_windows, test_windows, train_labels, test_labels


# Create train and test windows
train_windows, test_windows, train_labels, test_labels = make_train_test_splits(full_windows, full_labels)
len(train_windows), len(test_windows), len(train_labels), len(test_labels)



# Create a function to implement a ModelCheckpoint callback
def create_model_checkpoint(model_name, save_path="model_experiments"):
  return tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(save_path, model_name),
                                            verbose=2,
                                            save_best_only=True)


# 1. Construct model
model_1 = tf.keras.Sequential([
  layers.Dense(128, activation="relu"),
  layers.Dense(HORIZON, activation="linear") 
], name="model_1_dense") 

# 2. Compile model
model_1.compile(loss="mae",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["mae"]) 

# 3. Fit model
model_1.fit(x=train_windows,
            y=train_labels, 
            epochs=100,
            verbose=2,
            batch_size=128,
            validation_data=(test_windows, test_labels),
            callbacks=[create_model_checkpoint(model_name=model_1.name)])


# Evaluate model on test data
model_1.evaluate(test_windows, test_labels)

# Load in saved best perfomring model_1 and evaluate it on test data
model_1 = tf.keras.models.load_model("model_experiments/model_1_dense")
model_1.evaluate(test_windows, test_labels)



model_1_preds = make_preds(model_1, test_windows)
print(len(model_1_preds), model_1_preds[:10])

# We can now evaluate these forecasts with our evaluate_preds() function

model_1_results = evaluate_preds(y_true=test_labels, y_pred=model_1_preds)
print(model_1_results)


# # Plot model_1 predictions
# offset = 300
# plt.figure(figsize=(10, 7))

'''
# Account for the test_window offset and index into test_labels to ensure correct plotting
plot_time_series(timesteps=X_test[-len(test_windows):], 
                 values=test_labels[:, 0], start=offset, label="Test_data")

plot_time_series(timesteps=X_test[-len(test_windows):], 
                 values=model_1_preds, start=offset, format="-", label="model_1_preds")
'''