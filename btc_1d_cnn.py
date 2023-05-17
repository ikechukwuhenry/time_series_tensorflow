# Import and format historical Bitcoin data with Python
import csv
from datetime import datetime

timesteps = []
btc_price = []

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
  
  plt.show()


# test our plotting function
# plt.figure(figsize=(10,7))
plot_time_series(timesteps=X_train, values=y_train, label="Train Data")
plot_time_series(timesteps=X_test, values=y_test, label="Test Data")

# Create naive forecast
naive_forecast = y_test[:-1]

# If we look at these values, we see that with a horizon of 1 
# the naive forecast is simply predicting the previous timestep 
# as the next steps value.
#  Note we need to offset X_test by 1 for our naive forecast plot
#  since it won't have a value for the first timestep.

# plt.figure(figsize=(10,7))
plot_time_series(timesteps=X_train, values=y_train, label="Train data")
plot_time_series(timesteps=X_test, values=y_test, label="Test data")
plot_time_series(timesteps=X_test[1:], values=naive_forecast, format="-", label="Naive Forecast")

from MASE_with_tensorflow import mean_absolute_scaled_error, evaluate_preds

mase = mean_absolute_scaled_error(y_true=y_test[1:], y_pred=naive_forecast).numpy()
print(mase)

naive_results = evaluate_preds(y_true=y_test[1:],
                               y_pred=naive_forecast)

print(naive_results)

