import tensorflow as tf
from tensorflow.keras import layers
import os
from time_series_btc_eda import make_windows, make_train_test_splits
from MASE_with_tensorflow import evaluate_preds
from predictions import make_preds

timesteps = []
btc_price = []

HORIZON=1
WINDOW_SIZE=30

tf.random.set_seed(42)


btc_csv_path = '/Users/mac/Desktop/DATASETS/TIME_SERIES_DATA/BTC-USD.csv'


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

# Get bitcoin date array
timesteps = bitcoin_prices.index.to_numpy()
prices = bitcoin_prices["Price"].to_numpy()

full_windows, full_labels = make_windows(prices, window_size=WINDOW_SIZE, horizon=HORIZON)

# Make train & testing windows
train_windows, test_windows, train_labels, test_labels = make_train_test_splits(windows=full_windows, labels=full_windows, test_split=0.2)

# Create a function to implement a ModelCheckpoint callback
def create_model_checkpoint(model_name, save_path="model_experiments"):
  return tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(save_path, model_name),
                                            verbose=0,
                                            save_best_only=True)


# Create model
model_2 = tf.keras.Sequential([
    layers.Dense(128, activation="relu"),
    layers.Dense(HORIZON)  
], name="model_2_dense")

# Compile model
model_2.compile(loss="mae",
                optimizer=tf.keras.optimizers.Adam())

# Fit model
model_2.fit(train_windows,
            train_labels,
            epochs=100,
            batch_size=128,
            validation_data=(test_windows, test_labels),
            callbacks=[create_model_checkpoint(model_name=model_2.name)])


# Evaluate model 2 on test data
model_2.evaluate(test_windows, test_labels)

# Load in best performing model
model_2 = tf.keras.models.load_model("model_experiments/model_2_dense")
model_2.evaluate(test_windows, test_labels)

# Get forecast predictions
model_2_preds = make_preds(model_2, input_data=test_windows)


# Evaluate results for model 2 predictions
model_2_results = evaluate_preds(y_true=tf.squeeze(test_labels), # remove 1 dimension of test labels
                                 y_pred=model_2_preds)

print(model_2_results)