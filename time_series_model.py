import tensorflow as tf
from tensorflow.keras import layers
import os
from time_series_btc_eda import make_windows, make_train_test_splits
from predictions import make_preds
from MASE_with_tensorflow import evaluate_preds


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
            verbose=1,
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




