import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

data = yf.download('BRK-B', start='2000-01-01', end='2024-12-31')


ma_100_days = data.Close.rolling(100).mean()

plt.figure(figsize=(8, 6))
plt.plot(ma_100_days, 'r')
plt.plot(data.Close, 'g')
plt.show()

ma_200_days = data.Close.rolling(200).mean()

plt.figure(figsize=(8, 6))
plt.plot(ma_200_days, 'r')
plt.plot(data.Close, 'g')
plt.show()

data.dropna(inplace=True)

data_train = pd.DataFrame(data.Close[0: int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80): len(data)])

data_train.shape[0]

data_test.shape[0]

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))

data_train_scale = scaler.fit_transform(data_train)

x = []
y = []

for i in range(100, data_train_scale.shape[0]):
    x.append(data_train_scale[i - 100:i])
    y.append(data_train_scale[i, 0])

x, y = np.array(x), np.array(y)

from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

model = Sequential()
model.add(LSTM(units=50, activation='relu', return_sequences=True,
               input_shape=((x.shape[1], 1))))
model.add(Dropout(0, 2))

model.add(LSTM(units=60, activation='relu', return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(units=80, activation='relu', return_sequences=True))
model.add(Dropout(0.4))

model.add(LSTM(units=120, activation='relu', return_sequences=True))
model.add(Dropout(0.5))

model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x, y, epochs=50, batch_size=32, verbose=1)

past_100_days = data_train.tail(100)

data_test = pd.concat([past_100_days, data_test], ignore_index=True)

data_test_scale = scaler.fit_transform(data_test)

x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i - 100:i])
    y.append(data_test_scale[i, 0])
x, y = np.array(x), np.array(y)

y_predict = model.predict(x)

scale = 1 / scaler.scale_

y_predict = y_predict * scale

y = y * scale

print("y_predict shape: ", np.shape(y_predict))
print("y shape:", np.shape(y))
plt.figure(figsize=(10, 8))

# Assuming y_predict has shape (1207,) and you want to repeat it along the other dimensions
y_predict_reshaped = np.tile(y_predict, (1, 100, 1))

plt.plot(y_predict_reshaped[:, 0, 0], 'r', label='Predicted Price')
plt.plot(y[:, 0, 0], 'g', label='Original Price')

plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.ylim(min(np.min(y), np.min(y_predict)), max(np.max(y), np.max(y_predict)))
plt.show()

model.save('Stock Prediction Model.keras')