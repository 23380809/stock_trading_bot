from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from pytrends import dailydata
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
company = 'FB'
key = 'facebook'


start = dt.datetime(2012, 1, 1)
end = dt.datetime(2020, 1, 1)

data = web.DataReader(company, 'yahoo', start, end)
df = dailydata.get_daily_data(key, 2019, 1, 2019, 2, geo='US')

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data_c = scaler.fit_transform(data['Close'].values.reshape(-1,1))
scaled_data_h = scaler.fit_transform(data['High'].values.reshape(-1,1))
scaled_data_l = scaler.fit_transform(data['Low'].values.reshape(-1,1))
scaled_data_o = scaler.fit_transform(data['Open'].values.reshape(-1,1))


x_train = []
y_train = []


for x in range(21, len(scaled_data_c)):
    h_l = float(scaled_data_h[x] - scaled_data_l[x])
    o_c = float(scaled_data_o[x] - scaled_data_c[x])
    seven_a = np.average(scaled_data_c[x-7:x, 0])
    fourteen_a = np.average(scaled_data_c[x-14:x, 0])
    twenty_one_a = np.average(scaled_data_c[x-21:x, 0])
    seven_std = np.std(scaled_data_c[x-7:x, 0])
    arr = np.array([h_l, o_c, seven_a, fourteen_a, twenty_one_a, seven_std])
    x_train.append(arr)
    y_train.append(scaled_data_c[x, 0])


x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential()
model.add(LSTM(units=5, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.02))

model.add(LSTM(units=3))
model.add(Dropout(0.02))
model.add(Dense(units=1))#Prediction of the next closing price


model.compile(optimizer='rmsprop', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=30, batch_size=32)


test_start_actual = dt.datetime(2020,1,1)
test_start = dt.datetime(2019,11,29)
test_end = dt.datetime.now()

test_data = web.DataReader(company, 'yahoo', test_start, test_end)
test_data_actual = web.DataReader(company, 'yahoo', test_start_actual, test_end)
actual_prices = test_data_actual['Close'].values


total_dataset = pd.concat((data['Close'], test_data['Close']))


data_c = scaler.fit_transform(test_data['Close'].values.reshape(-1,1))
data_h = scaler.fit_transform(test_data['High'].values.reshape(-1,1))
data_l = scaler.fit_transform(test_data['Low'].values.reshape(-1,1))
data_o = scaler.fit_transform(test_data['Open'].values.reshape(-1,1))


x_test = []
for x in range(21, len(data_c)):
    h_l = float(data_h[x] - data_l[x])
    o_c = float(data_o[x] - data_c[x])
    seven_a = np.average(data_c[x-7:x, 0])
    fourteen_a = np.average(data_c[x-14:x, 0])
    twenty_one_a = np.average(data_c[x-21:x, 0])
    seven_std = np.std(data_c[x-7:x, 0])
    arr = np.array([h_l, o_c, seven_a, fourteen_a, twenty_one_a, seven_std])
    x_test.append(arr)


x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)


plt.plot(actual_prices, color ="black", label=f"Actual {company} price")
plt.plot(predicted_prices, color="green", label=f"predicted {company} price")
plt.title(f"{company} Stock Price")
plt.xlabel("Days")
plt.ylabel(f"{company} Stock Price")
plt.legend()


real_data = np.array(x_test[-1])

real_data = np.reshape(real_data, (1, 6, 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print("Prediction: ", prediction)

plt.show()

