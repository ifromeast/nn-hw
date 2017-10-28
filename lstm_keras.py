import numpy
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from math import sin, exp, pi, cos


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


length = 500
data_y = np.zeros([length])
data_y[0] = 1
data_y[1] = 1
for i in range(length):
    data_y[i] = 0.3 * data_y[i - 1] + 0.6 * data_y[i - 2] + 0.6 * sin(
        pi * cos(2 * pi * i / 250)) + 0.9 * exp(-data_y[i - 1] ** 2)

data_x = np.linspace(1, length, length)

data_y = np.transpose([data_y])
scaler = MinMaxScaler(feature_range=(0, 1))
data_y = scaler.fit_transform(data_y)

look_back = 1
trainX, trainY = create_dataset(data_y, look_back)
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
trainPredict = model.predict(trainX)
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])

trainPredictPlot = numpy.empty_like(data_y)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

plt.plot(scaler.inverse_transform(data_y))
plt.plot(trainPredictPlot)
plt.show()



