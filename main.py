import sys
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from LSTM import LongShortTermMemory
from Helper import *


def plot_graph(y_true, y_pred):
    plt.plot(y_true, label="True Value")
    plt.plot(y_pred, label="Predicted Value")
    plt.legend()
    plt.show()


if __name__ == '__main__':

    input_file = 'https://raw.githubusercontent.com/preerao7/FOREX-Rate-Prediction-using-RNN/refs/heads/main/dataset/EURUSD_D1_Sorted.csv'
    data = pd.read_csv(input_file)   # reading data from input file

    dates = data.iloc[:, [0]]   # Selecting dates from input data

    input_data = data.iloc[:, [1]]  # Selecting target feature

    # Normalizing data
    scaler = MinMaxScaler()
    scaler.fit(input_data)
    input_data = scaler.transform(input_data)
    input_data = pd.DataFrame(input_data)

    print('Split ratio: {}'.format(0.7))

    x, y = getSequences(input_data, 2)  # Creating sequences of [no_sequences, len_sequence, cols], [target, 1]
    x_train, x_test, y_train, y_test = split(x, y, 0.7)    # Splitting data into train and test set

    print('x_train\t{}'.format(x_train.shape))
    print('x_test\t{}'.format(x_test.shape))
    print('y_train\t{}'.format(y_train.shape))
    print('y_test\t{}'.format(y_test.shape))

    # Creating LSTM model
    model = LongShortTermMemory(learning_rate=0.01, max_iterations=100, time_step=2, input_shape=(1, 1), hidden_layers=5)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    print('R2: {}'.format(r2_score(y_test, y_pred)))
    print('MSE: {}'.format(mean_squared_error(y_test, y_pred, squared=True)))
    print('RMSE {}'.format(mean_squared_error(y_test, y_pred, squared=False)))

    plot_graph(y_test, y_pred)






