import time

import matplotlib.pyplot as plot  # To plot the data
import numpy as np
import pandas as pd  # For reading data from file


def read_data(path):
    # Reading Data
    data = pd.read_csv(path, sep=",", header=None)
    data.columns = ['exam1', 'exam2', 'status']
    return data


def plot_data_helper(data):
    admitted = data[data.status == 0]
    notAdmitted = data[data.status == 1]
    plot.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1],
                 color='y', marker='o', edgecolors='black', label='not admitted')
    plot.scatter(notAdmitted.iloc[:, 0], notAdmitted.iloc[:, 1],
                 color='black', marker='+', edgecolors='black', label='admitted')
    plot.legend(loc="lower left")

    plot.title('Data Plot')

    plot.xlabel("Exam 1")

    plot.ylabel('Exam 2')


def plot_data(data):
    plot_data_helper(data)
    plot.show()


def plot_prediction(data, thetas, X):
    plot_data_helper(data)
    x_value = np.array([np.min(X[:, 1]), np.max(X[:, 1])])
    y_value = -(thetas[0] + thetas[1] * x_value) / thetas[2]
    plot.plot(x_value, y_value, "g")
    plot.show()


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def normalize_features(data):
    feature = data.iloc[:, :-1].to_numpy()
    y = data.iloc[:, -1].to_numpy()
    mean = (sum(feature) / len(feature))
    std = np.std(feature, axis=0)
    feature = (feature - mean)
    feature /= std
    out = np.insert(feature, len(data.columns) - 1, y, axis=1)
    data = pd.DataFrame(out)
    data.columns = ['exam1', 'exam2', 'status']
    return [data, mean, std]


def vectorize_data(data, thetas, n):
    x = data.iloc[:, :len(data.columns) - 1].to_numpy()  # converting all columns except last one to numpy array
    x = np.insert(x, 0, 1, axis=1)  # appending an all 1 column to start
    y = data.iloc[:, len(data.columns) - 1:].to_numpy()  # converting last one to numpy array
    thetas = thetas.reshape(n, 1)
    return [x, y, thetas]


def calculate_error(thetas, x, y):
    m = len(x)
    prediction = sigmoid(np.dot(x, thetas))
    c = np.log(prediction)
    c1 = np.log(1 - prediction)
    error = np.dot(np.transpose(y), c) + np.dot(np.transpose(1 - y),
                                                c1)  # y(log(h(x)))+(1-y)(log(1-h(x)))
    return (-1 / m) * error


def update_thetas(thetas, alpha, x, y, n):
    N = len(x)
    prediction = sigmoid(np.dot(x, thetas))
    error = (prediction - y) * x
    error = sum(error)
    error = error.reshape(n, 1)
    thetas = thetas - (alpha / N) * error
    return thetas


def logistic_regression_with_gradient(data):
    n = len(data.columns)
    thetas = np.zeros(n)  # array of thetas
    alpha = 0.1
    prev_err = 1
    new_err = 0
    x, y, thetas = vectorize_data(data, thetas, n)
    i = 0
    while prev_err - new_err != 0:
        i += 1
        print("Iteration: {0} with prev_err: {1} and new_err: {2}".format(i, prev_err, new_err))
        prev_err = calculate_error(thetas, x, y)
        thetas = update_thetas(thetas, alpha, x, y, n)
        new_err = calculate_error(thetas, x, y)
    return thetas, x


def predict(thetas, mean, std):
    x_test = np.array([45, 85])
    x_test = (x_test - mean) / std
    x_test = np.append(np.ones(1), x_test)
    prediction = sigmoid(x_test.dot(thetas))
    print("Admission Probability is is: {0} %".format(prediction * 100))


if __name__ == '__main__':
    original_data = read_data("ex2data1.txt")
    # plot_data(original_data)
    start_time = time.time()
    data, mean, std = normalize_features(original_data)
    thetas, x = logistic_regression_with_gradient(data)
    print("--- %s seconds ---" % (time.time() - start_time))
    print(thetas)
    plot_prediction(data, thetas, x)
    predict(thetas, mean, std)
