import time

import matplotlib.pyplot as plot  # To plot the data
import numpy as np
import pandas as pd  # For reading data from file


def read_data(path):
    # Reading Data
    data = pd.read_csv(path, sep=",", header=None)
    data.columns = ['test1', 'test2', 'status']
    return data


def plot_data_helper(data):
    accept = data[data.status == 0]
    reject = data[data.status == 1]
    plot.scatter(accept.iloc[:, 0], accept.iloc[:, 1],
                 color='y', marker='o', edgecolors='black', label='y=0')
    plot.scatter(reject.iloc[:, 0], reject.iloc[:, 1],
                 color='black', marker='+', edgecolors='black', label='y=1')
    plot.legend(loc="lower left")

    plot.title('Data Plot')

    plot.xlabel("Test 1")

    plot.ylabel('Test 2')


def plot_data(data):
    plot_data_helper(data)
    plot.show()


def plot_prediction(data, thetas):
    plot_data_helper(data)
    u_vals = np.linspace(-1, 1.5, 50)
    v_vals = np.linspace(-1, 1.5, 50)
    z = np.zeros((len(u_vals), len(v_vals)))
    for i in range(len(u_vals)):
        for j in range(len(v_vals)):
            f, y = map_feature(0, 1, u_vals[i], v_vals[j])
            z[i, j] = f @ thetas
    plot.contour(u_vals, v_vals, z.T, 0)
    plot.xlabel("Test 1")
    plot.ylabel("Test 2")
    plot.legend(loc=0)
    plot.show()


def map_feature(data, flag, x1=0, x2=0, y=0):
    if flag == 0:
        x1 = data.iloc[:, 0].to_numpy().reshape(len(data), 1)
        x2 = data.iloc[:, 1].to_numpy().reshape(len(data), 1)
        y = data.iloc[:, -1].to_numpy().reshape(len(data), 1)
        features = np.ones(len(x1)).reshape(len(x1), 1)  # First column of all 1
    else:
        features = np.ones(1)

    for i in range(1, 7):
        for j in range(i + 1):
            terms = (x1 ** (i - j) * x2 ** j)
            if flag == 0:
                terms = terms.reshape(len(x1),
                                      1)
                # Formula to generate polynomials of terms x1,x2 up untill
            # 6th power. i.e when i=1, j=0 it will give x1. When i=2, j=1 will give x1x2
            features = np.hstack((features, terms))

    return features, y


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def calculate_error(thetas, x, y, r_lambda):
    # Formula= -1/m Sigma y*log(h(x))+(1-y)(1-log(h(x))) + lambda/2m sigma theta^2
    m = len(y)
    prediction = sigmoid(np.dot(x, thetas))
    J = (-1 / m) * (np.dot(y.T, np.log(prediction)) + np.dot((1 - y.T), np.log(1 - prediction)))
    reg = (r_lambda / (2 * m)) * np.dot(thetas[1:].T, thetas[1:])
    return J + reg


def update_thetas(thetas, x, y, alpha, r_lambda):
    N = len(x)
    prediction = sigmoid(np.dot(x, thetas))
    term = np.dot(x.transpose(), (prediction - y))
    theta0 = alpha / N * term[0]
    theta1 = alpha / N * term[1:] + (alpha * r_lambda / N) * thetas[1:]
    grad = np.vstack((theta0[:, np.newaxis], theta1))
    return grad


def logistic_regression_regularize(x, y):
    thetas = np.zeros((x.shape[1], 1))  # nx1 array of thetas
    alpha = 1
    r_lambda = 100
    prev_err = 1
    new_err = 0
    i = 0
    while prev_err - new_err > 0.00000001:
        i += 1
        print("Iteration: {0} with prev_err: {1} and new_err: {2}".format(i, prev_err, new_err))
        prev_err = calculate_error(thetas, x, y, r_lambda)
        thetas = thetas - (alpha * update_thetas(thetas, x, y, alpha, r_lambda))
        new_err = calculate_error(thetas, x, y, r_lambda)
    return thetas, x


if __name__ == '__main__':
    original_data = read_data("ex2data2.txt")
    # plot_data(original_data)
    x, y = map_feature(original_data, 0)
    start_time = time.time()
    thetas, x = logistic_regression_regularize(x, y)
    print("--- %s seconds ---" % (time.time() - start_time))
    print(thetas)
    plot_prediction(original_data, thetas)
