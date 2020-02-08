import statistics
import time

import matplotlib.pyplot as plot  # To plot the data
import numpy as np
import pandas as pd  # For reading data from file


def read_data(path, flag):
    # Reading Data
    data = pd.read_csv(path, sep=",", header=None)
    if flag == 0:
        data.columns = ["population", "profit"]
    else:
        data.columns = ["size", "bed", "price"]
    return data


def plot_data(data):
    plot.scatter(data['population'], data['profit'], color='r', marker='x')

    plot.title('Prediction Model')

    plot.xlabel("Population of City in 10,000's")

    plot.ylabel('Profit in $10,000s')


def plot_prediction(thetas, x):
    # predicted response vector
    y_pred = np.dot(x, thetas)
    x = x[:, 1]  # removing the one extra column of all 1
    # plotting the regression line
    plot.plot(x, y_pred, color="b")

    # function to show plot
    plot.show()


def normalize_features(data):
    feature1 = data['size']
    feature2 = data['bed']
    feature3 = data['price']
    mean1 = (sum(feature1) / len(feature1))
    mean2 = (sum(feature2) / len(feature2))
    mean3 = (sum(feature3) / len(feature3))
    std1 = statistics.stdev(feature1)
    std2 = statistics.stdev(feature2)
    std3 = statistics.stdev(feature3)
    means = np.array([mean1, mean2, mean3])
    std_d = np.array([std1, std2, std3])
    feature1 = (feature1 - mean1)
    feature2 = (feature2 - mean2)
    feature3 = (feature3 - mean3)
    feature1 /= std1
    feature2 /= std2
    feature3 /= std3
    return [pd.DataFrame({'size': feature1, 'bed': feature2, 'price': feature3}), means, std_d]


def vectorize_data(data, thetas, flag, n):
    if flag == 1:
        x1 = np.array(data['size'])
        x2 = np.array(data['bed'])
        x = np.array([x1, x2])
        x = np.transpose(x)
        x = np.insert(x, 0, 1, axis=1)
        y = np.array(data['price'])
        y = y.reshape(47, 1)
    else:
        x = np.array(data['population'])
        x = x.reshape(97, 1)
        x = np.insert(x, 0, 1, axis=1)
        y = np.array(data['profit'])
        y = y.reshape(97, 1)
    thetas = thetas.reshape(n, 1)
    return [x, y, thetas]


def calculate_error(thetas, x, y):
    m = len(data)
    prediction = np.dot(x, thetas)
    error = prediction - y
    error = np.dot(np.transpose(error), error)
    return (1 / (2 * m)) * error


def update_thetas(thetas, alpha, x, y, n):
    N = len(x)
    prediction = np.dot(x, thetas)
    error = (prediction - y) * x
    error = sum(error)
    error = error.reshape(n, 1)
    thetas = thetas - (alpha / N) * error
    return thetas


def linear_regression(data, flag):
    n = len(data.columns)
    thetas = np.zeros(n)
    alpha = 0.01  # optimal alpha when while loop convergence is used, outputs in 1 seconds
    i = 0
    prev_err = 10000000
    # iterations = 100000 # 90 seconds for 100k iterations
    # costs = np.zeros(iterations)
    x, y, thetas = vectorize_data(data, thetas, flag, n)
    # for i in range(iterations):
    while calculate_error(thetas, x, y) - prev_err < 0:
        # print("Theetas: {0}".format(thetas))
        # i += 1
        prev_err = calculate_error(thetas, x, y)
        # costs = np.insert(costs, i, prev_err, axis=0)
        # print("Epoch: {0}  error:  {1}".format(i, prev_err))
        thetas = update_thetas(thetas, alpha, x, y, n)

    # costs = costs[:iterations, ]
    # plot.plot(list(range(iterations)), costs, '-r')
    # plot.show()

    return [thetas, x]


if __name__ == '__main__':
    start_time = time.time()

    data = read_data("data/data1.txt", 0)
    # data, means, std_d = normalize_features(data)
    # plot_data(data, 1)

    thetas, x = linear_regression(data, 0)
    print(thetas)
    print("--- %s seconds ---" % (time.time() - start_time))  # Vectorized version takes nearly 1 second to run

    # To check if our predictions are correct#
    plot_data(data)
    plot_prediction(thetas, x)
    # while 1:
    #     x1 = float(input("Enter size of house: "))
    #     x2 = float(input("Enter bedrooms of house: "))
    #     x1 = (x1 - means[0])
    #     x1 /= std_d[0]
    #     x2 = (x2 - means[1])
    #     x2 /= std_d[1]
    #     x = np.array([x1, x2])
    #     x = np.insert(x, 0, 1, axis=0)
    #     profit = np.dot(x, thetas)
    #     profit *= std_d[2]
    #     profit += means[2]
    #     print("Price is: ${0}".format(profit))
