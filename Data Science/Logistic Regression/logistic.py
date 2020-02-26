import time as time

import matplotlib.pyplot as plot  # To plot the data
import numpy as np
import pandas as pd  # For reading data from file


def read_data(path):
    """[Function to read data from txt file]

    Arguments:
        path {[String]} -- [relative path to the dataset.txt]

    Returns:
        [Pandas Dataframe] -- [Dataframe contains columns of data]
    """
    # Reading Data
    data = pd.read_csv(path, sep=",", header=None)
    data.columns = ['exam1', 'exam2', 'status']
    return data


def plot_data(data):
    """[Function to plot 2D scatter plot]

    Arguments:
        data {[Pandas dataframe]} -- [Works only on 2D or 2 feature data]
    """
    admitted = data[data.status == 0]
    notAdmitted = data[data.status == 1]
    plot.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1],
                 color='y', marker='o', edgecolors='black', label='admitted')
    plot.scatter(notAdmitted.iloc[:, 0], notAdmitted.iloc[:, 1],
                 color='black', marker='+', edgecolors='black', label='not admitted')
    plot.legend(loc="upper right")

    plot.title('Data Plot')

    plot.xlabel("Exam1")

    plot.ylabel('Exam2')
    plot.show()


def plot_prediction(thetas, x, data):
    """[Function to plot predicted hyperplane with respect to original data]

    Arguments:
        thetas {[array/list of thetas]} -- [Optimal thetas obtained after gradient descent]
        x {[input vector]} -- [Input/population vector that will help in plotting line]
    """

    admitted = data[data.status == 0]
    notAdmitted = data[data.status == 1]
    plot.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1],
                 color='y', marker='o', edgecolors='black', label='admitted')
    plot.scatter(notAdmitted.iloc[:, 0], notAdmitted.iloc[:, 1],
                 color='black', marker='+', edgecolors='black', label='not admitted')
    plot.legend(loc="upper right")

    plot.title('Data Plot')

    plot.xlabel("Exam1")

    plot.ylabel('Exam2')
    # predicted response vector
    y_pred = np.dot(x, thetas)
    x = x[:, 1]  # removing the one extra column of all 1
    # plotting the regression line
    plot.plot(x, y_pred, color="b")

    # function to show plot
    plot.show()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def vectorize_data(data, thetas, n):
    """[function to vectorize data so that iterations are avoided]

    Arguments:
        data {[pandas dataframe]} -- [contains original data]
        thetas {list} -- [contains all thetas ]
        n {[int]} -- [indicates columns of dataset/features of dataset]

    Returns:
        [array] -- [vectorized x, y and thetas]
    """
    x = data.iloc[:, :len(data.columns) - 1].to_numpy()  # converting all columns except last one to numpy array
    x = np.insert(x, 0, 1, axis=1)  # appending an all 1 column to start
    y = data.iloc[:, len(data.columns) - 1:].to_numpy()  # converting last one to numpy array
    thetas = thetas.reshape(n, 1)
    return [x, y, thetas]


def calculate_error(thetas, x, y):
    """[function to calculate objective function/loss]

    Arguments:
        thetas {[list]} -- [contains thetas]
        x {[input vector/matrix]} -- [contains vectorized matrix of input]
        y {[Y/output matrix]} -- [contains vector of output feature]

    Returns:
        [int] -- [returns sum of errors for all samples]
    """
    m = len(x)
    prediction = np.dot(x, thetas)
    prediction = sigmoid(prediction)
    c = np.log(prediction)
    c1 = np.log(1 - prediction)
    error = np.dot(-np.transpose(y), c) - np.dot(np.transpose(1 - y),
                                                 c1)  # -y(log(h(x)))-(1-y)(log(1-h(x)))
    return (1 / m) * error


def update_thetas(thetas, alpha, x, y, n):
    """[function to update theta/gradient descent]

    Arguments:
        thetas {[list]} -- [contains list of thetas to be updated]
        alpha {[float]} -- [learning rate]
        x {[input vector/matrix]} -- [vectorized input matrix]
        y {[output vector]} -- [vectorized output matrix]
        n {[int]} -- [no. of columns in dataset]

    Returns:
        [list] -- [returns updated thetas]
    """
    N = len(x)
    prediction = np.dot(x, thetas)
    prediction = sigmoid(prediction)
    error = (prediction - y) * x
    error = sum(error)
    error = error.reshape(n, 1)
    thetas = thetas - (alpha / N) * error
    return thetas


def logistic_regression_with_gradient(data):
    """[driver function for logistic regression with gradient_descent]

    Arguments:
        data {[pandas dataframe]} -- [contains original data]

    Returns:
        [list] -- [returns optimal thetas and vectorized input]
    """

    n = len(data.columns)
    thetas = np.zeros(n)  # array of thetas
    alpha = 0.001  # optimal alpha when while loop convergence is used, outputs in 1 seconds
    # i = 0
    prev_err = 10000000  # dummy error var to check convergence

    # iterations = 100000 # 90 seconds for 100k iterations
    # costs = np.zeros(iterations)
    x, y, thetas = vectorize_data(data, thetas, n)

    #          Gradient Descent Method start       ##
    thetas=[-13.44591237, 0.11275064, 0.1067869]
    while calculate_error(thetas, x, y) > 0.24:
        print("error: {0} , thetas: {1}".format(calculate_error(thetas, x, y), thetas))
        prev_err = calculate_error(thetas, x, y)
        # costs = np.insert(costs, i, prev_err, axis=0)
        thetas = update_thetas(thetas, alpha, x, y, n)

    # to plot iterations vs cost function
    # costs = costs[:iterations, ]
    # plot.plot(list(range(iterations)), costs, '-r')
    # plot.show()

    #          Gradient Descent Method End       ##
    print(calculate_error(thetas, x, y))

    return [thetas, x]


if __name__ == '__main__':
    data = read_data("ex2data1.txt")
    # plot_data(data)
    start_time = time.time()
    thetas, x = logistic_regression_with_gradient(data)
    print("--- %s seconds ---" % (time.time() - start_time))
    print(thetas)
    plot_prediction(thetas, x, data)
