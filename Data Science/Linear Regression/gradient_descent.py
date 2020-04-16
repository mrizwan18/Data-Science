import time

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
    return data


def plot_data(data):
    """[Function to plot 2D scatter plot]

    Arguments:
        data {[Pandas dataframe]} -- [Works only on 2D or 2 feature data]
    """
    plot.scatter(data.iloc[:, 0], data.iloc[:, 1],
                 color='r', marker='x', label='Training Data')

    plot.title('Data Plot')

    plot.xlabel("Population of City in 10,000's")

    plot.ylabel('Profit in $10,000s')
    plot.show()


def plot_prediction(thetas, x, data):
    """[Function to plot predicted hyperplane with respect to original data]

    Arguments:
        thetas {[array/list of thetas]} -- [Optimal thetas obtained after gradient descent]
        x {[input vector]} -- [Input/population vector that will help in plotting line]
    """

    plot.scatter(data.iloc[:, 0], data.iloc[:, 1],
                 color='r', marker='x', label='Training Data')
    # predicted response vector
    y_pred = np.dot(x, thetas)
    x = x[:, 1]  # removing the one extra column of all 1
    # plotting the regression line
    plot.plot(x, y_pred, color="b", label='Linear Regression')
    plot.legend(loc="lower right")
    plot.title('Prediction Model')
    plot.xlabel("Population of City in 10,000's")
    plot.ylabel('Profit in $10,000s')
    # function to show plot
    plot.show()


def normalize_features(data):
    """[To normalize all features so that no single feature dominates]

    Arguments:
        data {[pandas dataframe]} -- [contains the columns of dataset2.txt]

    Returns:
        [array/list] -- [array of normalized data, array of std_deviation, means of all features]
    """
    feature = data.iloc[:, :].to_numpy()
    mean = (sum(feature) / len(feature))
    std = np.std(feature, axis=0)
    feature = (feature - mean)
    feature /= std
    return [pd.DataFrame(feature), mean, std]


def normal_equation(x, y):
    """[Implementation of normal eq, theta= (X^T . X)^-1 . X^T . Y]

    Arguments:
        x {[matrix]} -- [vectoized input matrix]
        y {[matrix]} -- [vectoized output matrix]

    Returns:
        [list] -- [returns a list of thetas]
    """
    # Takes 0.01s for dataset1 :O and 0.006 for dataset 2
    x_trans = np.transpose(x)
    x_trans_x = np.dot(x_trans, x)
    x_trans_y = np.dot(x_trans, y)
    inverse = np.linalg.inv(x_trans_x)
    return np.dot(inverse, x_trans_y)


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
    m = len(data)
    prediction = np.dot(x, thetas)
    error = prediction - y
    error = np.dot(np.transpose(error), error)
    return (1 / (2 * m)) * error


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
    error = (prediction - y) * x
    error = sum(error)
    error = error.reshape(n, 1)
    thetas = thetas - (alpha / N) * error
    return thetas


def linear_regression_normal_eq(data):
    """[Function to update thetas in one go using normal eq]

    Arguments:
        data {[dataframe]} -- [original dataframe]

    Returns:
        [list] -- [returns a list of optimal thetas and vectorized input]
    """
    n = len(data.columns)
    thetas = np.zeros(n)  # array of thetas
    x, y, thetas = vectorize_data(data, thetas, n)
    thetas = normal_equation(x, y)
    return [thetas, x]


def linear_regression_with_gradient(data):
    """[driver function for linear regression with gradient_descent]

    Arguments:
        data {[pandas dataframe]} -- [contains original data]

    Returns:
        [list] -- [returns optimal thetas and vectorized input]
    """

    n = len(data.columns)
    thetas = np.zeros(n)  # array of thetas
    alpha = 0.01  # optimal alpha when while loop convergence is used, outputs in 1 seconds
    # i = 0
    prev_err = 10000000  # dummy error var to check convergence

    # iterations = 100000 # 90 seconds for 100k iterations
    # costs = np.zeros(iterations)
    x, y, thetas = vectorize_data(data, thetas, n)

    #          Gradient Descent Method start       ##

    while calculate_error(thetas, x, y) - prev_err < 0:
        prev_err = calculate_error(thetas, x, y)
        # costs = np.insert(costs, i, prev_err, axis=0)
        thetas = update_thetas(thetas, alpha, x, y, n)

    # to plot iterations vs cost function
    # costs = costs[:iterations, ]
    # plot.plot(list(range(iterations)), costs, '-r')
    # plot.show()

    #          Gradient Descent Method End       ##

    return [thetas, x]


def prediction_test(means, std_d, thetas):
    """[To test the prediction on test data]

    Arguments:
        means {[list]} -- [list of means for all features when normalized]
        std_d {[list]} -- [list of std_deviation for all features when normalized]
        thetas {[list]} -- [list of thetas]
    """
    x1 = float(input("Enter size of house: "))
    x2 = float(input("Enter bedrooms of house: "))
    x1 = (x1 - means[0])
    x1 /= std_d[0]
    x2 = (x2 - means[1])
    x2 /= std_d[1]
    x = np.array([x1, x2])
    x = np.insert(x, 0, 1, axis=0)
    profit = np.dot(x, thetas)
    profit *= std_d[2]
    profit += means[2]
    # $293081.4643349 by normal eq and $293081.46845345 by gradient for size=1650, beds=3
    print("Price is: ${0}".format(profit))
    # descent


def prediction_test_with_normal(thetas):
    x1 = float(input("Enter size of house: "))
    x2 = float(input("Enter bedrooms of house: "))
    x_pred = np.array([x1, x2])
    x_pred = np.insert(x_pred, 0, 1, axis=0)
    profit = np.dot(x_pred, thetas)
    # $293081.4643349 by normal eq and $293081.46845345 by gradient for size=1650, beds=3
    print("Price is: ${0}".format(profit))


if __name__ == '__main__':
    data = read_data("data/data1.txt")
    plot_data(data)

    start_time = time.time()

    # Normalize when dataset 2 is to be used
    # data, means, std_d = normalize_features(data)
    thetas, x = linear_regression_normal_eq(data)
    # thetas, x = linear_regression_with_gradient(data)
    print("--- %s seconds ---" % (time.time() - start_time))
    # print(thetas)

    # To check if our predictions are correct for dataset 1
    plot_prediction(thetas, x, data)

    # prediction_test_with_normal(thetas)  # for dataset 2
    # prediction_test(means, std_d, thetas)  # for dataset 2
