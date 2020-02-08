import statistics
import time

import matplotlib.pyplot as plot  # To plot the data
import numpy as np
import pandas as pd  # For reading data from file


def read_data(path, flag):
    """[Function to read data from txt file]

    Arguments:
        path {[String]} -- [relative path to the dataset.txt]
        flag {[int]} -- [a flag to differentitate what dataset is used and how many columns to make]

    Returns:
        [Pandas Dataframe] -- [Dataframe contains columns of data depending upon the flag]
    """
    # Reading Data
    data = pd.read_csv(path, sep=",", header=None)
    if flag == 0:
        data.columns = ["population", "profit"]
    else:
        data.columns = ["size", "bed", "price"]
    return data


def plot_data(data):
    """[Function to plot 2D scatter plot]

    Arguments:
        data {[Pandas dataframe]} -- [Works only on 2D or 2 feature data]
    """
    plot.scatter(data['population'], data['profit'],
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

    plot.scatter(data['population'], data['profit'],
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


def vectorize_data(data, thetas, flag, n):
    """[function to vectorize data so that iterations are avoided]

    Arguments:
        data {[pandas dataframe]} -- [contains original data]
        thetas {list} -- [contains all thetas ]
        flag {[int]} -- [to decide what dataset is being used]
        n {[int]} -- [indicates columns of dataset/features of dataset]

    Returns:
        [array] -- [vectorized dataframe and thetas]
    """
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


def linear_regression_normal_eq(data, flag):
    """[Function to update thetas in one go using normal eq]

    Arguments:
        data {[dataframe]} -- [original dataframe]
        flag {[int]} -- [to decide what dataset is used while vectorizing]

    Returns:
        [list] -- [returns a list of optimal thetas and vectorized input]
    """
    n = len(data.columns)
    thetas = np.zeros(n)  # array of thetas
    x, y, thetas = vectorize_data(data, thetas, flag, n)
    thetas = normal_equation(x, y)
    return [thetas, x]


def linear_regression_with_gradient(data, flag):
    """[driver function for linear regression with gradient_descent]

    Arguments:
        data {[pandas dataframe]} -- [contains original data]
        flag {[int]} -- [to indicate what dataset is used]

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
    x, y, thetas = vectorize_data(data, thetas, flag, n)

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


if __name__ == '__main__':
    start_time = time.time()

    data = read_data("data/data2.txt", 1)
    # plot_data(data)

    # Uncomment when dataset 2 is to be used
    data, means, std_d = normalize_features(data)
    # thetas, x = linear_regression_normal_eq(data, 1)
    thetas, x = linear_regression_with_gradient(data, 1)

    print("--- %s seconds ---" % (time.time() - start_time))

    # To check if our predictions are correct for dataset 1
    # plot_prediction(thetas, x, data)

    prediction_test(means, std_d, thetas)  # for dataset 2
