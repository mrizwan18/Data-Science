import time

import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn.metrics import accuracy_score, hinge_loss

import plot_helper

C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
gamma = [0.1, 0.5, 1, 5, 10, 20, 50, 100]


def load_data():
    iris = datasets.load_iris()

    iris_X = np.asarray(iris.data)
    iris_y = np.asarray(iris.target)
    return iris_X, iris_y


def divide_data(iris_X, iris_y):
    indices = np.random.permutation(len(iris_X))
    iris_X_train = iris_X[indices[:-40]]
    iris_y_train = iris_y[indices[:-40]]
    iris_X_valid = iris_X[indices[-40:-20]]
    iris_y_valid = iris_y[indices[-40:-20]]
    iris_X_test = iris_X[indices[-20:]]
    iris_y_test = iris_y[indices[-20:]]
    return iris_X_train, iris_y_train, iris_X_valid, iris_y_valid, iris_X_test, iris_y_test


def find_c(iris_X, iris_y):
    iris_X_train, iris_y_train, iris_X_valid, iris_y_valid, iris_X_test, iris_y_test = divide_data(iris_X, iris_y)

    optimal_c = 0
    loss = float('inf')

    for c in C:
        l2 = apply_model_linear(iris_X_train, iris_y_train, iris_X_valid, iris_y_valid,
                                c)
        if l2 < loss:
            loss = l2
            optimal_c = c

    # print("Optimal C for minimum Hinge Loss= {}".format(optimal_c))
    return optimal_c, iris_X_train, iris_y_train, iris_X_valid, iris_y_valid, iris_X_test, iris_y_test


def find_c_g(iris_X, iris_y, to_plot=True):
    if to_plot:
        iris_X = iris_X[:, :2]
    iris_X_train, iris_y_train, iris_X_valid, iris_y_valid, iris_X_test, iris_y_test = divide_data(iris_X,
                                                                                                   iris_y)

    optimal_c = 0
    optimal_g = 0
    loss = float('inf')
    count = 0
    for c in C:
        for g in gamma:
            l2 = apply_model_rbf(iris_X_train, iris_y_train, iris_X_valid, iris_y_valid,
                                 c, g, count, to_plot)
            count += 1
            if l2 < loss:
                loss = l2
                optimal_c = c
                optimal_g = g

    # print("Optimal C and gamma for minimum Hinge Loss= {}, {}".format(optimal_c, optimal_g))
    return optimal_c, optimal_g, iris_X_train, iris_y_train, iris_X_valid, iris_y_valid, iris_X_test, iris_y_test


def apply_model_rbf(iris_X_train, iris_y_train, iris_X_valid, iris_y_valid, c, g, count, to_plot):
    svc = svm.SVC(kernel='rbf', gamma=g, C=c)
    svc.fit(iris_X_train, iris_y_train)

    # Validation loss and accuracy
    predictions = svc.predict(iris_X_valid)
    valid_score = accuracy_score(iris_y_valid, predictions)
    prediction_dec = svc.decision_function(iris_X_valid)
    h_loss_v = hinge_loss(iris_y_valid, prediction_dec)
    if to_plot:
        p = plot_helper.plot_helper(iris_X_train, iris_y_train, c, g, svc, count)
        p.plot()
    # print(
    #     "Validation Score and loss for C= {} and gamma={} is : {}, {}".format(c, g, valid_score * 100.0, h_loss_v))
    return h_loss_v


def apply_model_linear(iris_X_train, iris_y_train, iris_X_valid, iris_y_valid, c):
    svc = svm.SVC(kernel='linear', C=c)
    svc.fit(iris_X_train, iris_y_train)

    # Validation loss and accuracy
    predictions = svc.predict(iris_X_valid)
    valid_score = accuracy_score(iris_y_valid, predictions)
    prediction_dec = svc.decision_function(iris_X_valid)
    h_loss_v = hinge_loss(iris_y_valid, prediction_dec)
    #
    # print(
    #     "Validation Score and loss for C= {} is : {}, {}".format(c, valid_score * 100.0, h_loss_v))
    return h_loss_v


def rbf_test(iris_X, iris_y):
    optimal_c, optimal_g, iris_X_train, iris_y_train, iris_X_valid, iris_y_valid, iris_X_test, iris_y_test = find_c_g(
        iris_X, iris_y, True)

    # Test loss and accuracy for optimal c and g
    svc = svm.SVC(kernel='rbf', gamma=optimal_g, C=optimal_c)
    svc.fit(iris_X_train[:, :2], iris_y_train)
    predictions = svc.predict(iris_X_test[:, :2])
    test_score = accuracy_score(iris_y_test, predictions)
    prediction_dec = svc.decision_function(iris_X_test[:, :2])
    h_loss_t = hinge_loss(iris_y_test, prediction_dec)
    p = plot_helper.plot_helper(iris_X_train, iris_y_train, optimal_c, optimal_g, svc, 100)
    p.plot()
    print(" RBF>>>")
    print(
        "Testing Score and loss for Optimal C= {} and gamma={} is : {}, {}\n".format(optimal_c, optimal_g,
                                                                                     test_score * 100.0,
                                                                                     h_loss_t))


def linear_test(iris_X, iris_y):
    optimal_c, iris_X_train, iris_y_train, iris_X_valid, iris_y_valid, iris_X_test, iris_y_test = find_c(iris_X, iris_y)

    # Test loss and accuracy for optimal c
    svc = svm.SVC(kernel='linear', C=optimal_c)
    svc.fit(iris_X_train, iris_y_train)
    predictions = svc.predict(iris_X_test)
    test_score = accuracy_score(iris_y_test, predictions)
    prediction_dec = svc.decision_function(iris_X_test)
    h_loss_t = hinge_loss(iris_y_test, prediction_dec)
    print(" Linear>>>>")
    print(
        "Testing Score and loss for Optimal C= {} is : {}, {} \n".format(optimal_c, test_score * 100.0, h_loss_t))


if __name__ == '__main__':
    start_time = time.time()
    iris_X, iris_y = load_data()

    linear_test(iris_X, iris_y)

    rbf_test(iris_X, iris_y)

    print("--- %s seconds ---" % (time.time() - start_time))
