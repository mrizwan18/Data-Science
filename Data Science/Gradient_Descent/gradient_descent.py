import matplotlib.pyplot as plot  # To plot the data
import pandas as pd  # For reading data from file


def read_data(path):
    # Reading Data
    data = pd.read_csv(path, sep=",", header=None)
    data.columns = ["population", "profit"]
    return data


def plot_data(data):
    plot.scatter(data['population'], data['profit'], color='r', marker='x')

    plot.title('Prediction Model')

    plot.xlabel("Population of City in 10,000's")

    plot.ylabel('Profit in $10,000s')


def plot_prediction(theta_zero, theta_one, population_vector):
    # predicted response vector
    y_pred = theta_zero + theta_one * population_vector

    # plotting the regression line
    plot.plot(population_vector, y_pred, color="b")

    # function to show plot
    plot.show()


def calculate_error(theta_zero, theta_one, data):
    m = len(data)
    error = 0
    for x, y in data.values:
        prediction = theta_zero + theta_one * x
        temp = prediction - y
        error = error + (temp ** 2)
    return (1 / (2 * m)) * error


def update_thetas(theta_zero_initial, theta_one_initial, alpha, population, profit):
    N = len(population)
    error1 = 0
    error2 = 0
    for x, y in zip(population, profit):
        prediction = theta_zero_initial + theta_one_initial * x
        error1 += prediction - y
        error2 += (prediction - y) * x
    theta_zero = theta_zero_initial - (alpha / N) * error1
    theta_one = theta_one_initial - (alpha / N) * error2
    return [theta_zero, theta_one]


def linear_regression(data):
    theta_zero = 0
    theta_one = 0
    alpha = 0.01
    population_vector = data['population']
    profit_vector = data['profit']
    i = 0
    prev_err = 10000000
    while calculate_error(theta_zero, theta_one, data) - prev_err < 0:
        print("Theeta Zero={0}, theeta One= {1}".format(theta_zero, theta_one))
        i += 1
        prev_err = calculate_error(theta_zero, theta_one, data)
        print("Epoch: {0}  error:  {1}".format(i, prev_err))
        theta_zero, theta_one = update_thetas(theta_zero, theta_one, alpha, population_vector, profit_vector)

    plot_data(data)
    plot_prediction(theta_zero, theta_one, population_vector)


data = read_data('data/data1.txt')
linear_regression(data)