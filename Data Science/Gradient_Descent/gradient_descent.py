import matplotlib.pyplot as plot  # To plot the data
import pandas as pd  # For reading data from file

# Reading Data
data = pd.read_csv('data/data1.txt', sep=",", header=None)
data.columns = ["population", "profit"]

# Plotting it
plot.scatter(data['population'], data['profit'], color='r', marker='x')

plot.title('Visualizing Population vs Profit')

plot.xlabel("Population of City in 10,000's")

plot.ylabel('Profit in $10,000s')

plot.show()
