'''
- creates a best fit line relative to data points
- applied only on linear data
- r squared theory (coefficient of determination) is used to find the accuracy of linear regression line versus the mean (of the y values) line
- R^2 a number that indicates the proportion of the variance in the dependent variable that is predictable from the independent variable
'''

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

x_values = np.array([1,2,3,4,5,6], dtype=np.float64)
y_values = np.array([5,4,6,5,6,7], dtype=np.float64)

# = denotes default optional parameters
#hm : how many data points you wanna marketplace
#variance: how variable you want data to be
def create_dataset(hm, variance, step=2, correlation=False):
    val = 1
    y_values = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance) # some value in the given range
        y_values.append(y) #add to array
        if correlation and correlation=='pos':
            val += step
        elif correlation and correlation=='neg':
            val -= step
    x_values = [i for i in range(len(y_values))]
    return np.array(x_values, dtype=np.float64), np.array(y_values, dtype=np.float64 )


#calculates the line of best fit
#Accurate: how accurate is this best fit line relative to the data set we have?
#Confidence:
def best_fit_slope_and_intercept(x_values, y_values):
    top = ( (mean(x_values) * mean(y_values)) - mean(x_values*y_values) )
    bottom = ( mean(x_values)**2 - mean(x_values**2) )
    m = top / bottom
    b = mean(y_values) - m*mean(x_values)
    return m,b

#compares the actual best possible fit line to the "attempted" best fit line relative to the data set
#squred because it smoothes and amplifies outliers
def squared_error(y_original, y_line):
    return sum( (y_line - y_original)**2 )

#calculates how accurate best fit line is.
def coefficient_of_determination(y_original, y_line):
    y_mean_line = [mean(y_original) for y in y_original]
    squared_error_of_regression_line = squared_error(y_original, y_line)
    sqared_error_y_mean = squared_error(y_original, y_mean_line)
    return 1 - (squared_error_of_regression_line / sqared_error_y_mean ) # value of coefficient_of_determination, squared error.


x_values, y_values = create_dataset(40, 40, 2, correlation='pos')

m,b = best_fit_slope_and_intercept(x_values, y_values)

regression_line = [ (m*x)+b for x in x_values ] # list of y values, one line for loop
'''
same as:
for x in x_values
    regression_line.append((m*x) + b)
'''

print('x values:', x_values)
print('y values:', y_values)
print('regression_line:', regression_line)
print('m,b', m,b)

r_squared = coefficient_of_determination(y_values, regression_line)
print('R^2', r_squared )

plt.scatter(x_values, y_values)
plt.plot(x_values, regression_line)
plt.show()
