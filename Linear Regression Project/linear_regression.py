import os, sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

import statsmodels.api as sm

# --------------------------------------------------------------------------
# set up plotting parameters
# --------------------------------------------------------------------------
line_width_1 = 2
line_width_2 = 2
marker_1 = '.' # point
marker_2 = 'o' # circle
marker_size = 12
line_style_1 = ':' # dotted line
line_style_2 = '-' # solid line

# --------------------------------------------------------------------------
# other settings
# --------------------------------------------------------------------------
boolean_using_existing_data = True

def main():

    # Importing data files for testing
    in_file_name = "linear_regression_test_data.csv"

    dataIn = pd.read_csv(in_file_name)
    x = np.array(dataIn['x'])
    y = np.array(dataIn['y'])
    y_theoretical = np.array(dataIn['y_theoretical'])

    # plot the data
    fig, ax = plt.subplots()
    ax.scatter(x, y, color='red', marker=marker_1, linewidth=line_width_1)
    plt.show()

    # Fitting the model using least squared solution
    least_square_model = LinearRegression()
    least_square_model.fit_least_squared(x, y)

    # Predictions calculation
    x_new = np.array([6, 7, 8])
    y_pred_least_square = least_square_model.prediction(x_new)

    # R Squared calculation
    r2_score_least_square = least_square_model.r_square(x_new, y)

    # F statistic calculation
    f_stat = least_square_model.f_statistic(x_new, y)

    # Least squared results
    print("Least Squared Results: ")
    print("Coefficients: ", least_square_model.coefficient, least_square_model.intercept)
    print("Predictions: ", y_pred_least_square)
    print("R Squared Value: ", r2_score_least_square)
    # print("F statistic: ", f_stat)
    # print("P value: ", p_value)

    # Plotting least squared results
    n2 = len(x)
    x_bar2 = np.mean(x)
    y_bar2 = np.mean(y)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x, y, color='red', marker=marker_1, linewidth=line_width_1)
    ax.plot(x, y_theoretical, color='green', label='theoretical', linewidth=line_width_1)

    ax.plot(x, np.ones(n2) * y_bar2, color='black', linestyle=':', linewidth=line_width_1)
    ax.plot([x_bar2, x_bar2], [np.min(y), np.max(y)], color='black', linestyle=':', linewidth=line_width_1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title("Linear regression")
    ax.legend(loc='lower right', fontsize=9)
    plt.show()

    # Fitting the model using gradient solution
    gradient_model = LinearRegression()
    gradient_model.fit_gradient(x, y)

    # Predictions calculation
    x_new = np.array([6, 7, 8])
    y_pred_gradient = gradient_model.prediction(x_new)

    # R Squared calculation
    r2_score_gradient = gradient_model.r_square(x, y)

    # Gradient results
    print("Gradient Results: ")
    print("Coefficients: ", gradient_model.coefficient, gradient_model.intercept)
    print("Predictions: ", y_pred_gradient)
    print("R Squared Value: ", r2_score_gradient)



class LinearRegression:
    def __init__(self):
        self.intercept = None
        self.coefficient = None
        self.r_squared = None
        self.f_stat = None
        self.p_value = None

    def fit_least_squared(self, x, y):

        # Calculating and returning the least squared linear regression model
        # Fitting model to the data
        x = np.array(x)
        y = np.array(y)

        n = len(x)
        x_bar = np.mean(x)
        y_bar = np.mean(y)

        # Calculating coefficients
        num = np.sum((x - x_bar) * (y - y_bar))
        den = np.sum((x - x_bar) ** 2)

        self.coefficient = (num / den)
        self.intercept = y_bar - (self.coefficient * x_bar)

        # R Squared value calculation
        y_pred = self.intercept + (self.coefficient * x)
        sum_squares = np.sum((y - y_bar) ** 2)
        res_sum_squares = np.sum((y - y_pred) ** 2)
        self.r_squared = 1 - (res_sum_squares / sum_squares)

        return self

    # def f_statistic(self, x, y):
    #     # Fit using statsmodels for F statistics and P-value
    #     model = sm.OLS(y, sm.add_constant(x))  # Add a constant (intercept) to the model
    #     results = model.fit()
    #
    #     self.f_stat = results.fvalue
    #
    #     return self

    # def p_value(self, x, y):
    #     # Fit using statsmodels for F statistics and P-value
    #     model = sm.OLS(y, sm.add_constant(x))  # Add a constant (intercept) to the model
    #     results = model.fit()
    #
    #     self.p_value = results.pvalue
    #
    #     return self

    def fit_gradient(self, x, y):

        # Calculates and returns the gradient linear regression model
        n_sample = x.shape[0]
        n_feature = 1
        x = x.reshape(-1, 1)

        self.coefficient = np.zeros((n_feature, 1))
        self.intercept = 0.0

        # Gradient n iterations
        for i in range(1000):
            # Prediction
            y_pred = self.prediction(x)

            coef = -(2 / n_sample) * np.dot(x.T, (y - y_pred))
            inter = -(2 / n_sample) * np.sum(y - y_pred)

            self.coefficient -= (coef * 0.01)
            self.intercept -= (inter * 0.01)

        return self

    def prediction(self, x):

        # Calculates predictions based on the output of the specific linear regression model
        y_pred = np.dot(x, self.coefficient) + self.intercept

        return y_pred

    def r_square(self, x, y):

        # Calculates R Squared based on the specific linear regression model
        y_pred = self.prediction(x)

        sum_square_reg = np.sum((y_pred - np.mean(y)) ** 2)
        tot_sum_square = np.sum((y - np.mean(y)) ** 2)
        r_square = (sum_square_reg / tot_sum_square)

        return r_square

if __name__ == '__main__':
     main()
