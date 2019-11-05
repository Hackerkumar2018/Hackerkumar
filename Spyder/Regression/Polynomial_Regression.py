# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Fitting Linear Resgression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(x)
poly_reg.fit(x_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

# Visualising the linear Regression results
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg.predict(x), color = 'blue')
plt.title('Truth or bluf (Polynomial Regression)')
plt.xlabel('Position lavel')
plt.ylabel('Salary')
plt.show()


# Visualising Polynomial Regression results
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg.predict(poly_reg.fit_transform(x)), color = 'blue')
plt.title('Truth or bluf (Polynomial Regression)')
plt.xlabel('Position lavel')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
lin_reg.predict(6.5)

# predicting a new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))
