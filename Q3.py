# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('global_co2.csv')
X = dataset.iloc[:, 0:1].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""


# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 13)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('GLOBAL PRODUCTION OF CO2(linear regression)')
plt.xlabel('Year')
plt.ylabel('Production of CO2')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('GLOBAL PRODUCTION OF CO2(Polynomial Regression)')
plt.xlabel('Year')
plt.ylabel('Production of CO2')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('GLOBAL PRODUCTION OF CO2(Polynomial Regression)')
plt.xlabel('Year')
plt.ylabel('Production of CO2')
plt.show()


# Predicting a new result with Polynomial Regression
y_pred = lin_reg_2.predict(poly_reg.fit_transform([[2012]]))
print('Production of CO2 in 2011:')
print(y_pred)

y_pred1 = lin_reg_2.predict(poly_reg.fit_transform([[2012]]))
print('Production of CO2 in 2012:')
print(y_pred1)

y_pred2 = lin_reg_2.predict(poly_reg.fit_transform([[2013]]))
print('Production of CO2 in 2013:')
print(y_pred2)
