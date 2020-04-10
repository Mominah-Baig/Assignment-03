# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('annual_temp.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values
#Preprocessing data for GCAG
X1 = dataset.loc[(dataset.Source=="GCAG"),['Source','Year','Mean']]
print(X1)

XGCAG= X1.iloc[:,1:2].values
YGCAG= X1.iloc[:,2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""


# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(XGCAG, YGCAG)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
X_poly = poly_reg.fit_transform(XGCAG)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, YGCAG)


# Visualising the Linear Regression results
plt.scatter(XGCAG, YGCAG, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Annual Temperature GCAG (Linear Regression)')
plt.xlabel('Year')
plt.ylabel('Mean Temperature')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(XGCAG, YGCAG, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Annual Temperature GCAG(Polynomial Regression)')
plt.xlabel('Year')
plt.ylabel('Mean Temperature')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(XGCAG), max(XGCAG), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(XGCAG, YGCAG, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Annual Temperature GCAG(Polynomial Regression))')
plt.xlabel('Year')
plt.ylabel('Mean Temperature')
plt.show()


# Predicting a new result with Polynomial Regression
y_pred= lin_reg_2.predict(poly_reg.fit_transform([[2016]]))
print('Annual Temperature of GCAG in 2016:')
print(y_pred)
y_pred_1 = lin_reg_2.predict(poly_reg.fit_transform([[2017]]))
print('Annual Temperature of GCAG in 2017:')
print(y_pred_1)



#FOR GISTEMP
# Importing the dataset
dataset = pd.read_csv('annual_temp.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values
#Preprocessing data for GISTEMP
X1 = dataset.loc[(dataset.Source=="GISTEMP"),['Source','Year','Mean']]
print(X1)

XGISTEMP= X1.iloc[:,1:2].values
YGISTEMP= X1.iloc[:,2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""


# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(XGISTEMP, YGISTEMP)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
X_poly = poly_reg.fit_transform(XGISTEMP)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, YGISTEMP)


# Visualising the Linear Regression results
plt.scatter(XGISTEMP, YGISTEMP, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Annual Temperature GISTEMP (Linear Regression)')
plt.xlabel('Year')
plt.ylabel('Mean Temperature')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(XGISTEMP, YGISTEMP, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Annual Temperature GISTEMP(Polynomial Regression)')
plt.xlabel('Year')
plt.ylabel('Mean Temperature')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(XGISTEMP), max(XGISTEMP), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(XGISTEMP, YGISTEMP, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Annual Temperature GISTEMP(Polynomial Regression))')
plt.xlabel('Year')
plt.ylabel('Mean Temperature')
plt.show()


# Predicting a new result with Polynomial Regression
y_pred= lin_reg_2.predict(poly_reg.fit_transform([[2016]]))
print('Annual Temperature of GISTEMP in 2016:')
print(y_pred)
y_pred_1 = lin_reg_2.predict(poly_reg.fit_transform([[2017]]))
print('Annual Temperature of GISTEMP in 2017:')
print(y_pred_1)