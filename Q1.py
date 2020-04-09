# Decision Tree Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#FOR CALIFORNIA
# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = np.arange(17)
print(X)
X= dataset.iloc[:, 0:4]
y = dataset.iloc[:, 4].values
#dividing data for california
X1= dataset.loc [(dataset.State=="California"),['State','Profit']]
print(X1)
Ycal= X1.iloc[:,-1]
Xcal= np.arange(17).reshape(-1,1)

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""


# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(Xcal, Ycal)


# Visualising the Decision Tree Regression results (higher resolution)
plt.scatter(Xcal, Ycal, color = 'red')
plt.plot(Xcal, regressor.predict(Xcal), color = 'blue')
plt.title('PROFITS OF STARTUPS IN DIFFERENT California (Decision Tree Regression)')
plt.xlabel('startup number')
plt.ylabel('Profit')
plt.show()
Ycal_pred=regressor.predict([[17]])
print ('PROFIT BY 17TH STARTUP IN CALIFORNIA')
print(Ycal_pred)

#FOR FLORIDA
dataset = pd.read_csv('50_Startups.csv')
X = np.arange(16)
print(X)
X= dataset.iloc[:, 0:4]
y = dataset.iloc[:, 4].values

X1= dataset.loc [(dataset.State=="Florida"),['State','Profit']]
print(X1)
Yflo= X1.iloc[:,-1]
Xflo= np.arange(16).reshape(-1,1)

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""


# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(Xflo, Yflo)


# Visualising the Decision Tree Regression results (higher resolution)
plt.scatter(Xflo, Yflo, color = 'red')
plt.plot(Xflo, regressor.predict(Xflo), color = 'blue')
plt.title('PROFITS OF STARTUPS IN Florida (Decision Tree Regression)')
plt.xlabel('startup number')
plt.ylabel('Profit')
plt.show()
Ycal_pred=regressor.predict([[17]])
print ('PROFIT BY 17th STARTUP IN CALIFORNIA')
print(Ycal_pred)

print('RESULT: The startups in California will make more profit in the future as shown by predictions')
