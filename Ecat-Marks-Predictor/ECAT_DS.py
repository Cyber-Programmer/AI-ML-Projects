import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import linear_model

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
"""
Index(['Timestamp', 'Username', 'Name', 'Registration No.', 'Gender',
       '9th Marks', '10th Marks', '11th Marks', '12th Marks', 'ECAT Marks'],
      dtype='object')"""


# Reading the data
dataset = pd.read_csv("ECAT.csv")
# print(dataset)
# print(dataset.keys())
# print(dataset.info())
# print(dataset.describe())
# print(dataset.shape)
dataset['Gender'] = dataset['Gender'].map({'Male':1, 'Female':2})
dataset.drop(['Timestamp', 'Username', 'Name', 'Registration No.'], axis=1, inplace=True)
# print(dataset)

# Visualizing the data
#
# sb.heatmap(data=dataset.corr(),cmap='YlGnBu', annot=True)
# plt.show()

Y = dataset.loc[:, 'ECAT Marks']
dataset.drop(['ECAT Marks'], axis=1, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(dataset, Y, test_size=0.2, random_state=42)
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
print(regr.coef_)
print(regr.intercept_)
Predict = regr.predict(X_test)
print(f"Mean Squared Error: {mean_squared_error(Predict, y_test)}")

ECAT_input = np.array([[2, 450, 1016, 503, 1024]])
ECAT = regr.predict(ECAT_input)
print(f"ECAT Marks: {ECAT}")