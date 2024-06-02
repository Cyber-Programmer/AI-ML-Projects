import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

# Index(['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education',
#        'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
#        'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Loan_Status'],
#       dtype='object')

dataset=pd.read_csv('Loan.csv')
#print(dataset)
#print(dataset.iloc[3:6,2:5])
# print(dataset.keys())

print(dataset.info())
#print(dataset.describe())
#print(dataset.iloc[:,8:13].describe())

#print(dataset.duplicated().sum())

# print(dataset['Gender'].unique())
# print(dataset['Married'].unique())
# print(dataset['Dependents'].unique())
# print(dataset['Education'].unique())
# print(dataset['Self_Employed'].unique())
# print(dataset['Loan_Amount_Term'].unique())
# print(dataset['Credit_History'].unique())
print(dataset['Property_Area'].unique())
# print(dataset['Loan_Status'].unique())

# fig, axes = plt.subplots(3, 3)
# sb.countplot(data=dataset, x='Gender',ax=axes[0,0]).set(title='Gender')
# sb.countplot(data=dataset, x='Married',ax=axes[0,1]).set(title='Married')
# sb.countplot(data=dataset, x='Dependents',ax=axes[0,2]).set(title='Dependents')
#
# sb.countplot(data=dataset, x='Education',ax=axes[1,0]).set(title='Education')
# sb.countplot(data=dataset, x='Self_Employed',ax=axes[1,1]).set(title='Self_Employed')
# sb.countplot(data=dataset, x='Loan_Amount_Term',ax=axes[1,2]).set(title='Loan_Amount_Term')
#
# sb.countplot(data=dataset, x='Credit_History',ax=axes[2,0]).set(title='Credit_History')
# sb.countplot(data=dataset, x='Property_Area',ax=axes[2,1]).set(title='Property_Area')
# sb.countplot(data=dataset, x='Loan_Status',ax=axes[2,2]).set(title='Loan_Status')
#
# plt.show()
dataset['Gender'].replace(np.nan,'Male', inplace=True)
dataset['Married'].replace(np.nan,'Yes', inplace=True)
dataset['Dependents'].replace(np.nan,'0', inplace=True)
dataset['Dependents']=dataset['Dependents'].str.replace('3+','3')
dataset['Self_Employed'].replace(np.nan,'No', inplace=True)
dataset['LoanAmount'].replace(np.nan,'146', inplace=True)
dataset['Loan_Amount_Term'].replace(np.nan,'342', inplace=True)
dataset['Credit_History'].replace(np.nan,'0', inplace=True)

# print(dataset.isnull().sum())
# print(dataset.shape)
dataset.drop(['Loan_ID'], axis=1,inplace=True)
dataset.drop(['Loan_Status'], axis=1,inplace=True)
# print(dataset.shape)
dataset['Gender']=dataset['Gender'].map({'Male':1,'Female':2})
dataset['Married']=dataset['Married'].map({'No':0,'Yes':1})
dataset['Education']=dataset['Education'].map({'Not Graduate':0,'Graduate':1})
dataset['Self_Employed']=dataset['Self_Employed'].map({'No':0,'Yes':1})
dataset['Property_Area']=dataset['Property_Area'].map({'Urban':1,'Rural':1, 'Semiurban':0.5})

print(dataset.corr())

# print(dataset['Gender'].unique())
# print(dataset['Married'].unique())
# print(dataset['Dependents'].unique())
# print(dataset['Education'].unique())
# print(dataset['Self_Employed'].unique())
sb.heatmap(data=dataset.corr(), cmap="YlGnBu", annot=True)
plt.show()
dataset.drop(['Loan_Amount_Term'], axis=1,inplace=True)
dataset.drop(['Credit_History'], axis=1,inplace=True)
dataset.drop(['Property_Area'], axis=1,inplace=True)
#print(dataset.keys())
# Index(['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
#        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount'],
#       dtype='object')
Y=dataset.loc[:,'LoanAmount']
dataset.drop(['LoanAmount'], axis=1,inplace=True)
X_train, X_test, y_train, y_test = train_test_split(dataset, Y, test_size=0.2, random_state=42)
regr = linear_model.LinearRegression()
regr.fit(X_train,y_train)
print(regr.coef_)
print(regr.intercept_)
Predict=regr.predict(X_test)
print(f"MSE: {mean_squared_error(Predict,y_test)}")
# plt.scatter(Predict,y_test)
# plt.plot(Predict,y_test)
# plt.show()
loan_amount_input = np.array([[1, 0, 0, 1, 1, 55000, 40000]])
loan_amount = regr.predict(loan_amount_input)
print(loan_amount)