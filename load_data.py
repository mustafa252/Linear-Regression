
# libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
''' use another environment becuase of pyside '''
import matplotlib
matplotlib.use('tkagg')
# load data
from sklearn.datasets import  load_boston
# split
from sklearn.model_selection import train_test_split
# linear regression
from sklearn.linear_model import LinearRegression
# metrics
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score


# load boston
boston = load_boston()

# target values
print(boston.target)
# data values
print(boston.data)

print('........................................ shape of x,y', end='\n')
# target shape
print('target shape: ', boston.target.shape)
# data shape
print('data shape: ',boston.data.shape)


print('........................................ description', end='\n')

print(boston.DESCR)

print('........................................ Data & Target', end='\n')

# get data
Data = boston.data
# get features
Features = boston.feature_names
# get target
Target = boston.target

print(Data.shape, end='\n')
print(Target.shape, end='\n')
print(Features)

# make dataFrame
dataFrame = pd.DataFrame(data=Data, columns=Features)

# import target to dataFrame as new column
dataFrame['Price'] = Target

# show price
print(dataFrame['Price'])

print('........................................ Data info', end='\n')
# show info
print(dataFrame.describe())
print(dataFrame.info())
print(dataFrame.isnull().sum())

######################################################################################################################
########## Train & test model

print('........................................ Train & Test ', end='\n')

# get X
X = dataFrame.drop(['Price'], axis=1)
# get Y
Y = dataFrame['Price']

# split train & test set (20%)
x_train, x_test, y_train, y_test = train_test_split(X,Y,
                                                    test_size=0.2,
                                                    random_state=0)

# check shape
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# create model
model = LinearRegression()

# train model
model.fit(x_train, y_train)

# make prediction
y_predict = model.predict(x_test)

# show predicted values
print(y_predict)

######################################################################################################################
########## Evaluate model

print('........................................ Evaluate ', end='\n')

# r2
print(r2_score(y_test, y_predict))
# MAE
print(mean_absolute_error(y_test, y_predict))
# MSE
print(mean_squared_error(y_test, y_predict))
# RMSE
print(np.sqrt(mean_squared_error(y_test, y_predict)))

######################################################################################################################
########## PLOT

print('........................................ PLOT ', end='\n')

plt.subplots(figsize=(10,5))
x_points = list(range(len(y_test)))
plt.plot(x_points, y_test, label='y_true')
plt.plot(x_points, y_predict, label='y_predict')
plt.legend()
plt.show()