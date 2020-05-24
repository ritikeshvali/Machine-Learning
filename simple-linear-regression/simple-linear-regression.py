import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os,csv

X_train = pd.read_csv('Linear_X_Train.csv').values
X_test = pd.read_csv('Linear_X_Test.csv').values
Y_train = pd.read_csv('Linear_Y_Train.csv').values

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, Y_train)

Y_test = lr.predict(X_test)

path = r'/home/ironrobot10/Documents/ml-projects/simple-linear-regression/Linear_Y_Test.csv'

df = pd.DataFrame(Y_test)
df.to_csv(path, index = False, float_format = '%g')

print(Y_test)
# plt.scatter(X_train, Y_train, color = 'red')
# plt.plot(X_train, lr.predict(X_test), color = 'blue')
# plt.title('Working Time vs Score (Train)')
# plt.xlabel('Working Time')
# plt.ylabel('Score')
# plt.show()