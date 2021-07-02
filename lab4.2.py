from sklearn.linear_model import LinearRegression
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

path = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv'

DataFrame = pd.read_csv(path)
Y = DataFrame["price"]
X = DataFrame["highway-mpg"]
print(X.max(),X.min())
#Polinomial Regression:

import numpy as np
f = np.polyfit(X,Y,6)
p = np.poly1d(f)
print(p)

#Dummy variabel untuk x:
x_dum = np.linspace(X.min(),X.max(),100)
y_pred = [p(x1) for x1 in x_dum]

plt.plot(x_dum,y_pred)
plt.scatter(X,Y)
plt.show()

#Polynomial Regresion with More the One Dimenion: y = b0+b1 x1+b2x2+b3x1x2+b4(x1)^2+b5(x2)^2+...
from sklearn.preprocessing import PolynomialFeatures
Z = DataFrame[["horsepower","curb-weight","engine-size","highway-mpg"]]

model_poly = PolynomialFeatures(degree=2, include_bias=False)
Z_pr = model_poly.fit_transform(Z)

print(Z.shape) #The original data i of 201 samples and 4 features
print(Z_pr.shape)

#Using Pipeline to simplify our process
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

input = [('scale',StandardScaler()),('polynomial',PolynomialFeatures(include_bias=False)),('model',LinearRegression())]
pipe = Pipeline(input)
pipe.fit(Z,Y)
Y_predict_simple_fit = pipe.predict(Z)

"""
Menentukan keakuratan hasil prediksi yang telah dilakukanmenggunakan 2 metoe:
1. Mean Squared Error (MSE)
2. R-square (R^2)
"""

#Metode MSE:
from sklearn.metrics import mean_squared_error
print("MSE: ",mean_squared_error(DataFrame["price"],Y_predict_simple_fit))

#Metode R-squared/ R^2:
""" 
Comparing a regression model to a simple model i.e the mean of the data points
we find the R-squared value in Python bu using the score() method, in the linear regression
"""
X = DataFrame[["highway-mpg"]]
model= LinearRegression()
print(model.fit(X,Y))
print(model.score(X,Y))

print(pipe.score(Z,Y))

#--------------------------------#
# Prediction and Decision Making #
#--------------------------------#