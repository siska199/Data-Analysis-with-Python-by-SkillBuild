"""
We will learn about:
1. Simple linear regression
2. Multiple Linear Regression
3. Polynomial Regression
"""

#-------------------------------------------------#
#     Simple and Multiple Linear Regression       #
#-------------------------------------------------#

#Simple Linear Regression: y = a+bx

from sklearn.linear_model import LinearRegression
import pandas as pd

path = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv'

DataFrame = pd.read_csv(path)

X = DataFrame[["highway-mpg"]]
Y = DataFrame[["price"]]

model = LinearRegression()
model.fit(X,Y) #untuk memperoleh a dan b
a = model.intercept_
b = model.coef_
print("konstanta a: ",a)
print("konstanta b:",b)
print("Jadi hubungan antara highway-mpg dan price diberikan oleh persamaan\n:","y= {} + {}x".format(a[0],b[0][0]))

Y_predict = model.predict(X)
print("Hasil Prediksi:",Y_predict)


import seaborn as sns
import matplotlib.pyplot as plt
sns.regplot(x="highway-mpg",y="price",data=DataFrame)
plt.show()

#Residual plot akan menmbuat kita mengetahui model kita benar atau tidak
sns.residplot(DataFrame["highway-mpg"],DataFrame["price"])
plt.show()


#Multiple Linear Regression: y = b0 + b1 x1 + b2 x2+....
Z = DataFrame[["horsepower","curb-weight","engine-size","highway-mpg"]]


DataFrame.dropna(subset=["horsepower"],axis=0,inplace=True)
print(DataFrame["horsepower"].isnull().value_counts())

DataFrame.dropna(subset=["curb-weight"],axis=0,inplace=True)
print(DataFrame["curb-weight"].isnull().value_counts())

DataFrame.dropna(subset=["engine-size"],axis=0,inplace=True)
print(DataFrame["engine-size"].isnull().value_counts())

DataFrame.dropna(subset=["highway-mpg"],axis=0,inplace=True)
print(DataFrame["highway-mpg"].isnull().value_counts())

model.fit(Z,Y)
print(model.intercept_)
print(model.coef_)

Y_predict_multi = model.predict(Z)

#Residual Plot
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.residplot(DataFrame['highway-mpg'], DataFrame['price'])
plt.show()

#Distribution Plot: menngetahui perbedaan antara predict dan actual value:
plt.figure(figsize=(width, height))
ax1 = sns.distplot(DataFrame['price'], hist=False, color="r", label="Actual Value")
sns.distplot(Y_predict_multi, hist=False, color="b", label="Fitted Values" , ax=ax1)


plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')

plt.show()


#Polynomial Regression