
#------Output-----#
#Pakek terminal:  #
#python lab1.py   #
#-----------------#

#Import Data Set
import pandas as pd
import numpy as np
path = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/auto.csv"
DataFrame = pd.read_csv(path,header=None)
print(DataFrame.head(20))

headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
print("headers\n",headers)
DataFrame.columns=headers


DataFrame["city-mpg"] = 235/DataFrame["city-mpg"]

DataFrame.rename(columns={"city-mpg": "city-L/100km"}, inplace=True)

print(DataFrame.head(10))
print(DataFrame["price"].tail(5))
DataFrame.replace("?",np.nan, inplace=True)
DataFrame.dropna(subset=["price"],axis=0,inplace=True)
print(DataFrame["price"] )
#Untuk convert data types gunakan DataFrame.astype()

DataFrame["price"] = DataFrame["price"].astype('int')


#NORMALISASI DATA:
#1. Metode simple feature scaling:
print("sebelum normaliasi\n",DataFrame["length"])
DataFrame["length"]=DataFrame["length"]/DataFrame["length"].max()
print("setelah normalisasi metode simple feature scaling\n",DataFrame["length"])

#2. Metode MinMax:
DataFrame["length"]=(DataFrame["length"]-DataFrame["length"].min())/(DataFrame["length"].max()-DataFrame["length"].min())
print("setelah normalisasi metode MinMax\n",DataFrame["length"])

#3. Metode Z-score:
DataFrame["length"]=(DataFrame["length"]-DataFrame["length"].mean())/DataFrame["length"].std()
print("setelah normalisasi metode Z-Score\n",DataFrame["length"])

#BINNING PYTHON
import numpy as np
bins = np.linspace(min(DataFrame["price"]),max(DataFrame["price"]),4)
group_names =["low","mid","high"]
DataFrame["price-binned"] = pd.cut(DataFrame["price"],bins,labels=group_names,include_lowest=True)

import matplotlib.pyplot as plt
plt.show()
#Metode ONE-HOT-ENCODING:
variable_dummies = pd.get_dummies(DataFrame["fuel-type"])

print(variable_dummies)

#Menenpatkan data baru hasil metode one hot encoding kedalam dataet:
#Merge Data:
DataFrame = pd.concat([DataFrame,variable_dummies],axis=1)
DataFrame.drop("fuel-type",axis=1,inplace=True)
print(DataFrame)
print(DataFrame.columns)

#export dataset:
DataFrame.to_csv('Siska.csv')
print(DataFrame[["price","price-binned"]])
plt.bar(group_names,DataFrame["price-binned"].value_counts())
plt.show()
