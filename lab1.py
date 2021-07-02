
#------Output-----#
#Pakek terminal:  #
#python lab1.py   #
#-----------------#

#Import Data Set
import pandas as pd
path = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/auto.csv"
DataFrame = pd.read_csv(path,header=None)
print(DataFrame.head(20))

#Membuat headers list:
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
print("headers\n",headers)
DataFrame.columns=headers
print(DataFrame.head(10))
import numpy as np
DataFrame.replace("?",np.nan, inplace=True)

print(DataFrame)
#Menghapus baris dari kolom atribut yang memiliki nilai nol
DataFrame.dropna(subset=["price"],axis=0,inplace=True)
print("DataFrame without NaN",DataFrame)

#Mengetahui tipe data dari atribut:
print(DataFrame.dtypes)

#Melihat karakteristik statistik dari data numerik:
print(DataFrame.describe())

#Melihat seluruh karakteristik dari data numerik dan string:
print(DataFrame.describe(include="all"))

#Melihat ringkasan dari dataset kita
print(DataFrame.info)


#Meng-Eksport Dataset
path=("dataset.csv")
DataFrame.to_csv(path)

